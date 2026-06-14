"""
QUANTUM-FORGE Data Alignment
Precision data alignment for multi-source financial time-series

Features:
- Sub-microsecond timestamp alignment
- Forward-fill, backward-fill, and interpolation methods
- Multi-source data synchronization
- Market hours awareness
- Asynchronous data handling
- Corporate actions alignment
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dt_time
import pytz
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import bisect
import warnings

logger = logging.getLogger(__name__)

class AlignmentMethod(Enum):
    """Data alignment methods"""
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"
    LINEAR_INTERPOLATION = "linear"
    NEAREST_NEIGHBOR = "nearest"
    CUBIC_SPLINE = "cubic"
    TIME_WEIGHTED = "time_weighted"
    LAST_VALID = "last_valid"

class MarketSession(Enum):
    """Market session types"""
    PRE_MARKET = "pre_market"
    MARKET_HOURS = "market_hours"
    AFTER_HOURS = "after_hours"
    EXTENDED_HOURS = "extended_hours"
    TWENTY_FOUR_SEVEN = "24_7"

@dataclass
class MarketHours:
    """Market hours configuration"""
    timezone: str
    market_open: dt_time
    market_close: dt_time
    pre_market_start: Optional[dt_time] = None
    after_hours_end: Optional[dt_time] = None
    
    def is_market_hours(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within market hours"""
        local_time = timestamp.tz_convert(self.timezone).time()
        return self.market_open <= local_time <= self.market_close

@dataclass
class AlignmentConfig:
    """Configuration for data alignment"""
    method: AlignmentMethod = AlignmentMethod.FORWARD_FILL
    tolerance: Optional[pd.Timedelta] = None  # Max time gap to fill
    market_hours: Optional[MarketHours] = None
    fill_weekend: bool = False
    fill_holidays: bool = False
    max_gap_fill: Optional[pd.Timedelta] = None

class DataAligner:
    """
    Production-grade data alignment for financial time-series
    
    Handles:
    - Multi-source data alignment with different frequencies
    - Market hours awareness
    - Corporate actions alignment
    - High-frequency data synchronization
    - Cross-asset alignment
    """
    
    def __init__(self,
                 default_method: AlignmentMethod = AlignmentMethod.FORWARD_FILL,
                 precision: str = "1ms",  # Timestamp precision
                 market_aware: bool = True):
        
        self.default_method = default_method
        self.precision = precision
        self.market_aware = market_aware
        
        # Market hours configurations
        self.market_configs = self._init_market_configs()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Data aligner initialized with precision: {precision}")
    
    def _init_market_configs(self) -> Dict[str, MarketHours]:
        """Initialize common market hour configurations"""
        
        configs = {
            'NYSE': MarketHours(
                timezone='America/New_York',
                market_open=dt_time(9, 30),
                market_close=dt_time(16, 0),
                pre_market_start=dt_time(4, 0),
                after_hours_end=dt_time(20, 0)
            ),
            'NASDAQ': MarketHours(
                timezone='America/New_York',
                market_open=dt_time(9, 30),
                market_close=dt_time(16, 0),
                pre_market_start=dt_time(4, 0),
                after_hours_end=dt_time(20, 0)
            ),
            'LSE': MarketHours(
                timezone='Europe/London',
                market_open=dt_time(8, 0),
                market_close=dt_time(16, 30)
            ),
            'TSE': MarketHours(
                timezone='Asia/Tokyo',
                market_open=dt_time(9, 0),
                market_close=dt_time(15, 0)
            ),
            'CRYPTO': MarketHours(
                timezone='UTC',
                market_open=dt_time(0, 0),
                market_close=dt_time(23, 59)
            )
        }
        
        return configs
    
    # ==================== MAIN ALIGNMENT METHODS ====================
    
    def align_dataframes(self,
                        dataframes: Dict[str, pd.DataFrame],
                        timestamp_col: str = 'timestamp',
                        method: Optional[AlignmentMethod] = None,
                        config: Optional[AlignmentConfig] = None) -> pd.DataFrame:
        """
        Align multiple DataFrames on timestamp
        
        Args:
            dataframes: Dict of {source_name: DataFrame}
            timestamp_col: Name of timestamp column
            method: Alignment method
            config: Alignment configuration
        
        Returns:
            Aligned DataFrame with multi-level columns
        """
        
        if not dataframes:
            raise ValueError("No dataframes provided")
        
        if method is None:
            method = self.default_method
        
        if config is None:
            config = AlignmentConfig(method=method)
        
        # Extract timestamps from all sources
        all_timestamps = set()
        
        for source_name, df in dataframes.items():
            if timestamp_col not in df.columns:
                raise ValueError(f"Timestamp column '{timestamp_col}' not found in {source_name}")
            
            timestamps = pd.to_datetime(df[timestamp_col])
            all_timestamps.update(timestamps)
        
        # Create master timeline
        master_timeline = pd.Series(sorted(all_timestamps))
        master_timeline = self._filter_timeline_by_market_hours(master_timeline, config)
        
        # Align each DataFrame to master timeline
        aligned_dfs = {}
        
        for source_name, df in dataframes.items():
            logger.info(f"Aligning {source_name}: {len(df)} → {len(master_timeline)} rows")
            
            aligned_df = self._align_single_dataframe(
                df, master_timeline, timestamp_col, config
            )
            
            # Add source prefix to columns
            aligned_df.columns = [f"{source_name}_{col}" if col != timestamp_col else col 
                                for col in aligned_df.columns]
            
            aligned_dfs[source_name] = aligned_df
        
        # Merge all aligned DataFrames
        result_df = aligned_dfs[list(aligned_dfs.keys())[0]]
        
        for source_name in list(aligned_dfs.keys())[1:]:
            result_df = result_df.merge(
                aligned_dfs[source_name],
                on=timestamp_col,
                how='outer'
            )
        
        # Sort by timestamp
        result_df = result_df.sort_values(timestamp_col).reset_index(drop=True)
        
        logger.info(f"Final aligned data: {len(result_df)} rows")
        return result_df
    
    def _align_single_dataframe(self,
                               df: pd.DataFrame,
                               target_timeline: pd.Series,
                               timestamp_col: str,
                               config: AlignmentConfig) -> pd.DataFrame:
        """Align single DataFrame to target timeline"""
        
        # Ensure timestamp is datetime
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=[timestamp_col]).sort_values(timestamp_col)
        
        # Create target DataFrame
        target_df = pd.DataFrame({timestamp_col: target_timeline})
        
        # Merge with target timeline
        merged_df = target_df.merge(df, on=timestamp_col, how='left')
        
        # Apply alignment method
        if config.method == AlignmentMethod.FORWARD_FILL:
            merged_df = self._forward_fill(merged_df, timestamp_col, config)
        
        elif config.method == AlignmentMethod.BACKWARD_FILL:
            merged_df = self._backward_fill(merged_df, timestamp_col, config)
        
        elif config.method == AlignmentMethod.LINEAR_INTERPOLATION:
            merged_df = self._linear_interpolate(merged_df, timestamp_col, config)
        
        elif config.method == AlignmentMethod.NEAREST_NEIGHBOR:
            merged_df = self._nearest_neighbor(merged_df, df, timestamp_col, config)
        
        elif config.method == AlignmentMethod.TIME_WEIGHTED:
            merged_df = self._time_weighted_interpolate(merged_df, timestamp_col, config)
        
        elif config.method == AlignmentMethod.LAST_VALID:
            merged_df = self._last_valid_observation(merged_df, timestamp_col, config)
        
        else:
            raise ValueError(f"Unknown alignment method: {config.method}")
        
        return merged_df
    
    # ==================== ALIGNMENT METHOD IMPLEMENTATIONS ====================
    
    def _forward_fill(self,
                     df: pd.DataFrame,
                     timestamp_col: str,
                     config: AlignmentConfig) -> pd.DataFrame:
        """Forward fill missing values"""
        
        result_df = df.copy()
        
        # Get non-timestamp columns
        value_cols = [col for col in df.columns if col != timestamp_col]
        
        # Forward fill each column
        for col in value_cols:
            if config.tolerance:
                # Time-aware forward fill
                result_df[col] = self._time_aware_ffill(
                    result_df, col, timestamp_col, config.tolerance
                )
            else:
                # Simple forward fill
                result_df[col] = result_df[col].fillna(method='ffill')
        
        return result_df
    
    def _backward_fill(self,
                      df: pd.DataFrame,
                      timestamp_col: str,
                      config: AlignmentConfig) -> pd.DataFrame:
        """Backward fill missing values"""
        
        result_df = df.copy()
        value_cols = [col for col in df.columns if col != timestamp_col]
        
        for col in value_cols:
            if config.tolerance:
                # Time-aware backward fill
                result_df[col] = self._time_aware_bfill(
                    result_df, col, timestamp_col, config.tolerance
                )
            else:
                # Simple backward fill
                result_df[col] = result_df[col].fillna(method='bfill')
        
        return result_df
    
    def _linear_interpolate(self,
                           df: pd.DataFrame,
                           timestamp_col: str,
                           config: AlignmentConfig) -> pd.DataFrame:
        """Linear interpolation based on time"""
        
        result_df = df.copy()
        value_cols = [col for col in df.columns if col != timestamp_col]
        
        # Convert timestamp to numeric for interpolation
        numeric_time = result_df[timestamp_col].astype(np.int64)
        
        for col in value_cols:
            if result_df[col].dtype in [np.float64, np.int64]:
                # Create Series with numeric time index
                series = pd.Series(
                    result_df[col].values,
                    index=numeric_time
                )
                
                # Interpolate
                interpolated = series.interpolate(method='linear')
                result_df[col] = interpolated.values
        
        return result_df
    
    def _nearest_neighbor(self,
                         target_df: pd.DataFrame,
                         source_df: pd.DataFrame,
                         timestamp_col: str,
                         config: AlignmentConfig) -> pd.DataFrame:
        """Nearest neighbor alignment"""
        
        result_df = target_df.copy()
        value_cols = [col for col in source_df.columns if col != timestamp_col]
        
        source_timestamps = source_df[timestamp_col].values
        target_timestamps = target_df[timestamp_col].values
        
        for col in value_cols:
            source_values = source_df[col].values
            aligned_values = np.full(len(target_timestamps), np.nan)
            
            for i, target_ts in enumerate(target_timestamps):
                # Find nearest timestamp
                if config.tolerance:
                    # Find within tolerance
                    time_diffs = np.abs(source_timestamps - target_ts)
                    min_idx = np.argmin(time_diffs)
                    
                    if pd.Timedelta(time_diffs[min_idx]) <= config.tolerance:
                        aligned_values[i] = source_values[min_idx]
                else:
                    # Always find nearest
                    nearest_idx = np.argmin(np.abs(source_timestamps - target_ts))
                    aligned_values[i] = source_values[nearest_idx]
            
            result_df[col] = aligned_values
        
        return result_df
    
    def _time_weighted_interpolate(self,
                                  df: pd.DataFrame,
                                  timestamp_col: str,
                                  config: AlignmentConfig) -> pd.DataFrame:
        """Time-weighted interpolation"""
        
        result_df = df.copy()
        value_cols = [col for col in df.columns if col != timestamp_col]
        
        timestamps = result_df[timestamp_col]
        
        for col in value_cols:
            values = result_df[col]
            
            # Find valid (non-NaN) values
            valid_mask = values.notna()
            
            if valid_mask.sum() < 2:
                continue  # Need at least 2 points for interpolation
            
            valid_times = timestamps[valid_mask]
            valid_values = values[valid_mask]
            
            # Interpolate for missing values
            for i, ts in enumerate(timestamps):
                if pd.isna(values.iloc[i]):
                    # Find surrounding valid points
                    before_mask = valid_times <= ts
                    after_mask = valid_times >= ts
                    
                    if before_mask.any() and after_mask.any():
                        # Get closest before and after
                        before_idx = before_mask.idxmax() if before_mask.any() else None
                        after_idx = after_mask.idxmin() if after_mask.any() else None
                        
                        if before_idx is not None and after_idx is not None and before_idx != after_idx:
                            # Time-weighted interpolation
                            t1, v1 = valid_times.loc[before_idx], valid_values.loc[before_idx]
                            t2, v2 = valid_times.loc[after_idx], valid_values.loc[after_idx]
                            
                            # Calculate weights based on time distance
                            total_time = (t2 - t1).total_seconds()
                            if total_time > 0:
                                weight = (ts - t1).total_seconds() / total_time
                                interpolated_value = v1 + weight * (v2 - v1)
                                result_df.loc[i, col] = interpolated_value
        
        return result_df
    
    def _last_valid_observation(self,
                               df: pd.DataFrame,
                               timestamp_col: str,
                               config: AlignmentConfig) -> pd.DataFrame:
        """Last valid observation carried forward"""
        
        return self._forward_fill(df, timestamp_col, config)
    
    def _time_aware_ffill(self,
                         df: pd.DataFrame,
                         col: str,
                         timestamp_col: str,
                         tolerance: pd.Timedelta) -> pd.Series:
        """Forward fill with time tolerance"""
        
        result = df[col].copy()
        timestamps = df[timestamp_col]
        
        last_valid_value = None
        last_valid_time = None
        
        for i in range(len(result)):
            current_time = timestamps.iloc[i]
            current_value = result.iloc[i]
            
            if pd.notna(current_value):
                last_valid_value = current_value
                last_valid_time = current_time
            elif last_valid_value is not None and last_valid_time is not None:
                # Check if within tolerance
                time_diff = current_time - last_valid_time
                if time_diff <= tolerance:
                    result.iloc[i] = last_valid_value
        
        return result
    
    def _time_aware_bfill(self,
                         df: pd.DataFrame,
                         col: str,
                         timestamp_col: str,
                         tolerance: pd.Timedelta) -> pd.Series:
        """Backward fill with time tolerance"""
        
        result = df[col].copy()
        timestamps = df[timestamp_col]
        
        # Process in reverse order
        next_valid_value = None
        next_valid_time = None
        
        for i in range(len(result) - 1, -1, -1):
            current_time = timestamps.iloc[i]
            current_value = result.iloc[i]
            
            if pd.notna(current_value):
                next_valid_value = current_value
                next_valid_time = current_time
            elif next_valid_value is not None and next_valid_time is not None:
                # Check if within tolerance
                time_diff = next_valid_time - current_time
                if time_diff <= tolerance:
                    result.iloc[i] = next_valid_value
        
        return result
    
    # ==================== MARKET HOURS FILTERING ====================
    
    def _filter_timeline_by_market_hours(self,
                                        timeline: pd.Series,
                                        config: AlignmentConfig) -> pd.Series:
        """Filter timeline based on market hours"""
        
        if not self.market_aware or not config.market_hours:
            return timeline
        
        market_config = config.market_hours
        
        # Convert to market timezone
        timeline_tz = timeline.dt.tz_convert(market_config.timezone)
        
        # Filter based on market hours
        if not config.fill_weekend:
            # Remove weekends
            timeline_tz = timeline_tz[timeline_tz.dt.dayofweek < 5]
        
        # Filter by market hours
        market_mask = (
            (timeline_tz.dt.time >= market_config.market_open) &
            (timeline_tz.dt.time <= market_config.market_close)
        )
        
        if market_config.pre_market_start and market_config.after_hours_end:
            # Include extended hours
            extended_mask = (
                (timeline_tz.dt.time >= market_config.pre_market_start) &
                (timeline_tz.dt.time <= market_config.after_hours_end)
            )
            market_mask = market_mask | extended_mask
        
        return timeline[market_mask]
    
    # ==================== HIGH-FREQUENCY ALIGNMENT ====================
    
    def align_tick_data(self,
                       tick_dfs: Dict[str, pd.DataFrame],
                       timestamp_col: str = 'timestamp',
                       precision: str = '1ms') -> pd.DataFrame:
        """Align high-frequency tick data with microsecond precision"""
        
        # Round timestamps to specified precision
        aligned_dfs = {}
        
        for source, df in tick_dfs.items():
            df_copy = df.copy()
            df_copy[timestamp_col] = df_copy[timestamp_col].dt.round(precision)
            aligned_dfs[source] = df_copy
        
        # Use regular alignment with high precision
        config = AlignmentConfig(
            method=AlignmentMethod.LAST_VALID,
            tolerance=pd.Timedelta(precision)
        )
        
        return self.align_dataframes(aligned_dfs, timestamp_col, config=config)
    
    # ==================== CORPORATE ACTIONS ALIGNMENT ====================
    
    def align_with_corporate_actions(self,
                                    price_df: pd.DataFrame,
                                    corporate_actions: pd.DataFrame,
                                    timestamp_col: str = 'timestamp',
                                    price_col: str = 'price') -> pd.DataFrame:
        """Align price data with corporate actions (splits, dividends)"""
        
        result_df = price_df.copy()
        
        # Sort corporate actions by date
        ca_sorted = corporate_actions.sort_values(timestamp_col)
        
        for _, action in ca_sorted.iterrows():
            action_date = action[timestamp_col]
            action_type = action.get('type', 'split')
            adjustment_factor = action.get('factor', 1.0)
            
            # Apply adjustment to prices before the action date
            mask = result_df[timestamp_col] < action_date
            
            if action_type == 'split':
                # Stock split: adjust prices and volumes
                result_df.loc[mask, price_col] /= adjustment_factor
                if 'volume' in result_df.columns:
                    result_df.loc[mask, 'volume'] *= adjustment_factor
            
            elif action_type == 'dividend':
                # Dividend: adjust prices
                result_df.loc[mask, price_col] -= adjustment_factor
        
        return result_df
    
    # ==================== CROSS-ASSET ALIGNMENT ====================
    
    def align_cross_asset(self,
                         asset_dfs: Dict[str, pd.DataFrame],
                         reference_asset: str,
                         timestamp_col: str = 'timestamp',
                         method: AlignmentMethod = AlignmentMethod.FORWARD_FILL) -> pd.DataFrame:
        """Align multiple assets using one as reference timeline"""
        
        if reference_asset not in asset_dfs:
            raise ValueError(f"Reference asset '{reference_asset}' not found")
        
        # Use reference asset timeline
        reference_timeline = asset_dfs[reference_asset][timestamp_col]
        
        aligned_dfs = {}
        
        for asset_name, df in asset_dfs.items():
            config = AlignmentConfig(method=method)
            
            aligned_df = self._align_single_dataframe(
                df, reference_timeline, timestamp_col, config
            )
            
            # Rename columns with asset prefix
            aligned_df.columns = [f"{asset_name}_{col}" if col != timestamp_col else col 
                                for col in aligned_df.columns]
            
            aligned_dfs[asset_name] = aligned_df
        
        # Merge all assets
        result_df = aligned_dfs[reference_asset]
        
        for asset_name in asset_dfs.keys():
            if asset_name != reference_asset:
                result_df = result_df.merge(
                    aligned_dfs[asset_name],
                    on=timestamp_col,
                    how='left'
                )
        
        return result_df
    
    # ==================== QUALITY CHECKS ====================
    
    def validate_alignment(self,
                          aligned_df: pd.DataFrame,
                          timestamp_col: str = 'timestamp') -> Dict[str, Any]:
        """Validate alignment quality"""
        
        if timestamp_col not in aligned_df.columns:
            return {"error": f"Timestamp column '{timestamp_col}' not found"}
        
        timestamps = aligned_df[timestamp_col]
        
        # Check for duplicates
        duplicate_count = timestamps.duplicated().sum()
        
        # Check for gaps
        time_diffs = timestamps.diff().dropna()
        
        # Check for sorting
        is_sorted = timestamps.is_monotonic_increasing
        
        # Missing data analysis
        missing_data = aligned_df.isnull().sum()
        total_missing = missing_data.sum()
        
        validation_report = {
            "total_rows": len(aligned_df),
            "duplicate_timestamps": duplicate_count,
            "is_chronologically_sorted": is_sorted,
            "time_gaps": {
                "min_gap_seconds": time_diffs.min().total_seconds() if not time_diffs.empty else 0,
                "max_gap_seconds": time_diffs.max().total_seconds() if not time_diffs.empty else 0,
                "median_gap_seconds": time_diffs.median().total_seconds() if not time_diffs.empty else 0
            },
            "missing_data": {
                "total_missing_values": int(total_missing),
                "missing_percentage": float(total_missing / (len(aligned_df) * len(aligned_df.columns)) * 100),
                "columns_with_missing": missing_data[missing_data > 0].to_dict()
            },
            "timestamp_range": {
                "start": timestamps.min().isoformat() if not timestamps.empty else None,
                "end": timestamps.max().isoformat() if not timestamps.empty else None,
                "duration_hours": (timestamps.max() - timestamps.min()).total_seconds() / 3600 if len(timestamps) > 1 else 0
            }
        }
        
        return validation_report
    
    # ==================== UTILITIES ====================
    
    def get_market_config(self, exchange: str) -> Optional[MarketHours]:
        """Get market hours configuration for exchange"""
        return self.market_configs.get(exchange.upper())
    
    def add_market_config(self, exchange: str, config: MarketHours):
        """Add custom market hours configuration"""
        self.market_configs[exchange.upper()] = config
        logger.info(f"Added market config for {exchange}")
    
    def close(self):
        """Clean shutdown"""
        self.executor.shutdown(wait=True)
        logger.info("Data aligner shutdown complete")

# ==================== ALIGNMENT PIPELINE ====================

class AlignmentPipeline:
    """
    Pipeline for complex multi-step alignment
    """
    
    def __init__(self, aligner: DataAligner):
        self.aligner = aligner
        self.steps = []
    
    def add_step(self, name: str, func: Callable, **kwargs):
        """Add alignment step"""
        self.steps.append({
            'name': name,
            'func': func,
            'kwargs': kwargs
        })
        logger.info(f"Added alignment step: {name}")
    
    def execute(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Execute alignment pipeline"""
        
        result = data
        
        for step in self.steps:
            logger.info(f"Executing step: {step['name']}")
            
            if isinstance(result, dict):
                result = step['func'](result, **step['kwargs'])
            else:
                result = step['func'](result, **step['kwargs'])
        
        return result
