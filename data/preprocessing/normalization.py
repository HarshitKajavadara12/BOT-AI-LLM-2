"""
QUANTUM-FORGE Data Normalization
Advanced normalization techniques for financial time-series

Features:
- Multiple normalization methods (Z-score, Min-Max, Robust, Quantile)
- Cross-sectional and time-series normalization
- Regime-aware normalization
- Online/streaming normalization with drift adaptation
- Volatility-adjusted normalization
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import rankdata
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle
import warnings

logger = logging.getLogger(__name__)

class NormalizationMethod(Enum):
    """Supported normalization methods"""
    ZSCORE = "zscore"
    MINMAX = "minmax"
    ROBUST = "robust"  # Median and MAD
    QUANTILE = "quantile"
    RANK = "rank"
    LOG_RETURN = "log_return"
    VOLATILITY_ADJUSTED = "vol_adjusted"

@dataclass
class NormalizationParams:
    """Parameters for normalization"""
    method: NormalizationMethod
    window_size: Optional[int] = None  # For rolling normalization
    clip_outliers: bool = True
    outlier_threshold: float = 3.0
    center: bool = True
    online_decay: float = 0.99  # For online updates

class DataNormalizer:
    """
    Production-grade data normalizer for financial time-series
    
    Features:
    - Multiple normalization techniques
    - Cross-sectional (across assets) and time-series normalization
    - Online/streaming normalization
    - Regime-aware normalization
    - Volatility adjustment
    """
    
    def __init__(self,
                 default_method: NormalizationMethod = NormalizationMethod.ZSCORE,
                 cross_sectional: bool = True,
                 preserve_magnitude: bool = False):
        
        self.default_method = default_method
        self.cross_sectional = cross_sectional
        self.preserve_magnitude = preserve_magnitude
        
        # Store normalization statistics for inverse transform
        self.normalization_stats = {}
        
        # Online statistics for streaming
        self.online_stats = {}
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Data normalizer initialized with method: {default_method.value}")
    
    # ==================== MAIN NORMALIZATION METHODS ====================
    
    def normalize(self, 
                  data: Union[pd.DataFrame, pd.Series, np.ndarray],
                  method: Optional[NormalizationMethod] = None,
                  params: Optional[NormalizationParams] = None,
                  group_by: Optional[str] = None) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Normalize data using specified method
        
        Args:
            data: Input data (DataFrame, Series, or array)
            method: Normalization method
            params: Normalization parameters
            group_by: Column to group by (for cross-sectional normalization)
        """
        
        if method is None:
            method = self.default_method
        
        if params is None:
            params = NormalizationParams(method=method)
        
        # Handle different input types
        if isinstance(data, pd.DataFrame):
            return self._normalize_dataframe(data, params, group_by)
        elif isinstance(data, pd.Series):
            return self._normalize_series(data, params)
        elif isinstance(data, np.ndarray):
            return self._normalize_array(data, params)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _normalize_dataframe(self, 
                            df: pd.DataFrame,
                            params: NormalizationParams,
                            group_by: Optional[str] = None) -> pd.DataFrame:
        """Normalize DataFrame"""
        
        result_df = df.copy()
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            logger.warning("No numeric columns found for normalization")
            return result_df
        
        if group_by and group_by in df.columns:
            # Group-wise normalization (e.g., by timestamp for cross-sectional)
            for group_value, group_data in df.groupby(group_by):
                mask = df[group_by] == group_value
                
                for col in numeric_cols:
                    if col != group_by:
                        normalized_values = self._apply_normalization(
                            group_data[col].values, params, f"{group_value}_{col}"
                        )
                        result_df.loc[mask, col] = normalized_values
        
        else:
            # Column-wise normalization
            for col in numeric_cols:
                normalized_values = self._apply_normalization(
                    df[col].values, params, col
                )
                result_df[col] = normalized_values
        
        return result_df
    
    def _normalize_series(self, series: pd.Series, params: NormalizationParams) -> pd.Series:
        """Normalize pandas Series"""
        
        normalized_values = self._apply_normalization(
            series.values, params, series.name or "series"
        )
        
        return pd.Series(normalized_values, index=series.index, name=series.name)
    
    def _normalize_array(self, array: np.ndarray, params: NormalizationParams) -> np.ndarray:
        """Normalize numpy array"""
        
        return self._apply_normalization(array, params, "array")
    
    # ==================== NORMALIZATION IMPLEMENTATIONS ====================
    
    def _apply_normalization(self, 
                            values: np.ndarray,
                            params: NormalizationParams,
                            key: str) -> np.ndarray:
        """Apply specific normalization method"""
        
        # Remove NaN values for computation
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        
        if len(valid_values) == 0:
            return values
        
        # Apply normalization based on method
        if params.method == NormalizationMethod.ZSCORE:
            normalized_valid = self._zscore_normalize(valid_values, params, key)
        
        elif params.method == NormalizationMethod.MINMAX:
            normalized_valid = self._minmax_normalize(valid_values, params, key)
        
        elif params.method == NormalizationMethod.ROBUST:
            normalized_valid = self._robust_normalize(valid_values, params, key)
        
        elif params.method == NormalizationMethod.QUANTILE:
            normalized_valid = self._quantile_normalize(valid_values, params, key)
        
        elif params.method == NormalizationMethod.RANK:
            normalized_valid = self._rank_normalize(valid_values, params, key)
        
        elif params.method == NormalizationMethod.LOG_RETURN:
            normalized_valid = self._log_return_normalize(valid_values, params, key)
        
        elif params.method == NormalizationMethod.VOLATILITY_ADJUSTED:
            normalized_valid = self._volatility_adjusted_normalize(valid_values, params, key)
        
        else:
            raise ValueError(f"Unknown normalization method: {params.method}")
        
        # Handle outliers if requested
        if params.clip_outliers:
            normalized_valid = self._clip_outliers(normalized_valid, params.outlier_threshold)
        
        # Reconstruct full array
        result = values.copy().astype(float)
        result[valid_mask] = normalized_valid
        
        return result
    
    def _zscore_normalize(self, values: np.ndarray, params: NormalizationParams, key: str) -> np.ndarray:
        """Z-score normalization: (x - mean) / std"""
        
        if params.window_size:
            # Rolling Z-score
            return self._rolling_zscore(values, params.window_size)
        
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        # Store stats for inverse transform
        self.normalization_stats[key] = {
            'method': 'zscore',
            'mean': mean_val,
            'std': std_val
        }
        
        if std_val == 0:
            return np.zeros_like(values)
        
        return (values - mean_val) / std_val
    
    def _minmax_normalize(self, values: np.ndarray, params: NormalizationParams, key: str) -> np.ndarray:
        """Min-Max normalization: (x - min) / (max - min)"""
        
        min_val = np.min(values)
        max_val = np.max(values)
        
        # Store stats
        self.normalization_stats[key] = {
            'method': 'minmax',
            'min': min_val,
            'max': max_val
        }
        
        if max_val == min_val:
            return np.zeros_like(values)
        
        return (values - min_val) / (max_val - min_val)
    
    def _robust_normalize(self, values: np.ndarray, params: NormalizationParams, key: str) -> np.ndarray:
        """Robust normalization using median and MAD"""
        
        median_val = np.median(values)
        mad_val = np.median(np.abs(values - median_val))
        
        # Store stats
        self.normalization_stats[key] = {
            'method': 'robust',
            'median': median_val,
            'mad': mad_val
        }
        
        if mad_val == 0:
            return np.zeros_like(values)
        
        return (values - median_val) / mad_val
    
    def _quantile_normalize(self, values: np.ndarray, params: NormalizationParams, key: str) -> np.ndarray:
        """Quantile normalization (rank-based)"""
        
        # Sort values to get quantiles
        sorted_values = np.sort(values)
        
        # Store stats
        self.normalization_stats[key] = {
            'method': 'quantile',
            'sorted_values': sorted_values
        }
        
        # Map to uniform distribution [0, 1]
        ranks = rankdata(values, method='average')
        return (ranks - 1) / (len(values) - 1)
    
    def _rank_normalize(self, values: np.ndarray, params: NormalizationParams, key: str) -> np.ndarray:
        """Rank normalization to [-1, 1]"""
        
        ranks = rankdata(values, method='average')
        
        # Store stats
        self.normalization_stats[key] = {
            'method': 'rank',
            'n_values': len(values)
        }
        
        # Map to [-1, 1]
        return 2 * (ranks - 1) / (len(values) - 1) - 1
    
    def _log_return_normalize(self, values: np.ndarray, params: NormalizationParams, key: str) -> np.ndarray:
        """Log return normalization: log(x_t / x_{t-1})"""
        
        if len(values) < 2:
            return np.zeros_like(values)
        
        # Ensure positive values
        values = np.abs(values)
        values[values == 0] = 1e-8
        
        # Compute log returns
        log_returns = np.diff(np.log(values))
        
        # Pad with zero for first value
        result = np.zeros_like(values)
        result[1:] = log_returns
        
        # Store stats
        self.normalization_stats[key] = {
            'method': 'log_return',
            'first_value': values[0]
        }
        
        return result
    
    def _volatility_adjusted_normalize(self, values: np.ndarray, params: NormalizationParams, key: str) -> np.ndarray:
        """Volatility-adjusted normalization"""
        
        if len(values) < 10:
            return self._zscore_normalize(values, params, key)
        
        # Compute rolling volatility (20-period default)
        window = min(20, len(values) // 2)
        
        returns = np.diff(values) / values[:-1]
        volatility = pd.Series(returns).rolling(window).std().fillna(method='bfill').values
        
        # Pad volatility to match values length
        volatility = np.concatenate([[volatility[0]], volatility])
        
        # Avoid division by zero
        volatility[volatility == 0] = np.nanmedian(volatility[volatility > 0])
        
        # Store stats
        self.normalization_stats[key] = {
            'method': 'volatility_adjusted',
            'mean_vol': np.mean(volatility)
        }
        
        # Normalize by volatility
        return (values - np.mean(values)) / volatility
    
    def _rolling_zscore(self, values: np.ndarray, window_size: int) -> np.ndarray:
        """Rolling Z-score normalization"""
        
        series = pd.Series(values)
        
        rolling_mean = series.rolling(window_size, min_periods=1).mean()
        rolling_std = series.rolling(window_size, min_periods=1).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.fillna(1.0)
        rolling_std[rolling_std == 0] = 1.0
        
        return ((series - rolling_mean) / rolling_std).values
    
    def _clip_outliers(self, values: np.ndarray, threshold: float) -> np.ndarray:
        """Clip outliers beyond threshold standard deviations"""
        
        return np.clip(values, -threshold, threshold)
    
    # ==================== CROSS-SECTIONAL NORMALIZATION ====================
    
    def normalize_cross_sectional(self, 
                                 df: pd.DataFrame,
                                 timestamp_col: str = 'timestamp',
                                 value_cols: Optional[List[str]] = None,
                                 method: Optional[NormalizationMethod] = None) -> pd.DataFrame:
        """
        Cross-sectional normalization (across assets at each timestamp)
        
        This is crucial for factor models and relative value strategies
        """
        
        if method is None:
            method = self.default_method
        
        if value_cols is None:
            value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if timestamp_col in value_cols:
                value_cols.remove(timestamp_col)
        
        result_df = df.copy()
        
        # Group by timestamp and normalize across assets
        for timestamp, group in df.groupby(timestamp_col):
            for col in value_cols:
                if col in group.columns:
                    values = group[col].values
                    
                    if len(values) > 1:  # Need at least 2 values for normalization
                        params = NormalizationParams(method=method)
                        normalized_values = self._apply_normalization(
                            values, params, f"{col}_{timestamp}"
                        )
                        
                        # Update result
                        mask = df[timestamp_col] == timestamp
                        result_df.loc[mask, col] = normalized_values
        
        return result_df
    
    # ==================== ONLINE/STREAMING NORMALIZATION ====================
    
    def update_online_stats(self, 
                           key: str,
                           new_value: float,
                           method: NormalizationMethod = NormalizationMethod.ZSCORE,
                           decay: float = 0.99) -> float:
        """
        Update online normalization statistics and return normalized value
        
        Uses exponential decay for streaming data
        """
        
        if key not in self.online_stats:
            self.online_stats[key] = {
                'count': 0,
                'mean': 0.0,
                'var': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'method': method.value
            }
        
        stats = self.online_stats[key]
        
        # Update statistics using exponential moving average
        if stats['count'] == 0:
            stats['mean'] = new_value
            stats['var'] = 0.0
            stats['min'] = new_value
            stats['max'] = new_value
        else:
            # Update mean
            old_mean = stats['mean']
            stats['mean'] = decay * stats['mean'] + (1 - decay) * new_value
            
            # Update variance (Welford's online algorithm with decay)
            stats['var'] = decay * stats['var'] + (1 - decay) * (new_value - old_mean) * (new_value - stats['mean'])
            
            # Update min/max
            stats['min'] = min(stats['min'], new_value)
            stats['max'] = max(stats['max'], new_value)
        
        stats['count'] += 1
        
        # Normalize new value
        if method == NormalizationMethod.ZSCORE:
            std = np.sqrt(stats['var']) if stats['var'] > 0 else 1.0
            return (new_value - stats['mean']) / std
        
        elif method == NormalizationMethod.MINMAX:
            if stats['max'] != stats['min']:
                return (new_value - stats['min']) / (stats['max'] - stats['min'])
            else:
                return 0.0
        
        else:
            # For other methods, fall back to batch normalization
            return new_value
    
    # ==================== INVERSE TRANSFORMATION ====================
    
    def inverse_transform(self, 
                         normalized_values: np.ndarray,
                         key: str) -> np.ndarray:
        """
        Inverse transform normalized values back to original scale
        """
        
        if key not in self.normalization_stats:
            logger.warning(f"No normalization stats found for key: {key}")
            return normalized_values
        
        stats = self.normalization_stats[key]
        method = stats['method']
        
        if method == 'zscore':
            return normalized_values * stats['std'] + stats['mean']
        
        elif method == 'minmax':
            return normalized_values * (stats['max'] - stats['min']) + stats['min']
        
        elif method == 'robust':
            return normalized_values * stats['mad'] + stats['median']
        
        elif method == 'quantile':
            # Use inverse quantile mapping
            sorted_values = stats['sorted_values']
            percentiles = normalized_values * 100
            return np.percentile(sorted_values, percentiles)
        
        else:
            logger.warning(f"Inverse transform not implemented for method: {method}")
            return normalized_values
    
    # ==================== SPECIALIZED FINANCIAL NORMALIZATIONS ====================
    
    def normalize_returns(self, 
                         returns: pd.Series,
                         method: str = "volatility_scaling") -> pd.Series:
        """
        Specialized normalization for financial returns
        
        Methods:
        - volatility_scaling: Scale by rolling volatility
        - rank_based: Convert to ranks
        - quantile_uniform: Map to uniform distribution
        """
        
        if method == "volatility_scaling":
            # Scale by 20-day rolling volatility
            vol = returns.rolling(20, min_periods=10).std()
            vol = vol.fillna(method='bfill').fillna(method='ffill')
            vol[vol == 0] = returns.std()  # Fallback
            
            return returns / vol
        
        elif method == "rank_based":
            # Convert to cross-sectional ranks
            return returns.rank(pct=True) - 0.5  # Center around 0
        
        elif method == "quantile_uniform":
            # Map to uniform distribution
            ranks = returns.rank(method='average')
            return (ranks - 1) / (len(returns) - 1)
        
        else:
            raise ValueError(f"Unknown return normalization method: {method}")
    
    def normalize_prices_to_returns(self, 
                                   prices: pd.Series,
                                   return_type: str = "log") -> pd.Series:
        """Convert prices to normalized returns"""
        
        if return_type == "log":
            returns = np.log(prices / prices.shift(1))
        elif return_type == "simple":
            returns = prices.pct_change()
        else:
            raise ValueError(f"Unknown return type: {return_type}")
        
        return returns.dropna()
    
    # ==================== REGIME-AWARE NORMALIZATION ====================
    
    def normalize_regime_aware(self, 
                              data: pd.DataFrame,
                              regime_column: str,
                              value_columns: List[str],
                              method: Optional[NormalizationMethod] = None) -> pd.DataFrame:
        """
        Normalize data within each regime separately
        
        Useful when market conditions change significantly
        """
        
        if method is None:
            method = self.default_method
        
        result_df = data.copy()
        
        for regime in data[regime_column].unique():
            regime_mask = data[regime_column] == regime
            regime_data = data[regime_mask]
            
            for col in value_columns:
                if col in regime_data.columns:
                    params = NormalizationParams(method=method)
                    normalized_values = self._apply_normalization(
                        regime_data[col].values, params, f"{col}_regime_{regime}"
                    )
                    
                    result_df.loc[regime_mask, col] = normalized_values
        
        return result_df
    
    # ==================== UTILITIES ====================
    
    def get_normalization_stats(self) -> Dict[str, Dict]:
        """Get all stored normalization statistics"""
        return self.normalization_stats.copy()
    
    def save_stats(self, filepath: str):
        """Save normalization statistics to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.normalization_stats, f)
        logger.info(f"Normalization stats saved to {filepath}")
    
    def load_stats(self, filepath: str):
        """Load normalization statistics from file"""
        with open(filepath, 'rb') as f:
            self.normalization_stats = pickle.load(f)
        logger.info(f"Normalization stats loaded from {filepath}")
    
    def reset_stats(self):
        """Reset all normalization statistics"""
        self.normalization_stats.clear()
        self.online_stats.clear()
        logger.info("Normalization statistics reset")
    
    def close(self):
        """Clean shutdown"""
        self.executor.shutdown(wait=True)
        logger.info("Data normalizer shutdown complete")

# ==================== NORMALIZATION PIPELINE ====================

class NormalizationPipeline:
    """
    Pipeline for applying multiple normalization steps
    """
    
    def __init__(self):
        self.steps = []
        self.normalizer = DataNormalizer()
    
    def add_step(self, 
                name: str,
                method: NormalizationMethod,
                columns: Optional[List[str]] = None,
                params: Optional[Dict] = None):
        """Add normalization step to pipeline"""
        
        step = {
            'name': name,
            'method': method,
            'columns': columns,
            'params': params or {}
        }
        
        self.steps.append(step)
        logger.info(f"Added normalization step: {name}")
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all normalization steps"""
        
        result_df = df.copy()
        
        for step in self.steps:
            logger.info(f"Applying normalization step: {step['name']}")
            
            columns = step['columns'] or result_df.select_dtypes(include=[np.number]).columns
            
            for col in columns:
                if col in result_df.columns:
                    params = NormalizationParams(
                        method=step['method'],
                        **step['params']
                    )
                    
                    result_df[col] = self.normalizer.normalize(
                        result_df[col], params=params
                    )
        
        return result_df

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Initialize normalizer
    normalizer = DataNormalizer()
    
    # Generate sample financial data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
    
    # Simulate price data with different volatility regimes
    returns = np.random.normal(0, 0.01, 1000)
    returns[500:750] *= 2  # High volatility regime
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': dates,
        'symbol': 'BTCUSDT',
        'price': prices,
        'volume': np.random.exponential(100, 1000),
        'returns': returns
    })
    
    print("Original data stats:")
    print(df[['price', 'volume', 'returns']].describe())
    
    # Test different normalization methods
    methods = [
        NormalizationMethod.ZSCORE,
        NormalizationMethod.MINMAX,
        NormalizationMethod.ROBUST,
        NormalizationMethod.RANK
    ]
    
    for method in methods:
        print(f"\n=== {method.value.upper()} NORMALIZATION ===")
        
        normalized_df = normalizer.normalize(
            df[['price', 'volume', 'returns']],
            method=method
        )
        
        print(normalized_df.describe())
    
    # Test cross-sectional normalization
    print("\n=== CROSS-SECTIONAL NORMALIZATION ===")
    
    # Create multi-asset data
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
    multi_df = pd.concat([
        df.assign(symbol=symbol) for symbol in symbols
    ]).reset_index(drop=True)
    
    # Add some asset-specific scaling
    multi_df.loc[multi_df['symbol'] == 'ETHUSDT', 'price'] *= 0.1
    multi_df.loc[multi_df['symbol'] == 'ADAUSDT', 'price'] *= 0.01
    
    cross_norm_df = normalizer.normalize_cross_sectional(
        multi_df,
        timestamp_col='timestamp',
        value_cols=['price'],
        method=NormalizationMethod.ZSCORE
    )
    
    print("Cross-sectional normalized prices:")
    print(cross_norm_df.groupby('symbol')['price'].describe())
    
    # Test online normalization
    print("\n=== ONLINE NORMALIZATION ===")
    
    online_values = []
    for i, price in enumerate(df['price'].iloc[:100]):
        normalized = normalizer.update_online_stats(
            'BTCUSDT_price', price, NormalizationMethod.ZSCORE
        )
        online_values.append(normalized)
        
        if i % 20 == 0:
            print(f"Sample {i}: {price:.2f} → {normalized:.3f}")
    
    # Test volatility-adjusted normalization for returns
    print("\n=== VOLATILITY-ADJUSTED RETURNS ===")
    
    vol_adjusted = normalizer.normalize_returns(
        df['returns'], method="volatility_scaling"
    )
    
    print(f"Original returns std: {df['returns'].std():.4f}")
    print(f"Vol-adjusted returns std: {vol_adjusted.std():.4f}")
    
    # Test normalization pipeline
    print("\n=== NORMALIZATION PIPELINE ===")
    
    pipeline = NormalizationPipeline()
    pipeline.add_step("price_norm", NormalizationMethod.ZSCORE, ['price'])
    pipeline.add_step("volume_norm", NormalizationMethod.MINMAX, ['volume'])
    pipeline.add_step("returns_norm", NormalizationMethod.ROBUST, ['returns'])
    
    pipeline_result = pipeline.fit_transform(df)
    print(pipeline_result[['price', 'volume', 'returns']].describe())
    
    # Cleanup
    normalizer.close()