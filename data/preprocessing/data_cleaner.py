"""
QUANTUM-FORGE Data Cleaner
Advanced data cleaning and quality assurance for financial time-series

Features:
- Real-time outlier detection using statistical methods
- Missing data imputation with financial awareness
- Data quality scoring and alerting
- Trade/quote data validation
- Market microstructure noise filtering
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import signal
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

class DataQualityScore(Enum):
    """Data quality assessment levels"""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    UNUSABLE = 1

@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    timestamp: float
    symbol: str
    total_records: int
    missing_data_pct: float
    outlier_pct: float
    duplicate_pct: float
    quality_score: DataQualityScore
    issues: List[str]
    recommendations: List[str]

class DataCleaner:
    """
    Production-grade data cleaner for financial time-series
    
    Handles:
    - Tick data validation and cleaning
    - Order book data sanitization
    - Statistical outlier detection
    - Missing data imputation
    - Data quality scoring
    """
    
    def __init__(self,
                 outlier_method: str = "iqr",
                 outlier_threshold: float = 3.0,
                 price_change_limit: float = 0.1,  # 10% max price change
                 min_tick_size: float = 0.01,
                 max_spread_bps: float = 1000):  # 10% max spread
        
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.price_change_limit = price_change_limit
        self.min_tick_size = min_tick_size
        self.max_spread_bps = max_spread_bps
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Data cleaner initialized")
    
    # ==================== TICK DATA CLEANING ====================
    
    def clean_tick_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Clean tick data with comprehensive validation
        
        Expected columns: timestamp, symbol, price, size, side, exchange
        """
        
        start_time = time.time()
        original_count = len(df)
        issues = []
        recommendations = []
        
        if df.empty:
            return df, self._create_quality_report(df, issues, recommendations, DataQualityScore.UNUSABLE)
        
        cleaned_df = df.copy()
        
        # 1. Data type validation and conversion
        cleaned_df = self._validate_data_types(cleaned_df, issues)
        
        # 2. Remove duplicate ticks
        duplicates_mask = cleaned_df.duplicated(subset=['timestamp', 'symbol', 'price', 'size'])
        if duplicates_mask.any():
            duplicate_count = duplicates_mask.sum()
            cleaned_df = cleaned_df[~duplicates_mask]
            issues.append(f"Removed {duplicate_count} duplicate ticks")
        
        # 3. Sort by timestamp
        cleaned_df = cleaned_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        
        # 4. Price validation
        cleaned_df = self._validate_prices(cleaned_df, issues, recommendations)
        
        # 5. Size validation
        cleaned_df = self._validate_sizes(cleaned_df, issues)
        
        # 6. Timestamp validation
        cleaned_df = self._validate_timestamps(cleaned_df, issues)
        
        # 7. Outlier detection and handling
        cleaned_df = self._detect_and_handle_outliers(cleaned_df, issues)
        
        # 8. Missing data handling
        cleaned_df = self._handle_missing_data(cleaned_df, issues, recommendations)
        
        # Generate quality report
        processing_time = time.time() - start_time
        quality_score = self._calculate_quality_score(original_count, len(cleaned_df), issues)
        
        report = DataQualityReport(
            timestamp=time.time(),
            symbol=cleaned_df['symbol'].iloc[0] if not cleaned_df.empty else "UNKNOWN",
            total_records=original_count,
            missing_data_pct=((original_count - len(cleaned_df)) / original_count * 100) if original_count > 0 else 0,
            outlier_pct=len([i for i in issues if "outlier" in i.lower()]) / original_count * 100 if original_count > 0 else 0,
            duplicate_pct=duplicates_mask.sum() / original_count * 100 if original_count > 0 else 0,
            quality_score=quality_score,
            issues=issues,
            recommendations=recommendations
        )
        
        logger.info(f"Cleaned {original_count} → {len(cleaned_df)} ticks in {processing_time:.2f}s")
        
        return cleaned_df, report
    
    def _validate_data_types(self, df: pd.DataFrame, issues: List[str]) -> pd.DataFrame:
        """Validate and convert data types"""
        
        try:
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                if df['timestamp'].dtype == 'object':
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                elif df['timestamp'].dtype in ['int64', 'float64']:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Ensure numeric types
            numeric_cols = ['price', 'size']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ensure string types
            string_cols = ['symbol', 'side', 'exchange']
            for col in string_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            
            return df
            
        except Exception as e:
            issues.append(f"Data type conversion error: {e}")
            return df
    
    def _validate_prices(self, df: pd.DataFrame, issues: List[str], recommendations: List[str]) -> pd.DataFrame:
        """Validate price data"""
        
        if 'price' not in df.columns:
            return df
        
        initial_count = len(df)
        
        # Remove negative or zero prices
        positive_mask = df['price'] > 0
        if not positive_mask.all():
            invalid_count = (~positive_mask).sum()
            df = df[positive_mask]
            issues.append(f"Removed {invalid_count} records with non-positive prices")
        
        # Check for unrealistic price changes
        if len(df) > 1:
            df_sorted = df.sort_values(['symbol', 'timestamp'])
            price_changes = df_sorted.groupby('symbol')['price'].pct_change().abs()
            
            outlier_mask = price_changes > self.price_change_limit
            if outlier_mask.any():
                outlier_count = outlier_mask.sum()
                # Mark rather than remove (might be legitimate)
                df.loc[df_sorted[outlier_mask].index, 'price_change_outlier'] = True
                issues.append(f"Flagged {outlier_count} records with large price changes (>{self.price_change_limit*100:.1f}%)")
                recommendations.append("Review flagged large price changes for legitimacy")
        
        # Check tick size compliance
        if self.min_tick_size > 0:
            tick_remainder = df['price'] % self.min_tick_size
            invalid_tick_mask = tick_remainder > 1e-6  # Allow for floating point precision
            
            if invalid_tick_mask.any():
                invalid_count = invalid_tick_mask.sum()
                # Round to nearest valid tick
                df.loc[invalid_tick_mask, 'price'] = (
                    df.loc[invalid_tick_mask, 'price'] / self.min_tick_size
                ).round() * self.min_tick_size
                issues.append(f"Adjusted {invalid_count} prices to comply with tick size {self.min_tick_size}")
        
        return df
    
    def _validate_sizes(self, df: pd.DataFrame, issues: List[str]) -> pd.DataFrame:
        """Validate trade/order sizes"""
        
        if 'size' not in df.columns:
            return df
        
        # Remove negative or zero sizes
        positive_mask = df['size'] > 0
        if not positive_mask.all():
            invalid_count = (~positive_mask).sum()
            df = df[positive_mask]
            issues.append(f"Removed {invalid_count} records with non-positive sizes")
        
        # Check for extremely large sizes (potential errors)
        if len(df) > 10:
            size_p99 = df['size'].quantile(0.99)
            size_median = df['size'].median()
            
            # If 99th percentile is >100x median, flag outliers
            if size_p99 > 100 * size_median:
                outlier_threshold = 50 * size_median
                outlier_mask = df['size'] > outlier_threshold
                
                if outlier_mask.any():
                    outlier_count = outlier_mask.sum()
                    df.loc[outlier_mask, 'size_outlier'] = True
                    issues.append(f"Flagged {outlier_count} records with extremely large sizes")
        
        return df
    
    def _validate_timestamps(self, df: pd.DataFrame, issues: List[str]) -> pd.DataFrame:
        """Validate timestamp data"""
        
        if 'timestamp' not in df.columns:
            return df
        
        # Remove records with invalid timestamps
        valid_timestamp_mask = pd.notna(df['timestamp'])
        if not valid_timestamp_mask.all():
            invalid_count = (~valid_timestamp_mask).sum()
            df = df[valid_timestamp_mask]
            issues.append(f"Removed {invalid_count} records with invalid timestamps")
        
        # Check for timestamps in the future (beyond reasonable buffer)
        future_threshold = pd.Timestamp.now() + pd.Timedelta(hours=1)
        future_mask = df['timestamp'] > future_threshold
        
        if future_mask.any():
            future_count = future_mask.sum()
            df = df[~future_mask]
            issues.append(f"Removed {future_count} records with future timestamps")
        
        # Check for very old timestamps (potential errors)
        old_threshold = pd.Timestamp.now() - pd.Timedelta(days=30)
        old_mask = df['timestamp'] < old_threshold
        
        if old_mask.any() and len(df) > 1000:  # Only flag if substantial data
            old_count = old_mask.sum()
            if old_count / len(df) < 0.1:  # If <10% of data is old, might be errors
                df.loc[old_mask, 'timestamp_old'] = True
                issues.append(f"Flagged {old_count} records with very old timestamps")
        
        return df
    
    def _detect_and_handle_outliers(self, df: pd.DataFrame, issues: List[str]) -> pd.DataFrame:
        """Detect and handle statistical outliers"""
        
        if len(df) < 10:  # Need minimum data for outlier detection
            return df
        
        numeric_cols = ['price', 'size']
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            outlier_mask = self._detect_outliers(df[col])
            
            if outlier_mask.any():
                outlier_count = outlier_mask.sum()
                
                # Mark outliers instead of removing (preserve information)
                df.loc[outlier_mask, f'{col}_outlier'] = True
                issues.append(f"Flagged {outlier_count} {col} outliers using {self.outlier_method} method")
        
        return df
    
    def _detect_outliers(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using specified method"""
        
        if self.outlier_method == "iqr":
            return self._detect_outliers_iqr(series)
        elif self.outlier_method == "zscore":
            return self._detect_outliers_zscore(series)
        elif self.outlier_method == "isolation_forest":
            return self._detect_outliers_isolation_forest(series)
        else:
            logger.warning(f"Unknown outlier method: {self.outlier_method}")
            return np.zeros(len(series), dtype=bool)
    
    def _detect_outliers_iqr(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using Interquartile Range method"""
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using Z-score method"""
        
        z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
        return z_scores > self.outlier_threshold
    
    def _detect_outliers_isolation_forest(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using Isolation Forest"""
        
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            iso_forest = IsolationForest(
                contamination=0.1,  # Expect 10% outliers
                random_state=42
            )
            
            outlier_labels = iso_forest.fit_predict(X)
            return outlier_labels == -1  # -1 indicates outlier
            
        except ImportError:
            logger.warning("sklearn not available, falling back to IQR method")
            return self._detect_outliers_iqr(series)
    
    def _handle_missing_data(self, df: pd.DataFrame, issues: List[str], recommendations: List[str]) -> pd.DataFrame:
        """Handle missing data intelligently"""
        
        missing_counts = df.isnull().sum()
        
        if missing_counts.sum() == 0:
            return df
        
        for col, missing_count in missing_counts.items():
            if missing_count == 0:
                continue
            
            missing_pct = missing_count / len(df) * 100
            
            if missing_pct > 50:
                issues.append(f"Column {col} has {missing_pct:.1f}% missing data")
                recommendations.append(f"Consider dropping column {col} or investigating data source")
            
            elif missing_pct > 10:
                issues.append(f"Column {col} has {missing_pct:.1f}% missing data")
                recommendations.append(f"Investigate missing data pattern in {col}")
            
            # Handle missing data based on column type
            if col in ['price', 'size']:
                # For financial data, forward fill is often appropriate
                df[col] = df.groupby('symbol')[col].fillna(method='ffill')
                
                # If still missing, use interpolation
                remaining_missing = df[col].isnull().sum()
                if remaining_missing > 0:
                    df[col] = df.groupby('symbol')[col].interpolate(method='linear')
            
            elif col in ['side', 'exchange']:
                # For categorical data, use mode or forward fill
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'UNKNOWN'
                df[col] = df[col].fillna(mode_value)
        
        return df
    
    # ==================== ORDER BOOK CLEANING ====================
    
    def clean_orderbook_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
        """
        Clean order book data
        
        Expected columns: timestamp, symbol, level, bid_price, bid_size, ask_price, ask_size
        """
        
        if df.empty:
            return df, self._create_quality_report(df, [], [], DataQualityScore.UNUSABLE)
        
        original_count = len(df)
        issues = []
        recommendations = []
        
        cleaned_df = df.copy()
        
        # 1. Basic validation
        cleaned_df = self._validate_data_types(cleaned_df, issues)
        
        # 2. Order book specific validation
        if all(col in cleaned_df.columns for col in ['bid_price', 'ask_price']):
            # Remove crossed books (bid >= ask)
            valid_spread_mask = cleaned_df['bid_price'] < cleaned_df['ask_price']
            if not valid_spread_mask.all():
                crossed_count = (~valid_spread_mask).sum()
                cleaned_df = cleaned_df[valid_spread_mask]
                issues.append(f"Removed {crossed_count} crossed book entries")
            
            # Check for unrealistic spreads
            cleaned_df['spread_bps'] = (
                (cleaned_df['ask_price'] - cleaned_df['bid_price']) / 
                cleaned_df['bid_price'] * 10000
            )
            
            wide_spread_mask = cleaned_df['spread_bps'] > self.max_spread_bps
            if wide_spread_mask.any():
                wide_count = wide_spread_mask.sum()
                cleaned_df.loc[wide_spread_mask, 'wide_spread_flag'] = True
                issues.append(f"Flagged {wide_count} entries with spreads >{self.max_spread_bps} bps")
        
        # 3. Level validation (if present)
        if 'level' in cleaned_df.columns:
            # Ensure levels are sequential
            for symbol in cleaned_df['symbol'].unique():
                symbol_mask = cleaned_df['symbol'] == symbol
                symbol_data = cleaned_df[symbol_mask]
                
                for timestamp in symbol_data['timestamp'].unique():
                    ts_mask = symbol_data['timestamp'] == timestamp
                    levels = symbol_data[ts_mask]['level'].sort_values()
                    
                    expected_levels = range(1, len(levels) + 1)
                    if not levels.equals(pd.Series(expected_levels, index=levels.index)):
                        issues.append(f"Non-sequential levels found for {symbol} at {timestamp}")
        
        # Generate quality report
        quality_score = self._calculate_quality_score(original_count, len(cleaned_df), issues)
        
        report = DataQualityReport(
            timestamp=time.time(),
            symbol=cleaned_df['symbol'].iloc[0] if not cleaned_df.empty else "UNKNOWN",
            total_records=original_count,
            missing_data_pct=((original_count - len(cleaned_df)) / original_count * 100) if original_count > 0 else 0,
            outlier_pct=0,  # TODO: Implement for orderbook
            duplicate_pct=0,  # TODO: Implement for orderbook
            quality_score=quality_score,
            issues=issues,
            recommendations=recommendations
        )
        
        return cleaned_df, report
    
    # ==================== DATA QUALITY SCORING ====================
    
    def _calculate_quality_score(self, original_count: int, final_count: int, issues: List[str]) -> DataQualityScore:
        """Calculate overall data quality score"""
        
        if original_count == 0:
            return DataQualityScore.UNUSABLE
        
        # Base score on data retention rate
        retention_rate = final_count / original_count
        
        # Penalty for each issue type
        issue_penalty = len(issues) * 0.1
        
        # Calculate raw score
        raw_score = retention_rate - issue_penalty
        
        # Map to quality levels
        if raw_score >= 0.95:
            return DataQualityScore.EXCELLENT
        elif raw_score >= 0.85:
            return DataQualityScore.GOOD
        elif raw_score >= 0.70:
            return DataQualityScore.ACCEPTABLE
        elif raw_score >= 0.50:
            return DataQualityScore.POOR
        else:
            return DataQualityScore.UNUSABLE
    
    def _create_quality_report(self, df: pd.DataFrame, issues: List[str], 
                              recommendations: List[str], score: DataQualityScore) -> DataQualityReport:
        """Create a basic quality report"""
        
        return DataQualityReport(
            timestamp=time.time(),
            symbol="UNKNOWN",
            total_records=len(df),
            missing_data_pct=0.0,
            outlier_pct=0.0,
            duplicate_pct=0.0,
            quality_score=score,
            issues=issues,
            recommendations=recommendations
        )
    
    # ==================== NOISE FILTERING ====================
    
    def filter_microstructure_noise(self, df: pd.DataFrame, 
                                   window_size: int = 20,
                                   filter_type: str = "savgol") -> pd.DataFrame:
        """
        Filter microstructure noise from price series
        
        Methods:
        - savgol: Savitzky-Golay filter (preserves trends)
        - rolling: Rolling average
        - ewm: Exponentially weighted moving average
        """
        
        if 'price' not in df.columns or len(df) < window_size:
            return df
        
        cleaned_df = df.copy()
        
        for symbol in cleaned_df['symbol'].unique():
            symbol_mask = cleaned_df['symbol'] == symbol
            symbol_data = cleaned_df[symbol_mask].sort_values('timestamp')
            
            if len(symbol_data) < window_size:
                continue
            
            prices = symbol_data['price'].values
            
            if filter_type == "savgol":
                # Savitzky-Golay filter (polynomial order 2)
                if len(prices) >= window_size and window_size % 2 == 1:
                    filtered_prices = signal.savgol_filter(prices, window_size, 2)
                    cleaned_df.loc[symbol_mask, 'price_filtered'] = filtered_prices
            
            elif filter_type == "rolling":
                # Simple moving average
                filtered_prices = pd.Series(prices).rolling(
                    window=window_size, center=True
                ).mean().fillna(method='bfill').fillna(method='ffill')
                cleaned_df.loc[symbol_mask, 'price_filtered'] = filtered_prices.values
            
            elif filter_type == "ewm":
                # Exponentially weighted moving average
                alpha = 2 / (window_size + 1)
                filtered_prices = pd.Series(prices).ewm(alpha=alpha).mean()
                cleaned_df.loc[symbol_mask, 'price_filtered'] = filtered_prices.values
        
        return cleaned_df
    
    # ==================== BATCH PROCESSING ====================
    
    def process_batch(self, data_batches: List[pd.DataFrame]) -> List[Tuple[pd.DataFrame, DataQualityReport]]:
        """Process multiple data batches in parallel"""
        
        with self.executor as executor:
            futures = [
                executor.submit(self.clean_tick_data, batch)
                for batch in data_batches
            ]
            
            results = [future.result() for future in futures]
        
        return results
    
    def close(self):
        """Clean shutdown"""
        self.executor.shutdown(wait=True)
        logger.info("Data cleaner shutdown complete")

# ==================== QUALITY MONITORING ====================

class DataQualityMonitor:
    """
    Monitor data quality over time and generate alerts
    """
    
    def __init__(self, 
                 alert_threshold: DataQualityScore = DataQualityScore.POOR,
                 history_length: int = 100):
        
        self.alert_threshold = alert_threshold
        self.history_length = history_length
        self.quality_history = []
    
    def add_report(self, report: DataQualityReport) -> List[str]:
        """Add quality report and check for alerts"""
        
        self.quality_history.append(report)
        
        # Maintain history length
        if len(self.quality_history) > self.history_length:
            self.quality_history.pop(0)
        
        alerts = []
        
        # Quality degradation alert
        if report.quality_score.value <= self.alert_threshold.value:
            alerts.append(f"Data quality below threshold: {report.quality_score.name}")
        
        # Trend analysis (if enough history)
        if len(self.quality_history) >= 10:
            recent_scores = [r.quality_score.value for r in self.quality_history[-10:]]
            
            # Check for declining trend
            if len(set(recent_scores)) > 1:  # Not all same scores
                trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                
                if trend < -0.1:  # Declining trend
                    alerts.append("Data quality trending downward")
        
        return alerts
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get summary of data quality metrics"""
        
        if not self.quality_history:
            return {"message": "No quality data available"}
        
        recent_reports = self.quality_history[-10:]  # Last 10 reports
        
        return {
            "total_reports": len(self.quality_history),
            "avg_quality_score": np.mean([r.quality_score.value for r in recent_reports]),
            "avg_missing_data_pct": np.mean([r.missing_data_pct for r in recent_reports]),
            "avg_outlier_pct": np.mean([r.outlier_pct for r in recent_reports]),
            "common_issues": self._get_common_issues(recent_reports),
            "last_report_time": self.quality_history[-1].timestamp
        }
    
    def _get_common_issues(self, reports: List[DataQualityReport]) -> List[str]:
        """Identify most common issues"""
        
        issue_counts = {}
        
        for report in reports:
            for issue in report.issues:
                # Extract issue type (before the first number/details)
                issue_type = issue.split()[0:3]  # First 3 words
                issue_key = " ".join(issue_type)
                
                issue_counts[issue_key] = issue_counts.get(issue_key, 0) + 1
        
        # Return top 5 issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        return [issue for issue, count in sorted_issues[:5]]

# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    import random
    
    # Initialize cleaner
    cleaner = DataCleaner(
        outlier_method="iqr",
        outlier_threshold=2.5,
        price_change_limit=0.05  # 5% max change
    )
    
    # Generate sample dirty data
    def generate_dirty_data(n_rows: int = 1000) -> pd.DataFrame:
        data = []
        base_price = 50000
        base_time = time.time()
        
        for i in range(n_rows):
            # Add some data quality issues
            price = base_price + random.gauss(0, 100)
            
            # Introduce outliers (5% chance)
            if random.random() < 0.05:
                price *= random.choice([0.5, 2.0])  # 50% lower or 100% higher
            
            # Introduce missing data (2% chance)
            if random.random() < 0.02:
                price = None
            
            # Introduce negative prices (1% chance)
            if random.random() < 0.01:
                price = -abs(price) if price else -1000
            
            data.append({
                "timestamp": base_time + i * 0.1,
                "symbol": "BTCUSDT",
                "price": price,
                "size": max(0.01, random.expovariate(10)),  # Exponential distribution
                "side": random.choice(["B", "S"]),
                "exchange": "binance"
            })
        
        return pd.DataFrame(data)
    
    # Test cleaning
    print("Generating dirty data...")
    dirty_df = generate_dirty_data(10000)
    print(f"Generated {len(dirty_df)} dirty records")
    
    print("\nCleaning data...")
    clean_df, quality_report = cleaner.clean_tick_data(dirty_df)
    
    print(f"  Cleaned data: {len(dirty_df)} → {len(clean_df)} records")
    print(f"Quality Score: {quality_report.quality_score.name}")
    print(f"Issues Found: {len(quality_report.issues)}")
    
    for issue in quality_report.issues[:5]:  # Show first 5 issues
        print(f"  - {issue}")
    
    if quality_report.recommendations:
        print(f"\nRecommendations:")
        for rec in quality_report.recommendations[:3]:  # Show first 3
            print(f"  - {rec}")
    
    # Test noise filtering
    print(f"\nTesting noise filtering...")
    filtered_df = cleaner.filter_microstructure_noise(clean_df, window_size=21, filter_type="savgol")
    
    if 'price_filtered' in filtered_df.columns:
        original_std = clean_df['price'].std()
        filtered_std = filtered_df['price_filtered'].std()
        noise_reduction = (1 - filtered_std / original_std) * 100
        print(f"Noise reduction: {noise_reduction:.1f}%")
    
    # Test quality monitoring
    monitor = DataQualityMonitor()
    alerts = monitor.add_report(quality_report)
    
    if alerts:
        print(f"\n ️  Quality Alerts:")
        for alert in alerts:
            print(f"  - {alert}")
    
    summary = monitor.get_quality_summary()
    print(f"\n  Quality Summary: {summary}")
    
    # Cleanup
    cleaner.close()