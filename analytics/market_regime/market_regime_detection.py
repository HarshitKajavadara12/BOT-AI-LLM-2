"""
Market Regime Detection Engine for QUANTUM-FORGE
Implements sophisticated market regime identification using multiple approaches:
statistical methods, machine learning, volatility clustering, and structural breaks.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime, timedelta
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
import ruptures as rpt
import pickle
import json
warnings.filterwarnings('ignore')

class RegimeType(Enum):
    """Types of market regimes."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    MOMENTUM = "momentum"
    UNKNOWN = "unknown"

class DetectionMethod(Enum):
    """Regime detection methods."""
    HIDDEN_MARKOV_MODEL = "hmm"
    GAUSSIAN_MIXTURE_MODEL = "gmm"
    VOLATILITY_CLUSTERING = "volatility_clustering"
    STRUCTURAL_BREAKS = "structural_breaks"
    STATISTICAL_TESTS = "statistical_tests"
    MACHINE_LEARNING = "machine_learning"
    ENSEMBLE = "ensemble"

@dataclass
class RegimeDetectionResult:
    """Result of regime detection."""
    timestamp: pd.Timestamp
    regime: RegimeType
    confidence: float
    method: DetectionMethod
    features: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RegimeTransition:
    """Regime transition information."""
    from_regime: RegimeType
    to_regime: RegimeType
    transition_time: pd.Timestamp
    duration: int
    confidence: float
    trigger_features: Optional[Dict[str, float]] = None

class VolatilityModel:
    """GARCH-type volatility model for regime detection."""
    
    def __init__(self, p: int = 1, q: int = 1):
        """Initialize volatility model."""
        self.p = p  # ARCH terms
        self.q = q  # GARCH terms
        self.params = None
        self.fitted = False
        
    def fit(self, returns: np.ndarray) -> Dict[str, float]:
        """Fit GARCH model to returns."""
        from arch import arch_model
        
        # Fit GARCH model
        model = arch_model(returns * 100, vol='Garch', p=self.p, q=self.q)
        res = model.fit(disp='off')
        
        self.params = res.params
        self.fitted = True
        
        # Extract conditional volatility
        self.conditional_volatility = res.conditional_volatility / 100
        
        return {
            'omega': self.params.get('omega', 0),
            'alpha': [self.params.get(f'alpha[{i+1}]', 0) for i in range(self.p)],
            'beta': [self.params.get(f'beta[{i+1}]', 0) for i in range(self.q)],
            'loglikelihood': res.loglikelihood,
            'aic': res.aic,
            'bic': res.bic
        }
    
    def predict_volatility(self, n_periods: int = 1) -> np.ndarray:
        """Predict future volatility."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        # Simple forecast based on last volatility
        last_vol = self.conditional_volatility[-1]
        return np.array([last_vol] * n_periods)

class MarketRegimeDetector:
    """Comprehensive market regime detection engine."""
    
    def __init__(self, min_regime_length: int = 20):
        """Initialize regime detector."""
        self.min_regime_length = min_regime_length
        self.models = {}
        self.scalers = {}
        self.regime_history = []
        self.current_regime = RegimeType.UNKNOWN
        
    def extract_features(self, data: pd.DataFrame, 
                        lookback_windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Extract features for regime detection."""
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        if 'close' in data.columns:
            prices = data['close']
            
            # Returns at different horizons
            for window in lookback_windows:
                features[f'return_{window}d'] = prices.pct_change(window)
                features[f'volatility_{window}d'] = prices.pct_change().rolling(window).std()
                
                # Price momentum
                features[f'momentum_{window}d'] = prices / prices.shift(window) - 1
                
                # Moving average ratios
                ma = prices.rolling(window).mean()
                features[f'price_ma_ratio_{window}d'] = prices / ma - 1
        
        # Volume features (if available)
        if 'volume' in data.columns:
            volume = data['volume']
            
            for window in lookback_windows:
                features[f'volume_ma_{window}d'] = volume.rolling(window).mean()
                features[f'volume_ratio_{window}d'] = volume / features[f'volume_ma_{window}d']
        
        # Technical indicators
        returns = data['close'].pct_change() if 'close' in data.columns else data.iloc[:, 0].pct_change()
        
        # Realized volatility
        for window in lookback_windows:
            features[f'realized_vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Skewness and kurtosis
        for window in lookback_windows:
            features[f'skewness_{window}d'] = returns.rolling(window).skew()
            features[f'kurtosis_{window}d'] = returns.rolling(window).kurt()
        
        # VIX-like volatility of volatility
        for window in lookback_windows:
            vol = returns.rolling(window).std()
            features[f'vol_of_vol_{window}d'] = vol.rolling(window).std()
        
        # Drawdown features
        if 'close' in data.columns:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            for window in lookback_windows:
                features[f'max_drawdown_{window}d'] = drawdown.rolling(window).min()
                features[f'drawdown_duration_{window}d'] = self._calculate_drawdown_duration(
                    drawdown, window
                )
        
        # Trend strength
        for window in lookback_windows:
            # Linear regression slope
            slope = returns.rolling(window).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
            )
            features[f'trend_strength_{window}d'] = slope
        
        # Market efficiency measures
        for window in lookback_windows:
            # Hurst exponent approximation
            features[f'hurst_{window}d'] = returns.rolling(window).apply(
                self._calculate_hurst_exponent
            )
        
        # Cross-correlation features (if multiple assets)
        if len(data.columns) > 1:
            for i, col1 in enumerate(data.columns[:3]):  # Limit to avoid too many features
                for j, col2 in enumerate(data.columns[:3]):
                    if i < j:
                        ret1 = data[col1].pct_change()
                        ret2 = data[col2].pct_change()
                        
                        for window in lookback_windows:
                            corr = ret1.rolling(window).corr(ret2)
                            features[f'corr_{col1}_{col2}_{window}d'] = corr
        
        # Economic regime indicators
        # VIX-like fear index (using return volatility as proxy)
        for window in [5, 20]:
            vol = returns.rolling(window).std()
            vol_percentile = vol.rolling(252).rank(pct=True)  # Percentile over 1 year
            features[f'vol_percentile_{window}d'] = vol_percentile
        
        # Market stress indicators
        # Large negative returns
        features['large_negative_returns'] = (returns < -0.02).rolling(5).sum()
        features['large_positive_returns'] = (returns > 0.02).rolling(5).sum()
        
        # Gap detection
        if 'open' in data.columns and 'close' in data.columns:
            gaps = (data['open'] / data['close'].shift(1)) - 1
            features['gap_size'] = gaps.abs()
            features['gap_direction'] = np.sign(gaps)
        
        return features.dropna()
    
    def _calculate_drawdown_duration(self, drawdown: pd.Series, window: int) -> pd.Series:
        """Calculate drawdown duration within rolling window."""
        durations = []
        
        for i in range(len(drawdown)):
            if i < window - 1:
                durations.append(np.nan)
                continue
            
            window_dd = drawdown.iloc[i-window+1:i+1]
            
            # Find current drawdown duration
            duration = 0
            for j in range(len(window_dd)-1, -1, -1):
                if window_dd.iloc[j] < 0:
                    duration += 1
                else:
                    break
            
            durations.append(duration)
        
        return pd.Series(durations, index=drawdown.index)
    
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent for trend persistence."""
        try:
            if len(returns) < 10 or returns.isna().all():
                return np.nan
            
            returns = returns.dropna()
            if len(returns) < 10:
                return np.nan
            
            # Use R/S statistic method
            lags = range(2, min(len(returns)//2, 20))
            rs_values = []
            
            for lag in lags:
                # Split into chunks
                n_chunks = len(returns) // lag
                if n_chunks < 2:
                    continue
                
                rs_chunk = []
                for i in range(n_chunks):
                    chunk = returns.iloc[i*lag:(i+1)*lag]
                    if len(chunk) < lag:
                        continue
                    
                    # Mean-adjusted series
                    mean_adj = chunk - chunk.mean()
                    
                    # Cumulative sum
                    cum_sum = mean_adj.cumsum()
                    
                    # Range
                    R = cum_sum.max() - cum_sum.min()
                    
                    # Standard deviation
                    S = chunk.std()
                    
                    if S > 0:
                        rs_chunk.append(R / S)
                
                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))
            
            if len(rs_values) < 3:
                return 0.5  # Default to random walk
            
            # Linear regression on log-log plot
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            
            # Constrain to reasonable range
            return max(0.1, min(0.9, hurst))
            
        except:
            return 0.5  # Default to random walk
    
    def detect_volatility_regimes(self, returns: pd.Series, 
                                 n_regimes: int = 3) -> pd.Series:
        """Detect volatility regimes using GARCH + clustering."""
        
        # Fit GARCH model
        vol_model = VolatilityModel(p=1, q=1)
        
        try:
            vol_params = vol_model.fit(returns.values)
            conditional_vol = vol_model.conditional_volatility
        except:
            # Fallback to rolling volatility
            conditional_vol = returns.rolling(20).std().values
        
        # Cluster volatility levels
        vol_data = conditional_vol.reshape(-1, 1)
        
        # Remove NaN values
        valid_mask = ~np.isnan(vol_data.flatten())
        vol_clean = vol_data[valid_mask]
        
        if len(vol_clean) < n_regimes:
            return pd.Series(['low_vol'] * len(returns), index=returns.index)
        
        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
        vol_regimes = gmm.fit_predict(vol_clean)
        
        # Map back to full series
        full_regimes = np.full(len(returns), -1)
        full_regimes[valid_mask] = vol_regimes
        
        # Forward fill missing values
        regime_series = pd.Series(full_regimes, index=returns.index)
        regime_series = regime_series.replace(-1, np.nan).fillna(method='ffill').fillna(method='bfill')
        
        # Map numeric regimes to descriptive names
        vol_means = []
        for i in range(n_regimes):
            mask = vol_regimes == i
            if np.sum(mask) > 0:
                vol_means.append(np.mean(vol_clean[mask]))
            else:
                vol_means.append(0)
        
        # Sort regimes by volatility level
        regime_order = np.argsort(vol_means)
        
        regime_names = ['low_vol', 'medium_vol', 'high_vol'][:n_regimes]
        
        # Create mapping
        regime_mapping = {}
        for i, regime_idx in enumerate(regime_order):
            regime_mapping[regime_idx] = regime_names[i]
        
        regime_labels = regime_series.map(regime_mapping)
        
        return regime_labels
    
    def detect_hmm_regimes(self, features: pd.DataFrame, 
                          n_regimes: int = 3) -> Tuple[pd.Series, hmm.GaussianHMM]:
        """Detect regimes using Hidden Markov Models."""
        
        # Prepare data
        feature_data = features.select_dtypes(include=[np.number]).dropna()
        
        if len(feature_data) < n_regimes * 10:
            # Not enough data
            return pd.Series(['unknown'] * len(features), index=features.index), None
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_data)
        
        # Reduce dimensionality if too many features
        if scaled_features.shape[1] > 10:
            pca = PCA(n_components=10)
            scaled_features = pca.fit_transform(scaled_features)
        
        # Fit HMM
        try:
            model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", 
                                   n_iter=100, random_state=42)
            model.fit(scaled_features)
            
            # Predict regimes
            regime_sequence = model.predict(scaled_features)
            
            # Map back to full time series
            full_regimes = pd.Series(index=features.index, dtype=object)
            full_regimes.loc[feature_data.index] = regime_sequence
            full_regimes = full_regimes.fillna(method='ffill').fillna(method='bfill')
            
            # Assign regime names based on characteristics
            regime_names = self._assign_regime_names(regime_sequence, feature_data)
            regime_labels = full_regimes.map(regime_names)
            
            return regime_labels, model
            
        except Exception as e:
            print(f"HMM fitting failed: {e}")
            return pd.Series(['unknown'] * len(features), index=features.index), None
    
    def detect_structural_breaks(self, returns: pd.Series, 
                               method: str = 'pelt') -> List[int]:
        """Detect structural breaks in return series."""
        
        if len(returns) < 50:
            return []
        
        # Prepare data
        returns_clean = returns.dropna().values
        
        if len(returns_clean) < 50:
            return []
        
        try:
            # Use ruptures library for structural break detection
            if method == 'pelt':
                # PELT (Pruned Exact Linear Time)
                algo = rpt.Pelt(model="rbf").fit(returns_clean.reshape(-1, 1))
                breakpoints = algo.predict(pen=10)
            elif method == 'window':
                # Window-based method
                algo = rpt.Window(width=40, model="l2").fit(returns_clean.reshape(-1, 1))
                breakpoints = algo.predict(n_bkps=5)
            else:
                # Dynamic programming
                algo = rpt.Dynp(model="l2", min_size=20).fit(returns_clean.reshape(-1, 1))
                breakpoints = algo.predict(n_bkps=5)
            
            # Remove the last breakpoint (end of series)
            if breakpoints and breakpoints[-1] == len(returns_clean):
                breakpoints = breakpoints[:-1]
            
            # Map back to original index
            valid_indices = returns.dropna().index
            break_times = []
            
            for bp in breakpoints:
                if bp < len(valid_indices):
                    break_times.append(returns.index.get_loc(valid_indices[bp]))
            
            return break_times
            
        except Exception as e:
            print(f"Structural break detection failed: {e}")
            return []
    
    def detect_trend_regimes(self, prices: pd.Series, 
                           trend_window: int = 50) -> pd.Series:
        """Detect trend-based regimes."""
        
        if len(prices) < trend_window:
            return pd.Series(['sideways'] * len(prices), index=prices.index)
        
        # Calculate rolling trend strength
        trend_strength = prices.rolling(trend_window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == trend_window else np.nan
        )
        
        # Calculate rolling volatility for normalization
        returns = prices.pct_change()
        volatility = returns.rolling(trend_window).std()
        
        # Normalize trend by volatility
        normalized_trend = trend_strength / (volatility * prices)
        
        # Define regime thresholds
        trend_regimes = pd.Series(index=prices.index, dtype=object)
        
        # Dynamic thresholds based on historical distribution
        trend_percentiles = normalized_trend.rolling(252).quantile([0.3, 0.7])
        
        for i in range(len(prices)):
            if pd.isna(normalized_trend.iloc[i]):
                trend_regimes.iloc[i] = 'unknown'
            else:
                trend_val = normalized_trend.iloc[i]
                
                # Use rolling percentiles if available
                if i >= 252:
                    low_thresh = normalized_trend.iloc[max(0, i-252):i].quantile(0.3)
                    high_thresh = normalized_trend.iloc[max(0, i-252):i].quantile(0.7)
                else:
                    low_thresh = normalized_trend.iloc[:i+1].quantile(0.3)
                    high_thresh = normalized_trend.iloc[:i+1].quantile(0.7)
                
                if pd.isna(low_thresh) or pd.isna(high_thresh):
                    trend_regimes.iloc[i] = 'sideways'
                elif trend_val > high_thresh:
                    trend_regimes.iloc[i] = 'trending_up'
                elif trend_val < low_thresh:
                    trend_regimes.iloc[i] = 'trending_down'
                else:
                    trend_regimes.iloc[i] = 'sideways'
        
        return trend_regimes.fillna('sideways')
    
    def detect_crisis_regimes(self, returns: pd.Series, 
                            volatility_threshold: float = 0.5,
                            drawdown_threshold: float = -0.1) -> pd.Series:
        """Detect crisis/stress regimes."""
        
        # Calculate stress indicators
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)  # Annualized
        
        # Calculate rolling drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Crisis indicators
        high_vol = rolling_vol > volatility_threshold
        large_drawdown = drawdown < drawdown_threshold
        
        # Large negative returns
        large_negative = returns < -0.05  # 5% daily loss
        
        # Combine indicators
        crisis_regimes = pd.Series(index=returns.index, dtype=object)
        
        for i in range(len(returns)):
            if pd.isna(rolling_vol.iloc[i]):
                crisis_regimes.iloc[i] = 'normal'
            else:
                is_crisis = (high_vol.iloc[i] and large_drawdown.iloc[i]) or large_negative.iloc[i]
                
                if is_crisis:
                    crisis_regimes.iloc[i] = 'crisis'
                elif rolling_vol.iloc[i] > volatility_threshold * 0.7:
                    crisis_regimes.iloc[i] = 'stressed'
                else:
                    crisis_regimes.iloc[i] = 'normal'
        
        return crisis_regimes.fillna('normal')
    
    def _assign_regime_names(self, regime_sequence: np.ndarray, 
                           features: pd.DataFrame) -> Dict[int, str]:
        """Assign meaningful names to numeric regimes."""
        
        regime_names = {}
        n_regimes = len(np.unique(regime_sequence))
        
        # Calculate regime characteristics
        for regime_id in range(n_regimes):
            mask = regime_sequence == regime_id
            regime_features = features[mask]
            
            if len(regime_features) == 0:
                regime_names[regime_id] = 'unknown'
                continue
            
            # Analyze characteristics
            volatility_cols = [col for col in features.columns if 'volatility' in col or 'vol' in col]
            return_cols = [col for col in features.columns if 'return' in col]
            momentum_cols = [col for col in features.columns if 'momentum' in col]
            
            avg_vol = 0
            avg_return = 0
            avg_momentum = 0
            
            if volatility_cols:
                avg_vol = regime_features[volatility_cols].mean().mean()
            if return_cols:
                avg_return = regime_features[return_cols].mean().mean()
            if momentum_cols:
                avg_momentum = regime_features[momentum_cols].mean().mean()
            
            # Name assignment logic
            if avg_vol > 0.02:  # High volatility
                if avg_return < -0.01:
                    regime_names[regime_id] = 'bear_market'
                else:
                    regime_names[regime_id] = 'high_volatility'
            elif avg_return > 0.01:
                regime_names[regime_id] = 'bull_market'
            elif avg_return < -0.01:
                regime_names[regime_id] = 'bear_market'
            else:
                regime_names[regime_id] = 'sideways'
        
        return regime_names
    
    def ensemble_detection(self, data: pd.DataFrame, 
                          methods: List[DetectionMethod] = None) -> pd.DataFrame:
        """Ensemble regime detection using multiple methods."""
        
        if methods is None:
            methods = [
                DetectionMethod.VOLATILITY_CLUSTERING,
                DetectionMethod.HIDDEN_MARKOV_MODEL,
                DetectionMethod.STRUCTURAL_BREAKS
            ]
        
        # Extract features
        features = self.extract_features(data)
        returns = data['close'].pct_change() if 'close' in data.columns else data.iloc[:, 0].pct_change()
        prices = data['close'] if 'close' in data.columns else data.iloc[:, 0]
        
        regime_results = pd.DataFrame(index=data.index)
        
        # Apply each detection method
        for method in methods:
            if method == DetectionMethod.VOLATILITY_CLUSTERING:
                vol_regimes = self.detect_volatility_regimes(returns.dropna())
                regime_results[f'{method.value}_regime'] = vol_regimes.reindex(data.index)
                
            elif method == DetectionMethod.HIDDEN_MARKOV_MODEL:
                hmm_regimes, hmm_model = self.detect_hmm_regimes(features)
                regime_results[f'{method.value}_regime'] = hmm_regimes
                self.models['hmm'] = hmm_model
                
            elif method == DetectionMethod.STRUCTURAL_BREAKS:
                # Convert structural breaks to regimes
                breakpoints = self.detect_structural_breaks(returns)
                break_regimes = pd.Series(['stable'] * len(data), index=data.index)
                
                # Mark periods around breaks as transition
                for bp in breakpoints:
                    start_idx = max(0, bp - 5)
                    end_idx = min(len(data), bp + 5)
                    break_regimes.iloc[start_idx:end_idx] = 'transition'
                
                regime_results[f'{method.value}_regime'] = break_regimes
        
        # Add trend-based detection
        trend_regimes = self.detect_trend_regimes(prices)
        regime_results['trend_regime'] = trend_regimes
        
        # Add crisis detection
        crisis_regimes = self.detect_crisis_regimes(returns.dropna())
        regime_results['crisis_regime'] = crisis_regimes.reindex(data.index)
        
        # Ensemble voting
        regime_results['ensemble_regime'] = self._ensemble_vote(regime_results)
        
        return regime_results
    
    def _ensemble_vote(self, regime_results: pd.DataFrame) -> pd.Series:
        """Combine multiple regime predictions using voting."""
        
        ensemble_regimes = pd.Series(index=regime_results.index, dtype=object)
        
        for i in range(len(regime_results)):
            row = regime_results.iloc[i]
            
            # Count votes for each regime type
            regime_votes = {}
            
            for col in regime_results.columns:
                if col.endswith('_regime'):
                    regime = row[col]
                    if pd.notna(regime):
                        regime_votes[regime] = regime_votes.get(regime, 0) + 1
            
            # Assign regime with most votes
            if regime_votes:
                winning_regime = max(regime_votes.items(), key=lambda x: x[1])[0]
                ensemble_regimes.iloc[i] = winning_regime
            else:
                ensemble_regimes.iloc[i] = 'unknown'
        
        return ensemble_regimes
    
    def analyze_regime_transitions(self, regime_series: pd.Series) -> List[RegimeTransition]:
        """Analyze regime transitions and their characteristics."""
        
        transitions = []
        current_regime = None
        regime_start = None
        
        for i, (timestamp, regime) in enumerate(regime_series.items()):
            if regime != current_regime:
                # End of previous regime
                if current_regime is not None and regime_start is not None:
                    duration = i - regime_start
                    
                    transition = RegimeTransition(
                        from_regime=RegimeType(current_regime) if current_regime in [r.value for r in RegimeType] else RegimeType.UNKNOWN,
                        to_regime=RegimeType(regime) if regime in [r.value for r in RegimeType] else RegimeType.UNKNOWN,
                        transition_time=timestamp,
                        duration=duration,
                        confidence=1.0  # Could be calculated based on regime stability
                    )
                    
                    transitions.append(transition)
                
                # Start of new regime
                current_regime = regime
                regime_start = i
        
        return transitions
    
    def regime_persistence_analysis(self, regime_series: pd.Series) -> Dict[str, Dict[str, float]]:
        """Analyze regime persistence and transition probabilities."""
        
        # Calculate transition matrix
        unique_regimes = regime_series.unique()
        n_regimes = len(unique_regimes)
        
        transition_matrix = np.zeros((n_regimes, n_regimes))
        regime_to_idx = {regime: i for i, regime in enumerate(unique_regimes)}
        
        # Count transitions
        for i in range(len(regime_series) - 1):
            from_regime = regime_series.iloc[i]
            to_regime = regime_series.iloc[i + 1]
            
            if pd.notna(from_regime) and pd.notna(to_regime):
                from_idx = regime_to_idx[from_regime]
                to_idx = regime_to_idx[to_regime]
                transition_matrix[from_idx, to_idx] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        # Calculate persistence metrics
        persistence_metrics = {}
        
        for i, regime in enumerate(unique_regimes):
            if pd.isna(regime):
                continue
                
            # Self-transition probability (persistence)
            persistence = transition_matrix[i, i]
            
            # Expected duration
            if persistence < 1.0:
                expected_duration = 1 / (1 - persistence)
            else:
                expected_duration = np.inf
            
            # Most likely transition
            next_regime_idx = np.argmax(transition_matrix[i, :])
            if next_regime_idx != i:
                most_likely_transition = unique_regimes[next_regime_idx]
                transition_prob = transition_matrix[i, next_regime_idx]
            else:
                most_likely_transition = regime
                transition_prob = persistence
            
            persistence_metrics[regime] = {
                'persistence_probability': persistence,
                'expected_duration': expected_duration,
                'most_likely_transition': most_likely_transition,
                'transition_probability': transition_prob
            }
        
        return persistence_metrics
    
    def real_time_regime_detection(self, new_data: pd.Series, 
                                 lookback_window: int = 100) -> RegimeDetectionResult:
        """Real-time regime detection for streaming data."""
        
        if len(new_data) < lookback_window:
            return RegimeDetectionResult(
                timestamp=new_data.index[-1] if len(new_data) > 0 else pd.Timestamp.now(),
                regime=RegimeType.UNKNOWN,
                confidence=0.0,
                method=DetectionMethod.ENSEMBLE
            )
        
        # Use recent data window
        recent_data = new_data.tail(lookback_window)
        
        # Quick feature extraction
        returns = recent_data.pct_change().dropna()
        
        # Fast volatility regime detection
        current_vol = returns.tail(20).std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252)
        
        vol_ratio = current_vol / historical_vol
        
        # Fast trend detection
        trend_strength = np.polyfit(range(len(recent_data)), recent_data.values, 1)[0]
        
        # Fast momentum detection
        momentum = recent_data.iloc[-1] / recent_data.iloc[-20] - 1 if len(recent_data) >= 20 else 0
        
        # Simple regime classification
        if vol_ratio > 1.5:
            if momentum < -0.05:
                regime = RegimeType.CRISIS
                confidence = min(0.9, vol_ratio - 1.0)
            else:
                regime = RegimeType.HIGH_VOLATILITY
                confidence = min(0.8, vol_ratio - 1.0)
        elif abs(momentum) > 0.03:
            if momentum > 0:
                regime = RegimeType.BULL_MARKET
            else:
                regime = RegimeType.BEAR_MARKET
            confidence = min(0.8, abs(momentum) * 10)
        elif trend_strength > 0:
            regime = RegimeType.TRENDING_UP
            confidence = min(0.7, abs(trend_strength) * 1000)
        elif trend_strength < 0:
            regime = RegimeType.TRENDING_DOWN
            confidence = min(0.7, abs(trend_strength) * 1000)
        else:
            regime = RegimeType.SIDEWAYS
            confidence = 0.6
        
        # Update current regime
        self.current_regime = regime
        
        return RegimeDetectionResult(
            timestamp=new_data.index[-1],
            regime=regime,
            confidence=confidence,
            method=DetectionMethod.ENSEMBLE,
            features={
                'volatility_ratio': vol_ratio,
                'trend_strength': trend_strength,
                'momentum': momentum,
                'current_volatility': current_vol
            }
        )
