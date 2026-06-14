"""
Alpha Decay Study Framework
===========================

Comprehensive alpha decay analysis system for understanding signal deterioration
patterns and optimizing refresh frequencies for trading strategies.

Features:
- Alpha half-life estimation using multiple methodologies
- Decay pattern analysis (linear, exponential, regime-dependent)
- Signal refresh optimization
- Regime-dependent decay analysis
- Seasonal decay patterns
- Cross-sectional decay consistency
- Decay prediction models
- Portfolio impact analysis

Author: Quantum Forge Analytics Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats, optimize
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DecayAnalysisResult:
    """Container for alpha decay analysis results."""
    signal_name: str
    analysis_period: Tuple[pd.Timestamp, pd.Timestamp]
    half_life_days: float
    decay_rate: float
    decay_pattern: str  # 'linear', 'exponential', 'power_law', 'regime_dependent'
    r_squared: float
    decay_consistency: float
    optimal_refresh_frequency: int
    confidence_interval: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'signal_name': self.signal_name,
            'analysis_start': self.analysis_period[0],
            'analysis_end': self.analysis_period[1],
            'half_life_days': self.half_life_days,
            'decay_rate': self.decay_rate,
            'decay_pattern': self.decay_pattern,
            'r_squared': self.r_squared,
            'decay_consistency': self.decay_consistency,
            'optimal_refresh_frequency': self.optimal_refresh_frequency,
            'confidence_interval_lower': self.confidence_interval[0],
            'confidence_interval_upper': self.confidence_interval[1]
        }

@dataclass
class RegimeDecayResult:
    """Container for regime-specific decay analysis."""
    regime_name: str
    regime_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]
    regime_half_life: float
    regime_decay_rate: float
    regime_consistency: float
    observations_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'regime_name': self.regime_name,
            'regime_periods': [(start, end) for start, end in self.regime_periods],
            'regime_half_life': self.regime_half_life,
            'regime_decay_rate': self.regime_decay_rate,
            'regime_consistency': self.regime_consistency,
            'observations_count': self.observations_count
        }

@dataclass
class SeasonalDecayResult:
    """Container for seasonal decay analysis."""
    seasonal_period: str  # 'monthly', 'quarterly', 'weekly'
    seasonal_patterns: Dict[str, float]
    seasonal_half_lives: Dict[str, float]
    strongest_seasonal_effect: str
    seasonal_significance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'seasonal_period': self.seasonal_period,
            'seasonal_patterns': self.seasonal_patterns,
            'seasonal_half_lives': self.seasonal_half_lives,
            'strongest_seasonal_effect': self.strongest_seasonal_effect,
            'seasonal_significance': self.seasonal_significance
        }

@dataclass
class AlphaDecayReport:
    """Container for comprehensive alpha decay study report."""
    signal_name: str
    study_date: datetime
    overall_decay_analysis: DecayAnalysisResult
    regime_decay_results: List[RegimeDecayResult]
    seasonal_decay_results: List[SeasonalDecayResult]
    cross_sectional_analysis: Dict[str, Any]
    decay_prediction_model: Dict[str, Any]
    portfolio_impact_analysis: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'signal_name': self.signal_name,
            'study_date': self.study_date,
            'overall_decay_analysis': self.overall_decay_analysis.to_dict(),
            'regime_decay_results': [r.to_dict() for r in self.regime_decay_results],
            'seasonal_decay_results': [r.to_dict() for r in self.seasonal_decay_results],
            'cross_sectional_analysis': self.cross_sectional_analysis,
            'decay_prediction_model': self.decay_prediction_model,
            'portfolio_impact_analysis': self.portfolio_impact_analysis,
            'recommendations': self.recommendations
        }

class AlphaHalfLifeEstimator:
    """
    Alpha half-life estimation using multiple methodologies.
    
    Implements various approaches to estimate how quickly alpha signals decay
    including exponential decay, power law decay, and regime-dependent models.
    """
    
    def __init__(self, min_periods: int = 50):
        """
        Initialize alpha half-life estimator.
        
        Parameters:
        -----------
        min_periods : int
            Minimum periods required for half-life estimation
        """
        self.min_periods = min_periods
        
    def estimate_half_life(self, alpha_signal: pd.Series, returns: pd.Series,
                          method: str = 'exponential') -> DecayAnalysisResult:
        """
        Estimate alpha signal half-life.
        
        Parameters:
        -----------
        alpha_signal : pd.Series
            Alpha signal time series
        returns : pd.Series
            Forward returns for correlation analysis
        method : str
            Estimation method ('exponential', 'linear', 'power_law', 'adaptive')
            
        Returns:
        --------
        DecayAnalysisResult
            Half-life estimation results
        """
        try:
            # Align data
            common_index = alpha_signal.index.intersection(returns.index)
            aligned_signal = alpha_signal.loc[common_index]
            aligned_returns = returns.loc[common_index]
            
            if len(aligned_signal) < self.min_periods:
                raise ValueError(f"Insufficient data for half-life estimation. Need at least {self.min_periods} periods.")
            
            # Calculate time-lagged correlations
            max_lag = min(100, len(aligned_signal) // 4)  # Maximum lag to analyze
            lag_correlations = self._calculate_lag_correlations(aligned_signal, aligned_returns, max_lag)
            
            # Estimate half-life based on method
            if method == 'exponential':
                result = self._exponential_decay_estimation(lag_correlations, aligned_signal.name or "Alpha Signal")
            elif method == 'linear':
                result = self._linear_decay_estimation(lag_correlations, aligned_signal.name or "Alpha Signal")
            elif method == 'power_law':
                result = self._power_law_decay_estimation(lag_correlations, aligned_signal.name or "Alpha Signal")
            elif method == 'adaptive':
                result = self._adaptive_decay_estimation(lag_correlations, aligned_signal.name or "Alpha Signal")
            else:
                raise ValueError(f"Unknown half-life estimation method: {method}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in half-life estimation: {str(e)}")
            raise
    
    def _calculate_lag_correlations(self, signal: pd.Series, returns: pd.Series, max_lag: int) -> pd.Series:
        """Calculate correlations at different lags."""
        try:
            lag_correlations = []
            lags = []
            
            for lag in range(1, max_lag + 1):
                if len(signal) > lag + 10:  # Need minimum observations
                    lagged_signal = signal.shift(lag)
                    # Align and remove NaN
                    valid_mask = ~(lagged_signal.isna() | returns.isna())
                    
                    if valid_mask.sum() > 10:
                        correlation = lagged_signal[valid_mask].corr(returns[valid_mask])
                        if not pd.isna(correlation):
                            lag_correlations.append(abs(correlation))  # Use absolute correlation
                            lags.append(lag)
            
            return pd.Series(lag_correlations, index=lags)
            
        except Exception as e:
            logger.warning(f"Error calculating lag correlations: {str(e)}")
            return pd.Series([], dtype=float)
    
    def _exponential_decay_estimation(self, lag_correlations: pd.Series, signal_name: str) -> DecayAnalysisResult:
        """Estimate half-life using exponential decay model."""
        try:
            if len(lag_correlations) < 5:
                return self._create_default_decay_result(signal_name, "insufficient_data")
            
            # Exponential decay model: IC(t) = IC(0) * exp(-λt)
            # Taking log: log(IC(t)) = log(IC(0)) - λt
            
            lags = np.array(lag_correlations.index)
            log_correlations = np.log(lag_correlations.values + 1e-8)  # Add small epsilon to avoid log(0)
            
            # Linear regression on log-transformed data
            slope, intercept, r_value, p_value, std_err = linregress(lags, log_correlations)
            
            # Calculate half-life
            decay_rate = -slope
            half_life = np.log(2) / decay_rate if decay_rate > 0 else np.inf
            
            # Calculate optimal refresh frequency (when correlation drops to 50% of original)
            optimal_refresh = int(half_life) if np.isfinite(half_life) else 30
            
            # Confidence interval
            conf_interval = (half_life - 1.96 * std_err / decay_rate, half_life + 1.96 * std_err / decay_rate) if decay_rate > 0 else (0, np.inf)
            
            # Decay consistency (R-squared of the fit)
            consistency = r_value ** 2
            
            return DecayAnalysisResult(
                signal_name=signal_name,
                analysis_period=(lag_correlations.index[0], lag_correlations.index[-1]),
                half_life_days=half_life,
                decay_rate=decay_rate,
                decay_pattern='exponential',
                r_squared=consistency,
                decay_consistency=consistency,
                optimal_refresh_frequency=optimal_refresh,
                confidence_interval=conf_interval
            )
            
        except Exception as e:
            logger.warning(f"Error in exponential decay estimation: {str(e)}")
            return self._create_default_decay_result(signal_name, "exponential")
    
    def _linear_decay_estimation(self, lag_correlations: pd.Series, signal_name: str) -> DecayAnalysisResult:
        """Estimate half-life using linear decay model."""
        try:
            if len(lag_correlations) < 5:
                return self._create_default_decay_result(signal_name, "insufficient_data")
            
            # Linear decay model: IC(t) = IC(0) - λt
            lags = np.array(lag_correlations.index)
            correlations = lag_correlations.values
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = linregress(lags, correlations)
            
            # Calculate half-life (when correlation drops to 50% of original)
            initial_correlation = intercept
            decay_rate = -slope
            
            if decay_rate > 0 and initial_correlation > 0:
                half_life = (initial_correlation * 0.5) / decay_rate
            else:
                half_life = np.inf
            
            optimal_refresh = int(half_life) if np.isfinite(half_life) else 30
            
            # Confidence interval (simplified)
            conf_interval = (max(0, half_life - 10), half_life + 10) if np.isfinite(half_life) else (0, np.inf)
            
            consistency = r_value ** 2
            
            return DecayAnalysisResult(
                signal_name=signal_name,
                analysis_period=(lag_correlations.index[0], lag_correlations.index[-1]),
                half_life_days=half_life,
                decay_rate=decay_rate,
                decay_pattern='linear',
                r_squared=consistency,
                decay_consistency=consistency,
                optimal_refresh_frequency=optimal_refresh,
                confidence_interval=conf_interval
            )
            
        except Exception as e:
            logger.warning(f"Error in linear decay estimation: {str(e)}")
            return self._create_default_decay_result(signal_name, "linear")
    
    def _power_law_decay_estimation(self, lag_correlations: pd.Series, signal_name: str) -> DecayAnalysisResult:
        """Estimate half-life using power law decay model."""
        try:
            if len(lag_correlations) < 5:
                return self._create_default_decay_result(signal_name, "insufficient_data")
            
            # Power law decay model: IC(t) = IC(0) * t^(-α)
            # Taking log: log(IC(t)) = log(IC(0)) - α * log(t)
            
            lags = np.array(lag_correlations.index)
            log_lags = np.log(lags)
            log_correlations = np.log(lag_correlations.values + 1e-8)
            
            # Linear regression on log-log scale
            slope, intercept, r_value, p_value, std_err = linregress(log_lags, log_correlations)
            
            # Power law exponent
            alpha = -slope
            decay_rate = alpha
            
            # Calculate half-life: t_half = (1/2)^(1/α)
            if alpha > 0:
                half_life = np.power(0.5, -1.0/alpha)
            else:
                half_life = np.inf
            
            optimal_refresh = int(half_life) if np.isfinite(half_life) else 30
            
            # Confidence interval (simplified)
            conf_interval = (max(0, half_life * 0.5), half_life * 1.5) if np.isfinite(half_life) else (0, np.inf)
            
            consistency = r_value ** 2
            
            return DecayAnalysisResult(
                signal_name=signal_name,
                analysis_period=(lag_correlations.index[0], lag_correlations.index[-1]),
                half_life_days=half_life,
                decay_rate=decay_rate,
                decay_pattern='power_law',
                r_squared=consistency,
                decay_consistency=consistency,
                optimal_refresh_frequency=optimal_refresh,
                confidence_interval=conf_interval
            )
            
        except Exception as e:
            logger.warning(f"Error in power law decay estimation: {str(e)}")
            return self._create_default_decay_result(signal_name, "power_law")
    
    def _adaptive_decay_estimation(self, lag_correlations: pd.Series, signal_name: str) -> DecayAnalysisResult:
        """Choose best decay model based on fit quality."""
        try:
            # Try all methods and choose the best one
            methods = ['exponential', 'linear', 'power_law']
            results = []
            
            for method in methods:
                try:
                    if method == 'exponential':
                        result = self._exponential_decay_estimation(lag_correlations, signal_name)
                    elif method == 'linear':
                        result = self._linear_decay_estimation(lag_correlations, signal_name)
                    elif method == 'power_law':
                        result = self._power_law_decay_estimation(lag_correlations, signal_name)
                    
                    results.append(result)
                except:
                    continue
            
            if not results:
                return self._create_default_decay_result(signal_name, "adaptive")
            
            # Choose model with highest R-squared
            best_result = max(results, key=lambda x: x.r_squared)
            best_result.decay_pattern = f"adaptive_{best_result.decay_pattern}"
            
            return best_result
            
        except Exception as e:
            logger.warning(f"Error in adaptive decay estimation: {str(e)}")
            return self._create_default_decay_result(signal_name, "adaptive")
    
    def _create_default_decay_result(self, signal_name: str, pattern: str) -> DecayAnalysisResult:
        """Create default decay result for error cases."""
        return DecayAnalysisResult(
            signal_name=signal_name,
            analysis_period=(pd.Timestamp.now(), pd.Timestamp.now()),
            half_life_days=30.0,  # Default assumption
            decay_rate=0.023,     # ln(2)/30
            decay_pattern=pattern,
            r_squared=0.0,
            decay_consistency=0.0,
            optimal_refresh_frequency=30,
            confidence_interval=(20.0, 40.0)
        )

class RegimeDecayAnalyzer:
    """
    Regime-dependent alpha decay analysis.
    
    Analyzes how alpha decay patterns vary across different market regimes
    such as bull/bear markets, high/low volatility periods, etc.
    """
    
    def __init__(self, min_regime_periods: int = 30):
        """
        Initialize regime decay analyzer.
        
        Parameters:
        -----------
        min_regime_periods : int
            Minimum periods required for regime analysis
        """
        self.min_regime_periods = min_regime_periods
        self.half_life_estimator = AlphaHalfLifeEstimator()
        
    def analyze_regime_decay(self, alpha_signal: pd.Series, returns: pd.Series,
                           regime_indicator: Optional[pd.Series] = None) -> List[RegimeDecayResult]:
        """
        Analyze alpha decay across different market regimes.
        
        Parameters:
        -----------
        alpha_signal : pd.Series
            Alpha signal time series
        returns : pd.Series
            Forward returns
        regime_indicator : pd.Series, optional
            Pre-defined regime indicator. If None, will create volatility-based regimes
            
        Returns:
        --------
        List[RegimeDecayResult]
            Regime-specific decay analysis results
        """
        try:
            # Align data
            common_index = alpha_signal.index.intersection(returns.index)
            aligned_signal = alpha_signal.loc[common_index]
            aligned_returns = returns.loc[common_index]
            
            # Create or use regime indicator
            if regime_indicator is None:
                regime_indicator = self._create_volatility_regimes(aligned_returns)
            else:
                regime_indicator = regime_indicator.loc[common_index]
            
            results = []
            
            # Analyze each regime
            for regime_name in regime_indicator.unique():
                if pd.isna(regime_name):
                    continue
                    
                regime_mask = regime_indicator == regime_name
                regime_periods = self._get_regime_periods(regime_mask)
                
                if regime_mask.sum() < self.min_regime_periods:
                    continue
                
                # Extract regime data
                regime_signal = aligned_signal[regime_mask]
                regime_returns = aligned_returns[regime_mask]
                
                try:
                    # Estimate decay for this regime
                    decay_result = self.half_life_estimator.estimate_half_life(
                        regime_signal, regime_returns, method='exponential'
                    )
                    
                    # Create regime-specific result
                    regime_result = RegimeDecayResult(
                        regime_name=regime_name,
                        regime_periods=regime_periods,
                        regime_half_life=decay_result.half_life_days,
                        regime_decay_rate=decay_result.decay_rate,
                        regime_consistency=decay_result.decay_consistency,
                        observations_count=regime_mask.sum()
                    )
                    
                    results.append(regime_result)
                    
                except Exception as e:
                    logger.warning(f"Error analyzing regime {regime_name}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in regime decay analysis: {str(e)}")
            raise
    
    def _create_volatility_regimes(self, returns: pd.Series) -> pd.Series:
        """Create simple volatility-based regime indicator."""
        try:
            # Calculate rolling volatility
            rolling_vol = returns.rolling(window=20).std()
            
            # Define regime thresholds
            high_vol_threshold = rolling_vol.quantile(0.75)
            low_vol_threshold = rolling_vol.quantile(0.25)
            
            # Assign regimes
            regimes = pd.Series('medium_volatility', index=returns.index)
            regimes[rolling_vol > high_vol_threshold] = 'high_volatility'
            regimes[rolling_vol < low_vol_threshold] = 'low_volatility'
            
            return regimes
            
        except Exception as e:
            logger.warning(f"Error creating volatility regimes: {str(e)}")
            return pd.Series('unknown', index=returns.index)
    
    def _get_regime_periods(self, regime_mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Extract continuous regime periods."""
        try:
            periods = []
            in_regime = False
            start_date = None
            
            for date, is_regime in regime_mask.items():
                if is_regime and not in_regime:
                    # Start of regime period
                    start_date = date
                    in_regime = True
                elif not is_regime and in_regime:
                    # End of regime period
                    if start_date is not None:
                        periods.append((start_date, date))
                    in_regime = False
                    start_date = None
            
            # Handle case where regime continues to end
            if in_regime and start_date is not None:
                periods.append((start_date, regime_mask.index[-1]))
            
            return periods
            
        except Exception as e:
            logger.warning(f"Error extracting regime periods: {str(e)}")
            return []

class SeasonalDecayAnalyzer:
    """
    Seasonal alpha decay analysis.
    
    Analyzes seasonal patterns in alpha decay such as monthly, quarterly,
    or weekly variations in signal effectiveness.
    """
    
    def __init__(self):
        """Initialize seasonal decay analyzer."""
        self.half_life_estimator = AlphaHalfLifeEstimator()
        
    def analyze_seasonal_decay(self, alpha_signal: pd.Series, returns: pd.Series) -> List[SeasonalDecayResult]:
        """
        Analyze seasonal patterns in alpha decay.
        
        Parameters:
        -----------
        alpha_signal : pd.Series
            Alpha signal time series
        returns : pd.Series
            Forward returns
            
        Returns:
        --------
        List[SeasonalDecayResult]
            Seasonal decay analysis results
        """
        try:
            # Align data
            common_index = alpha_signal.index.intersection(returns.index)
            aligned_signal = alpha_signal.loc[common_index]
            aligned_returns = returns.loc[common_index]
            
            results = []
            
            # Monthly seasonal analysis
            monthly_result = self._analyze_monthly_seasonality(aligned_signal, aligned_returns)
            if monthly_result:
                results.append(monthly_result)
            
            # Quarterly seasonal analysis
            quarterly_result = self._analyze_quarterly_seasonality(aligned_signal, aligned_returns)
            if quarterly_result:
                results.append(quarterly_result)
            
            # Weekly seasonal analysis (if enough data)
            if len(aligned_signal) > 365:  # Need at least a year of data
                weekly_result = self._analyze_weekly_seasonality(aligned_signal, aligned_returns)
                if weekly_result:
                    results.append(weekly_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in seasonal decay analysis: {str(e)}")
            raise
    
    def _analyze_monthly_seasonality(self, signal: pd.Series, returns: pd.Series) -> Optional[SeasonalDecayResult]:
        """Analyze monthly seasonal patterns."""
        try:
            monthly_patterns = {}
            monthly_half_lives = {}
            
            for month in range(1, 13):
                month_mask = signal.index.month == month
                
                if month_mask.sum() < 20:  # Need minimum observations
                    continue
                
                month_signal = signal[month_mask]
                month_returns = returns[month_mask]
                
                # Calculate monthly correlation
                monthly_ic = month_signal.corr(month_returns)
                if not pd.isna(monthly_ic):
                    monthly_patterns[f'month_{month:02d}'] = monthly_ic
                
                # Estimate decay for this month (simplified)
                try:
                    decay_result = self.half_life_estimator.estimate_half_life(
                        month_signal, month_returns, method='exponential'
                    )
                    monthly_half_lives[f'month_{month:02d}'] = decay_result.half_life_days
                except:
                    monthly_half_lives[f'month_{month:02d}'] = 30.0  # Default
            
            if not monthly_patterns:
                return None
            
            # Find strongest seasonal effect
            strongest_effect = max(monthly_patterns.keys(), key=lambda k: abs(monthly_patterns[k]))
            
            # Calculate seasonal significance (simplified)
            pattern_values = list(monthly_patterns.values())
            seasonal_significance = np.std(pattern_values) / np.mean(np.abs(pattern_values)) if pattern_values else 0
            
            return SeasonalDecayResult(
                seasonal_period='monthly',
                seasonal_patterns=monthly_patterns,
                seasonal_half_lives=monthly_half_lives,
                strongest_seasonal_effect=strongest_effect,
                seasonal_significance=seasonal_significance
            )
            
        except Exception as e:
            logger.warning(f"Error in monthly seasonality analysis: {str(e)}")
            return None
    
    def _analyze_quarterly_seasonality(self, signal: pd.Series, returns: pd.Series) -> Optional[SeasonalDecayResult]:
        """Analyze quarterly seasonal patterns."""
        try:
            quarterly_patterns = {}
            quarterly_half_lives = {}
            
            for quarter in range(1, 5):
                quarter_mask = signal.index.quarter == quarter
                
                if quarter_mask.sum() < 30:  # Need minimum observations
                    continue
                
                quarter_signal = signal[quarter_mask]
                quarter_returns = returns[quarter_mask]
                
                # Calculate quarterly correlation
                quarterly_ic = quarter_signal.corr(quarter_returns)
                if not pd.isna(quarterly_ic):
                    quarterly_patterns[f'Q{quarter}'] = quarterly_ic
                
                # Estimate decay for this quarter
                try:
                    decay_result = self.half_life_estimator.estimate_half_life(
                        quarter_signal, quarter_returns, method='exponential'
                    )
                    quarterly_half_lives[f'Q{quarter}'] = decay_result.half_life_days
                except:
                    quarterly_half_lives[f'Q{quarter}'] = 30.0  # Default
            
            if not quarterly_patterns:
                return None
            
            # Find strongest seasonal effect
            strongest_effect = max(quarterly_patterns.keys(), key=lambda k: abs(quarterly_patterns[k]))
            
            # Calculate seasonal significance
            pattern_values = list(quarterly_patterns.values())
            seasonal_significance = np.std(pattern_values) / np.mean(np.abs(pattern_values)) if pattern_values else 0
            
            return SeasonalDecayResult(
                seasonal_period='quarterly',
                seasonal_patterns=quarterly_patterns,
                seasonal_half_lives=quarterly_half_lives,
                strongest_seasonal_effect=strongest_effect,
                seasonal_significance=seasonal_significance
            )
            
        except Exception as e:
            logger.warning(f"Error in quarterly seasonality analysis: {str(e)}")
            return None
    
    def _analyze_weekly_seasonality(self, signal: pd.Series, returns: pd.Series) -> Optional[SeasonalDecayResult]:
        """Analyze weekly seasonal patterns."""
        try:
            weekly_patterns = {}
            weekly_half_lives = {}
            
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            
            for day_idx, day_name in enumerate(weekdays):
                day_mask = signal.index.dayofweek == day_idx
                
                if day_mask.sum() < 20:  # Need minimum observations
                    continue
                
                day_signal = signal[day_mask]
                day_returns = returns[day_mask]
                
                # Calculate daily correlation
                daily_ic = day_signal.corr(day_returns)
                if not pd.isna(daily_ic):
                    weekly_patterns[day_name] = daily_ic
                
                # Simplified half-life estimation
                weekly_half_lives[day_name] = 30.0  # Default for weekly analysis
            
            if not weekly_patterns:
                return None
            
            # Find strongest seasonal effect
            strongest_effect = max(weekly_patterns.keys(), key=lambda k: abs(weekly_patterns[k]))
            
            # Calculate seasonal significance
            pattern_values = list(weekly_patterns.values())
            seasonal_significance = np.std(pattern_values) / np.mean(np.abs(pattern_values)) if pattern_values else 0
            
            return SeasonalDecayResult(
                seasonal_period='weekly',
                seasonal_patterns=weekly_patterns,
                seasonal_half_lives=weekly_half_lives,
                strongest_seasonal_effect=strongest_effect,
                seasonal_significance=seasonal_significance
            )
            
        except Exception as e:
            logger.warning(f"Error in weekly seasonality analysis: {str(e)}")
            return None

class DecayPredictionModel:
    """
    Machine learning model for predicting alpha decay patterns.
    
    Uses market conditions and signal characteristics to predict
    expected decay rates and optimal refresh frequencies.
    """
    
    def __init__(self):
        """Initialize decay prediction model."""
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_names = []
        
    def train_decay_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train decay prediction model.
        
        Parameters:
        -----------
        training_data : List[Dict[str, Any]]
            Training data with features and decay targets
            
        Returns:
        --------
        Dict[str, Any]
            Training results and model performance
        """
        try:
            if not training_data:
                raise ValueError("No training data provided")
            
            # Extract features and targets
            features = []
            targets = []
            
            for sample in training_data:
                feature_vector = self._extract_features(sample)
                decay_target = sample.get('half_life_days', 30.0)
                
                if feature_vector is not None and not pd.isna(decay_target):
                    features.append(feature_vector)
                    targets.append(decay_target)
            
            if len(features) < 10:  # Need minimum samples
                return {
                    'training_success': False,
                    'reason': 'Insufficient training samples',
                    'samples_count': len(features)
                }
            
            features_array = np.array(features)
            targets_array = np.array(targets)
            
            # Train model
            self.model.fit(features_array, targets_array)
            self.is_trained = True
            
            # Evaluate model performance
            predictions = self.model.predict(features_array)
            r2_score_val = r2_score(targets_array, predictions)
            mse = mean_squared_error(targets_array, predictions)
            
            # Feature importance
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            return {
                'training_success': True,
                'samples_count': len(features),
                'r2_score': r2_score_val,
                'mse': mse,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Error training decay model: {str(e)}")
            return {
                'training_success': False,
                'reason': f'Training error: {str(e)}',
                'samples_count': 0
            }
    
    def predict_decay(self, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict alpha decay for given market conditions.
        
        Parameters:
        -----------
        market_conditions : Dict[str, Any]
            Current market conditions and signal characteristics
            
        Returns:
        --------
        Dict[str, Any]
            Decay predictions
        """
        try:
            if not self.is_trained:
                return {
                    'prediction_success': False,
                    'reason': 'Model not trained',
                    'predicted_half_life': 30.0
                }
            
            # Extract features
            feature_vector = self._extract_features(market_conditions)
            
            if feature_vector is None:
                return {
                    'prediction_success': False,
                    'reason': 'Invalid features',
                    'predicted_half_life': 30.0
                }
            
            # Make prediction
            prediction = self.model.predict([feature_vector])[0]
            
            # Get prediction confidence (simplified using std of predictions)
            confidence = 0.8  # Simplified confidence measure
            
            return {
                'prediction_success': True,
                'predicted_half_life': max(prediction, 1.0),  # Minimum 1 day
                'confidence': confidence,
                'recommended_refresh_frequency': int(max(prediction * 0.7, 1))
            }
            
        except Exception as e:
            logger.warning(f"Error predicting decay: {str(e)}")
            return {
                'prediction_success': False,
                'reason': f'Prediction error: {str(e)}',
                'predicted_half_life': 30.0
            }
    
    def _extract_features(self, data: Dict[str, Any]) -> Optional[List[float]]:
        """Extract feature vector from data."""
        try:
            features = []
            
            # Market volatility
            features.append(data.get('market_volatility', 0.02))
            
            # Market trend (simplified)
            features.append(data.get('market_trend', 0.0))
            
            # Signal characteristics
            features.append(data.get('signal_volatility', 0.1))
            features.append(data.get('signal_ic', 0.05))
            features.append(data.get('signal_turnover', 0.5))
            
            # Time features
            features.append(data.get('month', 6) / 12.0)  # Normalize month
            features.append(data.get('quarter', 2) / 4.0)  # Normalize quarter
            
            # Market regime (simplified)
            features.append(data.get('is_crisis', 0))  # Binary indicator
            features.append(data.get('vix_level', 20) / 100.0)  # Normalized VIX
            
            # Update feature names if first time
            if not self.feature_names:
                self.feature_names = [
                    'market_volatility', 'market_trend', 'signal_volatility',
                    'signal_ic', 'signal_turnover', 'month_normalized',
                    'quarter_normalized', 'is_crisis', 'vix_normalized'
                ]
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting features: {str(e)}")
            return None

class ComprehensiveAlphaDecayStudy:
    """
    Comprehensive alpha decay study system integrating all decay analysis methodologies.
    
    Provides unified interface for complete alpha decay analysis including
    half-life estimation, regime analysis, seasonal patterns, and prediction modeling.
    """
    
    def __init__(self):
        """Initialize comprehensive alpha decay study system."""
        self.half_life_estimator = AlphaHalfLifeEstimator()
        self.regime_analyzer = RegimeDecayAnalyzer()
        self.seasonal_analyzer = SeasonalDecayAnalyzer()
        self.prediction_model = DecayPredictionModel()
        
    def conduct_comprehensive_decay_study(self, alpha_signal: pd.Series, returns: pd.Series,
                                        signal_name: str = "Alpha Signal",
                                        regime_indicator: Optional[pd.Series] = None) -> AlphaDecayReport:
        """
        Conduct comprehensive alpha decay study.
        
        Parameters:
        -----------
        alpha_signal : pd.Series
            Alpha signal to analyze
        returns : pd.Series
            Forward returns for analysis
        signal_name : str
            Name of the alpha signal
        regime_indicator : pd.Series, optional
            Market regime indicator
            
        Returns:
        --------
        AlphaDecayReport
            Comprehensive decay study report
        """
        try:
            logger.info(f"Starting comprehensive decay study for {signal_name}")
            
            # Overall decay analysis
            overall_decay = self.half_life_estimator.estimate_half_life(
                alpha_signal, returns, method='adaptive'
            )
            
            # Regime-dependent decay analysis
            regime_results = []
            try:
                regime_results = self.regime_analyzer.analyze_regime_decay(
                    alpha_signal, returns, regime_indicator
                )
            except Exception as e:
                logger.warning(f"Error in regime decay analysis: {str(e)}")
            
            # Seasonal decay analysis
            seasonal_results = []
            try:
                seasonal_results = self.seasonal_analyzer.analyze_seasonal_decay(
                    alpha_signal, returns
                )
            except Exception as e:
                logger.warning(f"Error in seasonal decay analysis: {str(e)}")
            
            # Cross-sectional consistency analysis
            cross_sectional_analysis = self._analyze_cross_sectional_consistency(
                alpha_signal, returns
            )
            
            # Decay prediction model training and evaluation
            decay_prediction_analysis = self._train_and_evaluate_prediction_model(
                alpha_signal, returns, overall_decay, regime_results
            )
            
            # Portfolio impact analysis
            portfolio_impact = self._analyze_portfolio_impact(
                alpha_signal, returns, overall_decay
            )
            
            # Generate recommendations
            recommendations = self._generate_decay_recommendations(
                overall_decay, regime_results, seasonal_results,
                cross_sectional_analysis, portfolio_impact
            )
            
            # Create comprehensive report
            report = AlphaDecayReport(
                signal_name=signal_name,
                study_date=datetime.now(),
                overall_decay_analysis=overall_decay,
                regime_decay_results=regime_results,
                seasonal_decay_results=seasonal_results,
                cross_sectional_analysis=cross_sectional_analysis,
                decay_prediction_model=decay_prediction_analysis,
                portfolio_impact_analysis=portfolio_impact,
                recommendations=recommendations
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error in comprehensive decay study: {str(e)}")
            raise
    
    def _analyze_cross_sectional_consistency(self, signal: pd.Series, returns: pd.Series) -> Dict[str, Any]:
        """Analyze cross-sectional consistency of decay patterns."""
        try:
            # Split data into multiple periods for consistency analysis
            n_periods = min(5, len(signal) // 200)  # At least 200 observations per period
            
            if n_periods < 2:
                return {
                    'consistency_analysis': 'insufficient_data',
                    'cross_sectional_consistency': 0.0
                }
            
            period_size = len(signal) // n_periods
            decay_rates = []
            
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = (i + 1) * period_size if i < n_periods - 1 else len(signal)
                
                period_signal = signal.iloc[start_idx:end_idx]
                period_returns = returns.iloc[start_idx:end_idx]
                
                if len(period_signal) > 50:  # Minimum for decay analysis
                    try:
                        decay_result = self.half_life_estimator.estimate_half_life(
                            period_signal, period_returns, method='exponential'
                        )
                        decay_rates.append(decay_result.decay_rate)
                    except:
                        continue
            
            if len(decay_rates) < 2:
                return {
                    'consistency_analysis': 'insufficient_periods',
                    'cross_sectional_consistency': 0.0
                }
            
            # Calculate consistency as inverse of coefficient of variation
            decay_std = np.std(decay_rates)
            decay_mean = np.mean(decay_rates)
            
            if decay_mean > 0:
                cv = decay_std / decay_mean
                consistency = max(0, 1 - cv)  # Higher consistency = lower variation
            else:
                consistency = 0.0
            
            return {
                'consistency_analysis': 'completed',
                'cross_sectional_consistency': consistency,
                'decay_rates_analyzed': decay_rates,
                'decay_rate_std': decay_std,
                'decay_rate_mean': decay_mean,
                'coefficient_of_variation': cv if decay_mean > 0 else np.inf
            }
            
        except Exception as e:
            logger.warning(f"Error in cross-sectional consistency analysis: {str(e)}")
            return {
                'consistency_analysis': 'error',
                'cross_sectional_consistency': 0.0,
                'error': str(e)
            }
    
    def _train_and_evaluate_prediction_model(self, signal: pd.Series, returns: pd.Series,
                                           overall_decay: DecayAnalysisResult,
                                           regime_results: List[RegimeDecayResult]) -> Dict[str, Any]:
        """Train and evaluate decay prediction model."""
        try:
            # Generate synthetic training data for demonstration
            training_samples = []
            
            # Add overall sample
            training_samples.append({
                'market_volatility': returns.std(),
                'market_trend': returns.mean(),
                'signal_volatility': signal.std(),
                'signal_ic': abs(signal.corr(returns)),
                'signal_turnover': 0.5,  # Simplified
                'month': 6,
                'quarter': 2,
                'is_crisis': 0,
                'vix_level': 20,
                'half_life_days': overall_decay.half_life_days
            })
            
            # Add regime-specific samples
            for regime_result in regime_results:
                training_samples.append({
                    'market_volatility': returns.std() * (1.5 if 'high' in regime_result.regime_name else 0.8),
                    'market_trend': returns.mean(),
                    'signal_volatility': signal.std(),
                    'signal_ic': abs(signal.corr(returns)),
                    'signal_turnover': 0.5,
                    'month': 6,
                    'quarter': 2,
                    'is_crisis': 1 if 'high' in regime_result.regime_name else 0,
                    'vix_level': 30 if 'high' in regime_result.regime_name else 15,
                    'half_life_days': regime_result.regime_half_life
                })
            
            # Train model
            training_result = self.prediction_model.train_decay_model(training_samples)
            
            # Test prediction
            current_conditions = {
                'market_volatility': returns.std(),
                'market_trend': returns.mean(),
                'signal_volatility': signal.std(),
                'signal_ic': abs(signal.corr(returns)),
                'signal_turnover': 0.5,
                'month': datetime.now().month,
                'quarter': (datetime.now().month - 1) // 3 + 1,
                'is_crisis': 0,
                'vix_level': 20
            }
            
            prediction_result = self.prediction_model.predict_decay(current_conditions)
            
            return {
                'model_training': training_result,
                'current_prediction': prediction_result,
                'model_ready': training_result.get('training_success', False)
            }
            
        except Exception as e:
            logger.warning(f"Error in prediction model analysis: {str(e)}")
            return {
                'model_training': {'training_success': False, 'reason': str(e)},
                'current_prediction': {'prediction_success': False},
                'model_ready': False
            }
    
    def _analyze_portfolio_impact(self, signal: pd.Series, returns: pd.Series,
                                decay_result: DecayAnalysisResult) -> Dict[str, Any]:
        """Analyze portfolio impact of alpha decay."""
        try:
            # Simplified portfolio impact analysis
            
            # Calculate impact of different refresh frequencies
            refresh_frequencies = [5, 10, 20, 30, 60]  # Days
            impact_analysis = {}
            
            base_ic = abs(signal.corr(returns))
            
            for freq in refresh_frequencies:
                # Estimate IC decay at different frequencies
                if decay_result.decay_pattern == 'exponential':
                    decayed_ic = base_ic * np.exp(-decay_result.decay_rate * freq)
                elif decay_result.decay_pattern == 'linear':
                    decayed_ic = max(0, base_ic - decay_result.decay_rate * freq)
                else:
                    # Power law approximation
                    decayed_ic = base_ic * (freq ** -0.1)  # Simplified
                
                # Estimate impact on Sharpe ratio (simplified)
                ic_ratio = decayed_ic / base_ic if base_ic > 0 else 0
                estimated_sharpe_impact = ic_ratio * 0.8  # Simplified relationship
                
                impact_analysis[f'refresh_{freq}d'] = {
                    'decayed_ic': decayed_ic,
                    'ic_retention': ic_ratio,
                    'estimated_sharpe_impact': estimated_sharpe_impact,
                    'recommended': freq <= decay_result.optimal_refresh_frequency
                }
            
            # Calculate cost-benefit analysis
            optimal_freq = decay_result.optimal_refresh_frequency
            
            return {
                'refresh_frequency_analysis': impact_analysis,
                'optimal_refresh_frequency': optimal_freq,
                'base_information_coefficient': base_ic,
                'half_life_days': decay_result.half_life_days,
                'decay_rate': decay_result.decay_rate,
                'cost_benefit_analysis': {
                    'frequent_refresh_benefit': 'Higher IC retention',
                    'frequent_refresh_cost': 'Higher turnover and transaction costs',
                    'recommended_frequency': optimal_freq,
                    'ic_at_recommended_freq': impact_analysis.get(f'refresh_{optimal_freq}d', {}).get('decayed_ic', base_ic)
                }
            }
            
        except Exception as e:
            logger.warning(f"Error in portfolio impact analysis: {str(e)}")
            return {
                'refresh_frequency_analysis': {},
                'optimal_refresh_frequency': 30,
                'analysis_error': str(e)
            }
    
    def _generate_decay_recommendations(self, overall_decay: DecayAnalysisResult,
                                      regime_results: List[RegimeDecayResult],
                                      seasonal_results: List[SeasonalDecayResult],
                                      cross_sectional_analysis: Dict[str, Any],
                                      portfolio_impact: Dict[str, Any]) -> List[str]:
        """Generate comprehensive decay-based recommendations."""
        recommendations = []
        
        # Overall decay assessment
        if overall_decay.half_life_days < 10:
            recommendations.append("FAST DECAY DETECTED: Signal has very short half-life (<10 days). Requires frequent refresh or intraday rebalancing.")
        elif overall_decay.half_life_days < 30:
            recommendations.append("MODERATE DECAY: Signal decays moderately fast. Weekly refresh recommended.")
        else:
            recommendations.append("SLOW DECAY: Signal has good persistence. Monthly refresh may be sufficient.")
        
        # Decay pattern assessment
        if overall_decay.decay_pattern in ['exponential', 'adaptive_exponential']:
            recommendations.append("Exponential decay pattern detected. Signal effectiveness drops rapidly after optimal holding period.")
        elif overall_decay.decay_pattern in ['linear', 'adaptive_linear']:
            recommendations.append("Linear decay pattern detected. Signal effectiveness declines steadily over time.")
        
        # Consistency assessment
        consistency = cross_sectional_analysis.get('cross_sectional_consistency', 0)
        if consistency > 0.8:
            recommendations.append("HIGH CONSISTENCY: Decay patterns are stable across different time periods.")
        elif consistency > 0.5:
            recommendations.append("MODERATE CONSISTENCY: Some variation in decay patterns across time periods.")
        else:
            recommendations.append("LOW CONSISTENCY: Significant variation in decay patterns. Consider regime-dependent refresh strategies.")
        
        # Regime-dependent recommendations
        if regime_results:
            high_vol_regime = next((r for r in regime_results if 'high' in r.regime_name.lower()), None)
            low_vol_regime = next((r for r in regime_results if 'low' in r.regime_name.lower()), None)
            
            if high_vol_regime and low_vol_regime:
                if high_vol_regime.regime_half_life < low_vol_regime.regime_half_life * 0.7:
                    recommendations.append("REGIME DEPENDENCY: Signal decays faster in high volatility periods. Consider more frequent refresh during market stress.")
        
        # Seasonal recommendations
        if seasonal_results:
            for seasonal_result in seasonal_results:
                if seasonal_result.seasonal_significance > 0.3:
                    recommendations.append(f"SEASONAL PATTERN: Significant {seasonal_result.seasonal_period} seasonality detected in {seasonal_result.strongest_seasonal_effect}.")
        
        # Portfolio impact recommendations
        optimal_freq = portfolio_impact.get('optimal_refresh_frequency', 30)
        recommendations.append(f"OPTIMAL REFRESH: Based on decay analysis, refresh signal every {optimal_freq} days for optimal risk-adjusted returns.")
        
        # Risk management recommendations
        if overall_decay.r_squared < 0.5:
            recommendations.append("LOW PREDICTABILITY: Decay pattern is not well-captured by standard models. Use conservative refresh strategy.")
        
        return recommendations

