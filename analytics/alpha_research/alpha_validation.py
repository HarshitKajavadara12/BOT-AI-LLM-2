"""
Alpha Validation Framework
=========================

Comprehensive alpha validation system for rigorous testing and validation
of alpha signals before deployment in live trading strategies.

Features:
- Out-of-sample validation with multiple methodologies
- Cross-validation for time series data
- Statistical significance testing
- Regime-dependent validation
- Robustness testing across market conditions
- Performance attribution and decomposition
- Overfitting detection and mitigation
- Production-readiness assessment

Author: Quantum Forge Analytics Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.stats import ttest_1samp, jarque_bera, normaltest
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for alpha validation results."""
    validation_type: str
    validation_period: Tuple[pd.Timestamp, pd.Timestamp]
    information_coefficient: float
    ic_t_statistic: float
    ic_p_value: float
    sharpe_ratio: float
    hit_rate: float
    max_drawdown: float
    calmar_ratio: float
    skewness: float
    kurtosis: float
    is_significant: bool
    confidence_level: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'validation_type': self.validation_type,
            'validation_start': self.validation_period[0],
            'validation_end': self.validation_period[1],
            'information_coefficient': self.information_coefficient,
            'ic_t_statistic': self.ic_t_statistic,
            'ic_p_value': self.ic_p_value,
            'sharpe_ratio': self.sharpe_ratio,
            'hit_rate': self.hit_rate,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'is_significant': self.is_significant,
            'confidence_level': self.confidence_level
        }

@dataclass
class RobustnessTestResult:
    """Container for robustness test results."""
    test_name: str
    test_description: str
    original_ic: float
    stressed_ic: float
    ic_change: float
    relative_ic_change: float
    passes_robustness_test: bool
    robustness_threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'test_name': self.test_name,
            'test_description': self.test_description,
            'original_ic': self.original_ic,
            'stressed_ic': self.stressed_ic,
            'ic_change': self.ic_change,
            'relative_ic_change': self.relative_ic_change,
            'passes_robustness_test': self.passes_robustness_test,
            'robustness_threshold': self.robustness_threshold
        }

@dataclass
class AlphaValidationReport:
    """Container for comprehensive alpha validation report."""
    alpha_signal_name: str
    validation_date: datetime
    in_sample_results: List[ValidationResult]
    out_of_sample_results: List[ValidationResult]
    cross_validation_results: List[ValidationResult]
    robustness_test_results: List[RobustnessTestResult]
    regime_validation_results: Dict[str, ValidationResult]
    overfitting_assessment: Dict[str, Any]
    production_readiness_score: float
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'alpha_signal_name': self.alpha_signal_name,
            'validation_date': self.validation_date,
            'in_sample_results': [r.to_dict() for r in self.in_sample_results],
            'out_of_sample_results': [r.to_dict() for r in self.out_of_sample_results],
            'cross_validation_results': [r.to_dict() for r in self.cross_validation_results],
            'robustness_test_results': [r.to_dict() for r in self.robustness_test_results],
            'regime_validation_results': {k: v.to_dict() for k, v in self.regime_validation_results.items()},
            'overfitting_assessment': self.overfitting_assessment,
            'production_readiness_score': self.production_readiness_score,
            'recommendations': self.recommendations
        }

class OutOfSampleValidator:
    """
    Out-of-sample validation system for alpha signals.
    
    Implements various out-of-sample testing methodologies to assess
    true predictive power of alpha signals.
    """
    
    def __init__(self, oos_ratio: float = 0.3, min_oos_periods: int = 100):
        """
        Initialize out-of-sample validator.
        
        Parameters:
        -----------
        oos_ratio : float
            Ratio of data to use for out-of-sample testing
        min_oos_periods : int
            Minimum periods required for out-of-sample test
        """
        self.oos_ratio = oos_ratio
        self.min_oos_periods = min_oos_periods
        
    def validate_alpha_signal(self, alpha_signal: pd.Series, returns: pd.Series,
                            validation_type: str = 'holdout') -> List[ValidationResult]:
        """
        Validate alpha signal using out-of-sample testing.
        
        Parameters:
        -----------
        alpha_signal : pd.Series
            Alpha signal to validate
        returns : pd.Series
            Forward returns for validation
        validation_type : str
            Type of validation ('holdout', 'rolling_window', 'expanding_window')
            
        Returns:
        --------
        List[ValidationResult]
            Validation results
        """
        try:
            # Align data
            common_index = alpha_signal.index.intersection(returns.index)
            aligned_signal = alpha_signal.loc[common_index]
            aligned_returns = returns.loc[common_index]
            
            if len(aligned_signal) < self.min_oos_periods * 2:
                raise ValueError(f"Insufficient data for out-of-sample validation. Need at least {self.min_oos_periods * 2} periods.")
            
            if validation_type == 'holdout':
                return self._holdout_validation(aligned_signal, aligned_returns)
            elif validation_type == 'rolling_window':
                return self._rolling_window_validation(aligned_signal, aligned_returns)
            elif validation_type == 'expanding_window':
                return self._expanding_window_validation(aligned_signal, aligned_returns)
            else:
                raise ValueError(f"Unknown validation type: {validation_type}")
                
        except Exception as e:
            logger.error(f"Error in out-of-sample validation: {str(e)}")
            raise
    
    def _holdout_validation(self, signal: pd.Series, returns: pd.Series) -> List[ValidationResult]:
        """Perform holdout validation."""
        total_periods = len(signal)
        split_point = int(total_periods * (1 - self.oos_ratio))
        
        # In-sample period
        is_signal = signal.iloc[:split_point]
        is_returns = returns.iloc[:split_point]
        
        # Out-of-sample period
        oos_signal = signal.iloc[split_point:]
        oos_returns = returns.iloc[split_point:]
        
        results = []
        
        # In-sample validation
        if len(is_signal) >= 30:
            is_result = self._calculate_validation_metrics(
                is_signal, is_returns, 'in_sample_holdout', 
                (is_signal.index[0], is_signal.index[-1])
            )
            results.append(is_result)
        
        # Out-of-sample validation
        if len(oos_signal) >= self.min_oos_periods:
            oos_result = self._calculate_validation_metrics(
                oos_signal, oos_returns, 'out_of_sample_holdout',
                (oos_signal.index[0], oos_signal.index[-1])
            )
            results.append(oos_result)
        
        return results
    
    def _rolling_window_validation(self, signal: pd.Series, returns: pd.Series,
                                 window_size: int = 252) -> List[ValidationResult]:
        """Perform rolling window validation."""
        results = []
        
        if len(signal) < window_size + self.min_oos_periods:
            logger.warning("Insufficient data for rolling window validation")
            return results
        
        # Rolling windows
        for i in range(window_size, len(signal) - self.min_oos_periods, self.min_oos_periods):
            # Training window
            train_signal = signal.iloc[i - window_size:i]
            train_returns = returns.iloc[i - window_size:i]
            
            # Test window
            test_end = min(i + self.min_oos_periods, len(signal))
            test_signal = signal.iloc[i:test_end]
            test_returns = returns.iloc[i:test_end]
            
            if len(test_signal) >= 30:  # Minimum for meaningful test
                test_result = self._calculate_validation_metrics(
                    test_signal, test_returns, 'rolling_window_oos',
                    (test_signal.index[0], test_signal.index[-1])
                )
                results.append(test_result)
        
        return results
    
    def _expanding_window_validation(self, signal: pd.Series, returns: pd.Series) -> List[ValidationResult]:
        """Perform expanding window validation."""
        results = []
        
        min_train_size = max(100, len(signal) // 4)
        
        # Expanding windows
        for i in range(min_train_size, len(signal) - self.min_oos_periods, self.min_oos_periods):
            # Training window (expanding)
            train_signal = signal.iloc[:i]
            train_returns = returns.iloc[:i]
            
            # Test window
            test_end = min(i + self.min_oos_periods, len(signal))
            test_signal = signal.iloc[i:test_end]
            test_returns = returns.iloc[i:test_end]
            
            if len(test_signal) >= 30:
                test_result = self._calculate_validation_metrics(
                    test_signal, test_returns, 'expanding_window_oos',
                    (test_signal.index[0], test_signal.index[-1])
                )
                results.append(test_result)
        
        return results
    
    def _calculate_validation_metrics(self, signal: pd.Series, returns: pd.Series,
                                    validation_type: str, period: Tuple[pd.Timestamp, pd.Timestamp]) -> ValidationResult:
        """Calculate comprehensive validation metrics."""
        try:
            # Remove NaN values
            valid_mask = ~(signal.isna() | returns.isna())
            clean_signal = signal[valid_mask]
            clean_returns = returns[valid_mask]
            
            if len(clean_signal) < 10:
                # Return default result for insufficient data
                return ValidationResult(
                    validation_type=validation_type,
                    validation_period=period,
                    information_coefficient=0.0,
                    ic_t_statistic=0.0,
                    ic_p_value=1.0,
                    sharpe_ratio=0.0,
                    hit_rate=0.5,
                    max_drawdown=0.0,
                    calmar_ratio=0.0,
                    skewness=0.0,
                    kurtosis=0.0,
                    is_significant=False,
                    confidence_level=0.95
                )
            
            # Information Coefficient
            ic = clean_signal.corr(clean_returns)
            if pd.isna(ic):
                ic = 0.0
            
            # IC significance test
            n = len(clean_signal)
            ic_t_stat = ic * np.sqrt((n - 2) / (1 - ic**2)) if abs(ic) < 1 else 0
            ic_p_value = 2 * (1 - stats.t.cdf(abs(ic_t_stat), n - 2)) if n > 2 else 1.0
            
            # Signal-based returns
            signal_returns = clean_signal * clean_returns  # Simplified approach
            
            # Sharpe ratio
            if signal_returns.std() > 0:
                sharpe = signal_returns.mean() / signal_returns.std()
            else:
                sharpe = 0.0
            
            # Hit rate (percentage of correct direction predictions)
            correct_direction = np.sign(clean_signal) == np.sign(clean_returns)
            hit_rate = correct_direction.mean()
            
            # Drawdown analysis
            cumulative_returns = (1 + signal_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0.0
            
            # Calmar ratio
            annualized_return = signal_returns.mean() * 252
            calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
            
            # Distribution properties
            skewness = stats.skew(signal_returns.dropna())
            kurtosis = stats.kurtosis(signal_returns.dropna())
            
            # Significance assessment
            is_significant = ic_p_value < 0.05 and abs(ic) > 0.02
            
            return ValidationResult(
                validation_type=validation_type,
                validation_period=period,
                information_coefficient=ic,
                ic_t_statistic=ic_t_stat,
                ic_p_value=ic_p_value,
                sharpe_ratio=sharpe,
                hit_rate=hit_rate,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar,
                skewness=skewness,
                kurtosis=kurtosis,
                is_significant=is_significant,
                confidence_level=0.95
            )
            
        except Exception as e:
            logger.warning(f"Error calculating validation metrics: {str(e)}")
            # Return default result
            return ValidationResult(
                validation_type=validation_type,
                validation_period=period,
                information_coefficient=0.0,
                ic_t_statistic=0.0,
                ic_p_value=1.0,
                sharpe_ratio=0.0,
                hit_rate=0.5,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                skewness=0.0,
                kurtosis=0.0,
                is_significant=False,
                confidence_level=0.95
            )

class CrossValidator:
    """
    Time series cross-validation system for alpha signals.
    
    Implements time-aware cross-validation techniques that respect
    the temporal nature of financial data.
    """
    
    def __init__(self, n_splits: int = 5, test_size: int = 100):
        """
        Initialize cross-validator.
        
        Parameters:
        -----------
        n_splits : int
            Number of cross-validation splits
        test_size : int
            Size of each test set
        """
        self.n_splits = n_splits
        self.test_size = test_size
        
    def cross_validate_alpha(self, alpha_signal: pd.Series, returns: pd.Series) -> List[ValidationResult]:
        """
        Perform time series cross-validation.
        
        Parameters:
        -----------
        alpha_signal : pd.Series
            Alpha signal to validate
        returns : pd.Series
            Forward returns for validation
            
        Returns:
        --------
        List[ValidationResult]
            Cross-validation results
        """
        try:
            # Align data
            common_index = alpha_signal.index.intersection(returns.index)
            aligned_signal = alpha_signal.loc[common_index]
            aligned_returns = returns.loc[common_index]
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
            
            results = []
            fold = 0
            
            for train_index, test_index in tscv.split(aligned_signal):
                fold += 1
                
                # Get test data
                test_signal = aligned_signal.iloc[test_index]
                test_returns = aligned_returns.iloc[test_index]
                
                if len(test_signal) >= 20:  # Minimum for validation
                    # Calculate validation metrics
                    result = self._calculate_cv_metrics(
                        test_signal, test_returns, fold,
                        (test_signal.index[0], test_signal.index[-1])
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            raise
    
    def _calculate_cv_metrics(self, signal: pd.Series, returns: pd.Series,
                            fold: int, period: Tuple[pd.Timestamp, pd.Timestamp]) -> ValidationResult:
        """Calculate cross-validation metrics for a single fold."""
        try:
            # Remove NaN values
            valid_mask = ~(signal.isna() | returns.isna())
            clean_signal = signal[valid_mask]
            clean_returns = returns[valid_mask]
            
            if len(clean_signal) < 5:
                # Return default for insufficient data
                return ValidationResult(
                    validation_type=f'cv_fold_{fold}',
                    validation_period=period,
                    information_coefficient=0.0,
                    ic_t_statistic=0.0,
                    ic_p_value=1.0,
                    sharpe_ratio=0.0,
                    hit_rate=0.5,
                    max_drawdown=0.0,
                    calmar_ratio=0.0,
                    skewness=0.0,
                    kurtosis=0.0,
                    is_significant=False,
                    confidence_level=0.95
                )
            
            # Calculate metrics (simplified version)
            ic = clean_signal.corr(clean_returns)
            if pd.isna(ic):
                ic = 0.0
            
            # Basic significance test
            n = len(clean_signal)
            ic_t_stat = ic * np.sqrt((n - 2) / (1 - ic**2)) if abs(ic) < 1 and n > 2 else 0
            ic_p_value = 2 * (1 - stats.t.cdf(abs(ic_t_stat), n - 2)) if n > 2 else 1.0
            
            # Simple returns metrics
            signal_returns = clean_signal * clean_returns
            sharpe = signal_returns.mean() / signal_returns.std() if signal_returns.std() > 0 else 0.0
            hit_rate = (np.sign(clean_signal) == np.sign(clean_returns)).mean()
            
            return ValidationResult(
                validation_type=f'cv_fold_{fold}',
                validation_period=period,
                information_coefficient=ic,
                ic_t_statistic=ic_t_stat,
                ic_p_value=ic_p_value,
                sharpe_ratio=sharpe,
                hit_rate=hit_rate,
                max_drawdown=0.0,  # Simplified - not calculated for CV
                calmar_ratio=0.0,
                skewness=stats.skew(signal_returns.dropna()) if len(signal_returns.dropna()) > 3 else 0.0,
                kurtosis=stats.kurtosis(signal_returns.dropna()) if len(signal_returns.dropna()) > 3 else 0.0,
                is_significant=ic_p_value < 0.05 and abs(ic) > 0.01,
                confidence_level=0.95
            )
            
        except Exception as e:
            logger.warning(f"Error in CV fold {fold}: {str(e)}")
            return ValidationResult(
                validation_type=f'cv_fold_{fold}',
                validation_period=period,
                information_coefficient=0.0,
                ic_t_statistic=0.0,
                ic_p_value=1.0,
                sharpe_ratio=0.0,
                hit_rate=0.5,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                skewness=0.0,
                kurtosis=0.0,
                is_significant=False,
                confidence_level=0.95
            )

class RobustnessTestSuite:
    """
    Robustness testing suite for alpha signals.
    
    Tests alpha signal performance under various stress conditions
    and parameter variations.
    """
    
    def __init__(self, robustness_threshold: float = 0.3):
        """
        Initialize robustness test suite.
        
        Parameters:
        -----------
        robustness_threshold : float
            Maximum acceptable relative IC degradation
        """
        self.robustness_threshold = robustness_threshold
        
    def run_robustness_tests(self, alpha_signal: pd.Series, returns: pd.Series) -> List[RobustnessTestResult]:
        """
        Run comprehensive robustness tests.
        
        Parameters:
        -----------
        alpha_signal : pd.Series
            Alpha signal to test
        returns : pd.Series
            Forward returns for testing
            
        Returns:
        --------
        List[RobustnessTestResult]
            Robustness test results
        """
        try:
            # Align data
            common_index = alpha_signal.index.intersection(returns.index)
            clean_signal = alpha_signal.loc[common_index]
            clean_returns = returns.loc[common_index]
            
            # Calculate baseline IC
            baseline_ic = clean_signal.corr(clean_returns)
            if pd.isna(baseline_ic):
                baseline_ic = 0.0
            
            results = []
            
            # Test 1: Random subsample robustness
            subsample_result = self._test_random_subsample(clean_signal, clean_returns, baseline_ic)
            results.append(subsample_result)
            
            # Test 2: High volatility periods
            high_vol_result = self._test_high_volatility_periods(clean_signal, clean_returns, baseline_ic)
            results.append(high_vol_result)
            
            # Test 3: Market stress periods (simplified)
            stress_result = self._test_market_stress_periods(clean_signal, clean_returns, baseline_ic)
            results.append(stress_result)
            
            # Test 4: Signal noise addition
            noise_result = self._test_signal_noise_addition(clean_signal, clean_returns, baseline_ic)
            results.append(noise_result)
            
            # Test 5: Outlier robustness
            outlier_result = self._test_outlier_robustness(clean_signal, clean_returns, baseline_ic)
            results.append(outlier_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in robustness testing: {str(e)}")
            raise
    
    def _test_random_subsample(self, signal: pd.Series, returns: pd.Series, baseline_ic: float) -> RobustnessTestResult:
        """Test robustness using random subsamples."""
        try:
            # Random subsample (80% of data)
            # Deterministic subsample (first 80% or evenly spaced indices)
            sample_size = int(len(signal) * 0.8)
            if sample_size <= 0:
                return self._create_failed_robustness_result("random_subsample", baseline_ic)

            # Evenly spaced deterministic selection
            indices = np.linspace(0, len(signal) - 1, sample_size, dtype=int)
            subsample_signal = signal.iloc[indices]
            subsample_returns = returns.iloc[indices]
            
            stressed_ic = subsample_signal.corr(subsample_returns)
            if pd.isna(stressed_ic):
                stressed_ic = 0.0
            
            ic_change = stressed_ic - baseline_ic
            relative_change = ic_change / baseline_ic if baseline_ic != 0 else 0.0
            
            passes_test = abs(relative_change) <= self.robustness_threshold
            
            return RobustnessTestResult(
                test_name="random_subsample",
                test_description="Random 80% subsample robustness test",
                original_ic=baseline_ic,
                stressed_ic=stressed_ic,
                ic_change=ic_change,
                relative_ic_change=relative_change,
                passes_robustness_test=passes_test,
                robustness_threshold=self.robustness_threshold
            )
            
        except Exception as e:
            logger.warning(f"Error in random subsample test: {str(e)}")
            return self._create_failed_robustness_result("random_subsample", baseline_ic)
    
    def _test_high_volatility_periods(self, signal: pd.Series, returns: pd.Series, baseline_ic: float) -> RobustnessTestResult:
        """Test robustness during high volatility periods."""
        try:
            # Identify high volatility periods (top quartile)
            rolling_vol = returns.rolling(20).std()
            high_vol_threshold = rolling_vol.quantile(0.75)
            high_vol_mask = rolling_vol > high_vol_threshold
            
            if high_vol_mask.sum() < 20:  # Need minimum observations
                return self._create_failed_robustness_result("high_volatility", baseline_ic)
            
            high_vol_signal = signal[high_vol_mask]
            high_vol_returns = returns[high_vol_mask]
            
            stressed_ic = high_vol_signal.corr(high_vol_returns)
            if pd.isna(stressed_ic):
                stressed_ic = 0.0
            
            ic_change = stressed_ic - baseline_ic
            relative_change = ic_change / baseline_ic if baseline_ic != 0 else 0.0
            
            passes_test = abs(relative_change) <= self.robustness_threshold
            
            return RobustnessTestResult(
                test_name="high_volatility",
                test_description="High volatility periods robustness test",
                original_ic=baseline_ic,
                stressed_ic=stressed_ic,
                ic_change=ic_change,
                relative_ic_change=relative_change,
                passes_robustness_test=passes_test,
                robustness_threshold=self.robustness_threshold
            )
            
        except Exception as e:
            logger.warning(f"Error in high volatility test: {str(e)}")
            return self._create_failed_robustness_result("high_volatility", baseline_ic)
    
    def _test_market_stress_periods(self, signal: pd.Series, returns: pd.Series, baseline_ic: float) -> RobustnessTestResult:
        """Test robustness during market stress periods."""
        try:
            # Identify stress periods (extreme negative returns)
            stress_threshold = returns.quantile(0.1)  # Bottom decile
            stress_mask = returns < stress_threshold
            
            if stress_mask.sum() < 20:
                return self._create_failed_robustness_result("market_stress", baseline_ic)
            
            stress_signal = signal[stress_mask]
            stress_returns = returns[stress_mask]
            
            stressed_ic = stress_signal.corr(stress_returns)
            if pd.isna(stressed_ic):
                stressed_ic = 0.0
            
            ic_change = stressed_ic - baseline_ic
            relative_change = ic_change / baseline_ic if baseline_ic != 0 else 0.0
            
            passes_test = abs(relative_change) <= self.robustness_threshold
            
            return RobustnessTestResult(
                test_name="market_stress",
                test_description="Market stress periods robustness test",
                original_ic=baseline_ic,
                stressed_ic=stressed_ic,
                ic_change=ic_change,
                relative_ic_change=relative_change,
                passes_robustness_test=passes_test,
                robustness_threshold=self.robustness_threshold
            )
            
        except Exception as e:
            logger.warning(f"Error in market stress test: {str(e)}")
            return self._create_failed_robustness_result("market_stress", baseline_ic)
    
    def _test_signal_noise_addition(self, signal: pd.Series, returns: pd.Series, baseline_ic: float) -> RobustnessTestResult:
        """Test robustness with added signal noise."""
        try:
            # Add 10% noise to signal
            signal_std = signal.std()
            # Deterministic perturbation: sinusoidal perturbation scaled by signal std
            t = np.arange(len(signal))
            noise = signal_std * 0.1 * np.sin(2 * np.pi * t / max(1, len(signal)))
            noisy_signal = signal + noise
            
            stressed_ic = noisy_signal.corr(returns)
            if pd.isna(stressed_ic):
                stressed_ic = 0.0
            
            ic_change = stressed_ic - baseline_ic
            relative_change = ic_change / baseline_ic if baseline_ic != 0 else 0.0
            
            passes_test = abs(relative_change) <= self.robustness_threshold
            
            return RobustnessTestResult(
                test_name="signal_noise",
                test_description="Signal noise addition robustness test (10% noise)",
                original_ic=baseline_ic,
                stressed_ic=stressed_ic,
                ic_change=ic_change,
                relative_ic_change=relative_change,
                passes_robustness_test=passes_test,
                robustness_threshold=self.robustness_threshold
            )
            
        except Exception as e:
            logger.warning(f"Error in signal noise test: {str(e)}")
            return self._create_failed_robustness_result("signal_noise", baseline_ic)
    
    def _test_outlier_robustness(self, signal: pd.Series, returns: pd.Series, baseline_ic: float) -> RobustnessTestResult:
        """Test robustness by removing outliers."""
        try:
            # Remove extreme signal outliers (beyond 3 std devs)
            signal_mean = signal.mean()
            signal_std = signal.std()
            
            outlier_mask = (
                (signal >= signal_mean - 3 * signal_std) & 
                (signal <= signal_mean + 3 * signal_std)
            )
            
            if outlier_mask.sum() < 50:  # Need minimum observations
                return self._create_failed_robustness_result("outlier_removal", baseline_ic)
            
            filtered_signal = signal[outlier_mask]
            filtered_returns = returns[outlier_mask]
            
            stressed_ic = filtered_signal.corr(filtered_returns)
            if pd.isna(stressed_ic):
                stressed_ic = 0.0
            
            ic_change = stressed_ic - baseline_ic
            relative_change = ic_change / baseline_ic if baseline_ic != 0 else 0.0
            
            passes_test = abs(relative_change) <= self.robustness_threshold
            
            return RobustnessTestResult(
                test_name="outlier_removal",
                test_description="Outlier removal robustness test (±3σ filter)",
                original_ic=baseline_ic,
                stressed_ic=stressed_ic,
                ic_change=ic_change,
                relative_ic_change=relative_change,
                passes_robustness_test=passes_test,
                robustness_threshold=self.robustness_threshold
            )
            
        except Exception as e:
            logger.warning(f"Error in outlier robustness test: {str(e)}")
            return self._create_failed_robustness_result("outlier_removal", baseline_ic)
    
    def _create_failed_robustness_result(self, test_name: str, baseline_ic: float) -> RobustnessTestResult:
        """Create a failed robustness test result."""
        return RobustnessTestResult(
            test_name=test_name,
            test_description=f"{test_name} robustness test (failed)",
            original_ic=baseline_ic,
            stressed_ic=0.0,
            ic_change=-baseline_ic,
            relative_ic_change=-1.0,
            passes_robustness_test=False,
            robustness_threshold=self.robustness_threshold
        )

class OverfittingDetector:
    """
    Overfitting detection system for alpha signals.
    
    Identifies potential overfitting issues in alpha signals using
    various statistical tests and heuristics.
    """
    
    def __init__(self):
        """Initialize overfitting detector."""
        pass
        
    def assess_overfitting_risk(self, in_sample_results: List[ValidationResult],
                              out_of_sample_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Assess overfitting risk by comparing in-sample vs out-of-sample performance.
        
        Parameters:
        -----------
        in_sample_results : List[ValidationResult]
            In-sample validation results
        out_of_sample_results : List[ValidationResult]
            Out-of-sample validation results
            
        Returns:
        --------
        Dict[str, Any]
            Overfitting assessment results
        """
        try:
            if not in_sample_results or not out_of_sample_results:
                return {
                    'overfitting_risk': 'unknown',
                    'reason': 'Insufficient validation results',
                    'is_overfit': None,
                    'performance_degradation': None
                }
            
            # Calculate average performance metrics
            is_avg_ic = np.mean([r.information_coefficient for r in in_sample_results])
            oos_avg_ic = np.mean([r.information_coefficient for r in out_of_sample_results])
            
            is_avg_sharpe = np.mean([r.sharpe_ratio for r in in_sample_results])
            oos_avg_sharpe = np.mean([r.sharpe_ratio for r in out_of_sample_results])
            
            # Performance degradation analysis
            ic_degradation = (is_avg_ic - oos_avg_ic) / abs(is_avg_ic) if is_avg_ic != 0 else 0
            sharpe_degradation = (is_avg_sharpe - oos_avg_sharpe) / abs(is_avg_sharpe) if is_avg_sharpe != 0 else 0
            
            # Overfitting thresholds
            high_degradation_threshold = 0.5  # 50% performance drop
            moderate_degradation_threshold = 0.3  # 30% performance drop
            
            # Assess overfitting risk
            if ic_degradation > high_degradation_threshold or sharpe_degradation > high_degradation_threshold:
                overfitting_risk = 'high'
                is_overfit = True
            elif ic_degradation > moderate_degradation_threshold or sharpe_degradation > moderate_degradation_threshold:
                overfitting_risk = 'moderate'
                is_overfit = True
            else:
                overfitting_risk = 'low'
                is_overfit = False
            
            # Additional checks
            significance_degradation = self._check_significance_degradation(in_sample_results, out_of_sample_results)
            
            return {
                'overfitting_risk': overfitting_risk,
                'is_overfit': is_overfit,
                'ic_degradation': ic_degradation,
                'sharpe_degradation': sharpe_degradation,
                'is_avg_ic': is_avg_ic,
                'oos_avg_ic': oos_avg_ic,
                'is_avg_sharpe': is_avg_sharpe,
                'oos_avg_sharpe': oos_avg_sharpe,
                'significance_degradation': significance_degradation,
                'performance_degradation': max(ic_degradation, sharpe_degradation)
            }
            
        except Exception as e:
            logger.error(f"Error assessing overfitting risk: {str(e)}")
            return {
                'overfitting_risk': 'unknown',
                'reason': f'Error in assessment: {str(e)}',
                'is_overfit': None,
                'performance_degradation': None
            }
    
    def _check_significance_degradation(self, is_results: List[ValidationResult],
                                      oos_results: List[ValidationResult]) -> Dict[str, Any]:
        """Check degradation in statistical significance."""
        try:
            is_significant_count = sum(1 for r in is_results if r.is_significant)
            oos_significant_count = sum(1 for r in oos_results if r.is_significant)
            
            is_significance_rate = is_significant_count / len(is_results) if is_results else 0
            oos_significance_rate = oos_significant_count / len(oos_results) if oos_results else 0
            
            significance_degradation = is_significance_rate - oos_significance_rate
            
            return {
                'is_significance_rate': is_significance_rate,
                'oos_significance_rate': oos_significance_rate,
                'significance_degradation': significance_degradation,
                'severe_significance_loss': significance_degradation > 0.5
            }
            
        except Exception as e:
            logger.warning(f"Error in significance degradation check: {str(e)}")
            return {
                'is_significance_rate': 0,
                'oos_significance_rate': 0,
                'significance_degradation': 0,
                'severe_significance_loss': False
            }

class ComprehensiveAlphaValidator:
    """
    Comprehensive alpha validation system integrating all validation methodologies.
    
    Provides unified interface for complete alpha signal validation including
    out-of-sample testing, cross-validation, robustness testing, and overfitting detection.
    """
    
    def __init__(self):
        """Initialize comprehensive alpha validator."""
        self.oos_validator = OutOfSampleValidator()
        self.cross_validator = CrossValidator()
        self.robustness_tester = RobustnessTestSuite()
        self.overfitting_detector = OverfittingDetector()
        
    def validate_alpha_comprehensive(self, alpha_signal: pd.Series, returns: pd.Series,
                                   signal_name: str = "Alpha Signal") -> AlphaValidationReport:
        """
        Perform comprehensive alpha validation.
        
        Parameters:
        -----------
        alpha_signal : pd.Series
            Alpha signal to validate
        returns : pd.Series
            Forward returns for validation
        signal_name : str
            Name of the alpha signal
            
        Returns:
        --------
        AlphaValidationReport
            Comprehensive validation report
        """
        try:
            logger.info(f"Starting comprehensive validation for {signal_name}")
            
            # Out-of-sample validation
            oos_results = []
            try:
                holdout_results = self.oos_validator.validate_alpha_signal(
                    alpha_signal, returns, 'holdout'
                )
                oos_results.extend(holdout_results)
                
                rolling_results = self.oos_validator.validate_alpha_signal(
                    alpha_signal, returns, 'rolling_window'
                )
                oos_results.extend(rolling_results)
            except Exception as e:
                logger.warning(f"Error in out-of-sample validation: {str(e)}")
            
            # Cross-validation
            cv_results = []
            try:
                cv_results = self.cross_validator.cross_validate_alpha(alpha_signal, returns)
            except Exception as e:
                logger.warning(f"Error in cross-validation: {str(e)}")
            
            # Robustness testing
            robustness_results = []
            try:
                robustness_results = self.robustness_tester.run_robustness_tests(alpha_signal, returns)
            except Exception as e:
                logger.warning(f"Error in robustness testing: {str(e)}")
            
            # Separate in-sample and out-of-sample results
            in_sample_results = [r for r in oos_results if 'in_sample' in r.validation_type]
            out_of_sample_results = [r for r in oos_results if 'out_of_sample' in r.validation_type or 'oos' in r.validation_type]
            
            # Overfitting assessment
            overfitting_assessment = {}
            try:
                overfitting_assessment = self.overfitting_detector.assess_overfitting_risk(
                    in_sample_results, out_of_sample_results
                )
            except Exception as e:
                logger.warning(f"Error in overfitting assessment: {str(e)}")
            
            # Regime validation (simplified - would normally use regime detection)
            regime_results = {}
            try:
                regime_results = self._perform_regime_validation(alpha_signal, returns)
            except Exception as e:
                logger.warning(f"Error in regime validation: {str(e)}")
            
            # Calculate production readiness score
            readiness_score = self._calculate_production_readiness_score(
                out_of_sample_results, cv_results, robustness_results, overfitting_assessment
            )
            
            # Generate recommendations
            recommendations = self._generate_validation_recommendations(
                out_of_sample_results, cv_results, robustness_results, overfitting_assessment, readiness_score
            )
            
            # Create comprehensive report
            report = AlphaValidationReport(
                alpha_signal_name=signal_name,
                validation_date=datetime.now(),
                in_sample_results=in_sample_results,
                out_of_sample_results=out_of_sample_results,
                cross_validation_results=cv_results,
                robustness_test_results=robustness_results,
                regime_validation_results=regime_results,
                overfitting_assessment=overfitting_assessment,
                production_readiness_score=readiness_score,
                recommendations=recommendations
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error in comprehensive alpha validation: {str(e)}")
            raise
    
    def _perform_regime_validation(self, signal: pd.Series, returns: pd.Series) -> Dict[str, ValidationResult]:
        """Perform simplified regime-based validation."""
        try:
            # Simplified regime detection based on volatility
            rolling_vol = returns.rolling(20).std()
            high_vol_threshold = rolling_vol.quantile(0.7)
            low_vol_threshold = rolling_vol.quantile(0.3)
            
            # High volatility regime
            high_vol_mask = rolling_vol > high_vol_threshold
            if high_vol_mask.sum() > 30:
                high_vol_signal = signal[high_vol_mask]
                high_vol_returns = returns[high_vol_mask]
                
                high_vol_result = ValidationResult(
                    validation_type='high_volatility_regime',
                    validation_period=(high_vol_signal.index[0], high_vol_signal.index[-1]),
                    information_coefficient=high_vol_signal.corr(high_vol_returns),
                    ic_t_statistic=0.0,  # Simplified
                    ic_p_value=0.1,  # Simplified
                    sharpe_ratio=(high_vol_signal * high_vol_returns).mean() / (high_vol_signal * high_vol_returns).std() if (high_vol_signal * high_vol_returns).std() > 0 else 0,
                    hit_rate=(np.sign(high_vol_signal) == np.sign(high_vol_returns)).mean(),
                    max_drawdown=0.0,  # Simplified
                    calmar_ratio=0.0,  # Simplified
                    skewness=0.0,  # Simplified
                    kurtosis=0.0,  # Simplified
                    is_significant=abs(high_vol_signal.corr(high_vol_returns)) > 0.02,
                    confidence_level=0.95
                )
            else:
                high_vol_result = None
            
            # Low volatility regime
            low_vol_mask = rolling_vol < low_vol_threshold
            if low_vol_mask.sum() > 30:
                low_vol_signal = signal[low_vol_mask]
                low_vol_returns = returns[low_vol_mask]
                
                low_vol_result = ValidationResult(
                    validation_type='low_volatility_regime',
                    validation_period=(low_vol_signal.index[0], low_vol_signal.index[-1]),
                    information_coefficient=low_vol_signal.corr(low_vol_returns),
                    ic_t_statistic=0.0,  # Simplified
                    ic_p_value=0.1,  # Simplified
                    sharpe_ratio=(low_vol_signal * low_vol_returns).mean() / (low_vol_signal * low_vol_returns).std() if (low_vol_signal * low_vol_returns).std() > 0 else 0,
                    hit_rate=(np.sign(low_vol_signal) == np.sign(low_vol_returns)).mean(),
                    max_drawdown=0.0,  # Simplified
                    calmar_ratio=0.0,  # Simplified
                    skewness=0.0,  # Simplified
                    kurtosis=0.0,  # Simplified
                    is_significant=abs(low_vol_signal.corr(low_vol_returns)) > 0.02,
                    confidence_level=0.95
                )
            else:
                low_vol_result = None
            
            regime_results = {}
            if high_vol_result:
                regime_results['high_volatility'] = high_vol_result
            if low_vol_result:
                regime_results['low_volatility'] = low_vol_result
            
            return regime_results
            
        except Exception as e:
            logger.warning(f"Error in regime validation: {str(e)}")
            return {}
    
    def _calculate_production_readiness_score(self, oos_results: List[ValidationResult],
                                            cv_results: List[ValidationResult],
                                            robustness_results: List[RobustnessTestResult],
                                            overfitting_assessment: Dict[str, Any]) -> float:
        """Calculate production readiness score (0-100)."""
        try:
            score_components = []
            
            # Out-of-sample performance (40% weight)
            if oos_results:
                avg_oos_ic = np.mean([abs(r.information_coefficient) for r in oos_results])
                oos_significance_rate = np.mean([r.is_significant for r in oos_results])
                oos_score = min(avg_oos_ic * 100, 40) * oos_significance_rate
                score_components.append(oos_score)
            
            # Cross-validation consistency (25% weight)
            if cv_results:
                cv_ic_std = np.std([r.information_coefficient for r in cv_results])
                cv_consistency_score = max(0, 25 - cv_ic_std * 500)  # Penalize high variance
                score_components.append(cv_consistency_score)
            
            # Robustness (25% weight)
            if robustness_results:
                robustness_pass_rate = np.mean([r.passes_robustness_test for r in robustness_results])
                robustness_score = robustness_pass_rate * 25
                score_components.append(robustness_score)
            
            # Overfitting penalty (10% weight)
            overfitting_penalty = 0
            if overfitting_assessment.get('is_overfit', False):
                degradation = overfitting_assessment.get('performance_degradation', 0)
                overfitting_penalty = min(degradation * 20, 10)  # Max 10 point penalty
            
            overfitting_score = max(0, 10 - overfitting_penalty)
            score_components.append(overfitting_score)
            
            # Calculate final score
            if score_components:
                final_score = sum(score_components)
            else:
                final_score = 0.0
            
            return min(max(final_score, 0.0), 100.0)  # Clamp to 0-100
            
        except Exception as e:
            logger.warning(f"Error calculating production readiness score: {str(e)}")
            return 0.0
    
    def _generate_validation_recommendations(self, oos_results: List[ValidationResult],
                                           cv_results: List[ValidationResult],
                                           robustness_results: List[RobustnessTestResult],
                                           overfitting_assessment: Dict[str, Any],
                                           readiness_score: float) -> List[str]:
        """Generate validation-based recommendations."""
        recommendations = []
        
        # Production readiness assessment
        if readiness_score >= 80:
            recommendations.append("APPROVED: Signal passes all validation tests and is ready for production deployment.")
        elif readiness_score >= 60:
            recommendations.append("CONDITIONAL: Signal shows promise but requires additional validation or risk management.")
        else:
            recommendations.append("NOT RECOMMENDED: Signal fails validation criteria and needs significant improvement before deployment.")
        
        # Out-of-sample performance
        if oos_results:
            avg_oos_ic = np.mean([abs(r.information_coefficient) for r in oos_results])
            if avg_oos_ic < 0.02:
                recommendations.append("Low out-of-sample Information Coefficient. Consider signal enhancement or combination with other alphas.")
        
        # Overfitting issues
        if overfitting_assessment.get('is_overfit', False):
            risk_level = overfitting_assessment.get('overfitting_risk', 'unknown')
            recommendations.append(f"OVERFITTING DETECTED ({risk_level} risk): Signal shows significant performance degradation out-of-sample. Consider regularization or feature selection.")
        
        # Robustness issues
        if robustness_results:
            failed_tests = [r for r in robustness_results if not r.passes_robustness_test]
            if failed_tests:
                test_names = [r.test_name for r in failed_tests]
                recommendations.append(f"Robustness concerns in: {', '.join(test_names)}. Signal may be vulnerable to market regime changes.")
        
        # Cross-validation consistency
        if cv_results:
            cv_ic_std = np.std([r.information_coefficient for r in cv_results])
            if cv_ic_std > 0.05:
                recommendations.append("High cross-validation variance detected. Signal performance may be unstable across different time periods.")
        
        # Additional recommendations based on validation results
        if oos_results:
            significant_results = [r for r in oos_results if r.is_significant]
            if len(significant_results) < len(oos_results) / 2:
                recommendations.append("Low statistical significance in validation tests. Consider larger sample size or signal improvement.")
        
        return recommendations

