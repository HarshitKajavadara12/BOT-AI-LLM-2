"""
Factor Decay Analysis Framework
Advanced factor decay analysis and signal degradation modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings
import math
from scipy import stats, optimize
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures


class DecayModel(Enum):
    """Types of decay models"""
    LINEAR = "LINEAR"                       # Linear decay
    EXPONENTIAL = "EXPONENTIAL"             # Exponential decay
    POWER_LAW = "POWER_LAW"                 # Power law decay
    LOGARITHMIC = "LOGARITHMIC"             # Logarithmic decay
    POLYNOMIAL = "POLYNOMIAL"               # Polynomial decay
    STEPWISE = "STEPWISE"                   # Step-wise decay
    PIECEWISE_LINEAR = "PIECEWISE_LINEAR"   # Piecewise linear
    ADAPTIVE = "ADAPTIVE"                   # Adaptive decay


class DecayMetric(Enum):
    """Metrics for measuring decay"""
    INFORMATION_COEFFICIENT = "INFORMATION_COEFFICIENT"  # IC decay
    RANK_CORRELATION = "RANK_CORRELATION"               # Rank correlation decay
    RETURN_CORRELATION = "RETURN_CORRELATION"           # Return correlation decay
    PREDICTIVE_POWER = "PREDICTIVE_POWER"               # R-squared decay
    SHARPE_RATIO = "SHARPE_RATIO"                       # Sharpe ratio decay
    HIT_RATE = "HIT_RATE"                               # Hit rate decay
    SIGNAL_STRENGTH = "SIGNAL_STRENGTH"                 # Signal strength decay


@dataclass
class DecayConfig:
    """Configuration for decay analysis"""
    
    # Analysis parameters
    max_horizon: int = 20                   # Maximum forecast horizon
    min_periods: int = 100                  # Minimum periods for analysis
    step_size: int = 1                      # Step size for horizon analysis
    
    # Decay modeling
    decay_metric: DecayMetric = DecayMetric.INFORMATION_COEFFICIENT
    decay_model: DecayModel = DecayModel.EXPONENTIAL
    confidence_level: float = 0.95          # Confidence level for intervals
    
    # Rolling analysis
    rolling_window: int = 252               # Rolling window size
    rolling_step: int = 63                  # Rolling step size
    
    # Statistical parameters
    correlation_method: str = "spearman"    # Correlation method
    significance_level: float = 0.05        # Statistical significance level
    
    # Modeling parameters
    polynomial_degree: int = 2              # Degree for polynomial decay
    n_segments: int = 3                     # Number of segments for piecewise
    
    # Other parameters
    normalize_factors: bool = True          # Normalize factors before analysis
    remove_outliers: bool = True            # Remove outliers
    outlier_threshold: float = 3.0          # Outlier threshold (std devs)


@dataclass
class DecayResult:
    """Results of decay analysis"""
    
    # Decay curve data
    horizons: List[int]                     # Forecast horizons
    decay_values: List[float]               # Decay metric values
    confidence_intervals: List[Tuple[float, float]]  # Confidence intervals
    
    # Model fitting results
    model_type: DecayModel
    model_parameters: Dict[str, float]
    model_r2: float                         # Model fit quality
    model_predictions: List[float]          # Model predictions
    
    # Decay characteristics
    half_life: Optional[float] = None       # Signal half-life
    decay_rate: Optional[float] = None      # Decay rate parameter
    initial_strength: float = 0.0          # Initial signal strength
    residual_strength: float = 0.0         # Residual signal strength
    
    # Statistical metrics
    significance_horizons: List[int] = field(default_factory=list)  # Significant horizons
    optimal_horizon: int = 1                # Optimal forecast horizon
    
    # Quality metrics
    stability_score: float = 0.0           # Stability of decay pattern
    monotonicity_score: float = 0.0        # Monotonicity of decay
    
    # Metadata
    factor_name: str = ""
    analysis_date: datetime = field(default_factory=datetime.now)
    sample_size: int = 0
    analysis_period: Tuple[datetime, datetime] = field(default_factory=lambda: (datetime.now(), datetime.now()))


class BaseDecayAnalyzer(ABC):
    """Base class for decay analysis"""
    
    def __init__(self, config: DecayConfig):
        self.config = config
        self.is_fitted = False
        
    @abstractmethod
    def analyze_decay(self, factor: pd.Series, returns: pd.Series) -> DecayResult:
        """Analyze factor decay"""
        pass
    
    def _calculate_decay_metric(self, factor: pd.Series, returns: pd.Series, 
                              horizon: int) -> Tuple[float, float]:
        """Calculate decay metric for given horizon"""
        
        # Shift returns for forward-looking analysis
        forward_returns = returns.shift(-horizon)
        
        # Align data
        aligned_data = pd.DataFrame({'factor': factor, 'returns': forward_returns}).dropna()
        
        if len(aligned_data) < self.config.min_periods:
            return 0.0, np.nan
        
        factor_values = aligned_data['factor']
        return_values = aligned_data['returns']
        
        # Calculate metric based on configuration
        if self.config.decay_metric == DecayMetric.INFORMATION_COEFFICIENT:
            if self.config.correlation_method == "spearman":
                metric_value, p_value = stats.spearmanr(factor_values, return_values)
            else:
                metric_value, p_value = stats.pearsonr(factor_values, return_values)
                
        elif self.config.decay_metric == DecayMetric.RANK_CORRELATION:
            metric_value, p_value = stats.spearmanr(factor_values, return_values)
            
        elif self.config.decay_metric == DecayMetric.RETURN_CORRELATION:
            metric_value, p_value = stats.pearsonr(factor_values, return_values)
            
        elif self.config.decay_metric == DecayMetric.PREDICTIVE_POWER:
            # R-squared from simple regression
            try:
                X = factor_values.values.reshape(-1, 1)
                y = return_values.values
                reg = LinearRegression().fit(X, y)
                metric_value = reg.score(X, y)
                p_value = 0.05  # Approximate
            except:
                metric_value, p_value = 0.0, 1.0
                
        elif self.config.decay_metric == DecayMetric.SHARPE_RATIO:
            # Factor-based Sharpe ratio
            if factor_values.std() > 0:
                metric_value = factor_values.mean() / factor_values.std()
                p_value = 0.05  # Approximate
            else:
                metric_value, p_value = 0.0, 1.0
                
        elif self.config.decay_metric == DecayMetric.HIT_RATE:
            # Hit rate (percentage of correct predictions)
            predictions = factor_values > factor_values.median()
            actuals = return_values > return_values.median()
            metric_value = (predictions == actuals).mean()
            
            # Binomial test for significance
            n_correct = (predictions == actuals).sum()
            n_total = len(predictions)
            p_value = stats.binom_test(n_correct, n_total, 0.5, alternative='greater')
            
        else:
            # Default to IC
            metric_value, p_value = stats.spearmanr(factor_values, return_values)
        
        # Handle NaN values
        if np.isnan(metric_value):
            metric_value = 0.0
        if np.isnan(p_value):
            p_value = 1.0
            
        return abs(metric_value), p_value  # Use absolute value for decay analysis
    
    def _preprocess_data(self, factor: pd.Series, returns: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Preprocess factor and return data"""
        
        # Align indices
        common_index = factor.index.intersection(returns.index)
        factor_aligned = factor.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        # Remove NaN values
        valid_data = pd.DataFrame({'factor': factor_aligned, 'returns': returns_aligned}).dropna()
        factor_clean = valid_data['factor']
        returns_clean = valid_data['returns']
        
        # Remove outliers if requested
        if self.config.remove_outliers:
            factor_clean = self._remove_outliers(factor_clean)
            returns_clean = self._remove_outliers(returns_clean)
        
        # Normalize factors if requested
        if self.config.normalize_factors:
            factor_clean = (factor_clean - factor_clean.mean()) / factor_clean.std()
        
        return factor_clean, returns_clean
    
    def _remove_outliers(self, series: pd.Series) -> pd.Series:
        """Remove outliers from series"""
        
        mean_val = series.mean()
        std_val = series.std()
        
        lower_bound = mean_val - self.config.outlier_threshold * std_val
        upper_bound = mean_val + self.config.outlier_threshold * std_val
        
        return series.clip(lower_bound, upper_bound)


class SimpleDecayAnalyzer(BaseDecayAnalyzer):
    """Simple decay analysis using direct metric calculation"""
    
    def analyze_decay(self, factor: pd.Series, returns: pd.Series) -> DecayResult:
        """Analyze factor decay using simple approach"""
        
        # Preprocess data
        factor_clean, returns_clean = self._preprocess_data(factor, returns)
        
        # Calculate decay curve
        horizons = list(range(1, self.config.max_horizon + 1, self.config.step_size))
        decay_values = []
        p_values = []
        
        for horizon in horizons:
            metric_value, p_value = self._calculate_decay_metric(factor_clean, returns_clean, horizon)
            decay_values.append(metric_value)
            p_values.append(p_value)
        
        # Calculate confidence intervals (approximate)
        confidence_intervals = []
        for i, (value, p_val) in enumerate(zip(decay_values, p_values)):
            # Simple approximation for confidence interval
            stderr = value * 0.1 if value > 0 else 0.01  # Rough estimate
            margin = stats.norm.ppf((1 + self.config.confidence_level) / 2) * stderr
            
            ci_lower = max(0, value - margin)
            ci_upper = value + margin
            confidence_intervals.append((ci_lower, ci_upper))
        
        # Fit decay model
        model_result = self._fit_decay_model(horizons, decay_values)
        
        # Calculate decay characteristics
        characteristics = self._calculate_decay_characteristics(horizons, decay_values, model_result)
        
        # Determine significant horizons
        significant_horizons = [h for h, p in zip(horizons, p_values) 
                              if p < self.config.significance_level]
        
        # Find optimal horizon (highest metric value)
        optimal_horizon = horizons[np.argmax(decay_values)] if decay_values else 1
        
        return DecayResult(
            horizons=horizons,
            decay_values=decay_values,
            confidence_intervals=confidence_intervals,
            model_type=self.config.decay_model,
            model_parameters=model_result['parameters'],
            model_r2=model_result['r2'],
            model_predictions=model_result['predictions'],
            half_life=characteristics.get('half_life'),
            decay_rate=characteristics.get('decay_rate'),
            initial_strength=decay_values[0] if decay_values else 0.0,
            residual_strength=decay_values[-1] if decay_values else 0.0,
            significance_horizons=significant_horizons,
            optimal_horizon=optimal_horizon,
            stability_score=self._calculate_stability_score(decay_values),
            monotonicity_score=self._calculate_monotonicity_score(decay_values),
            factor_name=factor.name if hasattr(factor, 'name') else "Unknown",
            sample_size=len(factor_clean),
            analysis_period=(factor_clean.index.min(), factor_clean.index.max())
        )
    
    def _fit_decay_model(self, horizons: List[int], decay_values: List[float]) -> Dict[str, Any]:
        """Fit decay model to observed values"""
        
        if not decay_values or len(decay_values) < 3:
            return {
                'parameters': {},
                'r2': 0.0,
                'predictions': [0.0] * len(horizons)
            }
        
        x = np.array(horizons)
        y = np.array(decay_values)
        
        try:
            if self.config.decay_model == DecayModel.LINEAR:
                # Linear decay: y = a - b*x
                def linear_func(x, a, b):
                    return np.maximum(0, a - b * x)
                
                popt, _ = optimize.curve_fit(linear_func, x, y, 
                                           bounds=([0, 0], [np.inf, np.inf]))
                predictions = linear_func(x, *popt)
                parameters = {'intercept': popt[0], 'slope': -popt[1]}
                
            elif self.config.decay_model == DecayModel.EXPONENTIAL:
                # Exponential decay: y = a * exp(-b*x)
                def exp_func(x, a, b):
                    return a * np.exp(-b * x)
                
                # Initial guess
                p0 = [y[0], 0.1]
                popt, _ = optimize.curve_fit(exp_func, x, y, p0=p0,
                                           bounds=([0, 0], [np.inf, np.inf]))
                predictions = exp_func(x, *popt)
                parameters = {'amplitude': popt[0], 'decay_rate': popt[1]}
                
            elif self.config.decay_model == DecayModel.POWER_LAW:
                # Power law decay: y = a * x^(-b)
                def power_func(x, a, b):
                    return a * np.power(x, -b)
                
                p0 = [y[0], 0.5]
                popt, _ = optimize.curve_fit(power_func, x, y, p0=p0,
                                           bounds=([0, 0], [np.inf, 2]))
                predictions = power_func(x, *popt)
                parameters = {'amplitude': popt[0], 'exponent': -popt[1]}
                
            elif self.config.decay_model == DecayModel.POLYNOMIAL:
                # Polynomial decay
                coeffs = np.polyfit(x, y, self.config.polynomial_degree)
                predictions = np.polyval(coeffs, x)
                parameters = {f'coeff_{i}': coeff for i, coeff in enumerate(coeffs)}
                
            else:
                # Default to exponential
                def exp_func(x, a, b):
                    return a * np.exp(-b * x)
                
                p0 = [y[0], 0.1]
                popt, _ = optimize.curve_fit(exp_func, x, y, p0=p0,
                                           bounds=([0, 0], [np.inf, np.inf]))
                predictions = exp_func(x, *popt)
                parameters = {'amplitude': popt[0], 'decay_rate': popt[1]}
            
            # Calculate R-squared
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return {
                'parameters': parameters,
                'r2': r2,
                'predictions': predictions.tolist()
            }
            
        except Exception as e:
            warnings.warn(f"Model fitting failed: {e}")
            return {
                'parameters': {},
                'r2': 0.0,
                'predictions': [0.0] * len(horizons)
            }
    
    def _calculate_decay_characteristics(self, horizons: List[int], decay_values: List[float],
                                       model_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate decay characteristics"""
        
        characteristics = {}
        
        if not decay_values or len(decay_values) < 2:
            return characteristics
        
        initial_value = decay_values[0]
        
        # Calculate half-life if we have exponential model
        if (self.config.decay_model == DecayModel.EXPONENTIAL and 
            'decay_rate' in model_result['parameters']):
            
            decay_rate = model_result['parameters']['decay_rate']
            if decay_rate > 0:
                half_life = np.log(2) / decay_rate
                characteristics['half_life'] = half_life
                characteristics['decay_rate'] = decay_rate
        
        # Calculate empirical half-life (when value drops to 50% of initial)
        if initial_value > 0:
            half_target = initial_value * 0.5
            
            for i, value in enumerate(decay_values):
                if value <= half_target:
                    if i > 0:
                        # Linear interpolation
                        x1, y1 = horizons[i-1], decay_values[i-1]
                        x2, y2 = horizons[i], decay_values[i]
                        
                        if y1 != y2:
                            half_life_empirical = x1 + (half_target - y1) * (x2 - x1) / (y2 - y1)
                            characteristics['empirical_half_life'] = half_life_empirical
                    break
        
        return characteristics
    
    def _calculate_stability_score(self, decay_values: List[float]) -> float:
        """Calculate stability score of decay pattern"""
        
        if len(decay_values) < 3:
            return 0.0
        
        # Calculate coefficient of variation of differences
        diffs = np.diff(decay_values)
        if len(diffs) > 0 and np.std(diffs) > 0:
            cv = np.std(diffs) / np.abs(np.mean(diffs)) if np.mean(diffs) != 0 else np.inf
            stability_score = 1.0 / (1.0 + cv)  # Higher score for lower CV
        else:
            stability_score = 1.0
        
        return stability_score
    
    def _calculate_monotonicity_score(self, decay_values: List[float]) -> float:
        """Calculate monotonicity score (how monotonic is the decay)"""
        
        if len(decay_values) < 2:
            return 1.0
        
        # Count non-monotonic transitions
        non_monotonic = 0
        total_transitions = len(decay_values) - 1
        
        for i in range(1, len(decay_values)):
            if decay_values[i] > decay_values[i-1]:  # Should be decreasing
                non_monotonic += 1
        
        monotonicity_score = 1.0 - (non_monotonic / total_transitions)
        
        return monotonicity_score


class RollingDecayAnalyzer(BaseDecayAnalyzer):
    """Rolling window decay analysis for time-varying patterns"""
    
    def analyze_decay(self, factor: pd.Series, returns: pd.Series) -> DecayResult:
        """Analyze decay using rolling windows"""
        
        # Preprocess data
        factor_clean, returns_clean = self._preprocess_data(factor, returns)
        
        # Rolling decay analysis
        rolling_results = self._rolling_decay_analysis(factor_clean, returns_clean)
        
        # Aggregate results
        aggregated_result = self._aggregate_rolling_results(rolling_results)
        
        return aggregated_result
    
    def _rolling_decay_analysis(self, factor: pd.Series, returns: pd.Series) -> List[DecayResult]:
        """Perform rolling decay analysis"""
        
        results = []
        window_size = self.config.rolling_window
        step_size = self.config.rolling_step
        
        for start_idx in range(0, len(factor) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            
            # Window data
            factor_window = factor.iloc[start_idx:end_idx]
            returns_window = returns.iloc[start_idx:end_idx]
            
            # Analyze window
            try:
                simple_analyzer = SimpleDecayAnalyzer(self.config)
                window_result = simple_analyzer.analyze_decay(factor_window, returns_window)
                results.append(window_result)
            except Exception as e:
                warnings.warn(f"Rolling window analysis failed: {e}")
                continue
        
        return results
    
    def _aggregate_rolling_results(self, rolling_results: List[DecayResult]) -> DecayResult:
        """Aggregate rolling analysis results"""
        
        if not rolling_results:
            # Return empty result
            return DecayResult(
                horizons=[],
                decay_values=[],
                confidence_intervals=[],
                model_type=self.config.decay_model,
                model_parameters={},
                model_r2=0.0,
                model_predictions=[]
            )
        
        # Aggregate decay curves
        n_horizons = len(rolling_results[0].horizons)
        horizons = rolling_results[0].horizons
        
        # Calculate statistics across rolling windows
        decay_matrix = np.array([result.decay_values for result in rolling_results])
        
        # Mean decay curve
        mean_decay = np.mean(decay_matrix, axis=0)
        
        # Confidence intervals from rolling results
        lower_percentile = (1 - self.config.confidence_level) / 2 * 100
        upper_percentile = (1 + self.config.confidence_level) / 2 * 100
        
        confidence_intervals = []
        for i in range(n_horizons):
            values = decay_matrix[:, i]
            ci_lower = np.percentile(values, lower_percentile)
            ci_upper = np.percentile(values, upper_percentile)
            confidence_intervals.append((ci_lower, ci_upper))
        
        # Fit model to mean decay curve
        simple_analyzer = SimpleDecayAnalyzer(self.config)
        model_result = simple_analyzer._fit_decay_model(horizons, mean_decay.tolist())
        
        # Aggregate characteristics
        half_lives = [result.half_life for result in rolling_results if result.half_life is not None]
        mean_half_life = np.mean(half_lives) if half_lives else None
        
        stability_scores = [result.stability_score for result in rolling_results]
        mean_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        monotonicity_scores = [result.monotonicity_score for result in rolling_results]
        mean_monotonicity = np.mean(monotonicity_scores) if monotonicity_scores else 0.0
        
        # Optimal horizon (most frequent)
        optimal_horizons = [result.optimal_horizon for result in rolling_results]
        optimal_horizon = max(set(optimal_horizons), key=optimal_horizons.count) if optimal_horizons else 1
        
        return DecayResult(
            horizons=horizons,
            decay_values=mean_decay.tolist(),
            confidence_intervals=confidence_intervals,
            model_type=self.config.decay_model,
            model_parameters=model_result['parameters'],
            model_r2=model_result['r2'],
            model_predictions=model_result['predictions'],
            half_life=mean_half_life,
            initial_strength=mean_decay[0] if len(mean_decay) > 0 else 0.0,
            residual_strength=mean_decay[-1] if len(mean_decay) > 0 else 0.0,
            optimal_horizon=optimal_horizon,
            stability_score=mean_stability,
            monotonicity_score=mean_monotonicity,
            sample_size=sum(result.sample_size for result in rolling_results)
        )


class FactorDecayAnalyzer:
    """
    Main factor decay analysis engine
    """
    
    def __init__(self):
        self.analysis_cache: Dict[str, DecayResult] = {}
        self.analyzer_instances: Dict[str, BaseDecayAnalyzer] = {}
        
    def analyze_factor_decay(self, factor: pd.Series, returns: pd.Series,
                           config: Optional[DecayConfig] = None,
                           analyzer_type: str = "simple") -> DecayResult:
        """Analyze factor decay"""
        
        if config is None:
            config = DecayConfig()
        
        # Create analyzer
        if analyzer_type.lower() == "rolling":
            analyzer = RollingDecayAnalyzer(config)
        else:
            analyzer = SimpleDecayAnalyzer(config)
        
        # Perform analysis
        try:
            result = analyzer.analyze_decay(factor, returns)
            
            # Cache result
            cache_key = f"{factor.name}_{analyzer_type}_{hash(str(config.__dict__))}"
            self.analysis_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Decay analysis failed: {e}")
    
    def analyze_multiple_factors(self, factors: pd.DataFrame, returns: pd.Series,
                               config: Optional[DecayConfig] = None,
                               analyzer_type: str = "simple") -> Dict[str, DecayResult]:
        """Analyze decay for multiple factors"""
        
        results = {}
        
        for factor_name in factors.columns:
            try:
                factor_series = factors[factor_name]
                factor_series.name = factor_name
                
                result = self.analyze_factor_decay(factor_series, returns, config, analyzer_type)
                results[factor_name] = result
                
            except Exception as e:
                warnings.warn(f"Decay analysis failed for factor {factor_name}: {e}")
                continue
        
        return results
    
    def compare_decay_patterns(self, factors: pd.DataFrame, returns: pd.Series,
                             config: Optional[DecayConfig] = None) -> pd.DataFrame:
        """Compare decay patterns across factors"""
        
        # Analyze all factors
        results = self.analyze_multiple_factors(factors, returns, config)
        
        if not results:
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_data = []
        
        for factor_name, result in results.items():
            comparison_data.append({
                'factor_name': factor_name,
                'initial_strength': result.initial_strength,
                'residual_strength': result.residual_strength,
                'half_life': result.half_life,
                'optimal_horizon': result.optimal_horizon,
                'stability_score': result.stability_score,
                'monotonicity_score': result.monotonicity_score,
                'model_r2': result.model_r2,
                'n_significant_horizons': len(result.significance_horizons),
                'decay_rate': result.decay_rate
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by initial strength
        comparison_df = comparison_df.sort_values('initial_strength', ascending=False)
        
        return comparison_df
    
    def plot_decay_curves(self, results: Dict[str, DecayResult], 
                         max_factors: int = 10, figsize: Tuple[int, int] = (12, 8)):
        """Plot decay curves for multiple factors"""
        
        import matplotlib.pyplot as plt
        
        # Limit number of factors for readability
        if len(results) > max_factors:
            # Sort by initial strength and take top factors
            sorted_results = sorted(results.items(), 
                                  key=lambda x: x[1].initial_strength, 
                                  reverse=True)
            results = dict(sorted_results[:max_factors])
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: Decay curves
        ax1 = axes[0]
        for factor_name, result in results.items():
            ax1.plot(result.horizons, result.decay_values, 
                    marker='o', label=factor_name, alpha=0.7)
        
        ax1.set_xlabel('Forecast Horizon')
        ax1.set_ylabel('Signal Strength')
        ax1.set_title('Factor Decay Curves')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Half-life distribution
        ax2 = axes[1]
        half_lives = [result.half_life for result in results.values() 
                     if result.half_life is not None and result.half_life < 50]
        
        if half_lives:
            ax2.hist(half_lives, bins=min(10, len(half_lives)), alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Half-life (periods)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Half-lives')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Initial vs Residual Strength
        ax3 = axes[2]
        initial_strengths = [result.initial_strength for result in results.values()]
        residual_strengths = [result.residual_strength for result in results.values()]
        
        ax3.scatter(initial_strengths, residual_strengths, alpha=0.7)
        ax3.plot([0, max(initial_strengths)], [0, max(initial_strengths)], 
                'r--', alpha=0.5, label='No decay line')
        
        ax3.set_xlabel('Initial Strength')
        ax3.set_ylabel('Residual Strength')
        ax3.set_title('Initial vs Residual Signal Strength')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model fit quality
        ax4 = axes[3]
        factor_names = list(results.keys())
        model_r2s = [result.model_r2 for result in results.values()]
        
        bars = ax4.bar(range(len(factor_names)), model_r2s, alpha=0.7)
        ax4.set_xlabel('Factors')
        ax4.set_ylabel('Model R²')
        ax4.set_title('Decay Model Fit Quality')
        ax4.set_xticks(range(len(factor_names)))
        ax4.set_xticklabels(factor_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Color bars based on R²
        for bar, r2 in zip(bars, model_r2s):
            if r2 > 0.8:
                bar.set_color('green')
            elif r2 > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.show()
    
    def get_decay_summary(self, results: Dict[str, DecayResult]) -> Dict[str, Any]:
        """Get summary statistics of decay analysis"""
        
        if not results:
            return {}
        
        # Extract metrics
        initial_strengths = [r.initial_strength for r in results.values()]
        half_lives = [r.half_life for r in results.values() if r.half_life is not None]
        stability_scores = [r.stability_score for r in results.values()]
        monotonicity_scores = [r.monotonicity_score for r in results.values()]
        optimal_horizons = [r.optimal_horizon for r in results.values()]
        
        summary = {
            'n_factors_analyzed': len(results),
            'initial_strength': {
                'mean': np.mean(initial_strengths),
                'median': np.median(initial_strengths),
                'std': np.std(initial_strengths),
                'min': np.min(initial_strengths),
                'max': np.max(initial_strengths)
            },
            'half_life': {
                'mean': np.mean(half_lives) if half_lives else None,
                'median': np.median(half_lives) if half_lives else None,
                'std': np.std(half_lives) if half_lives else None,
                'n_estimated': len(half_lives)
            },
            'stability': {
                'mean': np.mean(stability_scores),
                'median': np.median(stability_scores),
                'high_stability_pct': np.mean([s > 0.7 for s in stability_scores]) * 100
            },
            'monotonicity': {
                'mean': np.mean(monotonicity_scores),
                'median': np.median(monotonicity_scores),
                'monotonic_pct': np.mean([s > 0.8 for s in monotonicity_scores]) * 100
            },
            'optimal_horizon': {
                'mean': np.mean(optimal_horizons),
                'median': np.median(optimal_horizons),
                'mode': max(set(optimal_horizons), key=optimal_horizons.count)
            }
        }
        
        return summary
