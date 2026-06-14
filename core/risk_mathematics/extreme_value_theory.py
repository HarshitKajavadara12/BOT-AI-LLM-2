"""
Extreme Value Theory Engine for QUANTUM-FORGE
Implements sophisticated EVT models for tail risk analysis and extreme event modeling.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import gamma, gammainc
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from numba import jit, prange
from dataclasses import dataclass
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class EVTDistribution(Enum):
    """Extreme value theory distributions."""
    GUMBEL = "gumbel"           # ξ = 0
    FRECHET = "frechet"         # ξ > 0
    WEIBULL = "weibull"         # ξ < 0
    GENERALIZED_PARETO = "gpd"  # Generalized Pareto Distribution

@dataclass
class EVTParameters:
    """Parameters for EVT distributions."""
    location: float     # μ (location parameter)
    scale: float        # σ (scale parameter)  
    shape: float        # ξ (shape parameter)
    threshold: float    # u (threshold for POT)
    distribution: EVTDistribution

@dataclass
class TailRiskMetrics:
    """Comprehensive tail risk metrics."""
    var_95: float       # Value at Risk 95%
    var_99: float       # Value at Risk 99%
    var_999: float      # Value at Risk 99.9%
    es_95: float        # Expected Shortfall 95%
    es_99: float        # Expected Shortfall 99%
    es_999: float       # Expected Shortfall 99.9%
    tail_index: float   # Tail index (Hill estimator)
    return_level_10y: float    # 10-year return level
    return_level_100y: float   # 100-year return level

class GeneralizedExtremeValue:
    """Generalized Extreme Value (GEV) distribution for block maxima."""
    
    def __init__(self):
        """Initialize GEV model."""
        self.location = 0.0     # μ
        self.scale = 1.0        # σ  
        self.shape = 0.0        # ξ
        self.fitted = False
        
    def fit(self, data: np.ndarray, method: str = 'mle') -> bool:
        """
        Fit GEV distribution to block maxima data.
        
        Args:
            data: Block maxima observations
            method: Fitting method ('mle', 'mom', 'pwm')
        
        Returns:
            True if fitting successful
        """
        if len(data) < 10:
            return False
        
        try:
            if method == 'mle':
                # Maximum likelihood estimation
                def neg_log_likelihood(params):
                    mu, sigma, xi = params
                    if sigma <= 0:
                        return np.inf
                    
                    if abs(xi) < 1e-6:  # Gumbel case
                        z = (data - mu) / sigma
                        return len(data) * np.log(sigma) + np.sum(z) + np.sum(np.exp(-z))
                    else:  # General case
                        z = (data - mu) / sigma
                        t = 1 + xi * z
                        
                        if np.any(t <= 0):
                            return np.inf
                        
                        return (len(data) * np.log(sigma) + 
                               (1 + 1/xi) * np.sum(np.log(t)) + 
                               np.sum(np.power(t, -1/xi)))
                
                # Initial parameter estimates
                sample_mean = np.mean(data)
                sample_std = np.std(data)
                initial_params = [sample_mean, sample_std * 0.78, 0.1]
                
                # Optimization
                result = optimize.minimize(
                    neg_log_likelihood, 
                    initial_params, 
                    method='L-BFGS-B',
                    bounds=[(-np.inf, np.inf), (1e-6, np.inf), (-0.5, 0.5)]
                )
                
                if result.success:
                    self.location, self.scale, self.shape = result.x
                    self.fitted = True
                    return True
                    
            elif method == 'mom':
                # Method of moments
                sample_mean = np.mean(data)
                sample_var = np.var(data)
                sample_skew = stats.skew(data)
                
                # Solve for shape parameter using skewness
                if abs(sample_skew) < 1e-6:
                    self.shape = 0.0  # Gumbel
                else:
                    # Approximate relationship for shape parameter
                    self.shape = -sample_skew / 3.0
                    self.shape = np.clip(self.shape, -0.5, 0.5)
                
                # Calculate scale and location
                if abs(self.shape) < 1e-6:
                    self.scale = np.sqrt(6 * sample_var) / np.pi
                    self.location = sample_mean - 0.5772 * self.scale  # Euler's constant
                else:
                    gamma1 = gamma(1 - self.shape)
                    gamma2 = gamma(1 - 2 * self.shape)
                    
                    if gamma1 > 0 and gamma2 > 0:
                        self.scale = np.sqrt(sample_var * self.shape**2 / (gamma2 - gamma1**2))
                        self.location = sample_mean - self.scale * (gamma1 - 1) / self.shape
                    else:
                        return False
                
                self.fitted = True
                return True
                
        except Exception as e:
            return False
        
        return False
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Cumulative distribution function."""
        if not self.fitted:
            return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        
        z = (x - self.location) / self.scale
        
        if abs(self.shape) < 1e-6:  # Gumbel
            return np.exp(-np.exp(-z))
        else:  # General GEV
            t = 1 + self.shape * z
            t = np.maximum(t, 1e-10)  # Avoid domain issues
            return np.exp(-np.power(t, -1/self.shape))
    
    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Probability density function."""
        if not self.fitted:
            return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        
        z = (x - self.location) / self.scale
        
        if abs(self.shape) < 1e-6:  # Gumbel
            exp_z = np.exp(-z)
            return exp_z * np.exp(-exp_z) / self.scale
        else:  # General GEV
            t = 1 + self.shape * z
            t = np.maximum(t, 1e-10)
            
            return (np.power(t, -1/self.shape - 1) * 
                   np.exp(-np.power(t, -1/self.shape))) / self.scale
    
    def quantile(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function."""
        if not self.fitted:
            return np.zeros_like(p) if isinstance(p, np.ndarray) else 0.0
        
        if abs(self.shape) < 1e-6:  # Gumbel
            return self.location - self.scale * np.log(-np.log(p))
        else:  # General GEV
            return (self.location + 
                   self.scale * (np.power(-np.log(p), -self.shape) - 1) / self.shape)
    
    def return_level(self, return_period: float) -> float:
        """
        Calculate return level for given return period.
        
        Args:
            return_period: Return period (e.g., 100 for 100-year return level)
        
        Returns:
            Return level
        """
        if not self.fitted:
            return 0.0
        
        p = 1 - 1/return_period
        return self.quantile(p)

class GeneralizedParetoDistribution:
    """Generalized Pareto Distribution for peaks-over-threshold analysis."""
    
    def __init__(self):
        """Initialize GPD model."""
        self.scale = 1.0        # σ
        self.shape = 0.0        # ξ
        self.threshold = 0.0    # u
        self.fitted = False
        self.exceedance_rate = 0.0  # Rate of threshold exceedances
        
    def fit(self, data: np.ndarray, threshold: float, method: str = 'mle') -> bool:
        """
        Fit GPD to exceedances over threshold.
        
        Args:
            data: Full data series
            threshold: Threshold for exceedances
            method: Fitting method ('mle', 'mom')
        
        Returns:
            True if fitting successful
        """
        # Extract exceedances
        exceedances = data[data > threshold] - threshold
        
        if len(exceedances) < 10:
            return False
        
        self.threshold = threshold
        self.exceedance_rate = len(exceedances) / len(data)
        
        try:
            if method == 'mle':
                def neg_log_likelihood(params):
                    sigma, xi = params
                    if sigma <= 0:
                        return np.inf
                    
                    if abs(xi) < 1e-6:  # Exponential case
                        return len(exceedances) * np.log(sigma) + np.sum(exceedances) / sigma
                    else:  # General case
                        y = exceedances / sigma
                        t = 1 + xi * y
                        
                        if np.any(t <= 0):
                            return np.inf
                        
                        return len(exceedances) * np.log(sigma) + (1 + 1/xi) * np.sum(np.log(t))
                
                # Initial estimates
                sample_mean = np.mean(exceedances)
                sample_var = np.var(exceedances)
                
                initial_sigma = sample_mean / 2
                initial_xi = 0.5 * (sample_mean**2 / sample_var - 1)
                initial_xi = np.clip(initial_xi, -0.5, 0.5)
                
                # Optimization
                result = optimize.minimize(
                    neg_log_likelihood,
                    [initial_sigma, initial_xi],
                    method='L-BFGS-B',
                    bounds=[(1e-6, np.inf), (-0.5, 0.5)]
                )
                
                if result.success:
                    self.scale, self.shape = result.x
                    self.fitted = True
                    return True
                    
            elif method == 'mom':
                # Method of moments
                sample_mean = np.mean(exceedances)
                sample_var = np.var(exceedances)
                
                # GPD moment relationships
                if sample_var > sample_mean**2:
                    self.shape = 0.5 * (sample_mean**2 / sample_var - 1)
                    self.shape = np.clip(self.shape, -0.5, 0.5)
                    self.scale = sample_mean * (1 + self.shape)
                else:
                    # Exponential case
                    self.shape = 0.0
                    self.scale = sample_mean
                
                self.fitted = True
                return True
                
        except Exception as e:
            return False
        
        return False
    
    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """CDF of GPD for exceedances."""
        if not self.fitted:
            return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0
        
        x = np.asarray(x)
        result = np.zeros_like(x)
        
        # Only defined for x >= 0
        valid = x >= 0
        
        if abs(self.shape) < 1e-6:  # Exponential
            result[valid] = 1 - np.exp(-x[valid] / self.scale)
        else:  # General GPD
            t = 1 + self.shape * x[valid] / self.scale
            t = np.maximum(t, 1e-10)
            result[valid] = 1 - np.power(t, -1/self.shape)
        
        return result
    
    def quantile(self, p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Quantile function of GPD."""
        if not self.fitted:
            return np.zeros_like(p) if isinstance(p, np.ndarray) else 0.0
        
        if abs(self.shape) < 1e-6:  # Exponential
            return -self.scale * np.log(1 - p)
        else:  # General GPD
            return self.scale * (np.power(1 - p, -self.shape) - 1) / self.shape
    
    def var_estimate(self, confidence_level: float, n_observations: int) -> float:
        """
        Estimate VaR using GPD tail approximation.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95)
            n_observations: Total number of observations
        
        Returns:
            VaR estimate
        """
        if not self.fitted:
            return 0.0
        
        # Probability in the tail
        tail_prob = 1 - confidence_level
        
        # Conditional exceedance probability
        if tail_prob <= self.exceedance_rate:
            # Use GPD approximation
            conditional_prob = tail_prob / self.exceedance_rate
            exceedance_quantile = self.quantile(1 - conditional_prob)
            return self.threshold + exceedance_quantile
        else:
            # Use empirical quantile for less extreme levels
            return 0.0
    
    def expected_shortfall(self, confidence_level: float) -> float:
        """
        Calculate Expected Shortfall using GPD.
        
        Args:
            confidence_level: Confidence level
        
        Returns:
            Expected Shortfall estimate
        """
        if not self.fitted:
            return 0.0
        
        var = self.var_estimate(confidence_level, 1000)  # Simplified
        
        if abs(self.shape) < 1e-6:  # Exponential case
            return var + self.scale
        elif self.shape < 1:  # General case
            return (var + self.scale - self.shape * self.threshold) / (1 - self.shape)
        else:
            return np.inf  # ES doesn't exist for ξ >= 1

class HillEstimator:
    """Hill estimator for tail index estimation."""
    
    def __init__(self):
        """Initialize Hill estimator."""
        self.tail_index = 0.0
        self.optimal_k = 0
        self.fitted = False
    
    def fit(self, data: np.ndarray, method: str = 'adaptive') -> bool:
        """
        Estimate tail index using Hill estimator.
        
        Args:
            data: Data series (will use upper order statistics)
            method: Method for selecting k ('fixed', 'adaptive')
        
        Returns:
            True if estimation successful
        """
        if len(data) < 20:
            return False
        
        # Sort data in descending order
        sorted_data = np.sort(data)[::-1]
        n = len(sorted_data)
        
        try:
            if method == 'fixed':
                # Use fixed fraction of observations
                k = min(int(n * 0.1), 100)  # Use top 10% or 100 obs, whichever is smaller
                
                if k < 5:
                    return False
                
                # Hill estimator: average of log ratios
                log_ratios = np.log(sorted_data[:k]) - np.log(sorted_data[k])
                self.tail_index = np.mean(log_ratios)
                self.optimal_k = k
                
            elif method == 'adaptive':
                # Adaptive selection of k using bias-variance tradeoff
                min_k = max(5, int(n * 0.01))
                max_k = min(int(n * 0.3), 200)
                
                k_values = range(min_k, max_k + 1)
                hill_estimates = []
                
                for k in k_values:
                    if k < len(sorted_data):
                        log_ratios = np.log(sorted_data[:k]) - np.log(sorted_data[k])
                        hill_est = np.mean(log_ratios)
                        hill_estimates.append(hill_est)
                    else:
                        hill_estimates.append(np.nan)
                
                hill_estimates = np.array(hill_estimates)
                valid_estimates = hill_estimates[~np.isnan(hill_estimates)]
                
                if len(valid_estimates) == 0:
                    return False
                
                # Choose k that minimizes the variance (simplified approach)
                # In practice, more sophisticated methods like double bootstrap would be used
                variances = []
                for i, k in enumerate(k_values):
                    if i < len(hill_estimates) and not np.isnan(hill_estimates[i]):
                        # Bootstrap estimate of variance (simplified)
                        n_bootstrap = 100
                        bootstrap_estimates = []
                        
                        for b in range(n_bootstrap):
                            # Deterministic 'bootstrap' via rotations of top-k (reproducible)
                            rot = b % max(1, k)
                            bootstrap_sample = np.roll(sorted_data[:k], rot)
                            bootstrap_sample = np.sort(bootstrap_sample)[::-1]

                            if len(bootstrap_sample) > 1:
                                log_ratios = np.log(bootstrap_sample[:-1]) - np.log(bootstrap_sample[-1])
                                bootstrap_estimates.append(np.mean(log_ratios))
                        
                        if bootstrap_estimates:
                            variances.append(np.var(bootstrap_estimates))
                        else:
                            variances.append(np.inf)
                    else:
                        variances.append(np.inf)
                
                # Select k with minimum variance
                if variances:
                    optimal_idx = np.argmin(variances)
                    self.optimal_k = k_values[optimal_idx]
                    self.tail_index = hill_estimates[optimal_idx]
                else:
                    return False
            
            self.fitted = True
            return True
            
        except Exception as e:
            return False
    
    def get_tail_index(self) -> float:
        """Get estimated tail index."""
        return self.tail_index if self.fitted else 0.0
    
    def get_confidence_interval(self, confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Get confidence interval for tail index estimate.
        
        Args:
            confidence_level: Confidence level
        
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not self.fitted:
            return (0.0, 0.0)
        
        # Asymptotic normality of Hill estimator
        std_error = self.tail_index / np.sqrt(self.optimal_k)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        lower = self.tail_index - z_score * std_error
        upper = self.tail_index + z_score * std_error
        
        return (lower, upper)

class EVTAnalyzer:
    """Comprehensive extreme value theory analysis framework."""
    
    def __init__(self):
        """Initialize EVT analyzer."""
        self.gev_model = GeneralizedExtremeValue()
        self.gpd_model = GeneralizedParetoDistribution()
        self.hill_estimator = HillEstimator()
        
        self.block_maxima = []
        self.exceedances = []
        self.fitted_models = set()
    
    def fit_block_maxima_model(self, data: np.ndarray, block_size: int = 252) -> bool:
        """
        Fit GEV model to block maxima.
        
        Args:
            data: Time series data
            block_size: Size of blocks (e.g., 252 for yearly blocks of daily data)
        
        Returns:
            True if fitting successful
        """
        if len(data) < block_size * 3:  # Need at least 3 blocks
            return False
        
        # Create block maxima
        n_blocks = len(data) // block_size
        maxima = []
        
        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size
            block_data = data[start_idx:end_idx]
            maxima.append(np.max(block_data))
        
        self.block_maxima = np.array(maxima)
        
        # Fit GEV model
        success = self.gev_model.fit(self.block_maxima)
        if success:
            self.fitted_models.add('gev')
        
        return success
    
    def fit_peaks_over_threshold_model(self, data: np.ndarray, 
                                     threshold: Optional[float] = None,
                                     threshold_quantile: float = 0.95) -> bool:
        """
        Fit GPD model using peaks-over-threshold approach.
        
        Args:
            data: Time series data
            threshold: Explicit threshold (if None, use quantile)
            threshold_quantile: Quantile for automatic threshold selection
        
        Returns:
            True if fitting successful
        """
        if threshold is None:
            threshold = np.quantile(data, threshold_quantile)
        
        # Fit GPD model
        success = self.gpd_model.fit(data, threshold)
        if success:
            self.fitted_models.add('gpd')
        
        return success
    
    def fit_tail_index_model(self, data: np.ndarray) -> bool:
        """
        Estimate tail index using Hill estimator.
        
        Args:
            data: Time series data
        
        Returns:
            True if estimation successful
        """
        success = self.hill_estimator.fit(data, method='adaptive')
        if success:
            self.fitted_models.add('hill')
        
        return success
    
    def calculate_comprehensive_metrics(self, data: np.ndarray) -> TailRiskMetrics:
        """
        Calculate comprehensive tail risk metrics.
        
        Args:
            data: Time series data
        
        Returns:
            TailRiskMetrics object
        """
        # Fit all models if not already fitted
        if 'gev' not in self.fitted_models:
            self.fit_block_maxima_model(data)
        
        if 'gpd' not in self.fitted_models:
            self.fit_peaks_over_threshold_model(data)
        
        if 'hill' not in self.fitted_models:
            self.fit_tail_index_model(data)
        
        # Calculate VaR estimates
        var_95 = np.quantile(data, 0.05) if len(data) > 0 else 0.0  # Left tail
        var_99 = np.quantile(data, 0.01) if len(data) > 0 else 0.0
        var_999 = np.quantile(data, 0.001) if len(data) > 0 else 0.0
        
        # Use GPD for extreme quantiles if available
        if 'gpd' in self.fitted_models:
            var_99 = min(var_99, -self.gpd_model.var_estimate(0.99, len(data)))
            var_999 = min(var_999, -self.gpd_model.var_estimate(0.999, len(data)))
        
        # Calculate Expected Shortfall
        es_95 = np.mean(data[data <= var_95]) if np.any(data <= var_95) else var_95
        es_99 = np.mean(data[data <= var_99]) if np.any(data <= var_99) else var_99
        es_999 = np.mean(data[data <= var_999]) if np.any(data <= var_999) else var_999
        
        # Use GPD ES if available
        if 'gpd' in self.fitted_models:
            gpd_es_99 = -self.gpd_model.expected_shortfall(0.99)
            gpd_es_999 = -self.gpd_model.expected_shortfall(0.999)
            
            if not np.isinf(gpd_es_99):
                es_99 = min(es_99, gpd_es_99)
            if not np.isinf(gpd_es_999):
                es_999 = min(es_999, gpd_es_999)
        
        # Tail index
        tail_index = self.hill_estimator.get_tail_index() if 'hill' in self.fitted_models else 0.0
        
        # Return levels using GEV
        return_level_10y = 0.0
        return_level_100y = 0.0
        
        if 'gev' in self.fitted_models:
            return_level_10y = self.gev_model.return_level(10)
            return_level_100y = self.gev_model.return_level(100)
        
        return TailRiskMetrics(
            var_95=var_95,
            var_99=var_99,
            var_999=var_999,
            es_95=es_95,
            es_99=es_99,
            es_999=es_999,
            tail_index=tail_index,
            return_level_10y=return_level_10y,
            return_level_100y=return_level_100y
        )
    
    def generate_extreme_scenarios(self, n_scenarios: int = 1000, 
                                 horizon: int = 252) -> np.ndarray:
        """
        Generate extreme scenarios using fitted EVT models.
        
        Args:
            n_scenarios: Number of scenarios to generate
            horizon: Time horizon for scenarios
        
        Returns:
            Array of extreme scenarios
        """
        scenarios = []
        
        if 'gev' in self.fitted_models:
            # Generate scenarios using GEV model
            gev_scenarios = []
            for _ in range(n_scenarios):
                # Generate block maxima for the horizon
                n_blocks = max(1, horizon // 252)  # Yearly blocks
                block_maxima = []
                
                for bi in range(n_blocks):
                    # Deterministic grid for quantiles
                    u = float(bi + 1) / (n_blocks + 1)
                    extreme_value = self.gev_model.quantile(u)
                    block_maxima.append(extreme_value)
                
                scenarios.append(np.max(block_maxima))
        
        elif 'gpd' in self.fitted_models:
            # Generate scenarios using GPD model
            for si in range(n_scenarios):
                # Deterministic uniform quantiles across scenarios
                u = float(si + 1) / (n_scenarios + 1)
                exceedance = self.gpd_model.quantile(u)
                extreme_value = self.gpd_model.threshold + exceedance
                scenarios.append(extreme_value)
        
        else:
            # Fallback to empirical bootstrap
            top100 = np.sort(data)[-100:]
            # Deterministic repeated selection from top100
            tiled = np.tile(top100, int(np.ceil(n_scenarios / len(top100))))
            scenarios = tiled[:n_scenarios]
        
        return np.array(scenarios)
    
    def diagnostic_plots(self, data: np.ndarray) -> Dict[str, plt.Figure]:
        """
        Generate diagnostic plots for EVT model validation.
        
        Args:
            data: Time series data
        
        Returns:
            Dictionary of diagnostic plots
        """
        plots = {}
        
        try:
            # Q-Q plot for GEV
            if 'gev' in self.fitted_models and len(self.block_maxima) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Theoretical quantiles
                n = len(self.block_maxima)
                p_values = (np.arange(1, n+1) - 0.5) / n
                theoretical_quantiles = self.gev_model.quantile(p_values)
                
                # Empirical quantiles
                empirical_quantiles = np.sort(self.block_maxima)
                
                ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.7)
                ax.plot([np.min(theoretical_quantiles), np.max(theoretical_quantiles)],
                       [np.min(theoretical_quantiles), np.max(theoretical_quantiles)],
                       'r--', label='Perfect fit')
                
                ax.set_xlabel('Theoretical Quantiles (GEV)')
                ax.set_ylabel('Empirical Quantiles')
                ax.set_title('Q-Q Plot: GEV Model')
                ax.legend()
                ax.grid(True)
                
                plots['gev_qq'] = fig
            
            # Mean excess plot for threshold selection
            if len(data) > 100:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                thresholds = np.quantile(data, np.linspace(0.8, 0.99, 20))
                mean_excesses = []
                
                for threshold in thresholds:
                    exceedances = data[data > threshold] - threshold
                    if len(exceedances) > 5:
                        mean_excesses.append(np.mean(exceedances))
                    else:
                        mean_excesses.append(np.nan)
                
                valid_idx = ~np.isnan(mean_excesses)
                ax.plot(thresholds[valid_idx], np.array(mean_excesses)[valid_idx], 
                       'bo-', markersize=4)
                
                ax.set_xlabel('Threshold')
                ax.set_ylabel('Mean Excess')
                ax.set_title('Mean Excess Plot')
                ax.grid(True)
                
                plots['mean_excess'] = fig
            
            # Hill plot for tail index
            if 'hill' in self.fitted_models:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                sorted_data = np.sort(data)[::-1]
                n = len(sorted_data)
                k_values = range(10, min(200, n//2))
                hill_estimates = []
                
                for k in k_values:
                    if k < len(sorted_data):
                        log_ratios = np.log(sorted_data[:k]) - np.log(sorted_data[k])
                        hill_est = np.mean(log_ratios)
                        hill_estimates.append(hill_est)
                    else:
                        hill_estimates.append(np.nan)
                
                valid_idx = ~np.isnan(hill_estimates)
                ax.plot(np.array(k_values)[valid_idx], np.array(hill_estimates)[valid_idx], 
                       'b-', linewidth=2)
                
                # Mark optimal k
                ax.axvline(x=self.hill_estimator.optimal_k, color='r', linestyle='--',
                          label=f'Optimal k = {self.hill_estimator.optimal_k}')
                
                ax.set_xlabel('Number of Order Statistics (k)')
                ax.set_ylabel('Hill Estimate')
                ax.set_title('Hill Plot')
                ax.legend()
                ax.grid(True)
                
                plots['hill_plot'] = fig
        
        except Exception as e:
            pass  # Return empty plots dict if plotting fails
        
        return plots

# Example usage and testing
if __name__ == "__main__":
    print("Testing Extreme Value Theory Engine...")
    
    # Generate synthetic financial return data with fat tails (deterministic)
    n_observations = 2000

    # Base normal returns via deterministic inverse CDF grid
    u_base = np.linspace(1.0/(n_observations+1), n_observations/(n_observations+1), n_observations)
    normal_returns = norm.ppf(u_base) * 0.02

    # Add extreme events (2% deterministic extremes at evenly spaced indices)
    n_extremes = int(n_observations * 0.02)
    extreme_indices = np.linspace(0, n_observations - 1, n_extremes, dtype=int)

    # Deterministic extreme returns using t-quantiles on a grid of upper tail probs
    u_ext = np.linspace(0.995, 0.999, n_extremes)
    extreme_returns = t_dist.ppf(u_ext, df=3) * 0.05

    # Combine normal and extreme returns
    returns = normal_returns.copy()
    returns[extreme_indices] = extreme_returns
    
    print(f"Generated {n_observations} returns with {n_extremes} extreme events")
    print(f"Sample statistics: mean={np.mean(returns):.4f}, std={np.std(returns):.4f}")
    print(f"Skewness: {stats.skew(returns):.3f}, Kurtosis: {stats.kurtosis(returns):.3f}")
    
    print("\nTesting Generalized Extreme Value model...")
    gev = GeneralizedExtremeValue()
    
    # Create block maxima (monthly blocks)
    block_size = 21  # Daily data, monthly blocks
    n_blocks = len(returns) // block_size
    block_maxima = []
    
    for i in range(n_blocks):
        block_data = returns[i*block_size:(i+1)*block_size]
        block_maxima.append(np.max(block_data))
    
    block_maxima = np.array(block_maxima)
    
    success = gev.fit(block_maxima, method='mle')
    print(f"GEV fitting: {'successful' if success else 'failed'}")
    
    if success:
        print(f"GEV parameters: μ={gev.location:.4f}, σ={gev.scale:.4f}, ξ={gev.shape:.4f}")
        
        # Calculate return levels
        return_10y = gev.return_level(10)
        return_100y = gev.return_level(100)
        print(f"10-year return level: {return_10y:.4f}")
        print(f"100-year return level: {return_100y:.4f}")
    
    print("\nTesting Generalized Pareto Distribution...")
    gpd = GeneralizedParetoDistribution()
    
    # Use 95th percentile as threshold
    threshold = np.quantile(returns, 0.95)
    success = gpd.fit(returns, threshold, method='mle')
    
    print(f"GPD fitting: {'successful' if success else 'failed'}")
    
    if success:
        print(f"GPD parameters: σ={gpd.scale:.4f}, ξ={gpd.shape:.4f}")
        print(f"Threshold: {threshold:.4f}")
        print(f"Exceedance rate: {gpd.exceedance_rate:.3f}")
        
        # Calculate VaR estimates
        var_99 = gpd.var_estimate(0.99, len(returns))
        var_999 = gpd.var_estimate(0.999, len(returns))
        
        print(f"GPD VaR(99%): {var_99:.4f}")
        print(f"GPD VaR(99.9%): {var_999:.4f}")
        
        # Calculate Expected Shortfall
        es_99 = gpd.expected_shortfall(0.99)
        print(f"GPD ES(99%): {es_99:.4f}")
    
    print("\nTesting Hill estimator...")
    hill = HillEstimator()
    
    success = hill.fit(returns, method='adaptive')
    print(f"Hill estimation: {'successful' if success else 'failed'}")
    
    if success:
        tail_index = hill.get_tail_index()
        lower, upper = hill.get_confidence_interval(0.95)
        
        print(f"Tail index: {tail_index:.4f}")
        print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
        print(f"Optimal k: {hill.optimal_k}")
    
    print("\nTesting comprehensive EVT analysis...")
    evt_analyzer = EVTAnalyzer()
    
    # Fit all models
    success_gev = evt_analyzer.fit_block_maxima_model(returns, block_size=21)
    success_gpd = evt_analyzer.fit_peaks_over_threshold_model(returns, threshold_quantile=0.95)
    success_hill = evt_analyzer.fit_tail_index_model(returns)
    
    print(f"EVT model fitting: GEV={success_gev}, GPD={success_gpd}, Hill={success_hill}")
    
    # Calculate comprehensive metrics
    metrics = evt_analyzer.calculate_comprehensive_metrics(returns)
    
    print(f"\nTail Risk Metrics:")
    print(f"  VaR(95%): {metrics.var_95:.4f}")
    print(f"  VaR(99%): {metrics.var_99:.4f}")
    print(f"  VaR(99.9%): {metrics.var_999:.4f}")
    print(f"  ES(95%): {metrics.es_95:.4f}")
    print(f"  ES(99%): {metrics.es_99:.4f}")
    print(f"  ES(99.9%): {metrics.es_999:.4f}")
    print(f"  Tail Index: {metrics.tail_index:.4f}")
    print(f"  10-year return level: {metrics.return_level_10y:.4f}")
    print(f"  100-year return level: {metrics.return_level_100y:.4f}")
    
    # Generate extreme scenarios
    print("\nGenerating extreme scenarios...")
    scenarios = evt_analyzer.generate_extreme_scenarios(n_scenarios=100, horizon=252)
    
    print(f"Generated {len(scenarios)} extreme scenarios")
    print(f"Scenario statistics: mean={np.mean(scenarios):.4f}, max={np.max(scenarios):.4f}")
    
    # Compare with empirical quantiles
    empirical_99 = np.quantile(returns, 0.01)  # Left tail
    empirical_999 = np.quantile(returns, 0.001)
    
    print(f"\nComparison with empirical quantiles:")
    print(f"  Empirical VaR(99%): {empirical_99:.4f} vs EVT: {metrics.var_99:.4f}")
    print(f"  Empirical VaR(99.9%): {empirical_999:.4f} vs EVT: {metrics.var_999:.4f}")
    
    print("\nExtreme Value Theory engine test completed successfully!")