"""
Drawdown Mathematics Engine for QUANTUM-FORGE
Implements advanced drawdown analysis, recovery modeling, and risk metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from numba import jit, prange
from dataclasses import dataclass
from enum import Enum
from collections import deque
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class DrawdownType(Enum):
    """Types of drawdown measures."""
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    LOG = "logarithmic"
    CALMAR = "calmar"
    STERLING = "sterling"

class RecoveryType(Enum):
    """Types of recovery models."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    POWER_LAW = "power_law"
    MEAN_REVERTING = "mean_reverting"

@dataclass
class DrawdownPeriod:
    """Individual drawdown period information."""
    start_date: int
    end_date: int
    recovery_date: Optional[int]
    peak_value: float
    trough_value: float
    recovery_value: float
    max_drawdown: float
    duration: int
    recovery_time: Optional[int]
    total_duration: Optional[int]

@dataclass
class DrawdownMetrics:
    """Comprehensive drawdown metrics."""
    max_drawdown: float
    avg_drawdown: float
    drawdown_frequency: float
    avg_drawdown_duration: float
    avg_recovery_time: float
    calmar_ratio: float
    sterling_ratio: float
    burke_ratio: float
    pain_index: float
    ulcer_index: float
    drawdown_deviation: float

@dataclass
class RecoveryMetrics:
    """Recovery analysis metrics."""
    recovery_rate: float
    recovery_factor: float
    recovery_probability: float
    expected_recovery_time: float
    recovery_efficiency: float
    drawdown_recovery_ratio: float

class DrawdownAnalyzer:
    """Advanced drawdown analysis and measurement."""
    
    def __init__(self, price_series: Optional[np.ndarray] = None):
        """
        Initialize drawdown analyzer.
        
        Args:
            price_series: Time series of prices/values
        """
        self.price_series = price_series
        self.drawdown_series = None
        self.drawdown_periods = []
        self.fitted = False
        
    def calculate_drawdowns(self, series: Optional[np.ndarray] = None, 
                          drawdown_type: DrawdownType = DrawdownType.RELATIVE) -> np.ndarray:
        """
        Calculate drawdown series.
        
        Args:
            series: Price/value series (uses stored series if None)
            drawdown_type: Type of drawdown calculation
        
        Returns:
            Drawdown series
        """
        if series is None:
            series = self.price_series
        
        if series is None:
            raise ValueError("No price series provided")
        
        series = np.asarray(series)
        
        if drawdown_type == DrawdownType.ABSOLUTE:
            # Absolute drawdown from running maximum
            running_max = np.maximum.accumulate(series)
            drawdowns = series - running_max
            
        elif drawdown_type == DrawdownType.RELATIVE:
            # Relative drawdown as percentage
            running_max = np.maximum.accumulate(series)
            drawdowns = (series - running_max) / running_max
            
        elif drawdown_type == DrawdownType.LOG:
            # Log drawdown
            log_series = np.log(series)
            running_max = np.maximum.accumulate(log_series)
            drawdowns = log_series - running_max
            
        else:
            # Default to relative
            running_max = np.maximum.accumulate(series)
            drawdowns = (series - running_max) / running_max
        
        self.drawdown_series = drawdowns
        return drawdowns
    
    def identify_drawdown_periods(self, series: Optional[np.ndarray] = None,
                                 min_duration: int = 1) -> List[DrawdownPeriod]:
        """
        Identify individual drawdown periods.
        
        Args:
            series: Price series
            min_duration: Minimum duration to consider a drawdown
        
        Returns:
            List of DrawdownPeriod objects
        """
        if series is None:
            series = self.price_series
        
        if series is None:
            raise ValueError("No price series provided")
        
        series = np.asarray(series)
        drawdowns = self.calculate_drawdowns(series)
        
        periods = []
        in_drawdown = False
        start_idx = 0
        peak_value = series[0]
        peak_idx = 0
        
        for i, (value, dd) in enumerate(zip(series, drawdowns)):
            if not in_drawdown and dd < 0:
                # Start of drawdown
                in_drawdown = True
                start_idx = peak_idx
                
            elif in_drawdown and dd >= 0:
                # End of drawdown (recovery to new high)
                if i - start_idx >= min_duration:
                    # Find trough in this drawdown period
                    dd_period = drawdowns[start_idx:i]
                    trough_idx = start_idx + np.argmin(dd_period)
                    trough_value = series[trough_idx]
                    max_dd = np.min(dd_period)
                    
                    # Find recovery point
                    recovery_idx = i if value >= peak_value else None
                    recovery_value = value if recovery_idx else None
                    
                    period = DrawdownPeriod(
                        start_date=start_idx,
                        end_date=trough_idx,
                        recovery_date=recovery_idx,
                        peak_value=peak_value,
                        trough_value=trough_value,
                        recovery_value=recovery_value,
                        max_drawdown=max_dd,
                        duration=trough_idx - start_idx,
                        recovery_time=recovery_idx - trough_idx if recovery_idx else None,
                        total_duration=recovery_idx - start_idx if recovery_idx else None
                    )
                    
                    periods.append(period)
                
                in_drawdown = False
            
            # Update peak
            if value > peak_value:
                peak_value = value
                peak_idx = i
        
        # Handle ongoing drawdown at the end
        if in_drawdown and len(series) - start_idx >= min_duration:
            dd_period = drawdowns[start_idx:]
            trough_idx = start_idx + np.argmin(dd_period)
            trough_value = series[trough_idx]
            max_dd = np.min(dd_period)
            
            period = DrawdownPeriod(
                start_date=start_idx,
                end_date=trough_idx,
                recovery_date=None,
                peak_value=peak_value,
                trough_value=trough_value,
                recovery_value=None,
                max_drawdown=max_dd,
                duration=trough_idx - start_idx,
                recovery_time=None,
                total_duration=None
            )
            
            periods.append(period)
        
        self.drawdown_periods = periods
        return periods
    
    def calculate_comprehensive_metrics(self, series: Optional[np.ndarray] = None,
                                      returns: Optional[np.ndarray] = None,
                                      risk_free_rate: float = 0.02) -> DrawdownMetrics:
        """
        Calculate comprehensive drawdown metrics.
        
        Args:
            series: Price series
            returns: Return series (calculated if not provided)
            risk_free_rate: Annual risk-free rate
        
        Returns:
            DrawdownMetrics object
        """
        if series is None:
            series = self.price_series
        
        if series is None:
            raise ValueError("No price series provided")
        
        series = np.asarray(series)
        
        if returns is None:
            returns = np.diff(series) / series[:-1]
        
        # Calculate drawdowns
        drawdowns = self.calculate_drawdowns(series)
        
        # Identify periods
        periods = self.identify_drawdown_periods(series)
        
        # Basic metrics
        max_drawdown = np.min(drawdowns)
        avg_drawdown = np.mean(drawdowns[drawdowns < 0]) if np.any(drawdowns < 0) else 0.0
        
        # Frequency metrics
        total_periods = len(series)
        drawdown_periods_count = len(periods)
        drawdown_frequency = drawdown_periods_count / total_periods if total_periods > 0 else 0.0
        
        # Duration metrics
        durations = [p.duration for p in periods if p.duration is not None]
        recovery_times = [p.recovery_time for p in periods if p.recovery_time is not None]
        
        avg_drawdown_duration = np.mean(durations) if durations else 0.0
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0.0
        
        # Performance ratios
        annual_return = np.mean(returns) * 252 if len(returns) > 0 else 0.0
        
        # Calmar ratio: Annual return / Max drawdown
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # Sterling ratio: Annual return / Average max drawdown
        avg_max_dd = np.mean([abs(p.max_drawdown) for p in periods]) if periods else abs(max_drawdown)
        sterling_ratio = annual_return / avg_max_dd if avg_max_dd != 0 else 0.0
        
        # Burke ratio: (Annual return - Risk free rate) / sqrt(sum of squared drawdowns)
        squared_dd_sum = np.sum(drawdowns[drawdowns < 0]**2) if np.any(drawdowns < 0) else 1e-8
        burke_ratio = (annual_return - risk_free_rate) / np.sqrt(squared_dd_sum)
        
        # Pain Index: Average drawdown over entire period
        pain_index = np.mean(np.abs(drawdowns))
        
        # Ulcer Index: sqrt(mean of squared drawdowns)
        ulcer_index = np.sqrt(np.mean(drawdowns**2))
        
        # Drawdown deviation: Standard deviation of drawdowns
        drawdown_deviation = np.std(drawdowns)
        
        return DrawdownMetrics(
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            drawdown_frequency=drawdown_frequency,
            avg_drawdown_duration=avg_drawdown_duration,
            avg_recovery_time=avg_recovery_time,
            calmar_ratio=calmar_ratio,
            sterling_ratio=sterling_ratio,
            burke_ratio=burke_ratio,
            pain_index=pain_index,
            ulcer_index=ulcer_index,
            drawdown_deviation=drawdown_deviation
        )

class RecoveryAnalyzer:
    """Recovery pattern analysis and modeling."""
    
    def __init__(self):
        """Initialize recovery analyzer."""
        self.recovery_models = {}
        self.fitted_models = set()
        
    def fit_recovery_model(self, drawdown_periods: List[DrawdownPeriod],
                         model_type: RecoveryType = RecoveryType.EXPONENTIAL) -> bool:
        """
        Fit recovery model to historical drawdown periods.
        
        Args:
            drawdown_periods: List of historical drawdown periods
            model_type: Type of recovery model to fit
        
        Returns:
            True if fitting successful
        """
        # Extract recovery data
        recovery_data = []
        
        for period in drawdown_periods:
            if (period.recovery_time is not None and 
                period.max_drawdown is not None and
                period.recovery_time > 0):
                
                recovery_data.append({
                    'max_drawdown': abs(period.max_drawdown),
                    'recovery_time': period.recovery_time,
                    'drawdown_duration': period.duration
                })
        
        if len(recovery_data) < 5:  # Need minimum data points
            return False
        
        recovery_df = pd.DataFrame(recovery_data)
        
        try:
            if model_type == RecoveryType.LINEAR:
                # Linear recovery model: recovery_time = a * max_drawdown + b
                X = recovery_df['max_drawdown'].values
                y = recovery_df['recovery_time'].values
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                
                self.recovery_models['linear'] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value
                }
                
            elif model_type == RecoveryType.EXPONENTIAL:
                # Exponential recovery model: recovery_time = a * exp(b * max_drawdown)
                X = recovery_df['max_drawdown'].values
                y = recovery_df['recovery_time'].values
                
                # Log-linear regression
                log_y = np.log(np.maximum(y, 1e-6))
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, log_y)
                
                self.recovery_models['exponential'] = {
                    'a': np.exp(intercept),
                    'b': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value
                }
                
            elif model_type == RecoveryType.POWER_LAW:
                # Power law recovery: recovery_time = a * max_drawdown^b
                X = np.log(recovery_df['max_drawdown'].values)
                y = np.log(recovery_df['recovery_time'].values)
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
                
                self.recovery_models['power_law'] = {
                    'a': np.exp(intercept),
                    'b': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value
                }
                
            elif model_type == RecoveryType.MEAN_REVERTING:
                # Mean-reverting model with Ornstein-Uhlenbeck process
                # Simplified approach using half-life estimation
                
                # Calculate recovery rates
                recovery_rates = []
                for _, row in recovery_df.iterrows():
                    if row['recovery_time'] > 0:
                        # Recovery rate as drawdown per unit time
                        rate = row['max_drawdown'] / row['recovery_time']
                        recovery_rates.append(rate)
                
                if recovery_rates:
                    mean_rate = np.mean(recovery_rates)
                    std_rate = np.std(recovery_rates)
                    
                    # Estimate half-life (simplified)
                    half_life = np.log(2) / mean_rate if mean_rate > 0 else np.inf
                    
                    self.recovery_models['mean_reverting'] = {
                        'mean_rate': mean_rate,
                        'std_rate': std_rate,
                        'half_life': half_life
                    }
            
            self.fitted_models.add(model_type.value)
            return True
            
        except Exception as e:
            return False
    
    def predict_recovery_time(self, max_drawdown: float, 
                            model_type: RecoveryType = RecoveryType.EXPONENTIAL) -> float:
        """
        Predict recovery time for given drawdown magnitude.
        
        Args:
            max_drawdown: Maximum drawdown magnitude (positive)
            model_type: Recovery model to use
        
        Returns:
            Predicted recovery time
        """
        model_name = model_type.value
        
        if model_name not in self.fitted_models:
            return 0.0
        
        model = self.recovery_models[model_name]
        
        try:
            if model_type == RecoveryType.LINEAR:
                return model['slope'] * max_drawdown + model['intercept']
                
            elif model_type == RecoveryType.EXPONENTIAL:
                return model['a'] * np.exp(model['b'] * max_drawdown)
                
            elif model_type == RecoveryType.POWER_LAW:
                return model['a'] * (max_drawdown ** model['b'])
                
            elif model_type == RecoveryType.MEAN_REVERTING:
                # Simple exponential recovery based on mean rate
                mean_rate = model['mean_rate']
                return max_drawdown / mean_rate if mean_rate > 0 else np.inf
                
        except Exception as e:
            return 0.0
        
        return 0.0
    
    def calculate_recovery_metrics(self, drawdown_periods: List[DrawdownPeriod]) -> RecoveryMetrics:
        """
        Calculate comprehensive recovery metrics.
        
        Args:
            drawdown_periods: Historical drawdown periods
        
        Returns:
            RecoveryMetrics object
        """
        # Extract recovery information
        completed_recoveries = [p for p in drawdown_periods if p.recovery_time is not None]
        ongoing_drawdowns = [p for p in drawdown_periods if p.recovery_time is None]
        
        if not completed_recoveries:
            return RecoveryMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Recovery rate: Average drawdown recovered per unit time
        recovery_rates = []
        for period in completed_recoveries:
            if period.recovery_time > 0:
                rate = abs(period.max_drawdown) / period.recovery_time
                recovery_rates.append(rate)
        
        avg_recovery_rate = np.mean(recovery_rates) if recovery_rates else 0.0
        
        # Recovery factor: Ratio of recovery value to peak value
        recovery_factors = []
        for period in completed_recoveries:
            if period.peak_value > 0 and period.recovery_value is not None:
                factor = period.recovery_value / period.peak_value
                recovery_factors.append(factor)
        
        avg_recovery_factor = np.mean(recovery_factors) if recovery_factors else 0.0
        
        # Recovery probability: Fraction of drawdowns that recovered
        total_drawdowns = len(drawdown_periods)
        recovered_drawdowns = len(completed_recoveries)
        recovery_probability = recovered_drawdowns / total_drawdowns if total_drawdowns > 0 else 0.0
        
        # Expected recovery time
        recovery_times = [p.recovery_time for p in completed_recoveries]
        expected_recovery_time = np.mean(recovery_times) if recovery_times else 0.0
        
        # Recovery efficiency: Average drawdown magnitude / Average recovery time
        drawdown_magnitudes = [abs(p.max_drawdown) for p in completed_recoveries]
        avg_drawdown_magnitude = np.mean(drawdown_magnitudes) if drawdown_magnitudes else 0.0
        
        recovery_efficiency = (avg_drawdown_magnitude / expected_recovery_time 
                             if expected_recovery_time > 0 else 0.0)
        
        # Drawdown-recovery ratio: Average drawdown duration / Average recovery time
        drawdown_durations = [p.duration for p in completed_recoveries]
        avg_drawdown_duration = np.mean(drawdown_durations) if drawdown_durations else 0.0
        
        drawdown_recovery_ratio = (avg_drawdown_duration / expected_recovery_time
                                 if expected_recovery_time > 0 else 0.0)
        
        return RecoveryMetrics(
            recovery_rate=avg_recovery_rate,
            recovery_factor=avg_recovery_factor,
            recovery_probability=recovery_probability,
            expected_recovery_time=expected_recovery_time,
            recovery_efficiency=recovery_efficiency,
            drawdown_recovery_ratio=drawdown_recovery_ratio
        )

class DrawdownRiskModel:
    """Advanced drawdown risk modeling and simulation."""
    
    def __init__(self):
        """Initialize drawdown risk model."""
        self.drawdown_distribution = None
        self.duration_distribution = None
        self.fitted = False
        
    def fit_drawdown_distributions(self, drawdown_periods: List[DrawdownPeriod]) -> bool:
        """
        Fit statistical distributions to drawdown characteristics.
        
        Args:
            drawdown_periods: Historical drawdown periods
        
        Returns:
            True if fitting successful
        """
        if len(drawdown_periods) < 10:
            return False
        
        try:
            # Extract drawdown magnitudes (positive values)
            magnitudes = [abs(p.max_drawdown) for p in drawdown_periods if p.max_drawdown is not None]
            durations = [p.duration for p in drawdown_periods if p.duration is not None]
            
            if len(magnitudes) < 5 or len(durations) < 5:
                return False
            
            # Fit distributions to drawdown magnitudes
            # Try multiple distributions
            distributions = ['expon', 'gamma', 'lognorm', 'beta']
            best_magnitude_dist = None
            best_magnitude_params = None
            best_magnitude_aic = np.inf
            
            for dist_name in distributions:
                try:
                    dist = getattr(stats, dist_name)
                    
                    if dist_name == 'beta':
                        # Beta distribution needs values in [0,1]
                        scaled_magnitudes = np.array(magnitudes) / max(magnitudes)
                        params = dist.fit(scaled_magnitudes)
                    else:
                        params = dist.fit(magnitudes)
                    
                    # Calculate AIC
                    if dist_name == 'beta':
                        log_likelihood = np.sum(dist.logpdf(scaled_magnitudes, *params))
                    else:
                        log_likelihood = np.sum(dist.logpdf(magnitudes, *params))
                    
                    aic = -2 * log_likelihood + 2 * len(params)
                    
                    if aic < best_magnitude_aic:
                        best_magnitude_aic = aic
                        best_magnitude_dist = dist_name
                        best_magnitude_params = params
                        
                except:
                    continue
            
            # Fit distributions to durations
            best_duration_dist = None
            best_duration_params = None
            best_duration_aic = np.inf
            
            duration_distributions = ['expon', 'gamma', 'poisson']
            
            for dist_name in duration_distributions:
                try:
                    dist = getattr(stats, dist_name)
                    
                    if dist_name == 'poisson':
                        # Poisson for discrete durations
                        mu = np.mean(durations)
                        params = (mu,)
                        log_likelihood = np.sum(dist.logpmf(durations, mu))
                    else:
                        params = dist.fit(durations)
                        log_likelihood = np.sum(dist.logpdf(durations, *params))
                    
                    aic = -2 * log_likelihood + 2 * len(params)
                    
                    if aic < best_duration_aic:
                        best_duration_aic = aic
                        best_duration_dist = dist_name
                        best_duration_params = params
                        
                except:
                    continue
            
            # Store best fitting distributions
            if best_magnitude_dist and best_duration_dist:
                self.drawdown_distribution = {
                    'name': best_magnitude_dist,
                    'params': best_magnitude_params,
                    'aic': best_magnitude_aic
                }
                
                self.duration_distribution = {
                    'name': best_duration_dist,
                    'params': best_duration_params,
                    'aic': best_duration_aic
                }
                
                self.fitted = True
                return True
                
        except Exception as e:
            return False
        
        return False
    
    def simulate_drawdown_scenarios(self, n_scenarios: int = 1000, 
                                  time_horizon: int = 252) -> List[Dict]:
        """
        Simulate future drawdown scenarios.
        
        Args:
            n_scenarios: Number of scenarios to simulate
            time_horizon: Time horizon for simulation
        
        Returns:
            List of scenario dictionaries
        """
        if not self.fitted:
            return []
        
        scenarios = []
        
        for _ in range(n_scenarios):
            scenario = {
                'drawdowns': [],
                'max_drawdown': 0.0,
                'total_drawdown_days': 0,
                'number_of_drawdowns': 0
            }
            
            current_time = 0
            
            while current_time < time_horizon:
                # Deterministic time to next drawdown: use mean inter-arrival (30 days)
                time_to_next = 30.0
                current_time += time_to_next
                
                if current_time >= time_horizon:
                    break
                
                # Sample drawdown magnitude
                dd_dist = getattr(stats, self.drawdown_distribution['name'])
                dd_params = self.drawdown_distribution['params']
                
                # Deterministic drawdown magnitude: use distribution mean or median
                try:
                    magnitude = float(dd_dist.mean(*dd_params))
                except Exception:
                    try:
                        magnitude = float(dd_dist.ppf(0.5, *dd_params))
                    except Exception:
                        magnitude = 0.1
                if self.drawdown_distribution['name'] == 'beta':
                    magnitude = min(magnitude * 0.5, 0.8)
                
                magnitude = min(magnitude, 0.8)  # Cap at 80%
                
                # Sample duration
                dur_dist = getattr(stats, self.duration_distribution['name'])
                dur_params = self.duration_distribution['params']
                
                # Deterministic duration: use mean or median of distribution
                try:
                    dur_val = dur_dist.mean(*dur_params)
                except Exception:
                    try:
                        dur_val = dur_dist.ppf(0.5, *dur_params)
                    except Exception:
                        dur_val = 10
                duration = int(max(1, min(int(dur_val), time_horizon - current_time)))
                
                duration = max(1, min(duration, time_horizon - current_time))
                
                # Record drawdown
                drawdown_info = {
                    'start_time': current_time,
                    'magnitude': magnitude,
                    'duration': duration
                }
                
                scenario['drawdowns'].append(drawdown_info)
                scenario['max_drawdown'] = max(scenario['max_drawdown'], magnitude)
                scenario['total_drawdown_days'] += duration
                scenario['number_of_drawdowns'] += 1
                
                current_time += duration
            
            scenarios.append(scenario)
        
        return scenarios
    
    def calculate_var_drawdown(self, scenarios: List[Dict], 
                             confidence_level: float = 0.95) -> Dict:
        """
        Calculate Value-at-Risk for drawdown metrics.
        
        Args:
            scenarios: Simulated scenarios
            confidence_level: VaR confidence level
        
        Returns:
            Dictionary of VaR metrics
        """
        if not scenarios:
            return {}
        
        # Extract metrics from scenarios
        max_drawdowns = [s['max_drawdown'] for s in scenarios]
        total_dd_days = [s['total_drawdown_days'] for s in scenarios]
        num_drawdowns = [s['number_of_drawdowns'] for s in scenarios]
        
        # Calculate VaR and Expected Shortfall
        var_level = 1 - confidence_level
        
        max_dd_var = np.quantile(max_drawdowns, 1 - var_level)
        max_dd_es = np.mean([dd for dd in max_drawdowns if dd >= max_dd_var])
        
        dd_days_var = np.quantile(total_dd_days, 1 - var_level)
        dd_days_es = np.mean([days for days in total_dd_days if days >= dd_days_var])
        
        num_dd_var = np.quantile(num_drawdowns, 1 - var_level)
        num_dd_es = np.mean([num for num in num_drawdowns if num >= num_dd_var])
        
        return {
            'max_drawdown_var': max_dd_var,
            'max_drawdown_es': max_dd_es,
            'drawdown_days_var': dd_days_var,
            'drawdown_days_es': dd_days_es,
            'number_drawdowns_var': num_dd_var,
            'number_drawdowns_es': num_dd_es,
            'confidence_level': confidence_level
        }

class DrawdownOptimizer:
    """Portfolio optimization considering drawdown constraints."""
    
    def __init__(self, max_drawdown_limit: float = 0.15):
        """
        Initialize drawdown optimizer.
        
        Args:
            max_drawdown_limit: Maximum allowed drawdown
        """
        self.max_drawdown_limit = max_drawdown_limit
        self.optimal_weights = None
        self.optimization_results = {}
        
    def optimize_portfolio(self, expected_returns: np.ndarray, 
                         covariance_matrix: np.ndarray,
                         drawdown_estimates: np.ndarray,
                         target_return: Optional[float] = None) -> Dict:
        """
        Optimize portfolio weights considering drawdown constraints.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            drawdown_estimates: Expected drawdown for each asset
            target_return: Target portfolio return (optional)
        
        Returns:
            Dictionary with optimization results
        """
        n_assets = len(expected_returns)
        
        # Define objective function (minimize portfolio variance)
        def objective(weights):
            portfolio_variance = weights.T @ covariance_matrix @ weights
            return portfolio_variance
        
        # Constraints
        constraints = []
        
        # Weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        })
        
        # Drawdown constraint (simplified linear approximation)
        # Portfolio drawdown ≈ weighted average of individual drawdowns
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: self.max_drawdown_limit - np.dot(w, drawdown_estimates)
        })
        
        # Target return constraint (if specified)
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, expected_returns) - target_return
            })
        
        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Optimize
        try:
            result = optimize.minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'ftol': 1e-9, 'disp': False}
            )
            
            if result.success:
                self.optimal_weights = result.x
                
                # Calculate portfolio metrics
                portfolio_return = np.dot(self.optimal_weights, expected_returns)
                portfolio_variance = self.optimal_weights.T @ covariance_matrix @ self.optimal_weights
                portfolio_volatility = np.sqrt(portfolio_variance)
                portfolio_drawdown = np.dot(self.optimal_weights, drawdown_estimates)
                
                self.optimization_results = {
                    'success': True,
                    'weights': self.optimal_weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'expected_drawdown': portfolio_drawdown,
                    'sharpe_ratio': portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0,
                    'calmar_ratio': portfolio_return / portfolio_drawdown if portfolio_drawdown > 0 else 0,
                    'optimization_message': result.message
                }
                
                return self.optimization_results
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Optimization failed'}

# Example usage and testing
if __name__ == "__main__":
    print("Testing Drawdown Mathematics Engine...")
    
    # Generate synthetic price series deterministically with regime cycles
    n_observations = 1000

    # Deterministic regime cycle lengths and sequence
    regime_cycle = [0, 2, 0, 1]  # pattern of regimes
    cycle_lengths = {0: 60, 1: 40, 2: 30}  # days per regime type

    returns = np.zeros(n_observations)
    idx = 0
    cycle_pos = 0
    while idx < n_observations:
        regime = regime_cycle[cycle_pos % len(regime_cycle)]
        length = cycle_lengths.get(regime, 30)
        for j in range(length):
            if idx >= n_observations:
                break
            # Deterministic daily return: mean + deterministic sinusoidal perturbation
            if regime == 0:
                mu, vol = 0.0008, 0.015
            elif regime == 1:
                mu, vol = -0.0012, 0.025
            else:
                mu, vol = 0.0001, 0.012

            # Use sinusoidal deterministic perturbation to emulate variability
            phase = float(idx) / n_observations * 2 * np.pi
            perturb = vol * 0.5 * np.sin(5 * phase + 0.3 * j)
            daily_return = mu + perturb
            returns[idx] = daily_return
            idx += 1
        cycle_pos += 1
    
    returns = np.array(returns)
    
    # Convert to price series
    prices = np.zeros(n_observations + 1)
    prices[0] = 100
    
    for i in range(n_observations):
        prices[i + 1] = prices[i] * (1 + returns[i])
    
    print(f"Generated {n_observations} price observations")
    print(f"Price range: {np.min(prices):.2f} - {np.max(prices):.2f}")
    print(f"Total return: {(prices[-1] / prices[0] - 1) * 100:.2f}%")
    
    # Test DrawdownAnalyzer
    print("\nTesting DrawdownAnalyzer...")
    
    dd_analyzer = DrawdownAnalyzer(prices)
    
    # Calculate drawdowns
    drawdowns = dd_analyzer.calculate_drawdowns()
    print(f"Maximum drawdown: {np.min(drawdowns) * 100:.2f}%")
    print(f"Number of negative drawdown observations: {np.sum(drawdowns < 0)}")
    
    # Identify drawdown periods
    periods = dd_analyzer.identify_drawdown_periods()
    print(f"Number of drawdown periods identified: {len(periods)}")
    
    if periods:
        max_period = max(periods, key=lambda p: abs(p.max_drawdown))
        print(f"Largest drawdown: {max_period.max_drawdown * 100:.2f}% over {max_period.duration} days")
        
        if max_period.recovery_time:
            print(f"Recovery time: {max_period.recovery_time} days")
    
    # Calculate comprehensive metrics
    metrics = dd_analyzer.calculate_comprehensive_metrics(returns=returns)
    
    print(f"\nDrawdown Metrics:")
    print(f"  Max Drawdown: {metrics.max_drawdown * 100:.2f}%")
    print(f"  Average Drawdown: {metrics.avg_drawdown * 100:.2f}%")
    print(f"  Drawdown Frequency: {metrics.drawdown_frequency:.4f}")
    print(f"  Avg Drawdown Duration: {metrics.avg_drawdown_duration:.1f} days")
    print(f"  Avg Recovery Time: {metrics.avg_recovery_time:.1f} days")
    print(f"  Calmar Ratio: {metrics.calmar_ratio:.3f}")
    print(f"  Sterling Ratio: {metrics.sterling_ratio:.3f}")
    print(f"  Burke Ratio: {metrics.burke_ratio:.3f}")
    print(f"  Pain Index: {metrics.pain_index:.4f}")
    print(f"  Ulcer Index: {metrics.ulcer_index:.4f}")
    
    # Test RecoveryAnalyzer
    print("\nTesting RecoveryAnalyzer...")
    
    recovery_analyzer = RecoveryAnalyzer()
    
    # Fit recovery models
    success_exp = recovery_analyzer.fit_recovery_model(periods, RecoveryType.EXPONENTIAL)
    success_linear = recovery_analyzer.fit_recovery_model(periods, RecoveryType.LINEAR)
    
    print(f"Exponential recovery model fitting: {'successful' if success_exp else 'failed'}")
    print(f"Linear recovery model fitting: {'successful' if success_linear else 'failed'}")
    
    if success_exp:
        # Test prediction
        test_drawdown = 0.10  # 10% drawdown
        predicted_recovery = recovery_analyzer.predict_recovery_time(test_drawdown, RecoveryType.EXPONENTIAL)
        print(f"Predicted recovery time for 10% drawdown: {predicted_recovery:.1f} days")
    
    # Calculate recovery metrics
    recovery_metrics = recovery_analyzer.calculate_recovery_metrics(periods)
    
    print(f"\nRecovery Metrics:")
    print(f"  Recovery Rate: {recovery_metrics.recovery_rate:.6f}")
    print(f"  Recovery Factor: {recovery_metrics.recovery_factor:.3f}")
    print(f"  Recovery Probability: {recovery_metrics.recovery_probability:.3f}")
    print(f"  Expected Recovery Time: {recovery_metrics.expected_recovery_time:.1f} days")
    print(f"  Recovery Efficiency: {recovery_metrics.recovery_efficiency:.6f}")
    print(f"  Drawdown-Recovery Ratio: {recovery_metrics.drawdown_recovery_ratio:.3f}")
    
    # Test DrawdownRiskModel
    print("\nTesting DrawdownRiskModel...")
    
    risk_model = DrawdownRiskModel()
    
    # Fit distributions
    dist_success = risk_model.fit_drawdown_distributions(periods)
    print(f"Distribution fitting: {'successful' if dist_success else 'failed'}")
    
    if dist_success:
        print(f"Best drawdown distribution: {risk_model.drawdown_distribution['name']}")
        print(f"Best duration distribution: {risk_model.duration_distribution['name']}")
        
        # Simulate scenarios
        scenarios = risk_model.simulate_drawdown_scenarios(n_scenarios=1000, time_horizon=252)
        print(f"Generated {len(scenarios)} drawdown scenarios")
        
        # Calculate VaR
        var_results = risk_model.calculate_var_drawdown(scenarios, confidence_level=0.95)
        
        print(f"\nDrawdown VaR (95% confidence):")
        print(f"  Max Drawdown VaR: {var_results['max_drawdown_var'] * 100:.2f}%")
        print(f"  Max Drawdown ES: {var_results['max_drawdown_es'] * 100:.2f}%")
        print(f"  Drawdown Days VaR: {var_results['drawdown_days_var']:.0f} days")
        print(f"  Number of Drawdowns VaR: {var_results['number_drawdowns_var']:.0f}")
    
    # Test DrawdownOptimizer
    print("\nTesting DrawdownOptimizer...")
    
    # Create sample portfolio optimization problem
    n_assets = 5
    expected_returns = np.array([0.08, 0.10, 0.12, 0.06, 0.09])  # Annual returns
    
    # Sample covariance matrix
    correlations = np.array([
        [1.0, 0.3, 0.2, 0.1, 0.4],
        [0.3, 1.0, 0.5, 0.2, 0.3],
        [0.2, 0.5, 1.0, 0.1, 0.2],
        [0.1, 0.2, 0.1, 1.0, 0.1],
        [0.4, 0.3, 0.2, 0.1, 1.0]
    ])
    
    volatilities = np.array([0.15, 0.20, 0.25, 0.10, 0.18])
    covariance_matrix = np.outer(volatilities, volatilities) * correlations
    
    # Estimated drawdowns for each asset
    drawdown_estimates = np.array([0.12, 0.18, 0.22, 0.08, 0.15])
    
    optimizer = DrawdownOptimizer(max_drawdown_limit=0.15)
    
    # Optimize portfolio
    opt_results = optimizer.optimize_portfolio(
        expected_returns,
        covariance_matrix,
        drawdown_estimates,
        target_return=0.09
    )
    
    if opt_results.get('success', False):
        print("Portfolio optimization successful!")
        print(f"Optimal weights: {np.round(opt_results['weights'], 3)}")
        print(f"Expected return: {opt_results['expected_return'] * 100:.2f}%")
        print(f"Volatility: {opt_results['volatility'] * 100:.2f}%")
        print(f"Expected drawdown: {opt_results['expected_drawdown'] * 100:.2f}%")
        print(f"Sharpe ratio: {opt_results['sharpe_ratio']:.3f}")
        print(f"Calmar ratio: {opt_results['calmar_ratio']:.3f}")
    else:
        print(f"Portfolio optimization failed: {opt_results.get('error', 'Unknown error')}")
    
    print("\nDrawdown mathematics engine test completed successfully!")