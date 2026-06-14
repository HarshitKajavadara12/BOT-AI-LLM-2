"""
Performance Analytics Engine for QUANTUM-FORGE
Implements comprehensive performance measurement, attribution analysis, and risk metrics
for quantitative trading strategies with advanced statistical methods.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import pickle
import json
warnings.filterwarnings('ignore')

class PerformanceMetricType(Enum):
    """Types of performance metrics."""
    RETURN_BASED = "return_based"
    RISK_ADJUSTED = "risk_adjusted"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    ATTRIBUTION = "attribution"
    FACTOR_EXPOSURE = "factor_exposure"
    BENCHMARK_RELATIVE = "benchmark_relative"

class AnalyticsFrequency(Enum):
    """Analysis frequency types."""
    TICK = "tick"
    SECOND = "second"
    MINUTE = "minute"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"

@dataclass
class PerformanceResult:
    """Performance analysis result."""
    metric_name: str
    value: float
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AttributionResult:
    """Performance attribution result."""
    factor_name: str
    contribution: float
    exposure: float
    factor_return: float
    active_weight: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class PerformanceAnalytics:
    """Comprehensive performance analytics engine."""
    
    def __init__(self):
        """Initialize performance analytics."""
        self.analytics_cache = {}
        self.benchmark_data = {}
        self.factor_data = {}
        
    def compute_returns(self, prices: Union[pd.Series, np.ndarray], 
                       method: str = "simple") -> Union[pd.Series, np.ndarray]:
        """Compute returns from price series."""
        
        if isinstance(prices, pd.Series):
            if method == "simple":
                returns = prices.pct_change().dropna()
            elif method == "log":
                returns = np.log(prices / prices.shift(1)).dropna()
            else:
                raise ValueError(f"Unknown return method: {method}")
        else:  # numpy array
            if method == "simple":
                returns = np.diff(prices) / prices[:-1]
            elif method == "log":
                returns = np.diff(np.log(prices))
            else:
                raise ValueError(f"Unknown return method: {method}")
        
        return returns
    
    def total_return(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate total return."""
        if isinstance(returns, pd.Series):
            return (1 + returns).prod() - 1
        else:
            return np.prod(1 + returns) - 1
    
    def annualized_return(self, returns: Union[pd.Series, np.ndarray], 
                         periods_per_year: int = 252) -> float:
        """Calculate annualized return."""
        total_ret = self.total_return(returns)
        n_periods = len(returns)
        
        if n_periods == 0:
            return 0.0
        
        return (1 + total_ret) ** (periods_per_year / n_periods) - 1
    
    def volatility(self, returns: Union[pd.Series, np.ndarray], 
                  periods_per_year: int = 252) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0
        
        return np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    
    def sharpe_ratio(self, returns: Union[pd.Series, np.ndarray], 
                    risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        
        if np.std(excess_returns, ddof=1) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(periods_per_year)
    
    def sortino_ratio(self, returns: Union[pd.Series, np.ndarray], 
                     risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)
    
    def calmar_ratio(self, returns: Union[pd.Series, np.ndarray], 
                    periods_per_year: int = 252) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        annual_return = self.annualized_return(returns, periods_per_year)
        max_dd = self.max_drawdown(returns)
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0.0
        
        return annual_return / abs(max_dd)
    
    def information_ratio(self, returns: Union[pd.Series, np.ndarray], 
                         benchmark_returns: Union[pd.Series, np.ndarray], 
                         periods_per_year: int = 252) -> float:
        """Calculate information ratio."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        if isinstance(returns, pd.Series) and isinstance(benchmark_returns, pd.Series):
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        else:
            min_len = min(len(returns), len(benchmark_returns))
            aligned_returns = returns[:min_len]
            aligned_benchmark = benchmark_returns[:min_len]
        
        active_returns = aligned_returns - aligned_benchmark
        
        if len(active_returns) == 0 or np.std(active_returns, ddof=1) == 0:
            return 0.0
        
        return np.mean(active_returns) / np.std(active_returns, ddof=1) * np.sqrt(periods_per_year)
    
    def max_drawdown(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod() if isinstance(returns, pd.Series) else np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return np.min(drawdown)
    
    def drawdown_duration(self, returns: Union[pd.Series, np.ndarray]) -> Dict[str, int]:
        """Calculate drawdown duration statistics."""
        if len(returns) == 0:
            return {'max_duration': 0, 'current_duration': 0}
        
        cumulative = (1 + returns).cumprod() if isinstance(returns, pd.Series) else np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        
        # Find drawdown periods
        in_drawdown = cumulative < running_max
        
        # Calculate durations
        durations = []
        current_duration = 0
        max_duration = 0
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        # Check if we're currently in drawdown
        if current_duration > 0:
            durations.append(current_duration)
        
        return {
            'max_duration': max_duration,
            'current_duration': current_duration,
            'avg_duration': np.mean(durations) if durations else 0,
            'num_drawdowns': len(durations)
        }
    
    def value_at_risk(self, returns: Union[pd.Series, np.ndarray], 
                     confidence: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)."""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, confidence * 100)
    
    def conditional_value_at_risk(self, returns: Union[pd.Series, np.ndarray], 
                                 confidence: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0
        
        var = self.value_at_risk(returns, confidence)
        return np.mean(returns[returns <= var])
    
    def beta(self, returns: Union[pd.Series, np.ndarray], 
            market_returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate beta relative to market."""
        if len(returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        # Align returns
        if isinstance(returns, pd.Series) and isinstance(market_returns, pd.Series):
            aligned_returns, aligned_market = returns.align(market_returns, join='inner')
        else:
            min_len = min(len(returns), len(market_returns))
            aligned_returns = returns[:min_len]
            aligned_market = market_returns[:min_len]
        
        if len(aligned_returns) == 0 or np.var(aligned_market, ddof=1) == 0:
            return 0.0
        
        return np.cov(aligned_returns, aligned_market, ddof=1)[0, 1] / np.var(aligned_market, ddof=1)
    
    def alpha(self, returns: Union[pd.Series, np.ndarray], 
             market_returns: Union[pd.Series, np.ndarray], 
             risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Calculate Jensen's alpha."""
        if len(returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        # Align returns
        if isinstance(returns, pd.Series) and isinstance(market_returns, pd.Series):
            aligned_returns, aligned_market = returns.align(market_returns, join='inner')
        else:
            min_len = min(len(returns), len(market_returns))
            aligned_returns = returns[:min_len]
            aligned_market = market_returns[:min_len]
        
        if len(aligned_returns) == 0:
            return 0.0
        
        beta_val = self.beta(returns, market_returns)
        
        portfolio_return = np.mean(aligned_returns) * periods_per_year
        market_return = np.mean(aligned_market) * periods_per_year
        
        return portfolio_return - (risk_free_rate + beta_val * (market_return - risk_free_rate))
    
    def tracking_error(self, returns: Union[pd.Series, np.ndarray], 
                      benchmark_returns: Union[pd.Series, np.ndarray], 
                      periods_per_year: int = 252) -> float:
        """Calculate tracking error."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0
        
        # Align returns
        if isinstance(returns, pd.Series) and isinstance(benchmark_returns, pd.Series):
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        else:
            min_len = min(len(returns), len(benchmark_returns))
            aligned_returns = returns[:min_len]
            aligned_benchmark = benchmark_returns[:min_len]
        
        active_returns = aligned_returns - aligned_benchmark
        
        if len(active_returns) == 0:
            return 0.0
        
        return np.std(active_returns, ddof=1) * np.sqrt(periods_per_year)
    
    def omega_ratio(self, returns: Union[pd.Series, np.ndarray], 
                   threshold: float = 0.0) -> float:
        """Calculate Omega ratio."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - threshold
        positive_returns = excess_returns[excess_returns > 0]
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf if len(positive_returns) > 0 else 1.0
        
        if len(positive_returns) == 0:
            return 0.0
        
        return np.sum(positive_returns) / abs(np.sum(negative_returns))
    
    def kappa_ratio(self, returns: Union[pd.Series, np.ndarray], 
                   n: int = 3, threshold: float = 0.0) -> float:
        """Calculate Kappa ratio (generalized Sortino ratio)."""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - threshold
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0
        
        # Lower partial moment
        lpm = np.mean(np.abs(downside_returns) ** n) ** (1/n)
        
        if lpm == 0:
            return 0.0
        
        return np.mean(excess_returns) / lpm
    
    def sterling_ratio(self, returns: Union[pd.Series, np.ndarray], 
                      periods_per_year: int = 252) -> float:
        """Calculate Sterling ratio."""
        if len(returns) == 0:
            return 0.0
        
        annual_return = self.annualized_return(returns, periods_per_year)
        
        # Calculate average drawdown (excluding zero drawdowns)
        cumulative = (1 + returns).cumprod() if isinstance(returns, pd.Series) else np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        
        non_zero_drawdowns = drawdowns[drawdowns < 0]
        
        if len(non_zero_drawdowns) == 0:
            return np.inf if annual_return > 0 else 0.0
        
        avg_drawdown = abs(np.mean(non_zero_drawdowns))
        
        if avg_drawdown == 0:
            return 0.0
        
        return annual_return / avg_drawdown
    
    def burke_ratio(self, returns: Union[pd.Series, np.ndarray], 
                   periods_per_year: int = 252) -> float:
        """Calculate Burke ratio."""
        if len(returns) == 0:
            return 0.0
        
        annual_return = self.annualized_return(returns, periods_per_year)
        
        # Calculate drawdown magnitude
        cumulative = (1 + returns).cumprod() if isinstance(returns, pd.Series) else np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        
        # Sum of squared drawdowns
        drawdown_magnitude = np.sqrt(np.sum(drawdowns ** 2))
        
        if drawdown_magnitude == 0:
            return np.inf if annual_return > 0 else 0.0
        
        return annual_return / drawdown_magnitude
    
    def pain_index(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate Pain Index (average drawdown)."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod() if isinstance(returns, pd.Series) else np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        
        return abs(np.mean(drawdowns))
    
    def ulcer_index(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate Ulcer Index."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod() if isinstance(returns, pd.Series) else np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        
        return np.sqrt(np.mean(drawdowns ** 2))
    
    def tail_ratio(self, returns: Union[pd.Series, np.ndarray], 
                  percentile: float = 0.05) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        if len(returns) == 0:
            return 0.0
        
        upper_tail = np.percentile(returns, (1 - percentile) * 100)
        lower_tail = np.percentile(returns, percentile * 100)
        
        if lower_tail >= 0:
            return np.inf
        
        return abs(upper_tail / lower_tail)
    
    def gain_to_pain_ratio(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate Gain-to-Pain ratio."""
        if len(returns) == 0:
            return 0.0
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return np.inf if len(positive_returns) > 0 else 0.0
        
        if len(positive_returns) == 0:
            return 0.0
        
        gain = np.sum(positive_returns)
        pain = abs(np.sum(negative_returns))
        
        return gain / pain if pain > 0 else np.inf
    
    def common_sense_ratio(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate Common Sense Ratio."""
        if len(returns) == 0:
            return 0.0
        
        tail_ratio_val = self.tail_ratio(returns)
        gain_to_pain_val = self.gain_to_pain_ratio(returns)
        
        if tail_ratio_val == 0:
            return 0.0
        
        return gain_to_pain_val * tail_ratio_val
    
    def compute_performance_metrics(self, returns: Union[pd.Series, np.ndarray], 
                                  benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
                                  risk_free_rate: float = 0.0, 
                                  periods_per_year: int = 252) -> Dict[str, PerformanceResult]:
        """Compute comprehensive performance metrics."""
        
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = PerformanceResult(
            'total_return', self.total_return(returns)
        )
        
        metrics['annualized_return'] = PerformanceResult(
            'annualized_return', self.annualized_return(returns, periods_per_year)
        )
        
        metrics['volatility'] = PerformanceResult(
            'volatility', self.volatility(returns, periods_per_year)
        )
        
        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = PerformanceResult(
            'sharpe_ratio', self.sharpe_ratio(returns, risk_free_rate, periods_per_year)
        )
        
        metrics['sortino_ratio'] = PerformanceResult(
            'sortino_ratio', self.sortino_ratio(returns, risk_free_rate, periods_per_year)
        )
        
        metrics['calmar_ratio'] = PerformanceResult(
            'calmar_ratio', self.calmar_ratio(returns, periods_per_year)
        )
        
        # Drawdown metrics
        metrics['max_drawdown'] = PerformanceResult(
            'max_drawdown', self.max_drawdown(returns)
        )
        
        dd_duration = self.drawdown_duration(returns)
        metrics['max_drawdown_duration'] = PerformanceResult(
            'max_drawdown_duration', dd_duration['max_duration']
        )
        
        # Risk metrics
        metrics['value_at_risk_5'] = PerformanceResult(
            'value_at_risk_5', self.value_at_risk(returns, 0.05)
        )
        
        metrics['conditional_var_5'] = PerformanceResult(
            'conditional_var_5', self.conditional_value_at_risk(returns, 0.05)
        )
        
        # Advanced ratios
        metrics['omega_ratio'] = PerformanceResult(
            'omega_ratio', self.omega_ratio(returns)
        )
        
        metrics['sterling_ratio'] = PerformanceResult(
            'sterling_ratio', self.sterling_ratio(returns, periods_per_year)
        )
        
        metrics['pain_index'] = PerformanceResult(
            'pain_index', self.pain_index(returns)
        )
        
        metrics['ulcer_index'] = PerformanceResult(
            'ulcer_index', self.ulcer_index(returns)
        )
        
        metrics['tail_ratio'] = PerformanceResult(
            'tail_ratio', self.tail_ratio(returns)
        )
        
        # Benchmark-relative metrics
        if benchmark_returns is not None:
            metrics['beta'] = PerformanceResult(
                'beta', self.beta(returns, benchmark_returns)
            )
            
            metrics['alpha'] = PerformanceResult(
                'alpha', self.alpha(returns, benchmark_returns, risk_free_rate, periods_per_year)
            )
            
            metrics['information_ratio'] = PerformanceResult(
                'information_ratio', self.information_ratio(returns, benchmark_returns, periods_per_year)
            )
            
            metrics['tracking_error'] = PerformanceResult(
                'tracking_error', self.tracking_error(returns, benchmark_returns, periods_per_year)
            )
        
        return metrics
    
    def rolling_metrics(self, returns: Union[pd.Series, np.ndarray], 
                       window: int, metric: str = 'sharpe_ratio',
                       **kwargs) -> Union[pd.Series, np.ndarray]:
        """Calculate rolling performance metrics."""
        
        if len(returns) < window:
            return pd.Series([], dtype=float) if isinstance(returns, pd.Series) else np.array([])
        
        metric_func = getattr(self, metric)
        rolling_values = []
        
        for i in range(window, len(returns) + 1):
            window_returns = returns[i-window:i]
            value = metric_func(window_returns, **kwargs)
            rolling_values.append(value)
        
        if isinstance(returns, pd.Series):
            return pd.Series(rolling_values, index=returns.index[window-1:])
        else:
            return np.array(rolling_values)
    
    def performance_attribution(self, portfolio_returns: Union[pd.Series, np.ndarray],
                              factor_returns: pd.DataFrame,
                              factor_loadings: Optional[Union[pd.Series, np.ndarray]] = None) -> List[AttributionResult]:
        """Perform factor-based performance attribution."""
        
        if isinstance(portfolio_returns, np.ndarray):
            portfolio_returns = pd.Series(portfolio_returns)
        
        # Align data
        aligned_data = pd.concat([portfolio_returns, factor_returns], axis=1, join='inner')
        portfolio_ret = aligned_data.iloc[:, 0]
        factor_ret = aligned_data.iloc[:, 1:]
        
        if factor_loadings is None:
            # Estimate factor loadings using regression
            from sklearn.linear_model import LinearRegression
            
            reg = LinearRegression(fit_intercept=True)
            reg.fit(factor_ret.values, portfolio_ret.values)
            
            factor_loadings = reg.coef_
            alpha = reg.intercept_
        else:
            alpha = 0.0
        
        # Calculate attribution
        attribution_results = []
        
        for i, factor_name in enumerate(factor_ret.columns):
            factor_contribution = factor_loadings[i] * factor_ret.iloc[:, i].mean()
            
            attribution_results.append(AttributionResult(
                factor_name=factor_name,
                contribution=factor_contribution,
                exposure=factor_loadings[i],
                factor_return=factor_ret.iloc[:, i].mean(),
                metadata={'alpha': alpha}
            ))
        
        # Add alpha contribution
        attribution_results.append(AttributionResult(
            factor_name='Alpha',
            contribution=alpha,
            exposure=1.0,
            factor_return=alpha,
            metadata={'is_alpha': True}
        ))
        
        return attribution_results
    
    def generate_performance_report(self, returns: Union[pd.Series, np.ndarray],
                                  benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
                                  risk_free_rate: float = 0.0,
                                  periods_per_year: int = 252) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        # Compute all metrics
        metrics = self.compute_performance_metrics(
            returns, benchmark_returns, risk_free_rate, periods_per_year
        )
        
        # Convert to dictionary
        metrics_dict = {name: result.value for name, result in metrics.items()}
        
        # Add summary statistics
        if len(returns) > 0:
            summary_stats = {
                'total_periods': len(returns),
                'positive_periods': len(returns[returns > 0]) if hasattr(returns, '__len__') else sum(1 for r in returns if r > 0),
                'negative_periods': len(returns[returns < 0]) if hasattr(returns, '__len__') else sum(1 for r in returns if r < 0),
                'win_rate': len(returns[returns > 0]) / len(returns) if hasattr(returns, '__len__') else sum(1 for r in returns if r > 0) / len(list(returns)),
                'best_period': np.max(returns),
                'worst_period': np.min(returns),
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
            }
        else:
            summary_stats = {}
        
        # Drawdown analysis
        dd_stats = self.drawdown_duration(returns)
        
        report = {
            'performance_metrics': metrics_dict,
            'summary_statistics': summary_stats,
            'drawdown_analysis': dd_stats,
            'report_generated_at': datetime.now().isoformat(),
            'parameters': {
                'risk_free_rate': risk_free_rate,
                'periods_per_year': periods_per_year,
                'has_benchmark': benchmark_returns is not None
            }
        }
        
        return report
