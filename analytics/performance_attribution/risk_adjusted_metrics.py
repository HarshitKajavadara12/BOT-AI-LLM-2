"""
Advanced Risk-Adjusted Performance Metrics Framework
Comprehensive collection of risk-adjusted performance metrics for quantitative analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
import math
from collections import defaultdict
from scipy import stats
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class RiskMetricType(Enum):
    """Types of risk-adjusted metrics"""
    SHARPE_RATIO = "SHARPE_RATIO"
    SORTINO_RATIO = "SORTINO_RATIO"
    CALMAR_RATIO = "CALMAR_RATIO"
    STERLING_RATIO = "STERLING_RATIO"
    BURKE_RATIO = "BURKE_RATIO"
    INFORMATION_RATIO = "INFORMATION_RATIO"
    TREYNOR_RATIO = "TREYNOR_RATIO"
    JENSEN_ALPHA = "JENSEN_ALPHA"
    MODIGLIANI_RATIO = "MODIGLIANI_RATIO"
    VAN_SHARPE_RATIO = "VAN_SHARPE_RATIO"


class PerformancePeriod(Enum):
    """Performance measurement periods"""
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    YEARLY = "YEARLY"


@dataclass
class RiskAdjustedMetrics:
    """Container for risk-adjusted performance metrics"""
    
    # Basic ratios
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    sterling_ratio: float = 0.0
    burke_ratio: float = 0.0
    
    # Market-relative metrics
    information_ratio: float = 0.0
    treynor_ratio: float = 0.0
    jensen_alpha: float = 0.0
    tracking_error: float = 0.0
    beta: float = 0.0
    
    # Advanced metrics
    modigliani_ratio: float = 0.0
    van_sharpe_ratio: float = 0.0
    omega_ratio: float = 0.0
    kappa_three: float = 0.0
    gain_loss_ratio: float = 0.0
    
    # Statistical measures
    skewness: float = 0.0
    kurtosis: float = 0.0
    jarque_bera_stat: float = 0.0
    jarque_bera_p_value: float = 0.0
    
    # Drawdown-based metrics
    max_drawdown: float = 0.0
    average_drawdown: float = 0.0
    recovery_factor: float = 0.0
    ulcer_index: float = 0.0
    pain_index: float = 0.0
    
    # Value-at-Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    # Performance consistency
    hit_ratio: float = 0.0              # Percentage of positive periods
    profit_factor: float = 0.0          # Total profits / Total losses
    expectancy: float = 0.0             # Average win * win rate - Average loss * loss rate
    
    # Additional metadata
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    total_observations: int = 0
    risk_free_rate: float = 0.0
    benchmark_name: Optional[str] = None


@dataclass 
class PerformanceAttribution:
    """Performance attribution breakdown"""
    
    total_return: float = 0.0
    excess_return: float = 0.0
    
    # Attribution components
    security_selection: float = 0.0
    asset_allocation: float = 0.0
    interaction_effect: float = 0.0
    
    # Factor attributions
    factor_attributions: Dict[str, float] = field(default_factory=dict)
    
    # Period breakdown
    period_attributions: Dict[str, float] = field(default_factory=dict)


class RiskAdjustedCalculator(ABC):
    """Abstract base class for risk-adjusted metrics calculation"""
    
    @abstractmethod
    def calculate(self, 
                 returns: pd.Series,
                 benchmark_returns: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.0) -> RiskAdjustedMetrics:
        """Calculate risk-adjusted metrics"""
        pass


class ComprehensiveRiskCalculator(RiskAdjustedCalculator):
    """Comprehensive risk-adjusted metrics calculator"""
    
    def __init__(self, 
                 frequency: str = 'daily',
                 confidence_levels: List[float] = [0.95, 0.99]):
        self.frequency = frequency
        self.confidence_levels = confidence_levels
        self.annualization_factor = self._get_annualization_factor()
    
    def _get_annualization_factor(self) -> float:
        """Get annualization factor based on frequency"""
        factors = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'yearly': 1
        }
        return factors.get(self.frequency.lower(), 252)
    
    def calculate(self, 
                 returns: pd.Series,
                 benchmark_returns: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.0) -> RiskAdjustedMetrics:
        """Calculate comprehensive risk-adjusted metrics"""
        
        if len(returns) == 0:
            return RiskAdjustedMetrics()
        
        metrics = RiskAdjustedMetrics()
        
        # Basic statistics
        metrics.period_start = returns.index[0] if len(returns) > 0 else None
        metrics.period_end = returns.index[-1] if len(returns) > 0 else None
        metrics.total_observations = len(returns)
        metrics.risk_free_rate = risk_free_rate
        
        # Calculate basic metrics
        self._calculate_basic_ratios(returns, metrics, risk_free_rate)
        
        # Calculate market-relative metrics
        if benchmark_returns is not None:
            self._calculate_market_relative_metrics(returns, benchmark_returns, metrics, risk_free_rate)
        
        # Calculate advanced metrics
        self._calculate_advanced_metrics(returns, metrics, risk_free_rate)
        
        # Calculate drawdown metrics
        self._calculate_drawdown_metrics(returns, metrics)
        
        # Calculate statistical measures
        self._calculate_statistical_measures(returns, metrics)
        
        # Calculate VaR metrics
        self._calculate_var_metrics(returns, metrics)
        
        # Calculate performance consistency metrics
        self._calculate_consistency_metrics(returns, metrics)
        
        return metrics
    
    def _calculate_basic_ratios(self, 
                              returns: pd.Series, 
                              metrics: RiskAdjustedMetrics,
                              risk_free_rate: float):
        """Calculate basic risk-adjusted ratios"""
        
        if len(returns) < 2:
            return
        
        # Annualized return and volatility
        mean_return = returns.mean() * self.annualization_factor
        volatility = returns.std() * np.sqrt(self.annualization_factor)
        
        # Sharpe Ratio
        if volatility > 0:
            metrics.sharpe_ratio = (mean_return - risk_free_rate) / volatility
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_deviation = downside_returns.std() * np.sqrt(self.annualization_factor)
            if downside_deviation > 0:
                metrics.sortino_ratio = (mean_return - risk_free_rate) / downside_deviation
        
        # Calmar Ratio (requires drawdown calculation)
        max_dd = self._calculate_max_drawdown(returns)
        if max_dd > 0:
            metrics.calmar_ratio = mean_return / max_dd
        
        # Sterling Ratio (uses average drawdown)
        avg_dd = self._calculate_average_drawdown(returns)
        if avg_dd > 0:
            metrics.sterling_ratio = mean_return / avg_dd
        
        # Burke Ratio (uses square root of sum of squared drawdowns)
        burke_denominator = self._calculate_burke_denominator(returns)
        if burke_denominator > 0:
            metrics.burke_ratio = mean_return / burke_denominator
    
    def _calculate_market_relative_metrics(self, 
                                         returns: pd.Series,
                                         benchmark_returns: pd.Series,
                                         metrics: RiskAdjustedMetrics,
                                         risk_free_rate: float):
        """Calculate market-relative metrics"""
        
        # Align series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) < 2:
            return
        
        # Excess returns
        excess_returns = aligned_returns - aligned_benchmark
        
        # Information Ratio
        tracking_error = excess_returns.std() * np.sqrt(self.annualization_factor)
        metrics.tracking_error = tracking_error
        
        if tracking_error > 0:
            active_return = excess_returns.mean() * self.annualization_factor
            metrics.information_ratio = active_return / tracking_error
        
        # Beta and Treynor Ratio
        if aligned_benchmark.std() > 0:
            covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
            benchmark_variance = aligned_benchmark.var()
            
            metrics.beta = covariance / benchmark_variance
            
            # Treynor Ratio
            if metrics.beta != 0:
                portfolio_return = aligned_returns.mean() * self.annualization_factor
                benchmark_return = aligned_benchmark.mean() * self.annualization_factor
                metrics.treynor_ratio = (portfolio_return - risk_free_rate) / metrics.beta
        
        # Jensen's Alpha
        portfolio_return = aligned_returns.mean() * self.annualization_factor
        benchmark_return = aligned_benchmark.mean() * self.annualization_factor
        expected_return = risk_free_rate + metrics.beta * (benchmark_return - risk_free_rate)
        metrics.jensen_alpha = portfolio_return - expected_return
    
    def _calculate_advanced_metrics(self, 
                                  returns: pd.Series,
                                  metrics: RiskAdjustedMetrics,
                                  risk_free_rate: float):
        """Calculate advanced risk-adjusted metrics"""
        
        if len(returns) < 2:
            return
        
        # Modigliani-Modigliani Ratio (Risk-adjusted return)
        portfolio_return = returns.mean() * self.annualization_factor
        portfolio_vol = returns.std() * np.sqrt(self.annualization_factor)
        
        # Assume market volatility of 15% if not provided
        market_vol = 0.15
        
        if portfolio_vol > 0:
            metrics.modigliani_ratio = risk_free_rate + (portfolio_return - risk_free_rate) * (market_vol / portfolio_vol)
        
        # VAN (Value Added Monthly/Yearly) - simplified version
        if portfolio_vol > 0:
            metrics.van_sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
        
        # Omega Ratio
        threshold = 0.0  # Use 0 as threshold
        gains = returns[returns > threshold]
        losses = returns[returns <= threshold]
        
        if len(losses) > 0 and abs(losses.sum()) > 0:
            metrics.omega_ratio = gains.sum() / abs(losses.sum())
        
        # Kappa Three (third moment)
        if len(returns) > 2:
            excess_returns = returns - risk_free_rate / self.annualization_factor
            downside_deviation_cubed = ((excess_returns[excess_returns < 0] ** 3).mean()) ** (1/3)
            
            if abs(downside_deviation_cubed) > 0:
                metrics.kappa_three = excess_returns.mean() / abs(downside_deviation_cubed)
        
        # Gain-Loss Ratio
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            avg_gain = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            
            if avg_loss > 0:
                metrics.gain_loss_ratio = avg_gain / avg_loss
    
    def _calculate_drawdown_metrics(self, returns: pd.Series, metrics: RiskAdjustedMetrics):
        """Calculate drawdown-based metrics"""
        
        if len(returns) < 2:
            return
        
        # Convert returns to cumulative wealth
        cumulative_wealth = (1 + returns).cumprod()
        
        # Running maximum
        running_max = cumulative_wealth.expanding().max()
        
        # Drawdown series
        drawdown = (cumulative_wealth - running_max) / running_max
        
        # Maximum drawdown
        metrics.max_drawdown = abs(drawdown.min())
        
        # Average drawdown (of negative drawdowns)
        negative_drawdowns = drawdown[drawdown < 0]
        if len(negative_drawdowns) > 0:
            metrics.average_drawdown = abs(negative_drawdowns.mean())
        
        # Recovery factor
        total_return = cumulative_wealth.iloc[-1] - 1
        if metrics.max_drawdown > 0:
            metrics.recovery_factor = total_return / metrics.max_drawdown
        
        # Ulcer Index
        squared_drawdowns = drawdown ** 2
        metrics.ulcer_index = np.sqrt(squared_drawdowns.mean())
        
        # Pain Index
        metrics.pain_index = abs(drawdown).mean()
    
    def _calculate_statistical_measures(self, returns: pd.Series, metrics: RiskAdjustedMetrics):
        """Calculate statistical measures"""
        
        if len(returns) < 4:
            return
        
        # Skewness and Kurtosis
        metrics.skewness = stats.skew(returns)
        metrics.kurtosis = stats.kurtosis(returns, fisher=True)
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_p_value = stats.jarque_bera(returns)
            metrics.jarque_bera_stat = jb_stat
            metrics.jarque_bera_p_value = jb_p_value
        except:
            pass
    
    def _calculate_var_metrics(self, returns: pd.Series, metrics: RiskAdjustedMetrics):
        """Calculate Value-at-Risk metrics"""
        
        if len(returns) < 10:
            return
        
        # Historical VaR
        for confidence in self.confidence_levels:
            percentile = (1 - confidence) * 100
            var_value = np.percentile(returns, percentile)
            
            if confidence == 0.95:
                metrics.var_95 = abs(var_value)
            elif confidence == 0.99:
                metrics.var_99 = abs(var_value)
            
            # Conditional VaR (Expected Shortfall)
            tail_returns = returns[returns <= var_value]
            if len(tail_returns) > 0:
                cvar_value = tail_returns.mean()
                
                if confidence == 0.95:
                    metrics.cvar_95 = abs(cvar_value)
                elif confidence == 0.99:
                    metrics.cvar_99 = abs(cvar_value)
    
    def _calculate_consistency_metrics(self, returns: pd.Series, metrics: RiskAdjustedMetrics):
        """Calculate performance consistency metrics"""
        
        if len(returns) == 0:
            return
        
        # Hit ratio (percentage of positive periods)
        positive_periods = (returns > 0).sum()
        metrics.hit_ratio = positive_periods / len(returns)
        
        # Profit factor
        total_profits = returns[returns > 0].sum()
        total_losses = abs(returns[returns < 0].sum())
        
        if total_losses > 0:
            metrics.profit_factor = total_profits / total_losses
        
        # Expectancy
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())
            win_rate = len(positive_returns) / len(returns)
            loss_rate = len(negative_returns) / len(returns)
            
            metrics.expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        
        if len(returns) < 2:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
    
    def _calculate_average_drawdown(self, returns: pd.Series) -> float:
        """Calculate average drawdown"""
        
        if len(returns) < 2:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        negative_drawdowns = drawdown[drawdown < 0]
        return abs(negative_drawdowns.mean()) if len(negative_drawdowns) > 0 else 0.0
    
    def _calculate_burke_denominator(self, returns: pd.Series) -> float:
        """Calculate Burke ratio denominator"""
        
        if len(returns) < 2:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        squared_drawdowns = drawdown ** 2
        return np.sqrt(squared_drawdowns.sum())


class RollingRiskCalculator:
    """Calculate rolling risk-adjusted metrics"""
    
    def __init__(self, window: int = 252, calculator: Optional[RiskAdjustedCalculator] = None):
        self.window = window
        self.calculator = calculator or ComprehensiveRiskCalculator()
    
    def calculate_rolling_metrics(self, 
                                returns: pd.Series,
                                benchmark_returns: Optional[pd.Series] = None,
                                risk_free_rate: float = 0.0) -> pd.DataFrame:
        """Calculate rolling risk-adjusted metrics"""
        
        if len(returns) < self.window:
            raise ValueError(f"Need at least {self.window} observations")
        
        results = []
        
        for i in range(self.window, len(returns) + 1):
            window_returns = returns.iloc[i-self.window:i]
            window_benchmark = None
            
            if benchmark_returns is not None:
                window_benchmark = benchmark_returns.iloc[i-self.window:i]
            
            metrics = self.calculator.calculate(
                window_returns, window_benchmark, risk_free_rate
            )
            
            result_dict = {
                'date': returns.index[i-1],
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'calmar_ratio': metrics.calmar_ratio,
                'information_ratio': metrics.information_ratio,
                'max_drawdown': metrics.max_drawdown,
                'volatility': window_returns.std() * np.sqrt(252),
                'beta': metrics.beta if hasattr(metrics, 'beta') else 0.0
            }
            
            results.append(result_dict)
        
        return pd.DataFrame(results).set_index('date')


class PerformanceAttributor:
    """Performance attribution analysis"""
    
    def __init__(self):
        self.attribution_methods = {
            'brinson': self._brinson_attribution,
            'factor': self._factor_attribution,
            'holdings': self._holdings_attribution
        }
    
    def attribute_performance(self, 
                            portfolio_returns: pd.Series,
                            benchmark_returns: pd.Series,
                            method: str = 'brinson',
                            **kwargs) -> PerformanceAttribution:
        """Perform performance attribution analysis"""
        
        if method not in self.attribution_methods:
            raise ValueError(f"Unknown attribution method: {method}")
        
        return self.attribution_methods[method](
            portfolio_returns, benchmark_returns, **kwargs
        )
    
    def _brinson_attribution(self, 
                           portfolio_returns: pd.Series,
                           benchmark_returns: pd.Series,
                           **kwargs) -> PerformanceAttribution:
        """Brinson attribution (simplified)"""
        
        attribution = PerformanceAttribution()
        
        # Calculate total returns
        portfolio_total = (1 + portfolio_returns).prod() - 1
        benchmark_total = (1 + benchmark_returns).prod() - 1
        
        attribution.total_return = portfolio_total
        attribution.excess_return = portfolio_total - benchmark_total
        
        # Simplified attribution (would need weights for full implementation)
        attribution.security_selection = attribution.excess_return * 0.7
        attribution.asset_allocation = attribution.excess_return * 0.3
        attribution.interaction_effect = 0.0
        
        return attribution
    
    def _factor_attribution(self, 
                          portfolio_returns: pd.Series,
                          benchmark_returns: pd.Series,
                          factors: Optional[pd.DataFrame] = None,
                          **kwargs) -> PerformanceAttribution:
        """Factor-based attribution"""
        
        attribution = PerformanceAttribution()
        
        if factors is None:
            # Use simple market factor
            factors = pd.DataFrame({'market': benchmark_returns})
        
        # Align data
        aligned_data = pd.concat([portfolio_returns, factors], axis=1, join='inner')
        
        if len(aligned_data) < 10:
            return attribution
        
        # Regression analysis
        y = aligned_data.iloc[:, 0].values  # Portfolio returns
        X = aligned_data.iloc[:, 1:].values  # Factor returns
        
        if X.shape[1] > 0:
            try:
                reg = LinearRegression().fit(X, y)
                
                # Factor attributions
                for i, factor_name in enumerate(factors.columns):
                    factor_contribution = reg.coef_[i] * factors[factor_name].mean()
                    attribution.factor_attributions[factor_name] = factor_contribution
                
                # Alpha (unexplained return)
                attribution.factor_attributions['alpha'] = reg.intercept_
                
            except:
                pass
        
        return attribution
    
    def _holdings_attribution(self, 
                            portfolio_returns: pd.Series,
                            benchmark_returns: pd.Series,
                            **kwargs) -> PerformanceAttribution:
        """Holdings-based attribution (placeholder)"""
        
        attribution = PerformanceAttribution()
        
        # Would need actual holdings data for full implementation
        attribution.total_return = (1 + portfolio_returns).prod() - 1
        attribution.excess_return = attribution.total_return - ((1 + benchmark_returns).prod() - 1)
        
        return attribution


class RiskAdjustedReportGenerator:
    """Generate comprehensive risk-adjusted performance reports"""
    
    def __init__(self, calculator: Optional[RiskAdjustedCalculator] = None):
        self.calculator = calculator or ComprehensiveRiskCalculator()
    
    def generate_report(self, 
                       returns: pd.Series,
                       benchmark_returns: Optional[pd.Series] = None,
                       risk_free_rate: float = 0.0,
                       strategy_name: str = "Strategy") -> str:
        """Generate comprehensive performance report"""
        
        metrics = self.calculator.calculate(returns, benchmark_returns, risk_free_rate)
        
        report = []
        report.append("="*80)
        report.append(f"RISK-ADJUSTED PERFORMANCE ANALYSIS: {strategy_name}")
        report.append("="*80)
        
        # Period information
        if metrics.period_start and metrics.period_end:
            report.append(f"\nANALYSIS PERIOD:")
            report.append(f"  Start Date: {metrics.period_start.strftime('%Y-%m-%d')}")
            report.append(f"  End Date: {metrics.period_end.strftime('%Y-%m-%d')}")
            report.append(f"  Total Observations: {metrics.total_observations}")
        
        # Basic performance metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = total_return * (252 / len(returns)) if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252)
        
        report.append(f"\nBASIC PERFORMANCE METRICS:")
        report.append(f"  Total Return: {total_return:.2%}")
        report.append(f"  Annualized Return: {annualized_return:.2%}")
        report.append(f"  Annualized Volatility: {volatility:.2%}")
        report.append(f"  Risk-Free Rate: {risk_free_rate:.2%}")
        
        # Risk-adjusted ratios
        report.append(f"\nRISK-ADJUSTED RATIOS:")
        report.append(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        report.append(f"  Sortino Ratio: {metrics.sortino_ratio:.3f}")
        report.append(f"  Calmar Ratio: {metrics.calmar_ratio:.3f}")
        report.append(f"  Sterling Ratio: {metrics.sterling_ratio:.3f}")
        report.append(f"  Burke Ratio: {metrics.burke_ratio:.3f}")
        
        # Market-relative metrics
        if benchmark_returns is not None:
            benchmark_return = (1 + benchmark_returns).prod() - 1
            excess_return = total_return - benchmark_return
            
            report.append(f"\nMARKET-RELATIVE METRICS:")
            report.append(f"  Benchmark Return: {benchmark_return:.2%}")
            report.append(f"  Excess Return: {excess_return:.2%}")
            report.append(f"  Information Ratio: {metrics.information_ratio:.3f}")
            report.append(f"  Tracking Error: {metrics.tracking_error:.2%}")
            report.append(f"  Beta: {metrics.beta:.3f}")
            report.append(f"  Treynor Ratio: {metrics.treynor_ratio:.3f}")
            report.append(f"  Jensen's Alpha: {metrics.jensen_alpha:.2%}")
        
        # Advanced metrics
        report.append(f"\nADVANCED METRICS:")
        report.append(f"  Modigliani Ratio: {metrics.modigliani_ratio:.2%}")
        report.append(f"  Omega Ratio: {metrics.omega_ratio:.3f}")
        report.append(f"  Gain-Loss Ratio: {metrics.gain_loss_ratio:.3f}")
        
        # Drawdown analysis
        report.append(f"\nDRAWDOWN ANALYSIS:")
        report.append(f"  Maximum Drawdown: {metrics.max_drawdown:.2%}")
        report.append(f"  Average Drawdown: {metrics.average_drawdown:.2%}")
        report.append(f"  Recovery Factor: {metrics.recovery_factor:.2f}")
        report.append(f"  Ulcer Index: {metrics.ulcer_index:.4f}")
        report.append(f"  Pain Index: {metrics.pain_index:.4f}")
        
        # Statistical measures
        report.append(f"\nSTATISTICAL MEASURES:")
        report.append(f"  Skewness: {metrics.skewness:.3f}")
        report.append(f"  Kurtosis: {metrics.kurtosis:.3f}")
        report.append(f"  Jarque-Bera Stat: {metrics.jarque_bera_stat:.2f}")
        report.append(f"  JB P-Value: {metrics.jarque_bera_p_value:.4f}")
        
        # VaR metrics
        report.append(f"\nVALUE-AT-RISK METRICS:")
        report.append(f"  95% VaR: {metrics.var_95:.2%}")
        report.append(f"  99% VaR: {metrics.var_99:.2%}")
        report.append(f"  95% CVaR: {metrics.cvar_95:.2%}")
        report.append(f"  99% CVaR: {metrics.cvar_99:.2%}")
        
        # Performance consistency
        report.append(f"\nPERFORMANCE CONSISTENCY:")
        report.append(f"  Hit Ratio: {metrics.hit_ratio:.1%}")
        report.append(f"  Profit Factor: {metrics.profit_factor:.2f}")
        report.append(f"  Expectancy: {metrics.expectancy:.4f}")
        
        # Performance interpretation
        report.append(f"\nPERFORMANCE INTERPRETATION:")
        
        if metrics.sharpe_ratio > 2.0:
            report.append("  Excellent risk-adjusted performance (Sharpe > 2.0)")
        elif metrics.sharpe_ratio > 1.0:
            report.append("  Good risk-adjusted performance (Sharpe > 1.0)")
        elif metrics.sharpe_ratio > 0.5:
            report.append("  Acceptable risk-adjusted performance (Sharpe > 0.5)")
        else:
            report.append("  Poor risk-adjusted performance (Sharpe ≤ 0.5)")
        
        if metrics.max_drawdown < 0.05:
            report.append("  Low drawdown risk (Max DD < 5%)")
        elif metrics.max_drawdown < 0.15:
            report.append("  Moderate drawdown risk (Max DD < 15%)")
        else:
            report.append("  High drawdown risk (Max DD ≥ 15%)")
        
        if abs(metrics.skewness) < 0.5:
            report.append("  Returns are approximately normally distributed")
        elif metrics.skewness > 0.5:
            report.append("  Returns are positively skewed (right tail)")
        else:
            report.append("  Returns are negatively skewed (left tail)")
        
        return "\n".join(report)
    
    def plot_risk_metrics(self, 
                         returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         save_path: Optional[str] = None):
        """Create risk metrics visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Cumulative returns
        ax1 = axes[0, 0]
        cumulative_returns = (1 + returns).cumprod()
        ax1.plot(cumulative_returns.index, cumulative_returns.values, 'b-', 
                linewidth=2, label='Strategy')
        
        if benchmark_returns is not None:
            aligned_bench = benchmark_returns.reindex(returns.index).fillna(0)
            cumulative_bench = (1 + aligned_bench).cumprod()
            ax1.plot(cumulative_bench.index, cumulative_bench.values, 'r--', 
                    linewidth=2, label='Benchmark')
        
        ax1.set_title('Cumulative Returns')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Rolling Sharpe ratio
        ax2 = axes[0, 1]
        if len(returns) > 60:
            rolling_sharpe = returns.rolling(60).mean() / returns.rolling(60).std() * np.sqrt(252)
            ax2.plot(rolling_sharpe.index, rolling_sharpe.values, 'g-', linewidth=1)
            ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Sharpe = 1.0')
            ax2.set_title('Rolling Sharpe Ratio (60-day)')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Return distribution
        ax3 = axes[1, 0]
        ax3.hist(returns.values, bins=50, alpha=0.7, density=True, color='blue')
        ax3.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax3.set_title('Return Distribution')
        ax3.set_xlabel('Daily Returns')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Drawdown curve
        ax4 = axes[1, 1]
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        ax4.fill_between(drawdown.index, drawdown.values, 0, 
                        where=(drawdown.values < 0), color='red', alpha=0.7)
        ax4.plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
        ax4.set_title('Drawdown Curve')
        ax4.set_ylabel('Drawdown')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Risk metrics plot saved to {save_path}")
        
        plt.show()
