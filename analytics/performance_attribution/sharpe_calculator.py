"""
Advanced Sharpe Ratio Calculator Framework
Comprehensive Sharpe ratio analysis with multiple variations and statistical testing
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
from scipy.optimize import minimize_scalar
import concurrent.futures


class SharpeVariant(Enum):
    """Different variants of Sharpe ratio calculation"""
    TRADITIONAL = "TRADITIONAL"           # Standard Sharpe ratio
    ADJUSTED = "ADJUSTED"                 # Adjusted for skewness and kurtosis
    CONDITIONAL = "CONDITIONAL"           # Conditional Sharpe ratio
    PROBABILISTIC = "PROBABILISTIC"       # Probabilistic Sharpe ratio  
    MODIFIED = "MODIFIED"                 # Modified Sharpe ratio
    DOWNSIDE = "DOWNSIDE"                 # Downside Sharpe ratio
    ROLLING = "ROLLING"                   # Rolling Sharpe ratio
    RISK_ADJUSTED = "RISK_ADJUSTED"       # Risk-adjusted Sharpe ratio


class FrequencyAdjustment(Enum):
    """Frequency adjustments for annualization"""
    DAILY = 252
    WEEKLY = 52
    MONTHLY = 12
    QUARTERLY = 4
    ANNUAL = 1


@dataclass
class SharpeAnalysis:
    """Comprehensive Sharpe ratio analysis results"""
    sharpe_ratio: float
    variant: SharpeVariant
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    p_value: float = 0.0
    t_statistic: float = 0.0
    
    # Components
    excess_return: float = 0.0
    volatility: float = 0.0
    risk_free_rate: float = 0.0
    
    # Distribution properties
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Time series properties
    autocorrelation: float = 0.0
    var_95: float = 0.0
    max_drawdown: float = 0.0
    
    # Sample properties
    n_observations: int = 0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    frequency: FrequencyAdjustment = FrequencyAdjustment.DAILY


@dataclass
class SharpeComparison:
    """Comparison between two Sharpe ratios"""
    sharpe_1: float
    sharpe_2: float
    difference: float
    t_statistic: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


class SharpeCalculator(ABC):
    """Abstract base class for Sharpe ratio calculations"""
    
    @abstractmethod
    def calculate(self, returns: pd.Series, risk_free_rate: float = 0.02) -> SharpeAnalysis:
        """Calculate Sharpe ratio analysis"""
        pass


class TraditionalSharpeCalculator(SharpeCalculator):
    """Traditional Sharpe ratio calculator"""
    
    def __init__(self, frequency: FrequencyAdjustment = FrequencyAdjustment.DAILY):
        self.frequency = frequency
    
    def calculate(self, returns: pd.Series, risk_free_rate: float = 0.02) -> SharpeAnalysis:
        """Calculate traditional Sharpe ratio"""
        
        if len(returns) == 0:
            return SharpeAnalysis(
                sharpe_ratio=0.0,
                variant=SharpeVariant.TRADITIONAL,
                frequency=self.frequency
            )
        
        # Annualized metrics
        excess_returns = returns - risk_free_rate / self.frequency.value
        mean_excess_return = excess_returns.mean() * self.frequency.value
        volatility = returns.std() * np.sqrt(self.frequency.value)
        
        # Sharpe ratio
        sharpe_ratio = mean_excess_return / volatility if volatility > 0 else 0.0
        
        # Statistical properties
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns, fisher=True)  # Excess kurtosis
        
        # Confidence interval and significance testing
        n = len(returns)
        if n > 1:
            # Standard error of Sharpe ratio
            sharpe_se = np.sqrt((1 + 0.5 * sharpe_ratio**2) / n)
            
            # Confidence interval (normal approximation)
            ci_lower = sharpe_ratio - 1.96 * sharpe_se
            ci_upper = sharpe_ratio + 1.96 * sharpe_se
            
            # t-test for significance
            t_statistic = sharpe_ratio / sharpe_se
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n-1))
        else:
            ci_lower = ci_upper = sharpe_ratio
            t_statistic = p_value = 0.0
        
        # Additional metrics
        autocorr = self._calculate_autocorrelation(returns)
        var_95 = np.percentile(returns, 5) * self.frequency.value
        max_dd = self._calculate_max_drawdown(returns)
        
        return SharpeAnalysis(
            sharpe_ratio=sharpe_ratio,
            variant=SharpeVariant.TRADITIONAL,
            confidence_interval=(ci_lower, ci_upper),
            p_value=p_value,
            t_statistic=t_statistic,
            excess_return=mean_excess_return,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            skewness=skewness,
            kurtosis=kurtosis,
            autocorrelation=autocorr,
            var_95=var_95,
            max_drawdown=max_dd,
            n_observations=n,
            start_date=returns.index[0] if hasattr(returns.index[0], 'date') else None,
            end_date=returns.index[-1] if hasattr(returns.index[-1], 'date') else None,
            frequency=self.frequency
        )
    
    def _calculate_autocorrelation(self, returns: pd.Series, lag: int = 1) -> float:
        """Calculate autocorrelation of returns"""
        try:
            return returns.autocorr(lag=lag)
        except:
            return 0.0
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()


class AdjustedSharpeCalculator(SharpeCalculator):
    """
    Adjusted Sharpe ratio calculator accounting for skewness and kurtosis
    Based on Pezier and White (2006)
    """
    
    def __init__(self, frequency: FrequencyAdjustment = FrequencyAdjustment.DAILY):
        self.frequency = frequency
    
    def calculate(self, returns: pd.Series, risk_free_rate: float = 0.02) -> SharpeAnalysis:
        """Calculate adjusted Sharpe ratio"""
        
        # First get traditional Sharpe
        traditional_calc = TraditionalSharpeCalculator(self.frequency)
        base_analysis = traditional_calc.calculate(returns, risk_free_rate)
        
        if len(returns) < 4:  # Need sufficient observations
            return base_analysis
        
        # Calculate higher moments
        skewness = base_analysis.skewness
        kurtosis = base_analysis.kurtosis
        
        # Adjusted Sharpe ratio formula
        traditional_sharpe = base_analysis.sharpe_ratio
        
        adjustment = (1 + (skewness / 6) * traditional_sharpe - 
                     ((kurtosis - 3) / 24) * traditional_sharpe**2)
        
        adjusted_sharpe = traditional_sharpe * adjustment
        
        # Update confidence interval for adjusted Sharpe
        n = len(returns)
        if n > 3:
            # Adjusted standard error (approximation)
            base_se = np.sqrt((1 + 0.5 * traditional_sharpe**2) / n)
            adjustment_variance = (skewness**2 / 36) + ((kurtosis - 3)**2 / 576)
            adjusted_se = base_se * np.sqrt(1 + adjustment_variance)
            
            ci_lower = adjusted_sharpe - 1.96 * adjusted_se
            ci_upper = adjusted_sharpe + 1.96 * adjusted_se
            
            t_statistic = adjusted_sharpe / adjusted_se
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n-1))
        else:
            ci_lower = ci_upper = adjusted_sharpe
            t_statistic = p_value = 0.0
        
        # Update analysis
        base_analysis.sharpe_ratio = adjusted_sharpe
        base_analysis.variant = SharpeVariant.ADJUSTED
        base_analysis.confidence_interval = (ci_lower, ci_upper)
        base_analysis.t_statistic = t_statistic
        base_analysis.p_value = p_value
        
        return base_analysis


class ConditionalSharpeCalculator(SharpeCalculator):
    """
    Conditional Sharpe ratio calculator
    Calculates Sharpe ratio conditional on market regimes or volatility states
    """
    
    def __init__(self, frequency: FrequencyAdjustment = FrequencyAdjustment.DAILY,
                 condition_threshold: float = 0.0):
        self.frequency = frequency
        self.condition_threshold = condition_threshold
    
    def calculate(self, returns: pd.Series, risk_free_rate: float = 0.02) -> SharpeAnalysis:
        """Calculate conditional Sharpe ratio"""
        
        if len(returns) == 0:
            return SharpeAnalysis(
                sharpe_ratio=0.0,
                variant=SharpeVariant.CONDITIONAL,
                frequency=self.frequency
            )
        
        # Define condition (e.g., high volatility periods)
        rolling_vol = returns.rolling(window=min(20, len(returns)//2)).std()
        vol_threshold = rolling_vol.quantile(0.7)  # Top 30% volatility periods
        
        high_vol_mask = rolling_vol > vol_threshold
        
        # Calculate conditional Sharpe ratios
        if high_vol_mask.sum() > 5:  # Need sufficient observations
            high_vol_returns = returns[high_vol_mask]
            high_vol_excess = high_vol_returns - risk_free_rate / self.frequency.value
            high_vol_sharpe = (high_vol_excess.mean() * self.frequency.value / 
                              (high_vol_returns.std() * np.sqrt(self.frequency.value)))
        else:
            high_vol_sharpe = 0.0
        
        low_vol_mask = ~high_vol_mask
        if low_vol_mask.sum() > 5:
            low_vol_returns = returns[low_vol_mask]
            low_vol_excess = low_vol_returns - risk_free_rate / self.frequency.value
            low_vol_sharpe = (low_vol_excess.mean() * self.frequency.value / 
                             (low_vol_returns.std() * np.sqrt(self.frequency.value)))
        else:
            low_vol_sharpe = 0.0
        
        # Weighted conditional Sharpe ratio
        high_vol_weight = high_vol_mask.sum() / len(returns)
        low_vol_weight = 1 - high_vol_weight
        
        conditional_sharpe = (high_vol_weight * high_vol_sharpe + 
                            low_vol_weight * low_vol_sharpe)
        
        # Calculate other metrics using full sample
        traditional_calc = TraditionalSharpeCalculator(self.frequency)
        base_analysis = traditional_calc.calculate(returns, risk_free_rate)
        
        # Update with conditional Sharpe
        base_analysis.sharpe_ratio = conditional_sharpe
        base_analysis.variant = SharpeVariant.CONDITIONAL
        
        return base_analysis


class ProbabilisticSharpeCalculator(SharpeCalculator):
    """
    Probabilistic Sharpe ratio calculator
    Based on Marcos López de Prado (2016)
    """
    
    def __init__(self, frequency: FrequencyAdjustment = FrequencyAdjustment.DAILY,
                 benchmark_sharpe: float = 0.0):
        self.frequency = frequency
        self.benchmark_sharpe = benchmark_sharpe
    
    def calculate(self, returns: pd.Series, risk_free_rate: float = 0.02) -> SharpeAnalysis:
        """Calculate probabilistic Sharpe ratio"""
        
        # Get traditional Sharpe first
        traditional_calc = TraditionalSharpeCalculator(self.frequency)
        base_analysis = traditional_calc.calculate(returns, risk_free_rate)
        
        if len(returns) < 10:
            return base_analysis
        
        sharpe_ratio = base_analysis.sharpe_ratio
        skewness = base_analysis.skewness
        kurtosis = base_analysis.kurtosis
        n = len(returns)
        
        # Calculate standard error of Sharpe ratio
        sharpe_variance = (1 + 0.5 * sharpe_ratio**2 - skewness * sharpe_ratio + 
                          (kurtosis - 3) / 4 * sharpe_ratio**2) / n
        
        # Probabilistic Sharpe ratio
        if sharpe_variance > 0:
            psr = stats.norm.cdf((sharpe_ratio - self.benchmark_sharpe) / np.sqrt(sharpe_variance))
        else:
            psr = 0.5
        
        # The PSR itself becomes our "Sharpe ratio" for this variant
        base_analysis.sharpe_ratio = psr
        base_analysis.variant = SharpeVariant.PROBABILISTIC
        
        # Update confidence interval
        psr_se = np.sqrt(psr * (1 - psr) / n)  # Binomial approximation
        ci_lower = max(0, psr - 1.96 * psr_se)
        ci_upper = min(1, psr + 1.96 * psr_se)
        base_analysis.confidence_interval = (ci_lower, ci_upper)
        
        return base_analysis


class ModifiedSharpeCalculator(SharpeCalculator):
    """
    Modified Sharpe ratio using Value at Risk (VaR) instead of standard deviation
    """
    
    def __init__(self, frequency: FrequencyAdjustment = FrequencyAdjustment.DAILY,
                 confidence_level: float = 0.05):
        self.frequency = frequency
        self.confidence_level = confidence_level
    
    def calculate(self, returns: pd.Series, risk_free_rate: float = 0.02) -> SharpeAnalysis:
        """Calculate modified Sharpe ratio"""
        
        # Get base analysis
        traditional_calc = TraditionalSharpeCalculator(self.frequency)
        base_analysis = traditional_calc.calculate(returns, risk_free_rate)
        
        if len(returns) < 10:
            return base_analysis
        
        # Calculate VaR
        var_level = np.percentile(returns, self.confidence_level * 100)
        
        # Modified Sharpe ratio using VaR
        excess_return = base_analysis.excess_return
        modified_sharpe = excess_return / abs(var_level) if var_level != 0 else 0.0
        
        base_analysis.sharpe_ratio = modified_sharpe
        base_analysis.variant = SharpeVariant.MODIFIED
        base_analysis.var_95 = var_level * self.frequency.value
        
        return base_analysis


class DownsideSharpeCalculator(SharpeCalculator):
    """
    Downside Sharpe ratio (Sortino ratio) using downside deviation
    """
    
    def __init__(self, frequency: FrequencyAdjustment = FrequencyAdjustment.DAILY,
                 target_return: float = 0.0):
        self.frequency = frequency
        self.target_return = target_return
    
    def calculate(self, returns: pd.Series, risk_free_rate: float = 0.02) -> SharpeAnalysis:
        """Calculate downside Sharpe ratio (Sortino ratio)"""
        
        # Get base analysis
        traditional_calc = TraditionalSharpeCalculator(self.frequency)
        base_analysis = traditional_calc.calculate(returns, risk_free_rate)
        
        if len(returns) == 0:
            return base_analysis
        
        # Calculate downside deviation
        target_return_daily = self.target_return / self.frequency.value
        downside_returns = returns[returns < target_return_daily]
        
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean((downside_returns - target_return_daily)**2)) * np.sqrt(self.frequency.value)
        else:
            downside_deviation = 0.001  # Small positive value to avoid division by zero
        
        # Sortino ratio
        excess_return = base_analysis.excess_return
        sortino_ratio = excess_return / downside_deviation
        
        base_analysis.sharpe_ratio = sortino_ratio
        base_analysis.variant = SharpeVariant.DOWNSIDE
        base_analysis.volatility = downside_deviation  # Replace with downside deviation
        
        return base_analysis


class RollingSharpeCalculator(SharpeCalculator):
    """
    Rolling Sharpe ratio calculator
    """
    
    def __init__(self, frequency: FrequencyAdjustment = FrequencyAdjustment.DAILY,
                 window: int = 252):
        self.frequency = frequency
        self.window = window
    
    def calculate(self, returns: pd.Series, risk_free_rate: float = 0.02) -> SharpeAnalysis:
        """Calculate rolling Sharpe ratio statistics"""
        
        if len(returns) < self.window:
            # Fall back to traditional calculation
            traditional_calc = TraditionalSharpeCalculator(self.frequency)
            return traditional_calc.calculate(returns, risk_free_rate)
        
        # Calculate rolling Sharpe ratios
        rolling_sharpes = []
        
        for i in range(self.window, len(returns) + 1):
            window_returns = returns.iloc[i-self.window:i]
            
            window_excess = window_returns - risk_free_rate / self.frequency.value
            window_mean = window_excess.mean() * self.frequency.value
            window_std = window_returns.std() * np.sqrt(self.frequency.value)
            
            if window_std > 0:
                rolling_sharpe = window_mean / window_std
            else:
                rolling_sharpe = 0.0
            
            rolling_sharpes.append(rolling_sharpe)
        
        rolling_sharpes = pd.Series(rolling_sharpes, 
                                   index=returns.index[self.window-1:])
        
        # Statistics of rolling Sharpe ratios
        mean_rolling_sharpe = rolling_sharpes.mean()
        std_rolling_sharpe = rolling_sharpes.std()
        
        # Get base analysis for other metrics
        traditional_calc = TraditionalSharpeCalculator(self.frequency)
        base_analysis = traditional_calc.calculate(returns, risk_free_rate)
        
        # Update with rolling statistics
        base_analysis.sharpe_ratio = mean_rolling_sharpe
        base_analysis.variant = SharpeVariant.ROLLING
        
        # Confidence interval based on rolling Sharpe distribution
        if len(rolling_sharpes) > 1:
            ci_lower = rolling_sharpes.quantile(0.025)
            ci_upper = rolling_sharpes.quantile(0.975)
            base_analysis.confidence_interval = (ci_lower, ci_upper)
        
        return base_analysis


class AdvancedSharpeAnalyzer:
    """
    Advanced Sharpe ratio analysis framework
    """
    
    def __init__(self):
        self.calculators = {
            SharpeVariant.TRADITIONAL: TraditionalSharpeCalculator(),
            SharpeVariant.ADJUSTED: AdjustedSharpeCalculator(),
            SharpeVariant.CONDITIONAL: ConditionalSharpeCalculator(),
            SharpeVariant.PROBABILISTIC: ProbabilisticSharpeCalculator(),
            SharpeVariant.MODIFIED: ModifiedSharpeCalculator(),
            SharpeVariant.DOWNSIDE: DownsideSharpeCalculator(),
            SharpeVariant.ROLLING: RollingSharpeCalculator()
        }
    
    def comprehensive_analysis(self, 
                             returns: pd.Series,
                             risk_free_rate: float = 0.02,
                             frequency: FrequencyAdjustment = FrequencyAdjustment.DAILY) -> Dict[SharpeVariant, SharpeAnalysis]:
        """
        Perform comprehensive Sharpe ratio analysis using all variants
        """
        
        results = {}
        
        for variant, calculator in self.calculators.items():
            try:
                # Update frequency if calculator supports it
                if hasattr(calculator, 'frequency'):
                    calculator.frequency = frequency
                
                analysis = calculator.calculate(returns, risk_free_rate)
                results[variant] = analysis
                
            except Exception as e:
                print(f"Warning: {variant.value} calculation failed: {e}")
                continue
        
        return results
    
    def compare_sharpe_ratios(self, 
                            returns_1: pd.Series, 
                            returns_2: pd.Series,
                            risk_free_rate: float = 0.02,
                            frequency: FrequencyAdjustment = FrequencyAdjustment.DAILY) -> SharpeComparison:
        """
        Compare two Sharpe ratios with statistical significance testing
        Using Jobson-Korkie test
        """
        
        # Calculate Sharpe ratios
        calc = TraditionalSharpeCalculator(frequency)
        analysis_1 = calc.calculate(returns_1, risk_free_rate)
        analysis_2 = calc.calculate(returns_2, risk_free_rate)
        
        sharpe_1 = analysis_1.sharpe_ratio
        sharpe_2 = analysis_2.sharpe_ratio
        
        # Align returns for correlation calculation
        aligned_returns = pd.concat([returns_1, returns_2], axis=1, join='inner')
        
        if len(aligned_returns) < 10:
            return SharpeComparison(
                sharpe_1=sharpe_1,
                sharpe_2=sharpe_2,
                difference=sharpe_1 - sharpe_2,
                t_statistic=0.0,
                p_value=1.0,
                is_significant=False
            )
        
        aligned_1 = aligned_returns.iloc[:, 0]
        aligned_2 = aligned_returns.iloc[:, 1]
        
        # Calculate correlation
        correlation = aligned_1.corr(aligned_2)
        if pd.isna(correlation):
            correlation = 0.0
        
        # Jobson-Korkie test statistic
        n = len(aligned_returns)
        
        # Standard errors
        se_1 = np.sqrt((1 + 0.5 * sharpe_1**2) / n)
        se_2 = np.sqrt((1 + 0.5 * sharpe_2**2) / n)
        
        # Covariance term
        cov_term = correlation * se_1 * se_2
        
        # Test statistic
        variance_diff = se_1**2 + se_2**2 - 2 * cov_term
        
        if variance_diff > 0:
            t_statistic = (sharpe_1 - sharpe_2) / np.sqrt(variance_diff)
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n-1))
        else:
            t_statistic = 0.0
            p_value = 1.0
        
        is_significant = p_value < 0.05
        
        # Confidence interval for the difference
        if variance_diff > 0:
            margin_error = stats.t.ppf(0.975, df=n-1) * np.sqrt(variance_diff)
            ci_lower = (sharpe_1 - sharpe_2) - margin_error
            ci_upper = (sharpe_1 - sharpe_2) + margin_error
        else:
            ci_lower = ci_upper = sharpe_1 - sharpe_2
        
        return SharpeComparison(
            sharpe_1=sharpe_1,
            sharpe_2=sharpe_2,
            difference=sharpe_1 - sharpe_2,
            t_statistic=t_statistic,
            p_value=p_value,
            is_significant=is_significant,
            confidence_interval=(ci_lower, ci_upper)
        )
    
    def rolling_sharpe_analysis(self, 
                              returns: pd.Series,
                              windows: List[int] = [63, 126, 252],
                              risk_free_rate: float = 0.02,
                              frequency: FrequencyAdjustment = FrequencyAdjustment.DAILY) -> Dict[int, pd.Series]:
        """
        Calculate rolling Sharpe ratios for multiple window sizes
        """
        
        rolling_results = {}
        
        for window in windows:
            if len(returns) < window:
                continue
            
            rolling_sharpes = []
            
            for i in range(window, len(returns) + 1):
                window_returns = returns.iloc[i-window:i]
                
                calc = TraditionalSharpeCalculator(frequency)
                analysis = calc.calculate(window_returns, risk_free_rate)
                rolling_sharpes.append(analysis.sharpe_ratio)
            
            rolling_series = pd.Series(rolling_sharpes, 
                                     index=returns.index[window-1:])
            rolling_results[window] = rolling_series
        
        return rolling_results
    
    def sharpe_stability_analysis(self, 
                                returns: pd.Series,
                                risk_free_rate: float = 0.02,
                                frequency: FrequencyAdjustment = FrequencyAdjustment.DAILY,
                                n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Analyze Sharpe ratio stability using bootstrap methodology
        """
        
        if len(returns) < 50:
            return {'error': 'Insufficient data for stability analysis'}
        
        # Original Sharpe ratio
        calc = TraditionalSharpeCalculator(frequency)
        original_analysis = calc.calculate(returns, risk_free_rate)
        original_sharpe = original_analysis.sharpe_ratio
        
        # Bootstrap Sharpe ratios
        np.random.seed(42)  # For reproducibility
        bootstrap_sharpes = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_sample = returns.sample(n=len(returns), replace=True)
            
            try:
                bootstrap_analysis = calc.calculate(bootstrap_sample, risk_free_rate)
                bootstrap_sharpes.append(bootstrap_analysis.sharpe_ratio)
            except:
                continue
        
        if not bootstrap_sharpes:
            return {'error': 'Bootstrap analysis failed'}
        
        bootstrap_sharpes = np.array(bootstrap_sharpes)
        
        # Stability metrics
        stability_results = {
            'original_sharpe': original_sharpe,
            'bootstrap_mean': np.mean(bootstrap_sharpes),
            'bootstrap_std': np.std(bootstrap_sharpes),
            'bootstrap_median': np.median(bootstrap_sharpes),
            'confidence_interval_95': (np.percentile(bootstrap_sharpes, 2.5),
                                     np.percentile(bootstrap_sharpes, 97.5)),
            'stability_ratio': np.std(bootstrap_sharpes) / abs(np.mean(bootstrap_sharpes)) if np.mean(bootstrap_sharpes) != 0 else np.inf,
            'positive_probability': np.mean(bootstrap_sharpes > 0),
            'n_bootstrap': len(bootstrap_sharpes)
        }
        
        return stability_results
    
    def generate_sharpe_report(self, 
                             comprehensive_results: Dict[SharpeVariant, SharpeAnalysis],
                             stability_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive Sharpe ratio analysis report
        """
        
        report = []
        report.append("="*70)
        report.append("COMPREHENSIVE SHARPE RATIO ANALYSIS REPORT")
        report.append("="*70)
        
        if not comprehensive_results:
            report.append("No results available.")
            return "\n".join(report)
        
        # Get traditional analysis for base information
        traditional = comprehensive_results.get(SharpeVariant.TRADITIONAL)
        
        if traditional:
            report.append(f"\nDATA SUMMARY:")
            report.append(f"  Period: {traditional.start_date} to {traditional.end_date}" 
                         if traditional.start_date else "  Period: Custom")
            report.append(f"  Observations: {traditional.n_observations}")
            report.append(f"  Frequency: {traditional.frequency.name}")
            report.append(f"  Excess Return: {traditional.excess_return:.2%}")
            report.append(f"  Volatility: {traditional.volatility:.2%}")
            report.append(f"  Risk-free Rate: {traditional.risk_free_rate:.2%}")
        
        # Sharpe ratio variants
        report.append(f"\nSHARPE RATIO VARIANTS:")
        report.append(f"  {'Variant':<15} {'Sharpe':<8} {'95% CI':<20} {'P-Value':<8} {'Significance'}")
        report.append("  " + "-"*65)
        
        for variant, analysis in comprehensive_results.items():
            ci_str = f"[{analysis.confidence_interval[0]:.3f}, {analysis.confidence_interval[1]:.3f}]"
            significance = "***" if analysis.p_value < 0.01 else "**" if analysis.p_value < 0.05 else "*" if analysis.p_value < 0.1 else ""
            
            report.append(f"  {variant.value:<15} {analysis.sharpe_ratio:<8.3f} "
                         f"{ci_str:<20} {analysis.p_value:<8.3f} {significance}")
        
        # Distribution properties
        if traditional:
            report.append(f"\nDISTRIBUTION PROPERTIES:")
            report.append(f"  Skewness: {traditional.skewness:.3f}")
            report.append(f"  Kurtosis: {traditional.kurtosis:.3f}")
            report.append(f"  Autocorrelation: {traditional.autocorrelation:.3f}")
            report.append(f"  95% VaR: {traditional.var_95:.2%}")
            report.append(f"  Max Drawdown: {traditional.max_drawdown:.2%}")
        
        # Stability analysis
        if stability_results and 'error' not in stability_results:
            report.append(f"\nSTABILITY ANALYSIS:")
            report.append(f"  Bootstrap Mean: {stability_results['bootstrap_mean']:.3f}")
            report.append(f"  Bootstrap Std: {stability_results['bootstrap_std']:.3f}")
            report.append(f"  Stability Ratio: {stability_results['stability_ratio']:.3f}")
            report.append(f"  Positive Probability: {stability_results['positive_probability']:.1%}")
            
            ci_95 = stability_results['confidence_interval_95']
            report.append(f"  Bootstrap 95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
        
        # Recommendations
        report.append(f"\nRECOMMENDATIONS:")
        
        if traditional:
            if traditional.sharpe_ratio > 1.0:
                report.append("  • Excellent risk-adjusted performance")
            elif traditional.sharpe_ratio > 0.5:
                report.append("  • Good risk-adjusted performance")
            elif traditional.sharpe_ratio > 0.0:
                report.append("  • Modest risk-adjusted performance")
            else:
                report.append("  • Poor risk-adjusted performance")
            
            if traditional.p_value < 0.05:
                report.append("  • Sharpe ratio is statistically significant")
            else:
                report.append("  • Sharpe ratio is not statistically significant")
            
            if abs(traditional.skewness) > 1.0:
                report.append("  • Consider skewness-adjusted Sharpe ratio")
            
            if traditional.kurtosis > 3.0:
                report.append("  • High kurtosis detected - consider tail risk measures")
            
            if abs(traditional.autocorrelation) > 0.2:
                report.append("  • Significant autocorrelation detected")
        
        return "\n".join(report)
