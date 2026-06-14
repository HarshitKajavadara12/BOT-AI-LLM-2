"""
Advanced Drawdown Analysis Framework
Comprehensive drawdown analysis including maximum drawdown, underwater curves, and recovery analysis
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
from collections import defaultdict, deque
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns


class DrawdownType(Enum):
    """Types of drawdown measurements"""
    ABSOLUTE = "ABSOLUTE"                   # Absolute drawdown from peak
    RELATIVE = "RELATIVE"                   # Relative drawdown percentage
    LOGARITHMIC = "LOGARITHMIC"             # Log drawdown
    CONDITIONAL = "CONDITIONAL"             # Conditional drawdown (CVaR style)
    ULCER_INDEX = "ULCER_INDEX"             # Ulcer Index drawdown
    PAIN_INDEX = "PAIN_INDEX"               # Pain Index


class RecoveryType(Enum):
    """Types of recovery analysis"""
    TIME_TO_RECOVERY = "TIME_TO_RECOVERY"   # Time to recover from drawdown
    RECOVERY_FACTOR = "RECOVERY_FACTOR"     # Recovery strength factor
    RECOVERY_RATE = "RECOVERY_RATE"         # Rate of recovery
    PARTIAL_RECOVERY = "PARTIAL_RECOVERY"   # Partial recovery analysis


@dataclass
class DrawdownPeriod:
    """Individual drawdown period analysis"""
    start_date: datetime
    end_date: datetime
    peak_date: datetime
    trough_date: datetime
    recovery_date: Optional[datetime] = None
    
    peak_value: float = 0.0
    trough_value: float = 0.0
    recovery_value: float = 0.0
    
    drawdown_amount: float = 0.0
    drawdown_percent: float = 0.0
    duration_days: int = 0
    recovery_days: Optional[int] = None
    
    underwater_area: float = 0.0            # Area under the underwater curve
    pain_ratio: float = 0.0                 # Pain during drawdown period
    
    is_recovered: bool = False
    recovery_factor: float = 0.0


@dataclass
class DrawdownStatistics:
    """Comprehensive drawdown statistics"""
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    average_drawdown: float = 0.0
    drawdown_frequency: float = 0.0
    
    # Recovery statistics
    average_recovery_time: float = 0.0
    recovery_factor: float = 0.0
    
    # Advanced metrics
    ulcer_index: float = 0.0
    pain_index: float = 0.0
    lake_ratio: float = 0.0                 # Time underwater / total time
    gain_to_pain_ratio: float = 0.0
    
    # Distribution statistics
    drawdown_skewness: float = 0.0
    drawdown_kurtosis: float = 0.0
    
    # Conditional metrics
    conditional_drawdown_95: float = 0.0    # 95% Conditional Drawdown
    conditional_drawdown_99: float = 0.0    # 99% Conditional Drawdown
    
    # Period analysis
    drawdown_periods: List[DrawdownPeriod] = field(default_factory=list)
    total_periods: int = 0
    
    # Time series
    underwater_curve: Optional[pd.Series] = None
    rolling_max: Optional[pd.Series] = None


class DrawdownAnalyzer(ABC):
    """Abstract base class for drawdown analysis"""
    
    @abstractmethod
    def analyze(self, price_series: pd.Series) -> DrawdownStatistics:
        """Analyze drawdowns in price series"""
        pass


class StandardDrawdownAnalyzer(DrawdownAnalyzer):
    """Standard drawdown analysis implementation"""
    
    def __init__(self, min_periods: int = 1):
        self.min_periods = min_periods
        
    def analyze(self, price_series: pd.Series) -> DrawdownStatistics:
        """Analyze drawdowns using standard methodology"""
        
        if len(price_series) < 2:
            return DrawdownStatistics()
        
        # Calculate running maximum (peaks)
        rolling_max = price_series.expanding().max()
        
        # Calculate drawdown series
        drawdown_series = (price_series - rolling_max) / rolling_max
        
        # Underwater curve (negative drawdowns only)
        underwater_curve = drawdown_series.copy()
        
        # Find drawdown periods
        drawdown_periods = self._identify_drawdown_periods(
            price_series, rolling_max, drawdown_series
        )
        
        # Calculate statistics
        stats = self._calculate_statistics(
            drawdown_series, underwater_curve, drawdown_periods, price_series
        )
        
        stats.underwater_curve = underwater_curve
        stats.rolling_max = rolling_max
        stats.drawdown_periods = drawdown_periods
        stats.total_periods = len(drawdown_periods)
        
        return stats
    
    def _identify_drawdown_periods(self, 
                                 price_series: pd.Series,
                                 rolling_max: pd.Series,
                                 drawdown_series: pd.Series) -> List[DrawdownPeriod]:
        """Identify individual drawdown periods"""
        
        periods = []
        in_drawdown = False
        current_period = None
        
        for i, (date, price) in enumerate(price_series.items()):
            is_at_peak = abs(price - rolling_max.iloc[i]) < 1e-8
            
            if not in_drawdown and not is_at_peak:
                # Start of new drawdown
                in_drawdown = True
                peak_date = price_series.index[i-1] if i > 0 else date
                peak_value = rolling_max.iloc[i]
                
                current_period = DrawdownPeriod(
                    start_date=date,
                    peak_date=peak_date,
                    end_date=date,
                    trough_date=date,
                    peak_value=peak_value,
                    trough_value=price
                )
            
            elif in_drawdown:
                # Update current drawdown
                if current_period:
                    current_period.end_date = date
                    
                    # Update trough if this is a new low
                    if price < current_period.trough_value:
                        current_period.trough_date = date
                        current_period.trough_value = price
                    
                    # Check for recovery
                    if is_at_peak:
                        # End of drawdown period
                        current_period.recovery_date = date
                        current_period.recovery_value = price
                        current_period.is_recovered = True
                        
                        # Calculate period statistics
                        self._finalize_drawdown_period(current_period)
                        periods.append(current_period)
                        
                        in_drawdown = False
                        current_period = None
        
        # Handle ongoing drawdown
        if in_drawdown and current_period:
            current_period.is_recovered = False
            self._finalize_drawdown_period(current_period)
            periods.append(current_period)
        
        return periods
    
    def _finalize_drawdown_period(self, period: DrawdownPeriod):
        """Calculate final statistics for a drawdown period"""
        
        # Drawdown amount and percentage
        period.drawdown_amount = period.peak_value - period.trough_value
        if period.peak_value > 0:
            period.drawdown_percent = period.drawdown_amount / period.peak_value
        
        # Duration
        period.duration_days = (period.end_date - period.start_date).days
        
        # Recovery time
        if period.recovery_date:
            period.recovery_days = (period.recovery_date - period.trough_date).days
            
            # Recovery factor
            if period.trough_value > 0:
                period.recovery_factor = (period.recovery_value - period.trough_value) / period.trough_value
        
        # Pain ratio (simplified)
        period.pain_ratio = period.drawdown_percent * np.sqrt(period.duration_days / 252)
    
    def _calculate_statistics(self, 
                            drawdown_series: pd.Series,
                            underwater_curve: pd.Series,
                            drawdown_periods: List[DrawdownPeriod],
                            price_series: pd.Series) -> DrawdownStatistics:
        """Calculate comprehensive drawdown statistics"""
        
        stats = DrawdownStatistics()
        
        if len(drawdown_series) == 0:
            return stats
        
        # Basic drawdown metrics
        stats.max_drawdown = abs(drawdown_series.min())
        
        # Average drawdown (of non-zero drawdowns)
        negative_drawdowns = drawdown_series[drawdown_series < 0]
        if len(negative_drawdowns) > 0:
            stats.average_drawdown = abs(negative_drawdowns.mean())
        
        # Drawdown frequency (periods per year)
        if len(drawdown_periods) > 0:
            total_days = (price_series.index[-1] - price_series.index[0]).days
            stats.drawdown_frequency = len(drawdown_periods) * 365 / max(total_days, 1)
            
            # Max drawdown duration
            stats.max_drawdown_duration = max(period.duration_days for period in drawdown_periods)
            
            # Recovery statistics
            recovered_periods = [p for p in drawdown_periods if p.is_recovered and p.recovery_days is not None]
            if recovered_periods:
                stats.average_recovery_time = np.mean([p.recovery_days for p in recovered_periods])
                stats.recovery_factor = np.mean([p.recovery_factor for p in recovered_periods])
        
        # Advanced metrics
        stats.ulcer_index = self._calculate_ulcer_index(drawdown_series)
        stats.pain_index = self._calculate_pain_index(drawdown_series)
        stats.lake_ratio = self._calculate_lake_ratio(drawdown_series)
        
        # Gain to Pain ratio
        total_return = (price_series.iloc[-1] / price_series.iloc[0] - 1) if len(price_series) > 1 else 0
        stats.gain_to_pain_ratio = total_return / stats.pain_index if stats.pain_index > 0 else 0
        
        # Distribution statistics
        if len(negative_drawdowns) > 1:
            stats.drawdown_skewness = stats.skew(negative_drawdowns)
            stats.drawdown_kurtosis = stats.kurtosis(negative_drawdowns, fisher=True)
        
        # Conditional drawdowns
        if len(negative_drawdowns) > 0:
            stats.conditional_drawdown_95 = abs(np.percentile(negative_drawdowns, 5))
            stats.conditional_drawdown_99 = abs(np.percentile(negative_drawdowns, 1))
        
        return stats
    
    def _calculate_ulcer_index(self, drawdown_series: pd.Series) -> float:
        """Calculate Ulcer Index"""
        
        if len(drawdown_series) == 0:
            return 0.0
        
        # Ulcer Index = sqrt(mean(drawdown^2))
        squared_drawdowns = drawdown_series ** 2
        return np.sqrt(squared_drawdowns.mean())
    
    def _calculate_pain_index(self, drawdown_series: pd.Series) -> float:
        """Calculate Pain Index"""
        
        if len(drawdown_series) == 0:
            return 0.0
        
        # Pain Index = mean(abs(drawdown))
        return abs(drawdown_series).mean()
    
    def _calculate_lake_ratio(self, drawdown_series: pd.Series) -> float:
        """Calculate Lake Ratio (time underwater)"""
        
        if len(drawdown_series) == 0:
            return 0.0
        
        # Fraction of time with negative drawdowns
        underwater_periods = (drawdown_series < 0).sum()
        return underwater_periods / len(drawdown_series)


class ConditionalDrawdownAnalyzer(DrawdownAnalyzer):
    """Conditional Drawdown Analyzer (CDaR) similar to CVaR"""
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        
    def analyze(self, price_series: pd.Series) -> DrawdownStatistics:
        """Analyze conditional drawdowns"""
        
        # Get base analysis
        base_analyzer = StandardDrawdownAnalyzer()
        stats = base_analyzer.analyze(price_series)
        
        if len(price_series) < 2:
            return stats
        
        # Calculate drawdown series
        rolling_max = price_series.expanding().max()
        drawdown_series = (price_series - rolling_max) / rolling_max
        
        # Get negative drawdowns
        negative_drawdowns = drawdown_series[drawdown_series < 0]
        
        if len(negative_drawdowns) == 0:
            return stats
        
        # Calculate conditional drawdowns
        for confidence in self.confidence_levels:
            percentile = (1 - confidence) * 100
            threshold = np.percentile(negative_drawdowns, percentile)
            
            # Conditional expectation beyond threshold
            tail_drawdowns = negative_drawdowns[negative_drawdowns <= threshold]
            conditional_dd = abs(tail_drawdowns.mean()) if len(tail_drawdowns) > 0 else 0
            
            if confidence == 0.95:
                stats.conditional_drawdown_95 = conditional_dd
            elif confidence == 0.99:
                stats.conditional_drawdown_99 = conditional_dd
        
        return stats


class RollingDrawdownAnalyzer(DrawdownAnalyzer):
    """Rolling window drawdown analysis"""
    
    def __init__(self, window: int = 252):
        self.window = window
        
    def analyze(self, price_series: pd.Series) -> DrawdownStatistics:
        """Analyze rolling drawdowns"""
        
        if len(price_series) < self.window:
            # Fall back to standard analysis
            return StandardDrawdownAnalyzer().analyze(price_series)
        
        # Calculate rolling drawdowns
        rolling_drawdowns = []
        
        for i in range(self.window, len(price_series) + 1):
            window_prices = price_series.iloc[i-self.window:i]
            window_max = window_prices.max()
            window_current = window_prices.iloc[-1]
            
            if window_max > 0:
                rolling_dd = (window_current - window_max) / window_max
            else:
                rolling_dd = 0.0
            
            rolling_drawdowns.append(rolling_dd)
        
        rolling_dd_series = pd.Series(
            rolling_drawdowns, 
            index=price_series.index[self.window-1:]
        )
        
        # Calculate statistics on rolling drawdowns
        stats = DrawdownStatistics()
        
        if len(rolling_dd_series) > 0:
            stats.max_drawdown = abs(rolling_dd_series.min())
            stats.average_drawdown = abs(rolling_dd_series.mean())
            
            negative_dds = rolling_dd_series[rolling_dd_series < 0]
            if len(negative_dds) > 0:
                stats.drawdown_frequency = len(negative_dds) / len(rolling_dd_series)
                stats.conditional_drawdown_95 = abs(np.percentile(negative_dds, 5))
                stats.conditional_drawdown_99 = abs(np.percentile(negative_dds, 1))
        
        return stats


class AdvancedDrawdownAnalyzer:
    """
    Advanced drawdown analysis framework combining multiple methods
    """
    
    def __init__(self):
        self.analyzers = {
            'standard': StandardDrawdownAnalyzer(),
            'conditional': ConditionalDrawdownAnalyzer(),
            'rolling': RollingDrawdownAnalyzer()
        }
    
    def comprehensive_analysis(self, 
                             price_series: pd.Series,
                             return_series: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform comprehensive drawdown analysis
        
        Args:
            price_series: Price/equity curve series
            return_series: Optional return series
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        
        results = {}
        
        # Run different analysis methods
        for name, analyzer in self.analyzers.items():
            try:
                stats = analyzer.analyze(price_series)
                results[name] = stats
            except Exception as e:
                print(f"Warning: {name} analysis failed: {e}")
                continue
        
        # Additional analysis if return series provided
        if return_series is not None:
            results['return_based'] = self._analyze_return_based_drawdowns(return_series)
        
        # Comparative analysis
        results['comparison'] = self._compare_methods(results)
        
        # Risk-adjusted metrics
        results['risk_adjusted'] = self._calculate_risk_adjusted_metrics(
            price_series, results.get('standard')
        )
        
        return results
    
    def _analyze_return_based_drawdowns(self, return_series: pd.Series) -> Dict[str, Any]:
        """Analyze drawdowns based on return series"""
        
        # Convert returns to cumulative wealth
        cumulative_wealth = (1 + return_series).cumprod()
        
        # Analyze drawdowns on wealth curve
        analyzer = StandardDrawdownAnalyzer()
        stats = analyzer.analyze(cumulative_wealth)
        
        # Additional return-based metrics
        return_metrics = {
            'drawdown_statistics': stats,
            'negative_return_frequency': (return_series < 0).mean(),
            'worst_return': return_series.min(),
            'worst_return_date': return_series.idxmin(),
            'consecutive_losses': self._calculate_consecutive_losses(return_series),
            'recovery_ratios': self._calculate_recovery_ratios(return_series)
        }
        
        return return_metrics
    
    def _calculate_consecutive_losses(self, return_series: pd.Series) -> Dict[str, int]:
        """Calculate consecutive loss statistics"""
        
        losses = return_series < 0
        consecutive = []
        current_streak = 0
        
        for loss in losses:
            if loss:
                current_streak += 1
            else:
                if current_streak > 0:
                    consecutive.append(current_streak)
                current_streak = 0
        
        # Add final streak if ongoing
        if current_streak > 0:
            consecutive.append(current_streak)
        
        if consecutive:
            return {
                'max_consecutive_losses': max(consecutive),
                'avg_consecutive_losses': np.mean(consecutive),
                'total_loss_streaks': len(consecutive)
            }
        else:
            return {
                'max_consecutive_losses': 0,
                'avg_consecutive_losses': 0,
                'total_loss_streaks': 0
            }
    
    def _calculate_recovery_ratios(self, return_series: pd.Series) -> Dict[str, float]:
        """Calculate recovery ratio metrics"""
        
        cumulative = (1 + return_series).cumprod()
        
        # Find drawdown periods
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        
        # Recovery analysis
        in_drawdown = False
        drawdown_start = None
        recovery_ratios = []
        
        for i, (date, value) in enumerate(drawdowns.items()):
            if not in_drawdown and value < -0.01:  # 1% threshold
                in_drawdown = True
                drawdown_start = i
                max_dd_in_period = value
                
            elif in_drawdown:
                if value < max_dd_in_period:
                    max_dd_in_period = value
                    
                if abs(value) < 0.001:  # Recovery
                    # Calculate recovery ratio
                    if drawdown_start is not None:
                        drawdown_duration = i - drawdown_start
                        if drawdown_duration > 0:
                            recovery_ratio = abs(max_dd_in_period) / drawdown_duration
                            recovery_ratios.append(recovery_ratio)
                    
                    in_drawdown = False
                    drawdown_start = None
        
        if recovery_ratios:
            return {
                'avg_recovery_ratio': np.mean(recovery_ratios),
                'median_recovery_ratio': np.median(recovery_ratios),
                'min_recovery_ratio': np.min(recovery_ratios),
                'max_recovery_ratio': np.max(recovery_ratios)
            }
        else:
            return {
                'avg_recovery_ratio': 0.0,
                'median_recovery_ratio': 0.0,
                'min_recovery_ratio': 0.0,
                'max_recovery_ratio': 0.0
            }
    
    def _compare_methods(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results from different analysis methods"""
        
        comparison = {}
        
        # Extract max drawdown from each method
        max_drawdowns = {}
        for method, result in results.items():
            if isinstance(result, DrawdownStatistics):
                max_drawdowns[method] = result.max_drawdown
            elif isinstance(result, dict) and 'drawdown_statistics' in result:
                max_drawdowns[method] = result['drawdown_statistics'].max_drawdown
        
        if max_drawdowns:
            comparison['max_drawdown_comparison'] = max_drawdowns
            comparison['max_drawdown_range'] = max(max_drawdowns.values()) - min(max_drawdowns.values())
            comparison['max_drawdown_consensus'] = np.mean(list(max_drawdowns.values()))
        
        return comparison
    
    def _calculate_risk_adjusted_metrics(self, 
                                       price_series: pd.Series,
                                       drawdown_stats: Optional[DrawdownStatistics]) -> Dict[str, float]:
        """Calculate risk-adjusted drawdown metrics"""
        
        if drawdown_stats is None or len(price_series) < 2:
            return {}
        
        # Total return
        total_return = (price_series.iloc[-1] / price_series.iloc[0] - 1)
        
        # Risk-adjusted metrics
        metrics = {}
        
        # Calmar Ratio (Annual Return / Max Drawdown)
        if drawdown_stats.max_drawdown > 0:
            annualized_return = total_return * (252 / len(price_series))
            metrics['calmar_ratio'] = annualized_return / drawdown_stats.max_drawdown
        else:
            metrics['calmar_ratio'] = 0.0
        
        # Sterling Ratio (similar to Calmar but uses average drawdown)
        if drawdown_stats.average_drawdown > 0:
            metrics['sterling_ratio'] = annualized_return / drawdown_stats.average_drawdown
        else:
            metrics['sterling_ratio'] = 0.0
        
        # Burke Ratio (uses square root of sum of squared drawdowns)
        if drawdown_stats.underwater_curve is not None:
            squared_dds = (drawdown_stats.underwater_curve ** 2).sum()
            if squared_dds > 0:
                metrics['burke_ratio'] = annualized_return / np.sqrt(squared_dds)
            else:
                metrics['burke_ratio'] = 0.0
        
        # Pain Ratio
        if drawdown_stats.pain_index > 0:
            metrics['pain_ratio'] = annualized_return / drawdown_stats.pain_index
        else:
            metrics['pain_ratio'] = 0.0
        
        return metrics
    
    def plot_drawdown_analysis(self, 
                             price_series: pd.Series,
                             analysis_results: Dict[str, Any],
                             save_path: Optional[str] = None):
        """Create comprehensive drawdown visualization"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Price series with drawdown periods
        ax1 = axes[0, 0]
        ax1.plot(price_series.index, price_series.values, 'b-', linewidth=1)
        ax1.set_title('Price Series with Drawdown Periods')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        # Highlight major drawdown periods
        standard_stats = analysis_results.get('standard')
        if standard_stats and standard_stats.drawdown_periods:
            for period in standard_stats.drawdown_periods[:5]:  # Top 5 drawdowns
                if period.drawdown_percent > 0.05:  # >5% drawdown
                    ax1.axvspan(period.start_date, period.end_date, 
                              alpha=0.3, color='red', label='Drawdown' if period == standard_stats.drawdown_periods[0] else "")
        
        ax1.legend()
        
        # 2. Underwater curve
        ax2 = axes[0, 1]
        if standard_stats and standard_stats.underwater_curve is not None:
            underwater = standard_stats.underwater_curve * 100  # Convert to percentage
            ax2.fill_between(underwater.index, underwater.values, 0, 
                           where=(underwater.values < 0), color='red', alpha=0.7)
            ax2.plot(underwater.index, underwater.values, 'r-', linewidth=1)
            ax2.set_title('Underwater Curve (%)')
            ax2.set_ylabel('Drawdown %')
            ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown duration histogram
        ax3 = axes[1, 0]
        if standard_stats and standard_stats.drawdown_periods:
            durations = [period.duration_days for period in standard_stats.drawdown_periods]
            ax3.hist(durations, bins=min(20, len(durations)), alpha=0.7, color='orange')
            ax3.set_title('Drawdown Duration Distribution')
            ax3.set_xlabel('Duration (days)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
        
        # 4. Recovery time analysis
        ax4 = axes[1, 1]
        if standard_stats and standard_stats.drawdown_periods:
            recovery_times = [period.recovery_days for period in standard_stats.drawdown_periods 
                            if period.recovery_days is not None]
            
            if recovery_times:
                ax4.hist(recovery_times, bins=min(15, len(recovery_times)), 
                        alpha=0.7, color='green')
                ax4.set_title('Recovery Time Distribution')
                ax4.set_xlabel('Recovery Time (days)')
                ax4.set_ylabel('Frequency')
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'No Recovery Data', 
                        transform=ax4.transAxes, ha='center', va='center')
                ax4.set_title('Recovery Time Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Drawdown analysis plot saved to {save_path}")
        
        plt.show()
    
    def generate_drawdown_report(self, 
                               analysis_results: Dict[str, Any],
                               price_series: pd.Series) -> str:
        """Generate comprehensive drawdown analysis report"""
        
        report = []
        report.append("="*70)
        report.append("COMPREHENSIVE DRAWDOWN ANALYSIS REPORT")
        report.append("="*70)
        
        # Data summary
        if len(price_series) > 0:
            report.append(f"\nDATA SUMMARY:")
            report.append(f"  Period: {price_series.index[0]} to {price_series.index[-1]}")
            report.append(f"  Observations: {len(price_series)}")
            report.append(f"  Initial Value: {price_series.iloc[0]:.2f}")
            report.append(f"  Final Value: {price_series.iloc[-1]:.2f}")
            report.append(f"  Total Return: {(price_series.iloc[-1] / price_series.iloc[0] - 1):.2%}")
        
        # Standard drawdown analysis
        standard_stats = analysis_results.get('standard')
        if standard_stats:
            report.append(f"\nSTANDARD DRAWDOWN ANALYSIS:")
            report.append(f"  Maximum Drawdown: {standard_stats.max_drawdown:.2%}")
            report.append(f"  Average Drawdown: {standard_stats.average_drawdown:.2%}")
            report.append(f"  Max DD Duration: {standard_stats.max_drawdown_duration} days")
            report.append(f"  Average Recovery Time: {standard_stats.average_recovery_time:.1f} days")
            report.append(f"  Drawdown Frequency: {standard_stats.drawdown_frequency:.2f} periods/year")
            report.append(f"  Lake Ratio: {standard_stats.lake_ratio:.2%}")
            report.append(f"  Ulcer Index: {standard_stats.ulcer_index:.4f}")
            report.append(f"  Pain Index: {standard_stats.pain_index:.4f}")
            report.append(f"  Gain to Pain Ratio: {standard_stats.gain_to_pain_ratio:.2f}")
        
        # Conditional drawdown analysis
        conditional_stats = analysis_results.get('conditional')
        if conditional_stats:
            report.append(f"\nCONDITIONAL DRAWDOWN ANALYSIS:")
            report.append(f"  95% Conditional DD: {conditional_stats.conditional_drawdown_95:.2%}")
            report.append(f"  99% Conditional DD: {conditional_stats.conditional_drawdown_99:.2%}")
        
        # Risk-adjusted metrics
        risk_adjusted = analysis_results.get('risk_adjusted', {})
        if risk_adjusted:
            report.append(f"\nRISK-ADJUSTED METRICS:")
            for metric, value in risk_adjusted.items():
                metric_name = metric.replace('_', ' ').title()
                report.append(f"  {metric_name}: {value:.3f}")
        
        # Top drawdown periods
        if standard_stats and standard_stats.drawdown_periods:
            # Sort by magnitude
            top_periods = sorted(standard_stats.drawdown_periods, 
                               key=lambda x: x.drawdown_percent, reverse=True)[:5]
            
            report.append(f"\nTOP DRAWDOWN PERIODS:")
            report.append(f"  {'Start':<12} {'End':<12} {'Duration':<10} {'Drawdown':<10} {'Recovery':<10}")
            report.append("  " + "-"*60)
            
            for period in top_periods:
                start_str = period.start_date.strftime('%Y-%m-%d')
                end_str = period.end_date.strftime('%Y-%m-%d')
                duration_str = f"{period.duration_days}d"
                drawdown_str = f"{period.drawdown_percent:.1%}"
                recovery_str = f"{period.recovery_days}d" if period.recovery_days else "Ongoing"
                
                report.append(f"  {start_str:<12} {end_str:<12} {duration_str:<10} "
                             f"{drawdown_str:<10} {recovery_str:<10}")
        
        # Method comparison
        comparison = analysis_results.get('comparison', {})
        if 'max_drawdown_comparison' in comparison:
            report.append(f"\nMETHOD COMPARISON:")
            for method, max_dd in comparison['max_drawdown_comparison'].items():
                report.append(f"  {method.title()} Max DD: {max_dd:.2%}")
            
            if 'max_drawdown_consensus' in comparison:
                report.append(f"  Consensus Max DD: {comparison['max_drawdown_consensus']:.2%}")
        
        return "\n".join(report)
