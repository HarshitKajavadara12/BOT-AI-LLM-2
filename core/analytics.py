"""
Unified Analytics Engine
Phase 4D Component

This module serves as the SINGLE SOURCE OF TRUTH for all financial metrics.
It is used by:
1. The Live Trading Engine (for dashboards and reporting)
2. The Research Notebooks (for backtesting and analysis)

This ensures "Research <-> Live Parity". If it works in the notebook,
it works in production, because it is the SAME code.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float

class AnalyticsEngine:
    """
    Standardized calculation engine for financial metrics.
    """
    
    @staticmethod
    def calculate_metrics(returns: Union[List[float], np.ndarray, pd.Series], 
                          risk_free_rate: float = 0.0) -> PerformanceMetrics:
        """
        Calculates standard performance metrics from a series of returns.
        
        Args:
            returns: Array-like of period returns (e.g., daily % change)
            risk_free_rate: Annualized risk-free rate (default 0.0)
            
        Returns:
            PerformanceMetrics object
        """
        # Convert to numpy array for consistency
        if isinstance(returns, list):
            rets = np.array(returns)
        elif isinstance(returns, pd.Series):
            rets = returns.values
        else:
            rets = returns
            
        # Handle empty or insufficient data
        if len(rets) < 2:
            return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
        # 1. Total Return
        total_return = np.prod(1 + rets) - 1
        
        # 2. Volatility (Annualized, assuming daily data)
        volatility = np.std(rets, ddof=1) * np.sqrt(252)
        
        # 3. Sharpe Ratio
        excess_returns = rets - (risk_free_rate / 252)
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns, ddof=1)
        
        if std_excess_return > 0:
            sharpe = (mean_excess_return / std_excess_return) * np.sqrt(252)
        else:
            sharpe = 0.0
            
        # 4. Sortino Ratio (Downside deviation)
        downside_returns = rets[rets < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns, ddof=1)
            if downside_std > 0:
                sortino = (mean_excess_return / downside_std) * np.sqrt(252)
            else:
                sortino = 0.0
        else:
            sortino = 0.0 # No downside volatility or insufficient data
            
        # 5. Max Drawdown
        cumulative = np.cumprod(1 + rets)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # 6. Win Rate
        wins = np.sum(rets > 0)
        win_rate = wins / len(rets)
        
        # 7. Profit Factor
        gross_profit = np.sum(rets[rets > 0])
        gross_loss = np.abs(np.sum(rets[rets < 0]))
        
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
            
        return PerformanceMetrics(
            total_return=float(total_return),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=float(max_drawdown),
            volatility=float(volatility),
            win_rate=float(win_rate),
            profit_factor=float(profit_factor)
        )

# Global instance
_analytics = AnalyticsEngine()

def get_analytics_engine() -> AnalyticsEngine:
    return _analytics
