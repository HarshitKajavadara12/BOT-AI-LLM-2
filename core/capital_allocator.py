"""
Capital Allocator
Phase 5A Component — UPGRADED to Performance-Based Allocation

Decides how much capital each strategy gets based on:
1. Performance (Sharpe, Drawdown) — NOT equal-weight anymore
2. Market Regime (from Phase 5B)
3. Hard Constraints (Max allocation per strategy, min diversification)
4. Risk Parity (allocate inversely proportional to volatility)

Allocation Methods:
- EQUAL: Equal weight (baseline)
- PERFORMANCE: Weight by risk-adjusted returns (Sharpe)
- RISK_PARITY: Weight inversely by volatility
- HYBRID: Performance * Risk Parity (recommended)
"""

import numpy as np
from typing import Dict, Optional
from enum import Enum
from collections import deque
import logging

from core.strategy_interface import StrategyState
from core.regime_detector import MarketRegime, RegimeSignal

logger = logging.getLogger("CapitalAllocator")


class AllocationMethod(Enum):
    EQUAL = "EQUAL"
    PERFORMANCE = "PERFORMANCE"
    RISK_PARITY = "RISK_PARITY"
    HYBRID = "HYBRID"


class CapitalAllocator:
    def __init__(
        self,
        total_capital: float,
        method: AllocationMethod = AllocationMethod.HYBRID,
        max_single_allocation: float = 0.40,     # No strategy gets > 40%
        min_single_allocation: float = 0.05,     # Every active strategy gets at least 5%
        rebalance_threshold: float = 0.10,       # Rebalance when weights drift > 10%
    ):
        self.total_capital = total_capital
        self.method = method
        self.max_single_allocation = max_single_allocation
        self.min_single_allocation = min_single_allocation
        self.rebalance_threshold = rebalance_threshold
        
        self.strategy_weights: Dict[str, float] = {}
        self.current_regime = MarketRegime.NEUTRAL
        
        # Performance tracking per strategy
        self.strategy_returns: Dict[str, deque] = {}   # Rolling returns
        self.strategy_volatility: Dict[str, float] = {}
        self.strategy_sharpe: Dict[str, float] = {}
        self.strategy_drawdown: Dict[str, float] = {}
        
        logger.info(f"CapitalAllocator initialized: method={method.value}, "
                     f"capital=${total_capital:,.2f}, "
                     f"max_alloc={max_single_allocation:.0%}")

    def set_regime(self, regime_signal: RegimeSignal):
        """Update the allocator's view of the market regime."""
        old_regime = self.current_regime
        self.current_regime = regime_signal.regime
        if old_regime != self.current_regime:
            logger.info(f"Regime changed: {old_regime.value} → {self.current_regime.value}")

    def record_strategy_return(self, strategy_id: str, ret: float):
        """
        Record a strategy's return for performance tracking.
        Call this after each period to update performance-based weights.
        """
        if strategy_id not in self.strategy_returns:
            self.strategy_returns[strategy_id] = deque(maxlen=200)
        self.strategy_returns[strategy_id].append(ret)
        
        # Update derived metrics
        returns = np.array(self.strategy_returns[strategy_id])
        if len(returns) >= 5:
            self.strategy_volatility[strategy_id] = float(np.std(returns) * np.sqrt(252))
            avg_return = np.mean(returns) * 252
            vol = self.strategy_volatility[strategy_id]
            self.strategy_sharpe[strategy_id] = avg_return / vol if vol > 0 else 0.0
            
            # Max drawdown
            cumulative = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative)
            drawdowns = (peak - cumulative) / peak
            self.strategy_drawdown[strategy_id] = float(np.max(drawdowns))

    def _get_regime_multiplier(self) -> float:
        """
        Determine capital multiplier based on regime.
        HIGH_VOLATILITY -> Cut exposure significantly.
        BEAR -> Reduce exposure (defensive).
        BULL/NEUTRAL -> Full exposure.
        """
        if self.current_regime == MarketRegime.HIGH_VOLATILITY:
            return 0.3   # Cut capital by 70% in extreme vol
        elif self.current_regime == MarketRegime.BEAR:
            return 0.6   # 40% reduction in bear
        elif self.current_regime == MarketRegime.BULL:
            return 1.0   # Full exposure in bull
        return 0.85      # Slight caution in neutral

    def _compute_equal_weights(self, active: list) -> Dict[str, float]:
        """Equal weight allocation."""
        w = 1.0 / len(active)
        return {s: w for s in active}

    def _compute_performance_weights(self, active: list) -> Dict[str, float]:
        """
        Weight by risk-adjusted performance (Sharpe ratio).
        Strategies with higher Sharpe get more capital.
        Strategies with negative Sharpe get minimum allocation.
        """
        sharpes = {}
        for s in active:
            sharpe = self.strategy_sharpe.get(s, 0.0)
            # Penalize strategies in drawdown
            dd = self.strategy_drawdown.get(s, 0.0)
            adjusted = sharpe * (1.0 - dd)  # Reduce weight if in drawdown
            sharpes[s] = max(adjusted, 0.0)  # Floor at 0
        
        total = sum(sharpes.values())
        if total > 0:
            return {s: sharpes[s] / total for s in active}
        else:
            # All strategies have non-positive Sharpe → fall back to equal
            return self._compute_equal_weights(active)

    def _compute_risk_parity_weights(self, active: list) -> Dict[str, float]:
        """
        Risk parity: allocate inversely proportional to volatility.
        Lower vol strategies get MORE capital (to equalize risk contribution).
        """
        inv_vols = {}
        for s in active:
            vol = self.strategy_volatility.get(s, 0.0)
            inv_vols[s] = 1.0 / max(vol, 0.01)  # Cap at 100x
        
        total = sum(inv_vols.values())
        if total > 0:
            return {s: inv_vols[s] / total for s in active}
        else:
            return self._compute_equal_weights(active)

    def _compute_hybrid_weights(self, active: list) -> Dict[str, float]:
        """
        Hybrid: combine performance and risk parity.
        High Sharpe + Low Vol = highest allocation.
        """
        perf_weights = self._compute_performance_weights(active)
        risk_weights = self._compute_risk_parity_weights(active)
        
        # 60% performance, 40% risk parity
        combined = {}
        for s in active:
            combined[s] = 0.6 * perf_weights.get(s, 0) + 0.4 * risk_weights.get(s, 0)
        
        total = sum(combined.values())
        if total > 0:
            return {s: combined[s] / total for s in active}
        else:
            return self._compute_equal_weights(active)

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Apply hard constraints:
        - Max per strategy
        - Min per strategy  
        - Renormalize
        """
        # Apply max cap
        for s in weights:
            if weights[s] > self.max_single_allocation:
                weights[s] = self.max_single_allocation
        
        # Apply min floor
        for s in weights:
            if weights[s] < self.min_single_allocation:
                weights[s] = self.min_single_allocation
        
        # Renormalize so sum <= 1.0
        total = sum(weights.values())
        if total > 1.0:
            for s in weights:
                weights[s] /= total
        
        return weights

    def update_allocations(self, strategy_states: Dict[str, StrategyState]) -> Dict[str, float]:
        """
        Calculate new capital allocations for each strategy.
        Now uses performance-based weighting instead of equal-weight.
        """
        active_strategies = [s_id for s_id, state in strategy_states.items() if state.is_active]
        
        if not active_strategies:
            return {}
        
        # 1. Compute raw weights based on method
        if self.method == AllocationMethod.EQUAL:
            raw_weights = self._compute_equal_weights(active_strategies)
        elif self.method == AllocationMethod.PERFORMANCE:
            raw_weights = self._compute_performance_weights(active_strategies)
        elif self.method == AllocationMethod.RISK_PARITY:
            raw_weights = self._compute_risk_parity_weights(active_strategies)
        else:  # HYBRID
            raw_weights = self._compute_hybrid_weights(active_strategies)
        
        # 2. Apply constraints
        constrained_weights = self._apply_constraints(raw_weights)
        
        # 3. Apply regime multiplier (overall exposure reduction)
        regime_multiplier = self._get_regime_multiplier()
        
        # 4. Calculate final dollar allocations
        allocations = {}
        for s_id in active_strategies:
            weight = constrained_weights.get(s_id, 0.0) * regime_multiplier
            amount = self.total_capital * weight
            allocations[s_id] = amount
            self.strategy_weights[s_id] = weight
        
        # Log allocation changes
        for s_id, amount in allocations.items():
            sharpe = self.strategy_sharpe.get(s_id, 0.0)
            logger.debug(f"  {s_id}: ${amount:,.2f} "
                        f"(weight={self.strategy_weights[s_id]:.1%}, "
                        f"sharpe={sharpe:.2f}, regime={self.current_regime.value})")
        
        return allocations
    
    def get_allocation_summary(self) -> Dict:
        """Return a summary of current allocations for monitoring."""
        return {
            'method': self.method.value,
            'regime': self.current_regime.value,
            'regime_multiplier': self._get_regime_multiplier(),
            'total_capital': self.total_capital,
            'weights': dict(self.strategy_weights),
            'sharpes': dict(self.strategy_sharpe),
            'drawdowns': dict(self.strategy_drawdown),
            'volatilities': dict(self.strategy_volatility),
            'cash_held': self.total_capital * (1.0 - sum(self.strategy_weights.values())),
        }
