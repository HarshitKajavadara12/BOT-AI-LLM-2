"""
Strategy Multiplexer (Capital Allocator)
Phase 5A Component

Responsible for:
1. Managing the lifecycle of multiple strategies.
2. Routing market data to registered strategies.
3. Aggregating signals (but NOT executing them - that's the Execution Engine's job).
4. Enforcing capital limits per strategy.
"""

from typing import Dict, List, Any
import logging

from core.strategy_interface import IStrategy, StrategySignal, StrategyState
from core.shadow_tracker import ShadowTracker

class StrategyMultiplexer:
    def __init__(self):
        self.strategies: Dict[str, IStrategy] = {}
        self.shadow_strategies: Dict[str, IStrategy] = {} # Phase 5C
        self.allocations: Dict[str, float] = {}
        self.shadow_tracker = ShadowTracker()
        self.logger = logging.getLogger(__name__)

    def register_strategy(self, strategy: IStrategy, initial_capital: float = 0.0, is_shadow: bool = False):
        """Register a new strategy with the multiplexer."""
        if strategy.strategy_id in self.strategies or strategy.strategy_id in self.shadow_strategies:
            raise ValueError(f"Strategy {strategy.strategy_id} already registered")
        
        if is_shadow:
            self.shadow_strategies[strategy.strategy_id] = strategy
            self.shadow_tracker.initialize_strategy(strategy.strategy_id)
            self.logger.info(f"Registered SHADOW strategy: {strategy.strategy_id}")
        else:
            self.strategies[strategy.strategy_id] = strategy
            self.set_allocation(strategy.strategy_id, initial_capital)
            self.logger.info(f"Registered LIVE strategy: {strategy.strategy_id} with capital {initial_capital}")

    def set_allocation(self, strategy_id: str, amount: float):
        """Update capital allocation for a specific strategy."""
        if strategy_id in self.shadow_strategies:
            return # Shadow strategies don't get real capital allocation updates here
            
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        self.allocations[strategy_id] = amount
        self.strategies[strategy_id].set_capital_allocation(amount)
        self.logger.info(f"Updated allocation for {strategy_id}: {amount}")

    def process_market_data(self, data: Dict[str, Any]) -> List[StrategySignal]:
        """
        Route data to all active strategies (Live and Shadow).
        Returns ONLY signals from LIVE strategies.
        Shadow signals are processed internally by the ShadowTracker.
        """
        all_signals = []
        
        # 1. Process LIVE Strategies
        for strategy_id, strategy in self.strategies.items():
            try:
                signals = strategy.on_market_data(data)
                for signal in signals:
                    if signal.strategy_id != strategy_id:
                        continue
                    all_signals.append(signal)
            except Exception as e:
                print(f"DEBUG ERROR in strategy {strategy_id}: {str(e)}")
                self.logger.error(f"Error in strategy {strategy_id}: {str(e)}")
                continue
        
        # 2. Process SHADOW Strategies (Phase 5C)
        price = float(data.get('close', data.get('price', 0.0)))
        symbol = data.get('symbol', 'UNKNOWN')
        
        # Update tracker with latest price
        if price > 0:
            self.shadow_tracker.update_market_price(symbol, price)

        for strategy_id, strategy in self.shadow_strategies.items():
            try:
                # Shadow strategies also need "capital" to function internally if they check it
                # We can mock it or set it once.
                strategy.set_capital_allocation(100000.0) # Virtual capital
                
                signals = strategy.on_market_data(data)
                for signal in signals:
                    # Send to Shadow Tracker instead of returning
                    self.shadow_tracker.process_signal(signal, price)
                    
            except Exception as e:
                self.logger.error(f"Error in SHADOW strategy {strategy_id}: {str(e)}")
                continue

        return all_signals

    def get_system_state(self) -> Dict[str, StrategyState]:
        """Get the health/state of all strategies."""
        states = {}
        for strategy_id, strategy in self.strategies.items():
            try:
                states[strategy_id] = strategy.get_state()
            except Exception as e:
                self.logger.error(f"Failed to get state for {strategy_id}: {str(e)}")
        return states

    def rebalance_portfolio(self, total_capital: float):
        """
        Re-calculate and apply capital allocations across all strategies.
        """
        from core.capital_allocator import CapitalAllocator
        
        allocator = CapitalAllocator(total_capital)
        states = self.get_system_state()
        new_allocations = allocator.update_allocations(states)
        
        for strategy_id, amount in new_allocations.items():
            self.set_allocation(strategy_id, amount)
            
        self.logger.info(f"Rebalanced portfolio. Total Capital: {total_capital}")
    
    # ------------------------------------------------------------------
    # Shadow → Live promotion
    # ------------------------------------------------------------------
    def evaluate_promotions(
        self,
        min_return: float = 0.02,
        min_trades: int = 20,
        max_promote: int = 1,
    ) -> List[str]:
        """
        Compare shadow strategy performance against live.
        Promote the best-performing shadow to live if it meets thresholds.
        
        Args:
            min_return: Minimum cumulative return to consider promotion.
            min_trades: Minimum number of shadow trades.
            max_promote: Maximum strategies to promote per call.
        
        Returns:
            List of promoted strategy IDs.
        """
        promoted: List[str] = []
        candidates: List[tuple] = []
        
        for sid in list(self.shadow_strategies.keys()):
            try:
                perf = self.shadow_tracker.get_performance(sid)
                portfolio = self.shadow_tracker.portfolios.get(sid)
                n_trades = len(portfolio.trade_history) if portfolio else 0
                
                if perf >= min_return and n_trades >= min_trades:
                    candidates.append((sid, perf, n_trades))
            except Exception:
                continue
        
        # Sort by performance descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        for sid, perf, n_trades in candidates[:max_promote]:
            self._promote_shadow(sid)
            promoted.append(sid)
            self.logger.info(
                f"PROMOTED shadow→live: {sid} "
                f"(return={perf:+.2%}, trades={n_trades})"
            )
        
        return promoted
    
    def _promote_shadow(self, strategy_id: str):
        """Move a strategy from shadow to live."""
        strategy = self.shadow_strategies.pop(strategy_id)
        self.strategies[strategy_id] = strategy
        # Give it a small initial allocation (real capital will come via rebalance)
        self.allocations[strategy_id] = 0.0
        self.logger.info(f"Strategy {strategy_id} moved from shadow → live pool")
    
    def get_shadow_rankings(self) -> List[Dict[str, Any]]:
        """Return shadow strategies ranked by performance."""
        rankings = []
        for sid in self.shadow_strategies:
            try:
                perf = self.shadow_tracker.get_performance(sid)
                portfolio = self.shadow_tracker.portfolios.get(sid)
                rankings.append({
                    'strategy_id': sid,
                    'return': perf,
                    'equity': portfolio.total_equity if portfolio else 0,
                    'trades': len(portfolio.trade_history) if portfolio else 0,
                })
            except Exception:
                rankings.append({'strategy_id': sid, 'return': 0.0, 'equity': 0, 'trades': 0})
        rankings.sort(key=lambda x: x['return'], reverse=True)
        return rankings
