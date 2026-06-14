"""
Shadow Tracker (Phase 5C)

Responsible for tracking the virtual performance of strategies running in "Shadow Mode".
These strategies receive real market data but their signals are not executed on the real market.
Instead, this tracker simulates their execution and calculates theoretical PnL.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from core.strategy_interface import StrategySignal

@dataclass
class ShadowPosition:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float

@dataclass
class ShadowPortfolio:
    strategy_id: str
    cash: float
    positions: Dict[str, ShadowPosition]
    total_equity: float
    initial_capital: float

class ShadowTracker:
    def __init__(self):
        self.portfolios: Dict[str, ShadowPortfolio] = {}

    def initialize_strategy(self, strategy_id: str, initial_capital: float = 100000.0):
        self.portfolios[strategy_id] = ShadowPortfolio(
            strategy_id=strategy_id,
            cash=initial_capital,
            positions={},
            total_equity=initial_capital,
            initial_capital=initial_capital
        )

    def update_market_price(self, symbol: str, price: float):
        """Update the value of open positions based on new market price."""
        for pid, portfolio in self.portfolios.items():
            if symbol in portfolio.positions:
                pos = portfolio.positions[symbol]
                pos.current_price = price
                # Simple PnL: (Current - Entry) * Qty
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
                
            # Recalculate total equity
            equity = portfolio.cash
            for pos in portfolio.positions.values():
                equity += (pos.current_price * pos.quantity) # Market Value
            portfolio.total_equity = equity

    def process_signal(self, signal: StrategySignal, price: float):
        """
        Simulate execution of a signal.
        Assumes instant fill at current price (no slippage/fees for simplicity in this phase).
        """
        if signal.strategy_id not in self.portfolios:
            return

        portfolio = self.portfolios[signal.strategy_id]
        
        # Simple logic: BUY = use 10% of cash, SELL = sell 100% of position
        # In a real system, this would use the signal strength and risk manager
        
        if signal.signal_type == "BUY":
            if signal.symbol in portfolio.positions:
                return # Already have position
                
            allocation = portfolio.cash * 0.10 # Use 10% of cash
            quantity = allocation / price
            
            portfolio.cash -= allocation
            portfolio.positions[signal.symbol] = ShadowPosition(
                symbol=signal.symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                unrealized_pnl=0.0
            )
            
        elif signal.signal_type == "SELL":
            if signal.symbol not in portfolio.positions:
                return # No position to sell
                
            pos = portfolio.positions.pop(signal.symbol)
            proceeds = pos.quantity * price
            portfolio.cash += proceeds

    def get_performance(self, strategy_id: str) -> float:
        """Return percentage return."""
        if strategy_id not in self.portfolios:
            return 0.0
        p = self.portfolios[strategy_id]
        return (p.total_equity - p.initial_capital) / p.initial_capital
