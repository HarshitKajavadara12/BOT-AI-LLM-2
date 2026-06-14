"""
Momentum Strategy Implementation
Phase 5A Component

A reference implementation of IStrategy using a simple Moving Average Crossover logic.
This proves the end-to-end flow of the Strategy Multiplexer.
"""

from typing import Dict, Any, List, Deque
from collections import deque
from datetime import datetime
import numpy as np

from core.strategy_interface import IStrategy, StrategySignal, StrategyState

class MomentumStrategy(IStrategy):
    def __init__(self, strategy_id: str, fast_window: int = 10, slow_window: int = 50):
        self._id = strategy_id
        self.fast_window = fast_window
        self.slow_window = slow_window
        
        # State
        self.prices: Deque[float] = deque(maxlen=slow_window + 1)
        self.capital = 0.0
        self.position = 0.0  # Current position size (mock)
        self.last_signal = "FLAT"
        self.active = True

    @property
    def strategy_id(self) -> str:
        return self._id

    def set_capital_allocation(self, amount: float):
        self.capital = amount

    def on_market_data(self, data: Dict[str, Any]) -> List[StrategySignal]:
        """
        Process market data:
        1. Update price history.
        2. Calculate indicators (Fast/Slow MA).
        3. Generate signal if crossover detected.
        """
        if not self.active or self.capital <= 0:
            return []

        # Extract price (assuming standard format, e.g., from Binance)
        # Supporting both 'close' and 'price' keys for flexibility
        price = float(data.get('close', data.get('price', 0.0)))
        symbol = data.get('symbol', 'UNKNOWN')
        timestamp = data.get('timestamp', datetime.now())

        if price <= 0:
            return []

        self.prices.append(price)

        # Need enough data for the slow window
        if len(self.prices) < self.slow_window:
            return []

        # Calculate Moving Averages
        prices_array = np.array(self.prices)
        fast_ma = np.mean(prices_array[-self.fast_window:])
        slow_ma = np.mean(prices_array[-self.slow_window:])

        signal_type = "HOLD"
        strength = 0.0

        # Logic: Crossover
        # If Fast crosses above Slow -> BUY
        # If Fast crosses below Slow -> SELL
        
        # We need previous values to detect the *moment* of crossover
        # For simplicity in this reference impl, we just check current state vs last signal
        
        current_trend = "BULL" if fast_ma > slow_ma else "BEAR"
        
        if current_trend == "BULL" and self.last_signal != "BUY":
            signal_type = "BUY"
            strength = (fast_ma - slow_ma) / slow_ma * 100 # Normalized strength
            self.last_signal = "BUY"
            
        elif current_trend == "BEAR" and self.last_signal != "SELL":
            signal_type = "SELL"
            strength = (slow_ma - fast_ma) / slow_ma * 100
            self.last_signal = "SELL"

        if signal_type in ["BUY", "SELL"]:
            # Cap strength at 1.0
            strength = min(abs(strength), 1.0)
            
            return [StrategySignal(
                strategy_id=self._id,
                symbol=symbol,
                signal_type=signal_type,
                signal_strength=strength,
                timestamp=timestamp,
                metadata={
                    "fast_ma": fast_ma,
                    "slow_ma": slow_ma,
                    "price": price
                }
            )]
            
        return []

    def get_state(self) -> StrategyState:
        return StrategyState(
            strategy_id=self._id,
            is_active=self.active,
            current_drawdown=0.0, # TODO: Implement PnL tracking
            current_exposure=abs(self.position),
            open_positions=1 if self.position != 0 else 0,
            last_update=datetime.now()
        )
