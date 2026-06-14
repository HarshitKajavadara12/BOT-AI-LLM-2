"""
Strategy Interface Definition
Phase 5A Component

This module defines the strict contract that ALL strategies must adhere to.
This allows the Capital Allocator to manage them uniformly without knowing their internal logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StrategySignal:
    """Standardized output from any strategy."""
    strategy_id: str
    symbol: str
    signal_type: str  # BUY, SELL, HOLD, FLAT
    signal_strength: float  # 0.0 to 1.0
    timestamp: datetime
    metadata: Dict[str, Any] = None

@dataclass
class StrategyState:
    """Snapshot of strategy health for the Allocator."""
    strategy_id: str
    is_active: bool
    current_drawdown: float
    current_exposure: float
    open_positions: int
    last_update: datetime

class IStrategy(ABC):
    """
    The Interface that all strategies must implement.
    """
    
    @property
    @abstractmethod
    def strategy_id(self) -> str:
        pass
        
    @abstractmethod
    def on_market_data(self, data: Dict[str, Any]) -> List[StrategySignal]:
        """
        Process new market data and generate signals.
        Must be stateless or manage its own state internally.
        """
        pass
        
    @abstractmethod
    def get_state(self) -> StrategyState:
        """
        Report current health/state to the Capital Allocator.
        """
        pass
        
    @abstractmethod
    def set_capital_allocation(self, amount: float):
        """
        Receive capital allocation from the Allocator.
        The strategy must respect this limit.
        """
        pass
