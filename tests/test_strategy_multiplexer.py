import unittest
import logging
from datetime import datetime
from typing import List, Dict, Any
from core.strategy_interface import IStrategy, StrategySignal, StrategyState
from core.strategy_multiplexer import StrategyMultiplexer

class MockStrategy(IStrategy):
    def __init__(self, strategy_id: str):
        self._id = strategy_id
        self.capital = 0.0
        self.signals_to_emit = []

    @property
    def strategy_id(self) -> str:
        return self._id

    def on_market_data(self, data: Dict[str, Any]) -> List[StrategySignal]:
        # Return pre-configured signals for testing
        signals = self.signals_to_emit
        self.signals_to_emit = [] # Clear after emitting
        return signals

    def get_state(self) -> StrategyState:
        return StrategyState(
            strategy_id=self._id,
            is_active=True,
            current_drawdown=0.0,
            current_exposure=0.0,
            open_positions=0,
            last_update=datetime.now()
        )

    def set_capital_allocation(self, amount: float):
        self.capital = amount

class TestStrategyMultiplexer(unittest.TestCase):
    def setUp(self):
        self.multiplexer = StrategyMultiplexer()
        self.strategy_a = MockStrategy("STRAT_A")
        self.strategy_b = MockStrategy("STRAT_B")

    def test_registration_and_allocation(self):
        """Test that strategies are registered and receive capital."""
        self.multiplexer.register_strategy(self.strategy_a, initial_capital=10000.0)
        
        self.assertIn("STRAT_A", self.multiplexer.strategies)
        self.assertEqual(self.strategy_a.capital, 10000.0)
        
        # Update allocation
        self.multiplexer.set_allocation("STRAT_A", 20000.0)
        self.assertEqual(self.strategy_a.capital, 20000.0)

    def test_signal_aggregation(self):
        """Test that signals from multiple strategies are collected."""
        self.multiplexer.register_strategy(self.strategy_a)
        self.multiplexer.register_strategy(self.strategy_b)

        # Setup mock signals
        sig_a = StrategySignal("STRAT_A", "BTCUSDT", "BUY", 0.8, datetime.now())
        sig_b = StrategySignal("STRAT_B", "ETHUSDT", "SELL", 0.5, datetime.now())
        
        self.strategy_a.signals_to_emit = [sig_a]
        self.strategy_b.signals_to_emit = [sig_b]

        # Process dummy data
        results = self.multiplexer.process_market_data({"price": 100})

        self.assertEqual(len(results), 2)
        self.assertIn(sig_a, results)
        self.assertIn(sig_b, results)

    def test_error_isolation(self):
        """Test that one failing strategy doesn't crash the multiplexer."""
        class FailingStrategy(MockStrategy):
            def on_market_data(self, data):
                raise RuntimeError("Strategy crashed!")

        failing_strat = FailingStrategy("FAIL_STRAT")
        self.multiplexer.register_strategy(failing_strat)
        self.multiplexer.register_strategy(self.strategy_a)

        self.strategy_a.signals_to_emit = [StrategySignal("STRAT_A", "BTC", "BUY", 1.0, datetime.now())]

        # Suppress logging for this test to avoid confusing output
        logging.disable(logging.CRITICAL)
        try:
            # Should not raise exception
            results = self.multiplexer.process_market_data({})
        finally:
            logging.disable(logging.NOTSET)
        
        # Should still get result from the good strategy
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].strategy_id, "STRAT_A")

    def test_rebalancing(self):
        """Test that capital is distributed equally among active strategies."""
        self.multiplexer.register_strategy(self.strategy_a)
        self.multiplexer.register_strategy(self.strategy_b)
        
        # Initial state: 0 capital
        self.assertEqual(self.strategy_a.capital, 0.0)
        self.assertEqual(self.strategy_b.capital, 0.0)
        
        # Rebalance with 100k
        self.multiplexer.rebalance_portfolio(100000.0)
        
        # Should be 50k each
        self.assertEqual(self.strategy_a.capital, 50000.0)
        self.assertEqual(self.strategy_b.capital, 50000.0)

if __name__ == '__main__':
    unittest.main()
