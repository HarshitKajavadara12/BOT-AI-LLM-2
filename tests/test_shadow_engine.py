import unittest
from datetime import datetime
from core.strategy_multiplexer import StrategyMultiplexer
from core.strategies.momentum_strategy import MomentumStrategy

class TestShadowEngine(unittest.TestCase):
    def setUp(self):
        self.multiplexer = StrategyMultiplexer()
        self.live_strat = MomentumStrategy("LIVE_MOM", fast_window=3, slow_window=5)
        self.shadow_strat = MomentumStrategy("SHADOW_MOM", fast_window=3, slow_window=5)

    def test_shadow_registration(self):
        """Test that shadow strategies are registered separately."""
        self.multiplexer.register_strategy(self.live_strat, initial_capital=10000.0, is_shadow=False)
        self.multiplexer.register_strategy(self.shadow_strat, is_shadow=True)
        
        self.assertIn("LIVE_MOM", self.multiplexer.strategies)
        self.assertIn("SHADOW_MOM", self.multiplexer.shadow_strategies)
        self.assertNotIn("SHADOW_MOM", self.multiplexer.strategies)

    def test_shadow_execution_isolation(self):
        """
        Test that Shadow strategies:
        1. Receive data.
        2. Generate signals (internally).
        3. Do NOT return signals to the main execution loop.
        4. DO update the ShadowTracker.
        """
        self.multiplexer.register_strategy(self.live_strat, initial_capital=10000.0, is_shadow=False)
        self.multiplexer.register_strategy(self.shadow_strat, is_shadow=True)
        
        # Setup data to trigger BUY signal
        # Prices: 100, 102, 104, 106, 108 (Bullish)
        prices = [100, 102, 104, 106, 108]
        
        # Feed data
        # We need enough data points. Window is 5.
        # Feed 10 points to be safe and establish trend.
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        
        signals = []
        for p in prices:
            data = {"symbol": "BTC", "price": p, "timestamp": datetime.now()}
            # Accumulate signals from the last step
            batch = self.multiplexer.process_market_data(data)
            signals.extend(batch)
            
        # Check Live Signals
        live_signals = [s for s in signals if s.strategy_id == "LIVE_MOM"]
        shadow_signals_leaked = [s for s in signals if s.strategy_id == "SHADOW_MOM"]
        
        self.assertTrue(len(live_signals) > 0, "Live strategy should emit signal")
        self.assertEqual(len(shadow_signals_leaked), 0, "Shadow strategy signal leaked to execution!")
        
        # Check Shadow Tracker
        # Shadow strategy should have "bought" in the tracker
        portfolio = self.multiplexer.shadow_tracker.portfolios["SHADOW_MOM"]
        self.assertIn("BTC", portfolio.positions, "Shadow tracker should have executed the trade virtually")
        self.assertTrue(portfolio.positions["BTC"].quantity > 0)

if __name__ == '__main__':
    unittest.main()
