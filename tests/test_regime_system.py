import unittest
import numpy as np
from core.regime_detector import RegimeDetector, MarketRegime
from core.capital_allocator import CapitalAllocator
from core.strategy_interface import StrategyState
from datetime import datetime

class TestRegimeSystem(unittest.TestCase):
    def setUp(self):
        self.detector = RegimeDetector(window_size=5, vol_threshold=0.01)
        self.allocator = CapitalAllocator(total_capital=100000.0)

    def test_regime_detection_high_vol(self):
        """Test that high volatility is detected."""
        # Feed stable prices first
        prices = [100, 100.1, 100.2, 100.1, 100]
        for p in prices:
            signal = self.detector.on_market_data(p)
        
        self.assertEqual(signal.regime, MarketRegime.NEUTRAL)
        
        # Feed volatile prices
        # 100 -> 105 (+5%) -> 95 (-10%) -> 105 (+10%)
        vol_prices = [105, 95, 105, 95, 105]
        for p in vol_prices:
            signal = self.detector.on_market_data(p)
            
        self.assertEqual(signal.regime, MarketRegime.HIGH_VOLATILITY)

    def test_allocator_response_to_regime(self):
        """Test that Allocator reduces capital in dangerous regimes."""
        # Setup dummy strategy state
        strat_state = StrategyState("STRAT_A", True, 0.0, 0.0, 0, datetime.now())
        states = {"STRAT_A": strat_state}
        
        # 1. Neutral Regime -> 100% Allocation
        self.allocator.set_regime(self.detector.on_market_data(100)) # Reset/Init
        # Force Neutral manually for test clarity
        from core.regime_detector import RegimeSignal
        self.allocator.set_regime(RegimeSignal(MarketRegime.NEUTRAL, 1.0, 0.0))
        
        allocs = self.allocator.update_allocations(states)
        self.assertEqual(allocs["STRAT_A"], 100000.0)
        
        # 2. High Volatility -> 50% Allocation
        self.allocator.set_regime(RegimeSignal(MarketRegime.HIGH_VOLATILITY, 1.0, 0.05))
        allocs = self.allocator.update_allocations(states)
        self.assertEqual(allocs["STRAT_A"], 50000.0)
        
        # 3. Bear Market -> 80% Allocation
        self.allocator.set_regime(RegimeSignal(MarketRegime.BEAR, 1.0, 0.01))
        allocs = self.allocator.update_allocations(states)
        self.assertEqual(allocs["STRAT_A"], 80000.0)

if __name__ == '__main__':
    unittest.main()
