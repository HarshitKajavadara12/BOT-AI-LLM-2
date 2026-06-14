import unittest
from datetime import datetime
from core.strategies.momentum_strategy import MomentumStrategy

class TestMomentumStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = MomentumStrategy("MOM_01", fast_window=3, slow_window=5)
        self.strategy.set_capital_allocation(10000.0)

    def test_initialization(self):
        self.assertEqual(self.strategy.strategy_id, "MOM_01")
        self.assertEqual(self.strategy.capital, 10000.0)
        self.assertEqual(len(self.strategy.prices), 0)

    def test_insufficient_data(self):
        """Should not generate signals until enough data is collected."""
        data = {"symbol": "BTCUSDT", "price": 100.0, "timestamp": datetime.now()}
        signals = self.strategy.on_market_data(data)
        self.assertEqual(len(signals), 0)
        self.assertEqual(len(self.strategy.prices), 1)

    def test_buy_signal_generation(self):
        """Test that a Golden Cross generates a BUY signal."""
        # Feed data to establish a baseline (Slow > Fast -> Bearish)
        # Window is 5.
        prices = [100, 99, 98, 97, 96] # Mean=98
        for p in prices:
            self.strategy.on_market_data({"symbol": "BTC", "price": p})
            
        # Now pump the price to create a crossover
        # Fast window is 3.
        # New prices: 105, 110, 115
        # Fast MA (last 3): (105+110+115)/3 = 110
        # Slow MA (last 5): (98, 97, 96, 105, 110) -> 101.2 (approx)
        # Fast > Slow -> BUY
        
        signals = []
        signals.extend(self.strategy.on_market_data({"symbol": "BTC", "price": 105}))
        signals.extend(self.strategy.on_market_data({"symbol": "BTC", "price": 110}))
        
        # The crossover might happen here
        last_signals = self.strategy.on_market_data({"symbol": "BTC", "price": 115})
        signals.extend(last_signals)
        
        # We expect at least one BUY signal in this sequence
        buy_signals = [s for s in signals if s.signal_type == "BUY"]
        self.assertTrue(len(buy_signals) > 0)
        self.assertEqual(buy_signals[0].strategy_id, "MOM_01")
        self.assertEqual(buy_signals[0].symbol, "BTC")

    def test_sell_signal_generation(self):
        """Test that a Death Cross generates a SELL signal."""
        # 1. Establish a BULLISH trend first
        # Fast window 3, Slow 5
        # Prices increasing: 100, 102, 104, 106, 108
        # Fast(3): (104+106+108)/3 = 106
        # Slow(5): 104
        # Trend = BULL
        
        setup_prices = [100, 102, 104, 106, 108]
        for p in setup_prices:
            self.strategy.on_market_data({"symbol": "BTC", "price": p})
            
        # Verify we are in BULL mode (or at least last signal was BUY)
        # The strategy might have emitted BUY during setup
        self.strategy.last_signal = "BUY" 
        
        # 2. Crash the price to cause Death Cross
        # New prices: 90, 80, 70
        # After 90: Prices [102, 104, 106, 108, 90] (assuming maxlen=6, but window is 5)
        # Actually maxlen is 6.
        # Prices in deque: 100, 102, 104, 106, 108.
        # Add 90. Deque: 100, 102, 104, 106, 108, 90.
        # Fast(3): (106+108+90)/3 = 101.33
        # Slow(5): (102+104+106+108+90)/5 = 102
        # Fast < Slow -> BEAR -> SELL Signal
        
        signals = []
        signals.extend(self.strategy.on_market_data({"symbol": "BTC", "price": 90}))
        
        sell_signals = [s for s in signals if s.signal_type == "SELL"]
        self.assertTrue(len(sell_signals) > 0, "Should have generated SELL signal on crash")

if __name__ == '__main__':
    unittest.main()
