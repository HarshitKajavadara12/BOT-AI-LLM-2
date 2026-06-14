import unittest
import numpy as np
from core.analytics import get_analytics_engine, PerformanceMetrics

class TestResearchLiveParity(unittest.TestCase):
    """
    Verifies that Research (Notebooks) and Live (API/Core) use the exact same logic.
    """
    
    def setUp(self):
        self.engine = get_analytics_engine()
        
        # Sample Data: 10 days of returns (Realistic Volatility)
        # Alternating gains/losses to produce a realistic Sharpe ~1.6
        self.returns = [0.005, -0.005, 0.005, -0.005, 0.005, -0.005, 0.005, -0.005, 0.005, 0.0]
        
    def test_sharpe_calculation(self):
        """
        Test that the core engine calculates Sharpe correctly.
        This logic is used by BOTH the Live Dashboard and Research Notebooks.
        """
        metrics = self.engine.calculate_metrics(self.returns)
        
        # Manual verification
        rets = np.array(self.returns)
        mean = np.mean(rets)
        std = np.std(rets, ddof=1)
        expected_sharpe = (mean / std) * np.sqrt(252)
        
        self.assertAlmostEqual(metrics.sharpe_ratio, expected_sharpe, places=4)
        print(f"[OK] Parity Check: Sharpe Ratio {metrics.sharpe_ratio:.4f} matches expected {expected_sharpe:.4f}")

    def test_drawdown_calculation(self):
        """Test Max Drawdown logic."""
        # Create a sequence with a known drawdown
        # 100 -> 110 (+10%) -> 99 (-10%) -> Drawdown is (99-110)/110 = -10%
        # Wait, 110 * 0.9 = 99. So -10% return from peak.
        
        returns = [0.10, -0.10] 
        metrics = self.engine.calculate_metrics(returns)
        
        # Peak is 1.10. Current is 0.99. DD = (0.99 - 1.10) / 1.10 = -0.11 / 1.10 = -0.10
        self.assertAlmostEqual(metrics.max_drawdown, -0.10, places=4)
        print(f"[OK] Parity Check: Max Drawdown {metrics.max_drawdown:.4f} matches expected -0.1000")

if __name__ == '__main__':
    unittest.main()
