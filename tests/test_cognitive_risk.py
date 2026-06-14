import unittest
from core.risk_mathematics.cognitive_dampener import CognitiveDampener, CognitiveRiskConfig

class TestCognitiveRisk(unittest.TestCase):
    def setUp(self):
        self.dampener = CognitiveDampener()

    def test_invariant_never_exceeds_one(self):
        """CRITICAL: Multiplier must NEVER be > 1.0"""
        # Even with a "SUPER_BULL" regime that might imply 2.0
        self.dampener.config.regime_penalties["SUPER_BULL"] = 2.0
        
        multiplier = self.dampener.calculate_multiplier("SUPER_BULL", 1.0)
        self.assertLessEqual(multiplier, 1.0)
        self.assertEqual(multiplier, 1.0)

    def test_regime_penalties(self):
        """Verify that bad regimes reduce risk."""
        # High Volatility -> 0.5
        m_vol = self.dampener.calculate_multiplier("HIGH_VOLATILITY", 1.0)
        self.assertEqual(m_vol, 0.5)
        
        # Market Crash -> 0.2
        m_crash = self.dampener.calculate_multiplier("MARKET_CRASH", 1.0)
        self.assertEqual(m_crash, 0.2)

    def test_confidence_scaling(self):
        """Verify that low confidence reduces the multiplier further."""
        # Stable regime (1.0) but low confidence (0.35 vs threshold 0.7)
        # Factor = 0.35 / 0.7 = 0.5
        # Result = 1.0 * 0.5 = 0.5
        m_low_conf = self.dampener.calculate_multiplier("STABLE", 0.35)
        self.assertAlmostEqual(m_low_conf, 0.5)

    def test_min_floor(self):
        """Verify we don't go below the minimum floor."""
        # Crash (0.2) + Zero Confidence (0.0) -> Should be 0.0 theoretically
        # But clamped to min_multiplier (0.2)
        m_floor = self.dampener.calculate_multiplier("MARKET_CRASH", 0.0)
        self.assertEqual(m_floor, 0.2)

if __name__ == '__main__':
    unittest.main()
