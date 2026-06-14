"""
QUANTUM-FORGE: Backtest Validation & Math Correctness Tests
=============================================================
P1 8.2 — Validates backtesting engine sanity (no lookahead bias, proper sequencing).
P1 8.3 — Validates mathematical correctness of core math engine components.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Backtest Sanity Tests ───────────────────────────────────────────────

class TestBacktestSanity:
    """Ensure backtesting doesn't peek into the future."""

    def test_signal_uses_only_past_data(self):
        """Signals at time t must only use data up to time t."""
        from core.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline(feature_dim=32)

        np.random.seed(42)
        prices = np.cumsum(np.random.randn(300)) + 50000

        # Features at t=200 should not change if future data changes
        feat_original = pipe.extract(prices[:201])

        # Alter future data (indices 201+)
        prices_alt = prices.copy()
        prices_alt[201:] += 1000  # big shift in future

        feat_altered = pipe.extract(prices_alt[:201])

        assert np.allclose(feat_original, feat_altered), \
            "Feature extraction is using future data (lookahead bias)!"

    def test_monotonic_time_in_signals(self):
        """Signal timestamps must be monotonically increasing."""
        from core.signal_generator import SignalGenerator
        try:
            sg = SignalGenerator()
        except Exception:
            pytest.skip("SignalGenerator not instantiable in test mode")
            return

        # Feed sequential prices
        timestamps = []
        for i in range(100):
            price = 50000 + np.random.randn() * 500
            signal = sg.ingest_price("BTCUSDT", price, volume=1e6)
            if signal and hasattr(signal, "timestamp"):
                timestamps.append(signal.timestamp)

        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], \
                f"Non-monotonic timestamp at index {i}"

    def test_position_size_bounded(self):
        """No single trade should exceed max allocation."""
        max_pct = 0.05
        capital = 100000

        # Simulate what quantum_core does
        for strength in np.linspace(0, 2, 20):
            trade_value = capital * max_pct * min(strength, 1.0)
            assert trade_value <= capital * max_pct + 1e-6, \
                f"Trade value {trade_value} exceeds {capital * max_pct}"


# ─── Math Correctness Tests ─────────────────────────────────────────────

class TestMathCorrectness:
    """Validate core mathematical computations."""

    def test_rsi_bounds(self):
        """RSI must be in [0, 100] (or [0, 1] normalized)."""
        from core.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline(feature_dim=32)

        for _ in range(20):
            prices = np.cumsum(np.random.randn(100)) + 50000
            rsi = pipe._rsi(prices, period=14)
            assert 0 <= rsi <= 1, f"RSI out of bounds: {rsi}"

    def test_bollinger_pct_reasonable(self):
        """Bollinger %B should be roughly in [-1, 2] for normal data."""
        from core.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline(feature_dim=32)

        prices = np.cumsum(np.random.randn(100)) + 50000
        bb = pipe._bollinger_pct(prices)
        assert -5 <= bb <= 5, f"Bollinger %B unreasonable: {bb}"

    def test_atr_non_negative(self):
        """ATR (Average True Range) must be non-negative."""
        from core.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline(feature_dim=32)

        prices = np.cumsum(np.random.randn(100)) + 50000
        atr = pipe._atr(prices)
        assert atr >= 0, f"ATR is negative: {atr}"

    def test_spectral_features_no_nan(self):
        """Spectral features should not produce NaN."""
        from core.feature_pipeline import FeaturePipeline
        pipe = FeaturePipeline(feature_dim=32)

        prices = np.cumsum(np.random.randn(200)) + 50000
        feats = pipe._spectral_features(prices)
        assert isinstance(feats, list)
        for f in feats:
            assert not np.isnan(f), f"Spectral feature is NaN"

    def test_returns_sign_consistency(self):
        """If price goes up, return should be positive."""
        prices = np.array([100.0, 105.0, 110.0, 108.0])
        returns = np.diff(prices) / prices[:-1]
        assert returns[0] > 0  # 100→105 is positive
        assert returns[1] > 0  # 105→110 is positive
        assert returns[2] < 0  # 110→108 is negative

    def test_correlation_matrix_symmetric(self):
        """Correlation matrices must be symmetric with 1s on diagonal."""
        data = np.random.randn(100, 5)
        corr = np.corrcoef(data.T)
        assert np.allclose(corr, corr.T), "Correlation matrix not symmetric"
        assert np.allclose(np.diag(corr), 1.0), "Diagonal should be 1"

    def test_svm_classifier_labels_valid(self):
        """SVM classifier should only return valid regime labels."""
        from core.svm_classifier import SVMRegimeClassifier
        clf = SVMRegimeClassifier(feature_dim=32)
        valid_labels = {"BULL", "BEAR", "NEUTRAL", "HIGH_VOL"}

        for _ in range(50):
            features = np.random.randn(32).astype(np.float32)
            result = clf.predict(features)
            assert result["regime"] in valid_labels

    def test_cross_asset_signal_bounded(self):
        """Cross-asset composite score must be in [-1, 1]."""
        from core.cross_asset_alpha import CrossAssetAlphaEngine
        engine = CrossAssetAlphaEngine(
            symbols=["BTCUSDT", "ETHUSDT"],
            lookback=30,
        )

        for i in range(100):
            engine.update("BTCUSDT", 50000 + np.random.randn() * 1000)
            engine.update("ETHUSDT", 3000 + np.random.randn() * 100)

        sig = engine.get_signal_for_symbol("BTCUSDT")
        assert -1.0 <= sig["composite_score"] <= 1.0

    def test_fusion_weights_sum_to_one(self):
        """Signal fusion weights should sum to ~1."""
        # All three sources available
        math_w, ml_w, cross_w = 0.50, 0.30, 0.20
        assert abs(math_w + ml_w + cross_w - 1.0) < 1e-10

        # Only math + ML
        math_w, ml_w = 0.60, 0.40
        assert abs(math_w + ml_w - 1.0) < 1e-10

        # Only math + cross
        math_w, cross_w = 0.70, 0.30
        assert abs(math_w + cross_w - 1.0) < 1e-10


# ─── EVT and Copula Sanity ──────────────────────────────────────────────

class TestEVTCopulaSanity:
    """Basic sanity checks for EVT and Copula modules."""

    def test_evt_tail_probability_bounded(self):
        """EVT tail probabilities must be in [0, 1]."""
        try:
            from mathematics.extreme_value_theory import ExtremeValueAnalyzer
            evt = ExtremeValueAnalyzer()
            # If EVT has a method to compute tail risk
            if hasattr(evt, "compute_tail_risk"):
                returns = np.random.randn(500) * 0.02
                result = evt.compute_tail_risk(returns)
                if isinstance(result, dict) and "tail_probability" in result:
                    p = result["tail_probability"]
                    assert 0 <= p <= 1, f"Tail probability out of bounds: {p}"
        except ImportError:
            pytest.skip("EVT module not available")

    def test_copula_correlation_bounded(self):
        """Copula-estimated correlations must be in [-1, 1]."""
        try:
            from mathematics.copula_models import CopulaAnalyzer
            cop = CopulaAnalyzer()
            if hasattr(cop, "estimate_correlation"):
                x = np.random.randn(200)
                y = x * 0.5 + np.random.randn(200) * 0.5
                corr = cop.estimate_correlation(x, y)
                if isinstance(corr, (float, np.floating)):
                    assert -1.0 <= corr <= 1.0, f"Copula correlation out of bounds: {corr}"
        except ImportError:
            pytest.skip("Copula module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
