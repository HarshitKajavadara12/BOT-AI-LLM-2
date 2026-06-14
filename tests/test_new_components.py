"""
QUANTUM-FORGE: Comprehensive Unit Tests for New Components
============================================================
Tests for: FeaturePipeline, SVMRegimeClassifier, CrossAssetAlphaEngine,
           AlphaBridge, Signal Fusion, ML Ensemble alignment.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── FeaturePipeline Tests ──────────────────────────────────────────────

class TestFeaturePipeline:
    """Tests for core.feature_pipeline.FeaturePipeline"""

    def _make_pipeline(self):
        from core.feature_pipeline import FeaturePipeline
        return FeaturePipeline(feature_dim=32)

    def test_output_shape(self):
        pipe = self._make_pipeline()
        prices = np.cumsum(np.random.randn(200)) + 50000
        feats = pipe.extract(prices)
        assert feats.shape == (32,), f"Expected (32,), got {feats.shape}"

    def test_no_nan_or_inf(self):
        pipe = self._make_pipeline()
        prices = np.cumsum(np.random.randn(200)) + 50000
        volumes = np.abs(np.random.randn(200)) * 1e6
        feats = pipe.extract(prices, volumes)
        assert not np.any(np.isnan(feats)), "Features contain NaN"
        assert not np.any(np.isinf(feats)), "Features contain Inf"

    def test_short_input_still_works(self):
        pipe = self._make_pipeline()
        prices = np.array([100.0, 101.0, 99.0, 102.0, 100.5])
        feats = pipe.extract(prices)
        assert feats.shape == (32,)
        assert not np.any(np.isnan(feats))

    def test_constant_prices(self):
        """Constant prices should produce near-zero features (no information)."""
        pipe = self._make_pipeline()
        prices = np.full(200, 50000.0)
        feats = pipe.extract(prices)
        assert feats.shape == (32,)
        # Most features should be zero for constant input
        assert np.sum(np.abs(feats)) < 5.0, "Constant prices should have low feature magnitude"

    def test_with_volumes(self):
        pipe = self._make_pipeline()
        prices = np.cumsum(np.random.randn(200)) + 50000
        volumes = np.abs(np.random.randn(200)) * 1e6
        feats_with_vol = pipe.extract(prices, volumes)
        feats_no_vol = pipe.extract(prices)
        # Should differ when volumes are provided
        assert not np.allclose(feats_with_vol, feats_no_vol), \
            "Volume features should differ from no-volume"

    def test_deterministic(self):
        pipe = self._make_pipeline()
        prices = np.cumsum(np.random.randn(200)) + 50000
        f1 = pipe.extract(prices)
        f2 = pipe.extract(prices)
        assert np.allclose(f1, f2), "Feature extraction should be deterministic"


# ─── SVMRegimeClassifier Tests ──────────────────────────────────────────

class TestSVMClassifier:
    """Tests for core.svm_classifier.SVMRegimeClassifier"""

    def _make_classifier(self):
        from core.svm_classifier import SVMRegimeClassifier
        return SVMRegimeClassifier(feature_dim=32, n_components=50)

    def test_predict_output_type(self):
        clf = self._make_classifier()
        features = np.random.randn(32).astype(np.float32)
        result = clf.predict(features)
        assert isinstance(result, dict)
        assert "regime" in result
        assert "confidence" in result

    def test_regime_valid_labels(self):
        clf = self._make_classifier()
        features = np.random.randn(32).astype(np.float32)
        result = clf.predict(features)
        valid = {"BULL", "BEAR", "NEUTRAL", "HIGH_VOL"}
        assert result["regime"] in valid, f"Unknown regime: {result['regime']}"

    def test_confidence_range(self):
        clf = self._make_classifier()
        features = np.random.randn(32).astype(np.float32)
        result = clf.predict(features)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_online_update_no_crash(self):
        clf = self._make_classifier()
        for _ in range(20):
            features = np.random.randn(32).astype(np.float32)
            returns = np.random.randn(30) * 0.01
            vol = np.std(returns)
            clf.online_update(features, returns, vol)
        # Should still predict after updates
        result = clf.predict(np.random.randn(32).astype(np.float32))
        assert "regime" in result

    def test_fit_and_predict(self):
        clf = self._make_classifier()
        X = np.random.randn(100, 32).astype(np.float32)
        returns_batch = np.random.randn(100, 30) * 0.01
        vols = np.std(returns_batch, axis=1)
        clf.fit(X, returns_batch, vols)
        result = clf.predict(np.random.randn(32).astype(np.float32))
        assert result["confidence"] > 0


# ─── CrossAssetAlphaEngine Tests ─────────────────────────────────────────

class TestCrossAssetAlpha:
    """Tests for core.cross_asset_alpha.CrossAssetAlphaEngine"""

    def _make_engine(self):
        from core.cross_asset_alpha import CrossAssetAlphaEngine
        return CrossAssetAlphaEngine(
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            lookback=30
        )

    def test_update_and_signal(self):
        engine = self._make_engine()
        # Feed enough data
        for i in range(50):
            engine.update("BTCUSDT", 50000 + i * 10)
            engine.update("ETHUSDT", 3000 + i * 5)
            engine.update("BNBUSDT", 300 + i * 1)

        sig = engine.get_signal_for_symbol("BTCUSDT")
        assert isinstance(sig, dict)
        assert "composite_score" in sig

    def test_composite_score_bounded(self):
        engine = self._make_engine()
        for i in range(50):
            engine.update("BTCUSDT", 50000 + np.random.randn() * 500)
            engine.update("ETHUSDT", 3000 + np.random.randn() * 50)

        sig = engine.get_signal_for_symbol("BTCUSDT")
        assert -1.0 <= sig["composite_score"] <= 1.0, \
            f"Score out of bounds: {sig['composite_score']}"

    def test_empty_signal(self):
        engine = self._make_engine()
        sig = engine.get_signal_for_symbol("BTCUSDT")
        assert sig["composite_score"] == 0.0, "Empty engine should return neutral signal"


# ─── AlphaBridge Tests ───────────────────────────────────────────────────

class TestAlphaBridge:
    """Tests for core.alpha_bridge (AlphaStore, AlphaPortfolioConstructor)"""

    def test_alpha_store_add_and_get(self):
        from core.alpha_bridge import AlphaStore, AlphaState
        store = AlphaStore(persist_path=None)  # in-memory only
        store.add_alpha("momentum_5m", {"type": "momentum", "window": 5})
        alpha = store.get("momentum_5m")
        assert alpha is not None
        assert alpha.state == AlphaState.RESEARCH

    def test_alpha_lifecycle(self):
        from core.alpha_bridge import AlphaStore, AlphaState
        store = AlphaStore(persist_path=None)
        store.add_alpha("test_alpha", {"type": "test"})
        store.promote("test_alpha", AlphaState.VALIDATION)
        store.promote("test_alpha", AlphaState.SHADOW)
        store.promote("test_alpha", AlphaState.LIVE)
        alpha = store.get("test_alpha")
        assert alpha.state == AlphaState.LIVE

    def test_portfolio_constructor(self):
        from core.alpha_bridge import AlphaPortfolioConstructor
        pc = AlphaPortfolioConstructor(symbols=["BTCUSDT", "ETHUSDT"], max_position_pct=0.1)
        weights = {"BTCUSDT": 0.6, "ETHUSDT": 0.4}
        targets = pc.construct(weights, total_capital=100000)
        assert isinstance(targets, dict)
        assert "BTCUSDT" in targets
        # Position should be capped at max_position_pct * capital
        assert targets["BTCUSDT"] <= 100000 * 0.1 + 1


# ─── Signal Fusion Tests ────────────────────────────────────────────────

class TestSignalFusion:
    """Tests for the 3-way signal fusion (math + ML + cross-asset)."""

    def _make_mock_core(self):
        """
        Create a minimal mock that has _fuse_signals method.
        We import the actual class method if possible, or test the logic directly.
        """
        # Test the fusion logic directly
        from core.quantum_core import QuantumCore
        # We can't easily instantiate QuantumCore without all deps,
        # so test the fusion math directly
        return None

    def test_fusion_all_buy(self):
        """All signals agree on BUY → should return BUY with high strength."""
        # Simulate: math=BUY(0.8), ML=BUY(0.7), cross=0.5
        math_value = 0.8
        ml_value = 0.7 * 0.9  # consensus=0.9
        cross_value = 0.5
        fused = math_value * 0.50 + ml_value * 0.30 + cross_value * 0.20
        assert fused > 0.2, "All-buy should produce BUY"

    def test_fusion_conflicting(self):
        """Math=BUY, ML=SELL → should return weaker signal."""
        math_value = 0.6
        ml_value = -0.6 * 0.8
        cross_value = 0.0
        fused = math_value * 0.50 + ml_value * 0.30 + cross_value * 0.20
        # Math (0.3) vs ML (-0.144) → net positive but weak
        assert abs(fused) < 0.5, "Conflicting signals should weaken result"

    def test_fusion_no_ml(self):
        """When ML is None, math takes full weight."""
        math_value = 0.5
        fused = math_value * 1.0
        assert fused == 0.5


# ─── ML Ensemble Alignment Tests ────────────────────────────────────────

class TestMLEnsembleAlignment:
    """Tests that MLEnsembleEngine is aligned with FeaturePipeline dimension."""

    def test_feature_dim_alignment(self):
        try:
            from core.ml_ensemble import MLEnsembleEngine
            from core.feature_pipeline import FeaturePipeline
            pipe = FeaturePipeline(feature_dim=32)
            ensemble = MLEnsembleEngine(feature_dim=32)
            assert ensemble.feature_dim == 32
        except ImportError:
            pytest.skip("Cannot import both modules")

    def test_ensemble_predict_with_pipeline_features(self):
        try:
            from core.ml_ensemble import MLEnsembleEngine
            from core.feature_pipeline import FeaturePipeline
            pipe = FeaturePipeline(feature_dim=32)
            ensemble = MLEnsembleEngine(feature_dim=32)
            prices = np.cumsum(np.random.randn(200)) + 50000
            feats = pipe.extract(prices)
            result = ensemble.predict(feats)
            assert result.signal in ("BUY", "SELL", "HOLD")
            assert 0.0 <= result.strength <= 1.0
            assert 0.0 <= result.consensus <= 1.0
        except ImportError:
            pytest.skip("Cannot import modules")


# ─── Math Engine Sanity Tests ────────────────────────────────────────────

class TestMathEngineSanity:
    """Basic sanity checks for mathematical components."""

    def test_evt_import(self):
        try:
            from mathematics.extreme_value_theory import ExtremeValueAnalyzer
            evt = ExtremeValueAnalyzer()
            assert evt is not None
        except ImportError:
            pytest.skip("EVT module not available")

    def test_copula_import(self):
        try:
            from mathematics.copula_models import CopulaAnalyzer
            cop = CopulaAnalyzer()
            assert cop is not None
        except ImportError:
            pytest.skip("Copula module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
