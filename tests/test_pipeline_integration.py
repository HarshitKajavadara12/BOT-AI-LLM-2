"""
Quantum-Forge: Pipeline Integration Tests
============================================
End-to-end tests that verify the full core pipeline works as a unit.
Complements existing test_integration.py (which tests individual analytics modules).
"""

import os
import sys
import unittest
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFullPipelineInit(unittest.TestCase):
    """Test that the orchestrator initializes all subsystems."""

    def test_orchestrator_creates(self):
        from core.quantum_core import QuantumCoreOrchestrator
        orch = QuantumCoreOrchestrator(
            symbols=["BTCUSDT"],
            initial_capital=10000.0,
            enable_ml=False,
            enable_llm=False,
        )
        self.assertFalse(orch.is_running)
        self.assertEqual(orch.cash, 10000.0)
        self.assertEqual(len(orch.symbols), 1)

    def test_signal_generator_pipeline(self):
        from core.signal_generator import SignalGenerator
        sg = SignalGenerator(min_history=10, signal_threshold=0.2)
        prices = [100 + i * 0.5 + np.sin(i / 3) for i in range(30)]
        for p in prices:
            sg.ingest_price("BTCUSDT", p)
        signal = sg.generate_signal("BTCUSDT")
        self.assertIsNotNone(signal)
        self.assertIn(signal.signal_type.value, ["BUY", "SELL", "HOLD"])

    def test_regime_detector_pipeline(self):
        from core.regime_detector import RegimeDetector
        rd = RegimeDetector(window_size=20)
        for p in [100 + i * 0.1 for i in range(30)]:
            result = rd.on_market_data(p)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.regime)


class TestExecutionPipeline(unittest.TestCase):

    def test_paper_trade_roundtrip(self):
        from core.execution_manager import ExecutionManager, ExecutionMode
        em = ExecutionManager(mode=ExecutionMode.PAPER)
        result = em.execute(
            symbol="BTCUSDT", side="BUY", quantity=0.01,
            price=50000.0, signal_strength=0.6, volatility=0.02,
        )
        self.assertEqual(result.status, "FILLED")
        self.assertGreater(result.filled_quantity, 0)
        stats = em.get_execution_stats()
        self.assertEqual(stats["total_orders"], 1)

    def test_algo_selection_market_small(self):
        from core.execution_manager import ExecutionManager, ExecutionMode, AlgoType
        em = ExecutionManager(mode=ExecutionMode.PAPER)
        algo = em.select_algorithm(order_value_usd=100, volatility=0.02, signal_strength=0.5)
        self.assertEqual(algo, AlgoType.MARKET)


class TestCircuitBreaker(unittest.TestCase):

    def test_opens_after_failures(self):
        from core.quantum_core import CircuitBreaker
        cb = CircuitBreaker(max_failures=3, cooldown_seconds=0.5)
        for _ in range(3):
            cb.record_failure("BTCUSDT")
        self.assertTrue(cb.is_open("BTCUSDT"))
        self.assertFalse(cb.is_open("ETHUSDT"))

    def test_resets_after_cooldown(self):
        from core.quantum_core import CircuitBreaker
        cb = CircuitBreaker(max_failures=2, cooldown_seconds=0.2)
        cb.record_failure("X")
        cb.record_failure("X")
        self.assertTrue(cb.is_open("X"))
        time.sleep(0.3)
        self.assertFalse(cb.is_open("X"))

    def test_success_resets(self):
        from core.quantum_core import CircuitBreaker
        cb = CircuitBreaker(max_failures=3, cooldown_seconds=60)
        cb.record_failure("X")
        cb.record_failure("X")
        cb.record_success("X")
        cb.record_failure("X")
        self.assertFalse(cb.is_open("X"))


class TestRiskGate(unittest.TestCase):

    def test_crisis_blocks(self):
        from core.quantum_core import RiskGate
        from core.regime_detector import MarketRegime
        rg = RiskGate()
        result = rg.check("BTCUSDT", "BUY", 0.9, MarketRegime.CRISIS, 50000)
        self.assertFalse(result["approved"])

    def test_neutral_approves(self):
        from core.quantum_core import RiskGate
        from core.regime_detector import MarketRegime
        rg = RiskGate()
        result = rg.check("BTCUSDT", "BUY", 0.5, MarketRegime.NEUTRAL, 50000)
        self.assertTrue(result["approved"])


class TestStorageCoordinator(unittest.TestCase):

    def test_store_trade(self):
        from core.storage_coordinator import StorageCoordinator, TradeRecord
        sc = StorageCoordinator()
        tr = TradeRecord(
            timestamp="2024-01-01T00:00:00", symbol="BTCUSDT",
            side="BUY", quantity=0.01, price=50000.0, fill_price=50010.0,
            slippage_bps=2.0, fees=5.0, algorithm="MARKET",
            signal_strength=0.6, regime="NEUTRAL",
        )
        sc.store_trade(tr)
        stats = sc.get_stats()
        self.assertGreaterEqual(stats["trades_stored"], 1)
        sc.close()


class TestStrategyMultiplexer(unittest.TestCase):

    def test_shadow_rankings(self):
        from core.strategy_multiplexer import StrategyMultiplexer
        mux = StrategyMultiplexer()
        rankings = mux.get_shadow_rankings()
        self.assertIsInstance(rankings, list)

    def test_evaluate_promotions_empty(self):
        from core.strategy_multiplexer import StrategyMultiplexer
        mux = StrategyMultiplexer()
        promoted = mux.evaluate_promotions()
        self.assertEqual(promoted, [])


class TestAlertSystem(unittest.TestCase):

    def test_system_alert_no_crash(self):
        from core.alert_system import AlertSystem, AlertLevel
        als = AlertSystem()
        als.system_alert("test message", AlertLevel.INFO)


class TestMetricsServer(unittest.TestCase):

    def test_app_importable(self):
        from core.metrics_server import app
        self.assertIsNotNone(app)

    def test_set_orchestrator(self):
        from core.metrics_server import set_orchestrator
        set_orchestrator(None)  # Should not raise


if __name__ == "__main__":
    unittest.main()
