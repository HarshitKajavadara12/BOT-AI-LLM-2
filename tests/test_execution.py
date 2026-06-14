"""
Test Suite for Execution Algorithms and Market Microstructure

Tests against the actual class APIs in the codebase:
  - TWAPAlgorithm, VWAPAlgorithm, ArrivalPriceAlgorithm
  - ImplementationShortfallAlgorithm
  - LimitOrderBook, OrderBookAnalyzer
  - AmihudModel, KyleModel, LiquidityAnalyzer
  - ToxicityDetector
  - ExecutionManager (new P2 module)
"""

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTWAPAlgorithm(unittest.TestCase):
    """Test TWAP algorithm with actual API."""

    def setUp(self):
        from core.execution_algorithms.twap_algorithm import TWAPAlgorithm, TWAPParameters, MarketData
        self.TWAPAlgorithm = TWAPAlgorithm
        self.TWAPParameters = TWAPParameters
        self.MarketData = MarketData

    def test_instantiate(self):
        params = self.TWAPParameters(
            total_quantity=1000,
            duration_minutes=60,
            participation_rate=0.1,
            min_order_size=10,
            max_order_size=500,
            price_tolerance=0.002,
        )
        algo = self.TWAPAlgorithm(params)
        self.assertIsNotNone(algo)

    def test_initialize_order(self):
        params = self.TWAPParameters(total_quantity=1000, duration_minutes=60)
        algo = self.TWAPAlgorithm(params)
        md = self.MarketData(
            timestamp=datetime.now(), bid_price=100.0, ask_price=100.05,
            bid_size=50, ask_size=50, last_price=100.02, volume=10000, volatility=0.02,
        )
        algo.initialize_order('buy', 'BTCUSDT', md)

    def test_market_data_update(self):
        params = self.TWAPParameters(total_quantity=500, duration_minutes=30)
        algo = self.TWAPAlgorithm(params)
        md = self.MarketData(
            timestamp=datetime.now(), bid_price=100.0, ask_price=100.05,
            bid_size=50, ask_size=50, last_price=100.02, volume=10000, volatility=0.02,
        )
        algo.initialize_order('buy', 'BTCUSDT', md)
        algo.update_market_data(md)


class TestVWAPAlgorithm(unittest.TestCase):
    """Test VWAP algorithm."""

    def setUp(self):
        from core.execution_algorithms.vwap_algorithm import VWAPAlgorithm, VWAPParameters
        self.VWAPAlgorithm = VWAPAlgorithm
        self.VWAPParameters = VWAPParameters

    def test_instantiate(self):
        params = self.VWAPParameters(total_quantity=5000, duration_minutes=120)
        algo = self.VWAPAlgorithm(params)
        self.assertIsNotNone(algo)

    def test_initialize_order(self):
        from core.execution_algorithms.twap_algorithm import MarketData
        params = self.VWAPParameters(total_quantity=5000, duration_minutes=120)
        algo = self.VWAPAlgorithm(params)
        md = MarketData(
            timestamp=datetime.now(), bid_price=100.0, ask_price=100.05,
            bid_size=50, ask_size=50, last_price=100.02, volume=10000, volatility=0.02,
        )
        algo.initialize_order('buy', 'BTCUSDT', md)


class TestArrivalPriceAlgorithm(unittest.TestCase):
    """Test Arrival Price algorithm."""

    def test_instantiate(self):
        from core.execution_algorithms.arrival_price import ArrivalPriceAlgorithm, ArrivalPriceParameters
        params = ArrivalPriceParameters(total_quantity=5000, duration_minutes=60)
        algo = ArrivalPriceAlgorithm(params)
        self.assertIsNotNone(algo)


class TestImplementationShortfallAlgorithm(unittest.TestCase):
    """Test IS algorithm with correct class name."""

    def setUp(self):
        from core.execution_algorithms.implementation_shortfall import (
            ImplementationShortfallAlgorithm, ISParameters
        )
        self.ISAlgorithm = ImplementationShortfallAlgorithm
        self.ISParameters = ISParameters

    def test_instantiate(self):
        params = self.ISParameters(total_quantity=10000, duration_minutes=60)
        algo = self.ISAlgorithm(params)
        self.assertIsNotNone(algo)

    def test_initialize_order(self):
        from core.execution_algorithms.twap_algorithm import MarketData
        params = self.ISParameters(total_quantity=10000, duration_minutes=60)
        algo = self.ISAlgorithm(params)
        md = MarketData(
            timestamp=datetime.now(), bid_price=100.0, ask_price=100.05,
            bid_size=50, ask_size=50, last_price=100.02, volume=10000, volatility=0.02,
        )
        algo.initialize_order('buy', 'BTCUSDT', md)


class TestLimitOrderBook(unittest.TestCase):
    """Test LimitOrderBook (actual class name)."""

    def setUp(self):
        from core.market_microstructure.orderbook_dynamics import (
            LimitOrderBook, OrderBookAnalyzer, OrderSide, OrderType
        )
        self.book = LimitOrderBook(tick_size=0.01)
        self.OrderSide = OrderSide
        self.OrderType = OrderType
        self.OrderBookAnalyzer = OrderBookAnalyzer

    def test_empty_book(self):
        self.assertEqual(len(self.book.orders), 0)
        self.assertIsNone(self.book.get_best_bid())
        self.assertIsNone(self.book.get_best_ask())

    def test_add_limit_order(self):
        oid = self.book.add_order(
            self.OrderSide.BID, self.OrderType.LIMIT, 45000.0, 100, "t1",
        )
        self.assertIn(oid, self.book.orders)

    def test_best_bid_ask(self):
        self.book.add_order(self.OrderSide.BID, self.OrderType.LIMIT, 45000.0, 100, "t1")
        self.book.add_order(self.OrderSide.BID, self.OrderType.LIMIT, 44950.0, 50, "t2")
        self.book.add_order(self.OrderSide.ASK, self.OrderType.LIMIT, 45050.0, 80, "t3")
        self.assertEqual(self.book.get_best_bid(), 45000.0)
        self.assertEqual(self.book.get_best_ask(), 45050.0)

    def test_mid_price(self):
        self.book.add_order(self.OrderSide.BID, self.OrderType.LIMIT, 100.0, 100, "t1")
        self.book.add_order(self.OrderSide.ASK, self.OrderType.LIMIT, 102.0, 100, "t2")
        self.assertAlmostEqual(self.book.get_mid_price(), 101.0)

    def test_analyzer_instantiate(self):
        analyzer = self.OrderBookAnalyzer()
        self.assertIsNotNone(analyzer)


class TestLiquidityModels(unittest.TestCase):
    """Test actual liquidity model classes."""

    def test_amihud_model(self):
        from core.market_microstructure.liquidity_models import AmihudModel
        self.assertIsNotNone(AmihudModel())

    def test_kyle_model(self):
        from core.market_microstructure.liquidity_models import KyleModel
        self.assertIsNotNone(KyleModel())

    def test_liquidity_analyzer(self):
        from core.market_microstructure.liquidity_models import LiquidityAnalyzer
        self.assertIsNotNone(LiquidityAnalyzer())


class TestPriceFormation(unittest.TestCase):
    """Test actual price formation classes."""

    def test_hasbrouck_model(self):
        from core.market_microstructure.price_formation import HasbrouckModel
        self.assertIsNotNone(HasbrouckModel())

    def test_price_formation_analyzer(self):
        from core.market_microstructure.price_formation import PriceFormationAnalyzer
        self.assertIsNotNone(PriceFormationAnalyzer())


class TestToxicityDetector(unittest.TestCase):
    """Test ToxicityDetector."""

    def test_instantiate(self):
        from core.market_microstructure.toxicity_detection import ToxicityDetector
        self.assertIsNotNone(ToxicityDetector())


class TestExecutionManager(unittest.TestCase):
    """Test the ExecutionManager (P2 wiring layer)."""

    def setUp(self):
        from core.execution_manager import ExecutionManager, ExecutionMode, AlgoType
        self.em = ExecutionManager(mode=ExecutionMode.PAPER)
        self.AlgoType = AlgoType
        self.ExecutionMode = ExecutionMode

    def test_paper_mode(self):
        self.assertEqual(self.em.mode, self.ExecutionMode.PAPER)

    def test_select_algo_small_order(self):
        algo = self.em.select_algorithm(100, 0.02, 0.5)
        self.assertEqual(algo, self.AlgoType.MARKET)

    def test_select_algo_high_urgency(self):
        algo = self.em.select_algorithm(10000, 0.02, 0.9)
        self.assertEqual(algo, self.AlgoType.MARKET)

    def test_execute_paper_buy(self):
        r = self.em.execute("BTCUSDT", "BUY", 0.01, 50000.0)
        self.assertEqual(r.status, "FILLED")
        self.assertGreater(r.filled_quantity, 0)
        self.assertGreater(r.fees_paid, 0)

    def test_execute_paper_sell(self):
        r = self.em.execute("ETHUSDT", "SELL", 1.0, 3000.0)
        self.assertEqual(r.status, "FILLED")
        self.assertAlmostEqual(r.filled_quantity, 1.0)

    def test_slippage_positive(self):
        r = self.em.execute("BTCUSDT", "BUY", 0.5, 50000.0)
        self.assertGreater(r.slippage, 0)  # BUY slippage is positive

    def test_execution_stats(self):
        self.em.execute("BTCUSDT", "BUY", 0.1, 50000.0)
        self.em.execute("ETHUSDT", "SELL", 1.0, 3000.0)
        stats = self.em.get_execution_stats()
        self.assertEqual(stats['total_orders'], 2)
        self.assertEqual(stats['filled'], 2)
        self.assertGreater(stats['total_fees'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
