"""
Test Suite for Execution Algorithms and Market Microstructure

This module contains comprehensive unit tests for execution algorithms (TWAP, VWAP,
Arrival Price, Implementation Shortfall), market microstructure models, order book
dynamics, liquidity models, and toxicity detection.
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.execution_algorithms.twap_algorithm import TWAPAlgorithm
from core.execution_algorithms.vwap_algorithm import VWAPAlgorithm
from core.execution_algorithms.arrival_price import ArrivalPriceAlgorithm
from core.execution_algorithms.implementation_shortfall import ImplementationShortfall
from core.market_microstructure.orderbook_dynamics import OrderBook, OrderBookAnalyzer
from core.market_microstructure.liquidity_models import LiquidityModel, ImpactModel
from core.market_microstructure.price_formation import PriceFormationModel
from core.market_microstructure.toxicity_detection import ToxicityDetector


class TestTWAPAlgorithm(unittest.TestCase):
    """Test cases for Time-Weighted Average Price algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.twap = TWAPAlgorithm()
        self.order_size = 10000
        self.time_horizon = 60  # minutes
        
    def test_uniform_schedule(self):
        """Test uniform time slicing."""
        schedule = self.twap.generate_schedule(
            total_quantity=self.order_size,
            time_horizon=self.time_horizon,
            n_slices=10
        )
        
        # Each slice should have equal quantity
        self.assertEqual(len(schedule), 10)
        expected_slice = self.order_size / 10
        for slice_qty in schedule:
            assert_almost_equal(slice_qty, expected_slice, decimal=2)
            
    def test_total_quantity_preserved(self):
        """Test that total quantity is preserved."""
        schedule = self.twap.generate_schedule(
            total_quantity=self.order_size,
            time_horizon=self.time_horizon,
            n_slices=7
        )
        
        total_executed = sum(schedule)
        assert_almost_equal(total_executed, self.order_size, decimal=2)
        
    def test_execution_times(self):
        """Test execution time spacing."""
        n_slices = 12
        times = self.twap.get_execution_times(
            start_time=datetime(2024, 1, 1, 9, 30),
            time_horizon=self.time_horizon,
            n_slices=n_slices
        )
        
        self.assertEqual(len(times), n_slices)
        
        # Check uniform spacing
        intervals = [(times[i+1] - times[i]).total_seconds() / 60 
                    for i in range(len(times)-1)]
        expected_interval = self.time_horizon / (n_slices - 1)
        
        for interval in intervals:
            assert_almost_equal(interval, expected_interval, decimal=1)
            
    def test_adaptive_twap(self):
        """Test adaptive TWAP with volume participation."""
        # Historical volume pattern (higher volume at open/close)
        volume_pattern = np.array([1.5, 1.2, 1.0, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6, 1.4])
        
        schedule = self.twap.generate_adaptive_schedule(
            total_quantity=self.order_size,
            volume_pattern=volume_pattern,
            participation_rate=0.1
        )
        
        # Schedule should follow volume pattern
        self.assertEqual(len(schedule), len(volume_pattern))
        total_executed = sum(schedule)
        assert_almost_equal(total_executed, self.order_size, decimal=2)
        
    def test_execution_with_slippage(self):
        """Test TWAP execution with slippage model."""
        market_prices = np.linspace(100, 102, 10)  # Trending market
        
        result = self.twap.execute(
            total_quantity=self.order_size,
            n_slices=10,
            market_prices=market_prices,
            spread=0.02
        )
        
        self.assertEqual(result['total_quantity'], self.order_size)
        self.assertGreater(result['average_price'], market_prices[0])
        self.assertLess(result['average_price'], market_prices[-1])
        
    def test_twap_vs_benchmark(self):
        """Test TWAP performance vs benchmark."""
        arrival_price = 100.0
        market_prices = np.random.normal(100, 0.5, 10)
        
        result = self.twap.execute(
            total_quantity=self.order_size,
            n_slices=10,
            market_prices=market_prices,
            spread=0.01
        )
        
        # Calculate implementation shortfall
        shortfall = result['average_price'] - arrival_price
        self.assertIsNotNone(shortfall)


class TestVWAPAlgorithm(unittest.TestCase):
    """Test cases for Volume-Weighted Average Price algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vwap = VWAPAlgorithm()
        self.order_size = 50000
        
    def test_volume_based_schedule(self):
        """Test schedule based on historical volume profile."""
        # Typical intraday volume pattern
        volume_profile = pd.Series([
            0.15, 0.12, 0.10, 0.08, 0.08,  # Morning
            0.07, 0.08, 0.08, 0.09, 0.15   # Afternoon
        ])
        
        schedule = self.vwap.generate_schedule(
            total_quantity=self.order_size,
            volume_profile=volume_profile,
            participation_rate=0.1
        )
        
        # Schedule should follow volume profile
        self.assertEqual(len(schedule), len(volume_profile))
        
        # Higher volume periods should have more quantity
        max_idx = volume_profile.argmax()
        self.assertEqual(schedule.argmax(), max_idx)
        
        # Total preserved
        assert_almost_equal(schedule.sum(), self.order_size, decimal=2)
        
    def test_participation_rate_constraint(self):
        """Test that participation rate is respected."""
        volume_profile = pd.Series([1000, 1500, 2000, 1800, 1200])
        participation_rate = 0.2
        
        schedule = self.vwap.generate_schedule(
            total_quantity=self.order_size,
            volume_profile=volume_profile,
            participation_rate=participation_rate
        )
        
        # Each slice should not exceed participation rate
        expected_volumes = volume_profile * participation_rate
        for i, (exec_qty, exp_vol) in enumerate(zip(schedule, expected_volumes)):
            self.assertLessEqual(exec_qty, exp_vol * 1.01)  # Allow 1% tolerance
            
    def test_calculate_vwap_benchmark(self):
        """Test VWAP benchmark calculation."""
        prices = np.array([100, 101, 102, 101, 100])
        volumes = np.array([1000, 1500, 2000, 1800, 1200])
        
        vwap_benchmark = self.vwap.calculate_vwap(prices, volumes)
        
        # Manual calculation
        expected_vwap = np.sum(prices * volumes) / np.sum(volumes)
        assert_almost_equal(vwap_benchmark, expected_vwap, decimal=6)
        
    def test_execution_tracking(self):
        """Test tracking of VWAP execution."""
        prices = np.array([100, 101, 102, 101, 100])
        volumes = np.array([1000, 1500, 2000, 1800, 1200])
        execution_qty = np.array([200, 300, 400, 360, 240])
        
        result = self.vwap.execute_and_track(
            prices=prices,
            volumes=volumes,
            execution_schedule=execution_qty
        )
        
        self.assertIn('execution_vwap', result)
        self.assertIn('benchmark_vwap', result)
        self.assertIn('slippage', result)
        
    def test_aggressive_participation(self):
        """Test behavior with high participation rate."""
        volume_profile = pd.Series([100, 150, 200, 180, 120])
        participation_rate = 0.5  # Aggressive
        
        schedule = self.vwap.generate_schedule(
            total_quantity=5000,
            volume_profile=volume_profile,
            participation_rate=participation_rate
        )
        
        # Should execute more aggressively in high volume periods
        total = schedule.sum()
        self.assertGreater(total, 0)
        
    def test_intraday_vwap_curve(self):
        """Test intraday VWAP curve calculation."""
        # Simulate intraday data
        n_periods = 20
        prices = 100 + np.random.randn(n_periods).cumsum() * 0.1
        volumes = np.random.randint(1000, 5000, n_periods)
        
        vwap_curve = self.vwap.calculate_intraday_vwap(prices, volumes)
        
        self.assertEqual(len(vwap_curve), n_periods)
        # VWAP should be relatively smooth
        self.assertLess(np.std(np.diff(vwap_curve)), np.std(np.diff(prices)))


class TestArrivalPriceAlgorithm(unittest.TestCase):
    """Test cases for Arrival Price algorithm."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.algo = ArrivalPriceAlgorithm()
        
    def test_minimize_implementation_shortfall(self):
        """Test algorithm minimizes implementation shortfall."""
        arrival_price = 100.0
        order_size = 10000
        risk_aversion = 0.5
        
        # Simulate price path
        time_steps = 10
        volatility = 0.02
        price_path = arrival_price + np.random.normal(0, volatility, time_steps).cumsum()
        
        strategy = self.algo.optimize_strategy(
            arrival_price=arrival_price,
            order_size=order_size,
            risk_aversion=risk_aversion,
            time_horizon=time_steps,
            volatility=volatility
        )
        
        self.assertEqual(len(strategy), time_steps)
        assert_almost_equal(strategy.sum(), order_size, decimal=2)
        
    def test_almgren_chriss_model(self):
        """Test Almgren-Chriss optimal execution model."""
        params = {
            'arrival_price': 100,
            'total_shares': 100000,
            'risk_aversion': 1e-6,
            'volatility': 0.02,
            'eta': 0.0001,  # Temporary impact
            'gamma': 0.00001,  # Permanent impact
            'time_horizon': 10
        }
        
        trajectory = self.algo.almgren_chriss_trajectory(**params)
        
        # Check trajectory properties
        self.assertEqual(len(trajectory), params['time_horizon'] + 1)
        self.assertEqual(trajectory[0], params['total_shares'])
        self.assertEqual(trajectory[-1], 0)
        
        # Should be monotonically decreasing
        self.assertTrue(np.all(np.diff(trajectory) <= 0))
        
    def test_price_impact_model(self):
        """Test price impact calculation."""
        trade_size = 1000
        daily_volume = 100000
        volatility = 0.02
        
        impact = self.algo.calculate_impact(
            trade_size=trade_size,
            daily_volume=daily_volume,
            volatility=volatility
        )
        
        # Impact should be positive for buy orders
        self.assertGreater(impact, 0)
        
        # Impact should increase with trade size
        larger_impact = self.algo.calculate_impact(
            trade_size=trade_size * 2,
            daily_volume=daily_volume,
            volatility=volatility
        )
        self.assertGreater(larger_impact, impact)
        
    def test_risk_aversion_effect(self):
        """Test effect of risk aversion on strategy."""
        base_params = {
            'arrival_price': 100,
            'total_shares': 10000,
            'volatility': 0.02,
            'time_horizon': 10
        }
        
        # Low risk aversion - more aggressive
        aggressive = self.algo.optimize_strategy(
            **base_params,
            risk_aversion=0.1
        )
        
        # High risk aversion - more conservative
        conservative = self.algo.optimize_strategy(
            **base_params,
            risk_aversion=10.0
        )
        
        # Conservative should spread execution more evenly
        aggressive_std = np.std(aggressive)
        conservative_std = np.std(conservative)
        self.assertLess(conservative_std, aggressive_std)


class TestImplementationShortfall(unittest.TestCase):
    """Test cases for Implementation Shortfall analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.is_analyzer = ImplementationShortfall()
        
    def test_calculate_implementation_shortfall(self):
        """Test basic IS calculation."""
        decision_price = 100.0
        execution_prices = np.array([100.1, 100.2, 100.15, 100.3])
        quantities = np.array([250, 300, 250, 200])
        
        is_value = self.is_analyzer.calculate(
            decision_price=decision_price,
            execution_prices=execution_prices,
            quantities=quantities
        )
        
        # Calculate expected value
        total_qty = quantities.sum()
        avg_price = np.sum(execution_prices * quantities) / total_qty
        expected_is = (avg_price - decision_price) * total_qty
        
        assert_almost_equal(is_value, expected_is, decimal=2)
        
    def test_decompose_is_components(self):
        """Test IS decomposition into components."""
        decision_price = 100.0
        execution_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:30', periods=5, freq='1min'),
            'price': [100.0, 100.1, 100.2, 100.15, 100.3],
            'quantity': [200, 250, 300, 250, 200]
        })
        
        components = self.is_analyzer.decompose(
            decision_price=decision_price,
            execution_data=execution_data
        )
        
        self.assertIn('delay_cost', components)
        self.assertIn('market_impact', components)
        self.assertIn('timing_cost', components)
        self.assertIn('opportunity_cost', components)
        
        # Total should equal implementation shortfall
        total = sum(components.values())
        is_value = self.is_analyzer.calculate(
            decision_price=decision_price,
            execution_prices=execution_data['price'].values,
            quantities=execution_data['quantity'].values
        )
        assert_almost_equal(total, is_value, decimal=2)
        
    def test_market_impact_component(self):
        """Test market impact isolation."""
        arrival_price = 100.0
        execution_prices = np.array([100.05, 100.08, 100.06])
        quantities = np.array([300, 400, 300])
        
        impact = self.is_analyzer.calculate_market_impact(
            arrival_price=arrival_price,
            execution_prices=execution_prices,
            quantities=quantities
        )
        
        # Impact should be positive (adverse price movement)
        self.assertGreater(impact, 0)
        
    def test_timing_cost_component(self):
        """Test timing cost calculation."""
        decision_time = datetime(2024, 1, 1, 9, 30)
        execution_times = [
            datetime(2024, 1, 1, 9, 31),
            datetime(2024, 1, 1, 9, 33),
            datetime(2024, 1, 1, 9, 35)
        ]
        price_path = np.array([100.0, 100.1, 100.15, 100.2])
        
        timing_cost = self.is_analyzer.calculate_timing_cost(
            decision_time=decision_time,
            execution_times=execution_times,
            price_path=price_path
        )
        
        self.assertIsNotNone(timing_cost)


class TestOrderBook(unittest.TestCase):
    """Test cases for order book dynamics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.orderbook = OrderBook()
        
    def test_add_limit_order(self):
        """Test adding limit orders for BTCUSDT."""
        self.orderbook.add_order('buy', 45000.0, 0.5, order_id='1')
        self.orderbook.add_order('buy', 44950.0, 0.3, order_id='2')
        self.orderbook.add_order('sell', 45050.0, 0.4, order_id='3')
        
        self.assertEqual(len(self.orderbook.bids), 2)
        self.assertEqual(len(self.orderbook.asks), 1)
        
    def test_best_bid_ask(self):
        """Test best bid/ask retrieval for crypto."""
        self.orderbook.add_order('buy', 45000.0, 0.5, order_id='1')
        self.orderbook.add_order('buy', 44950.0, 0.3, order_id='2')
        self.orderbook.add_order('sell', 45050.0, 0.4, order_id='3')
        self.orderbook.add_order('sell', 45100.0, 0.2, order_id='4')
        
        best_bid = self.orderbook.get_best_bid()
        best_ask = self.orderbook.get_best_ask()
        
        self.assertEqual(best_bid['price'], 45000.0)
        self.assertEqual(best_ask['price'], 45050.0)
        
    def test_spread_calculation(self):
        """Test bid-ask spread calculation for crypto."""
        self.orderbook.add_order('buy', 45000.0, 0.5, order_id='1')
        self.orderbook.add_order('sell', 45050.0, 0.4, order_id='2')
        
        spread = self.orderbook.get_spread()
        self.assertEqual(spread, 50.0)
        
    def test_market_order_execution(self):
        """Test market order execution for crypto."""
        self.orderbook.add_order('sell', 45050.0, 0.3, order_id='1')
        self.orderbook.add_order('sell', 45100.0, 0.2, order_id='2')
        
        # Execute market buy order for 0.4 BTC
        result = self.orderbook.execute_market_order('buy', 0.4)
        
        self.assertEqual(result['total_quantity'], 0.4)
        self.assertGreater(result['average_price'], 45050.0)
        
    def test_order_book_depth(self):
        """Test order book depth calculation for crypto."""
        self.orderbook.add_order('buy', 45000.0, 0.5, order_id='1')
        self.orderbook.add_order('buy', 44950.0, 0.3, order_id='2')
        self.orderbook.add_order('buy', 44900.0, 0.2, order_id='3')
        
        depth = self.orderbook.get_depth(side='buy', levels=3)
        
        self.assertEqual(len(depth), 3)
        self.assertEqual(depth[0]['quantity'], 0.5)
        
    def test_order_book_imbalance(self):
        """Test order book imbalance calculation for crypto."""
        self.orderbook.add_order('buy', 45000.0, 1.0, order_id='1')
        self.orderbook.add_order('sell', 45050.0, 0.5, order_id='2')
        
        imbalance = self.orderbook.calculate_imbalance()
        
        # More buy volume - positive imbalance
        self.assertGreater(imbalance, 0)
        
    def test_order_cancellation(self):
        """Test order cancellation for crypto."""
        self.orderbook.add_order('buy', 45000.0, 0.5, order_id='1')
        self.assertEqual(len(self.orderbook.bids), 1)
        
        self.orderbook.cancel_order('1')
        self.assertEqual(len(self.orderbook.bids), 0)


class TestLiquidityModels(unittest.TestCase):
    """Test cases for liquidity models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.liquidity_model = LiquidityModel()
        self.impact_model = ImpactModel()
        
    def test_bid_ask_spread_model(self):
        """Test bid-ask spread modeling."""
        volatility = 0.02
        volume = 1000000
        price = 100.0
        
        spread = self.liquidity_model.estimate_spread(
            volatility=volatility,
            volume=volume,
            price=price
        )
        
        self.assertGreater(spread, 0)
        
        # Higher volatility should increase spread
        high_vol_spread = self.liquidity_model.estimate_spread(
            volatility=0.04,
            volume=volume,
            price=price
        )
        self.assertGreater(high_vol_spread, spread)
        
    def test_kyle_lambda(self):
        """Test Kyle's lambda (price impact coefficient)."""
        volatility = 0.02
        trading_volume = 1000000
        
        lambda_value = self.impact_model.kyle_lambda(
            volatility=volatility,
            trading_volume=trading_volume
        )
        
        self.assertGreater(lambda_value, 0)
        
    def test_temporary_impact(self):
        """Test temporary market impact."""
        trade_size = 10000
        daily_volume = 1000000
        
        impact = self.impact_model.temporary_impact(
            trade_size=trade_size,
            daily_volume=daily_volume
        )
        
        self.assertGreater(impact, 0)
        
        # Larger trades have more impact
        large_impact = self.impact_model.temporary_impact(
            trade_size=trade_size * 2,
            daily_volume=daily_volume
        )
        self.assertGreater(large_impact, impact)
        
    def test_permanent_impact(self):
        """Test permanent market impact."""
        trade_size = 10000
        daily_volume = 1000000
        
        impact = self.impact_model.permanent_impact(
            trade_size=trade_size,
            daily_volume=daily_volume
        )
        
        self.assertGreater(impact, 0)
        
        # Should be less than temporary impact
        temp_impact = self.impact_model.temporary_impact(
            trade_size=trade_size,
            daily_volume=daily_volume
        )
        self.assertLess(impact, temp_impact)
        
    def test_liquidity_cost_analysis(self):
        """Test comprehensive liquidity cost analysis."""
        order_size = 50000
        avg_daily_volume = 1000000
        volatility = 0.02
        spread = 0.01
        
        costs = self.liquidity_model.analyze_costs(
            order_size=order_size,
            avg_daily_volume=avg_daily_volume,
            volatility=volatility,
            spread=spread
        )
        
        self.assertIn('spread_cost', costs)
        self.assertIn('impact_cost', costs)
        self.assertIn('total_cost', costs)
        
        # Total should be sum of components
        expected_total = costs['spread_cost'] + costs['impact_cost']
        assert_almost_equal(costs['total_cost'], expected_total, decimal=6)


class TestToxicityDetection(unittest.TestCase):
    """Test cases for order flow toxicity detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = ToxicityDetector()
        
    def test_vpin_calculation(self):
        """Test Volume-Synchronized Probability of Informed Trading (VPIN)."""
        # Generate sample trade data
        n_trades = 1000
        volumes = np.random.exponential(100, n_trades)
        price_changes = np.random.choice([-1, 1], n_trades)
        
        vpin = self.detector.calculate_vpin(
            volumes=volumes,
            price_changes=price_changes,
            n_buckets=50
        )
        
        self.assertGreater(vpin, 0)
        self.assertLess(vpin, 1)
        
    def test_order_flow_imbalance(self):
        """Test order flow imbalance metric."""
        buy_volume = 60000
        sell_volume = 40000
        
        ofi = self.detector.calculate_order_flow_imbalance(
            buy_volume=buy_volume,
            sell_volume=sell_volume
        )
        
        # Should be positive (more buy volume)
        self.assertGreater(ofi, 0)
        
        # Magnitude check
        self.assertLess(abs(ofi), 1)
        
    def test_adverse_selection_indicator(self):
        """Test adverse selection detection."""
        # Trades followed by adverse price movement
        trade_prices = np.array([100, 100.05, 100.1, 100.15])
        subsequent_prices = np.array([100.2, 100.25, 100.3, 100.35])
        trade_sides = np.array([1, 1, 1, 1])  # All buys
        
        adverse_selection = self.detector.calculate_adverse_selection(
            trade_prices=trade_prices,
            subsequent_prices=subsequent_prices,
            trade_sides=trade_sides
        )
        
        # Should detect adverse selection (price moved against trader)
        self.assertGreater(adverse_selection, 0)
        
    def test_toxicity_score(self):
        """Test composite toxicity score."""
        trade_data = {
            'volumes': np.random.exponential(100, 100),
            'price_changes': np.random.choice([-1, 0, 1], 100),
            'spreads': np.random.uniform(0.01, 0.05, 100),
            'trade_sizes': np.random.exponential(500, 100)
        }
        
        toxicity = self.detector.calculate_toxicity_score(**trade_data)
        
        self.assertGreater(toxicity, 0)
        self.assertLess(toxicity, 1)


def run_tests():
    """Run all execution tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTWAPAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestVWAPAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestArrivalPriceAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestImplementationShortfall))
    suite.addTests(loader.loadTestsFromTestCase(TestOrderBook))
    suite.addTests(loader.loadTestsFromTestCase(TestLiquidityModels))
    suite.addTests(loader.loadTestsFromTestCase(TestToxicityDetection))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
