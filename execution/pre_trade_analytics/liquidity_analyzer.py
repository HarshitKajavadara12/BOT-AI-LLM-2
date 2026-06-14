"""
Liquidity Analyzer
Advanced market liquidity assessment and analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import time
from collections import deque, defaultdict
import warnings


class LiquidityMeasure(Enum):
    """Types of liquidity measures"""
    BID_ASK_SPREAD = "BID_ASK_SPREAD"           # Bid-ask spread measures
    MARKET_DEPTH = "MARKET_DEPTH"               # Order book depth
    PRICE_IMPACT = "PRICE_IMPACT"               # Price impact measures  
    VOLUME_TURNOVER = "VOLUME_TURNOVER"         # Volume and turnover
    RESILIENCE = "RESILIENCE"                   # Market resilience
    IMMEDIACY = "IMMEDIACY"                     # Speed of execution
    TRADING_COSTS = "TRADING_COSTS"             # Overall trading costs


class LiquidityRegime(Enum):
    """Liquidity regime classification"""
    VERY_HIGH = "VERY_HIGH"     # Excellent liquidity conditions
    HIGH = "HIGH"               # Good liquidity conditions  
    NORMAL = "NORMAL"           # Average liquidity conditions
    LOW = "LOW"                 # Poor liquidity conditions
    VERY_LOW = "VERY_LOW"       # Very poor liquidity conditions
    STRESSED = "STRESSED"       # Crisis/stress conditions


@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Spread measures
    bid_ask_spread_dollars: float = 0.0
    bid_ask_spread_bps: float = 0.0
    relative_spread: float = 0.0          # Spread relative to mid price
    effective_spread: float = 0.0         # Actual execution spread
    
    # Depth measures  
    bid_depth_shares: float = 0.0         # Shares at bid
    ask_depth_shares: float = 0.0         # Shares at ask
    total_depth_shares: float = 0.0       # Total depth (bid + ask)
    depth_dollars: float = 0.0            # Dollar depth
    depth_imbalance: float = 0.0          # (Ask depth - Bid depth) / Total depth
    
    # Price impact measures
    temporary_impact_bps: float = 0.0     # Temporary price impact
    permanent_impact_bps: float = 0.0     # Permanent price impact
    market_impact_coefficient: float = 0.0 # Impact coefficient
    kyle_lambda: float = 0.0              # Kyle's lambda (price impact)
    
    # Volume measures
    volume_turnover_ratio: float = 0.0    # Volume / Shares outstanding
    average_trade_size: float = 0.0       # Average trade size
    trade_frequency: float = 0.0          # Trades per unit time
    volume_concentration: float = 0.0     # Volume concentration measure
    
    # Resilience measures  
    price_reversal_speed: float = 0.0     # Speed of price mean reversion
    order_flow_toxicity: float = 0.0      # Toxicity of order flow
    resilience_score: float = 0.0         # Overall resilience
    
    # Immediacy measures
    time_to_execution: float = 0.0        # Expected time to execute
    execution_probability: float = 0.0    # Probability of immediate execution
    fill_rate: float = 0.0                # Order fill rate
    
    # Composite measures
    liquidity_score: float = 0.0          # Overall liquidity score (0-1)
    liquidity_percentile: float = 50.0    # Percentile vs. historical
    liquidity_regime: LiquidityRegime = LiquidityRegime.NORMAL
    
    # Risk measures
    liquidity_risk: float = 0.0           # Liquidity risk estimate
    funding_liquidity_risk: float = 0.0   # Funding liquidity risk
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.bid_depth_shares > 0 and self.ask_depth_shares > 0:
            self.total_depth_shares = self.bid_depth_shares + self.ask_depth_shares
            self.depth_imbalance = (
                (self.ask_depth_shares - self.bid_depth_shares) / 
                self.total_depth_shares
            )
    
    @property
    def is_liquid(self) -> bool:
        """Check if market is reasonably liquid"""
        return (
            self.liquidity_score > 0.4 and
            self.bid_ask_spread_bps < 50 and
            self.total_depth_shares > 1000
        )
    
    @property
    def is_favorable_for_execution(self) -> bool:
        """Check if conditions are favorable for large execution"""
        return (
            self.is_liquid and
            abs(self.depth_imbalance) < 0.3 and
            self.price_reversal_speed > 0.5
        )


@dataclass
class OrderBookSnapshot:
    """Order book snapshot for liquidity analysis"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Bid side (buyers)
    bid_prices: List[float] = field(default_factory=list)
    bid_sizes: List[float] = field(default_factory=list)
    
    # Ask side (sellers)  
    ask_prices: List[float] = field(default_factory=list)
    ask_sizes: List[float] = field(default_factory=list)
    
    # Market data
    mid_price: float = 0.0
    last_trade_price: float = 0.0
    last_trade_size: float = 0.0
    
    def __post_init__(self):
        """Calculate mid price if not provided"""
        if self.mid_price == 0.0 and self.bid_prices and self.ask_prices:
            self.mid_price = (self.bid_prices[0] + self.ask_prices[0]) / 2.0
    
    @property
    def best_bid(self) -> float:
        """Best bid price"""
        return self.bid_prices[0] if self.bid_prices else 0.0
    
    @property
    def best_ask(self) -> float:
        """Best ask price"""
        return self.ask_prices[0] if self.ask_prices else 0.0
    
    @property
    def spread(self) -> float:
        """Bid-ask spread"""
        return self.best_ask - self.best_bid if self.best_bid and self.best_ask else 0.0
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points"""
        if self.mid_price > 0 and self.spread > 0:
            return (self.spread / self.mid_price) * 10000
        return 0.0
    
    def calculate_depth_at_level(self, levels: int = 5) -> Tuple[float, float]:
        """Calculate cumulative depth at specified levels"""
        
        bid_depth = sum(self.bid_sizes[:min(levels, len(self.bid_sizes))])
        ask_depth = sum(self.ask_sizes[:min(levels, len(self.ask_sizes))])
        
        return bid_depth, ask_depth
    
    def calculate_price_impact(self, quantity: float, side: str = "buy") -> float:
        """Calculate price impact for given quantity"""
        
        if side.lower() == "buy":
            # Buying: walk up the ask side
            prices = self.ask_prices
            sizes = self.ask_sizes
            reference_price = self.best_bid
        else:
            # Selling: walk down the bid side  
            prices = self.bid_prices
            sizes = self.bid_sizes
            reference_price = self.best_ask
        
        if not prices or not sizes or reference_price == 0:
            return 0.0
        
        remaining_quantity = quantity
        total_cost = 0.0
        
        for price, size in zip(prices, sizes):
            if remaining_quantity <= 0:
                break
                
            executed_quantity = min(remaining_quantity, size)
            total_cost += executed_quantity * price
            remaining_quantity -= executed_quantity
        
        if remaining_quantity > 0:
            # Not enough liquidity - penalize heavily
            return 1000.0  # 1000 bps impact
        
        if quantity > 0:
            average_price = total_cost / quantity
            impact = abs(average_price - reference_price) / reference_price
            return impact * 10000  # Convert to bps
        
        return 0.0


class LiquidityDataManager:
    """
    Manages liquidity data collection and storage
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        
        # Data storage
        self.order_book_history: deque = deque(maxlen=max_history)
        self.trade_history: deque = deque(maxlen=max_history)
        self.liquidity_metrics_history: deque = deque(maxlen=max_history)
        
        # Real-time data
        self.current_order_book: Optional[OrderBookSnapshot] = None
        self.last_update_time: Optional[datetime] = None
        
        # Thread safety
        self.data_lock = threading.Lock()
        
    def update_order_book(
        self,
        bid_prices: List[float],
        bid_sizes: List[float], 
        ask_prices: List[float],
        ask_sizes: List[float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update order book data"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        snapshot = OrderBookSnapshot(
            timestamp=timestamp,
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes
        )
        
        with self.data_lock:
            self.current_order_book = snapshot
            self.order_book_history.append(snapshot)
            self.last_update_time = timestamp
    
    def add_trade(
        self,
        price: float,
        size: float,
        timestamp: Optional[datetime] = None,
        side: Optional[str] = None
    ) -> None:
        """Add trade data"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        trade_data = {
            'timestamp': timestamp,
            'price': price,
            'size': size,
            'side': side
        }
        
        with self.data_lock:
            self.trade_history.append(trade_data)
    
    def get_recent_order_books(self, count: int = 100) -> List[OrderBookSnapshot]:
        """Get recent order book snapshots"""
        
        with self.data_lock:
            recent_count = min(count, len(self.order_book_history))
            return list(self.order_book_history)[-recent_count:]
    
    def get_recent_trades(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent trades within specified minutes"""
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.data_lock:
            recent_trades = [
                trade for trade in self.trade_history
                if trade['timestamp'] >= cutoff_time
            ]
        
        return recent_trades
    
    def calculate_volume_statistics(self, minutes: int = 60) -> Dict[str, float]:
        """Calculate volume statistics"""
        
        recent_trades = self.get_recent_trades(minutes)
        
        if not recent_trades:
            return {
                'total_volume': 0.0,
                'trade_count': 0,
                'average_trade_size': 0.0,
                'volume_weighted_price': 0.0
            }
        
        total_volume = sum(trade['size'] for trade in recent_trades)
        total_value = sum(trade['price'] * trade['size'] for trade in recent_trades)
        
        stats = {
            'total_volume': total_volume,
            'trade_count': len(recent_trades),
            'average_trade_size': total_volume / len(recent_trades),
            'volume_weighted_price': total_value / total_volume if total_volume > 0 else 0.0
        }
        
        return stats


class LiquidityAnalyzer:
    """
    Main liquidity analysis engine
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.data_manager = LiquidityDataManager()
        
        # Analysis parameters
        self.analysis_window_minutes = 60
        self.depth_levels = 10
        self.impact_quantity_sizes = [1000, 5000, 10000, 25000, 50000]
        
        # Historical benchmarks
        self.liquidity_percentiles = defaultdict(list)
        self.regime_thresholds = {
            LiquidityRegime.VERY_HIGH: 0.8,
            LiquidityRegime.HIGH: 0.6, 
            LiquidityRegime.NORMAL: 0.4,
            LiquidityRegime.LOW: 0.2,
            LiquidityRegime.VERY_LOW: 0.0
        }
        
        # Real-time monitoring
        self.current_metrics: Optional[LiquidityMetrics] = None
        self.alert_thresholds = {
            'min_liquidity_score': 0.3,
            'max_spread_bps': 100,
            'min_depth_shares': 500
        }
        
    def analyze_current_liquidity(self, detailed: bool = True) -> LiquidityMetrics:
        """
        Analyze current liquidity conditions
        
        Args:
            detailed: Whether to perform detailed analysis
            
        Returns:
            Current liquidity metrics
        """
        
        if not self.data_manager.current_order_book:
            return LiquidityMetrics()
        
        order_book = self.data_manager.current_order_book
        
        # Initialize metrics
        metrics = LiquidityMetrics(timestamp=order_book.timestamp)
        
        # Basic spread measures
        metrics.bid_ask_spread_dollars = order_book.spread
        metrics.bid_ask_spread_bps = order_book.spread_bps
        
        if order_book.mid_price > 0:
            metrics.relative_spread = order_book.spread / order_book.mid_price
        
        # Depth measures
        bid_depth, ask_depth = order_book.calculate_depth_at_level(self.depth_levels)
        metrics.bid_depth_shares = bid_depth
        metrics.ask_depth_shares = ask_depth
        metrics.total_depth_shares = bid_depth + ask_depth
        metrics.depth_dollars = metrics.total_depth_shares * order_book.mid_price
        
        if detailed:
            # Price impact analysis
            metrics = self._calculate_price_impact_measures(metrics, order_book)
            
            # Volume and turnover analysis
            metrics = self._calculate_volume_measures(metrics)
            
            # Resilience analysis
            metrics = self._calculate_resilience_measures(metrics)
            
            # Composite liquidity score
            metrics.liquidity_score = self._calculate_liquidity_score(metrics)
            
            # Regime classification
            metrics.liquidity_regime = self._classify_liquidity_regime(metrics)
            
            # Risk measures
            metrics = self._calculate_liquidity_risks(metrics)
        
        # Update current metrics
        self.current_metrics = metrics
        self.data_manager.liquidity_metrics_history.append(metrics)
        
        return metrics
    
    def _calculate_price_impact_measures(
        self,
        metrics: LiquidityMetrics,
        order_book: OrderBookSnapshot
    ) -> LiquidityMetrics:
        """Calculate price impact measures"""
        
        impact_measurements = []
        
        for quantity in self.impact_quantity_sizes:
            buy_impact = order_book.calculate_price_impact(quantity, "buy")
            sell_impact = order_book.calculate_price_impact(quantity, "sell")
            
            avg_impact = (buy_impact + sell_impact) / 2.0
            impact_measurements.append((quantity, avg_impact))
        
        if impact_measurements:
            # Calculate average impact
            total_impact = sum(impact for _, impact in impact_measurements)
            metrics.temporary_impact_bps = total_impact / len(impact_measurements)
            
            # Estimate Kyle's lambda (linear regression would be better)
            if len(impact_measurements) >= 2:
                # Simple slope calculation
                quantities = [q for q, _ in impact_measurements]
                impacts = [i for _, i in impact_measurements]
                
                # Convert to sqrt quantities (common in microstructure)
                sqrt_quantities = [np.sqrt(q) for q in quantities]
                
                if len(sqrt_quantities) > 1:
                    # Simple linear fit
                    n = len(sqrt_quantities)
                    sum_x = sum(sqrt_quantities)
                    sum_y = sum(impacts)
                    sum_xy = sum(x * y for x, y in zip(sqrt_quantities, impacts))
                    sum_x2 = sum(x * x for x in sqrt_quantities)
                    
                    if n * sum_x2 - sum_x * sum_x != 0:
                        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                        metrics.kyle_lambda = max(0, slope)
        
        return metrics
    
    def _calculate_volume_measures(self, metrics: LiquidityMetrics) -> LiquidityMetrics:
        """Calculate volume and turnover measures"""
        
        volume_stats = self.data_manager.calculate_volume_statistics(
            self.analysis_window_minutes
        )
        
        metrics.average_trade_size = volume_stats['average_trade_size']
        metrics.trade_frequency = volume_stats['trade_count'] / (self.analysis_window_minutes / 60.0)
        
        # Volume turnover (would need shares outstanding data)
        # For now, use a proxy based on recent volume
        if volume_stats['total_volume'] > 0:
            # Simplified turnover estimate
            estimated_float = volume_stats['total_volume'] * 24  # Rough estimate
            metrics.volume_turnover_ratio = volume_stats['total_volume'] / estimated_float
        
        return metrics
    
    def _calculate_resilience_measures(self, metrics: LiquidityMetrics) -> LiquidityMetrics:
        """Calculate market resilience measures""" 
        
        recent_books = self.data_manager.get_recent_order_books(50)
        
        if len(recent_books) < 10:
            return metrics
        
        # Price reversal analysis
        mid_prices = [book.mid_price for book in recent_books if book.mid_price > 0]
        
        if len(mid_prices) >= 10:
            # Calculate price mean reversion speed
            returns = [
                (mid_prices[i] - mid_prices[i-1]) / mid_prices[i-1]
                for i in range(1, len(mid_prices))
            ]
            
            # Simple AR(1) estimation for mean reversion
            if len(returns) >= 5:
                lagged_returns = returns[:-1]
                current_returns = returns[1:]
                
                # Calculate correlation (proxy for mean reversion)
                if len(lagged_returns) > 1:
                    correlation = np.corrcoef(lagged_returns, current_returns)[0, 1]
                    
                    # Negative correlation indicates mean reversion
                    metrics.price_reversal_speed = max(0, -correlation)
        
        # Order flow toxicity (simplified)
        recent_trades = self.data_manager.get_recent_trades(30)  # 30 minutes
        
        if len(recent_trades) >= 10:
            # Calculate price impact of recent trades (simplified)
            impacts = []
            
            for i, trade in enumerate(recent_trades[1:], 1):
                prev_trade = recent_trades[i-1]
                price_change = (trade['price'] - prev_trade['price']) / prev_trade['price']
                size_impact = trade['size'] / metrics.average_trade_size if metrics.average_trade_size > 0 else 1
                
                normalized_impact = abs(price_change) / size_impact if size_impact > 0 else 0
                impacts.append(normalized_impact)
            
            if impacts:
                metrics.order_flow_toxicity = sum(impacts) / len(impacts)
        
        # Overall resilience score
        resilience_factors = [
            metrics.price_reversal_speed,
            1.0 - min(1.0, metrics.order_flow_toxicity),
            min(1.0, metrics.total_depth_shares / 10000)  # Depth factor
        ]
        
        metrics.resilience_score = sum(resilience_factors) / len(resilience_factors)
        
        return metrics
    
    def _calculate_liquidity_score(self, metrics: LiquidityMetrics) -> float:
        """Calculate composite liquidity score"""
        
        # Component scores (0-1 scale)
        spread_score = max(0, 1.0 - metrics.bid_ask_spread_bps / 100.0)  # Penalize spreads > 100 bps
        depth_score = min(1.0, metrics.total_depth_shares / 50000)        # Scale by 50k shares
        impact_score = max(0, 1.0 - metrics.temporary_impact_bps / 50.0)  # Penalize impact > 50 bps
        resilience_score = metrics.resilience_score
        
        # Volume score
        volume_score = 0.5  # Default
        if metrics.trade_frequency > 0:
            volume_score = min(1.0, metrics.trade_frequency / 10.0)  # Scale by 10 trades/hour
        
        # Weighted combination
        weights = {
            'spread': 0.25,
            'depth': 0.25,
            'impact': 0.20,
            'resilience': 0.20,
            'volume': 0.10
        }
        
        composite_score = (
            weights['spread'] * spread_score +
            weights['depth'] * depth_score +
            weights['impact'] * impact_score +
            weights['resilience'] * resilience_score +
            weights['volume'] * volume_score
        )
        
        return max(0.0, min(1.0, composite_score))
    
    def _classify_liquidity_regime(self, metrics: LiquidityMetrics) -> LiquidityRegime:
        """Classify current liquidity regime"""
        
        score = metrics.liquidity_score
        
        if score >= self.regime_thresholds[LiquidityRegime.VERY_HIGH]:
            return LiquidityRegime.VERY_HIGH
        elif score >= self.regime_thresholds[LiquidityRegime.HIGH]:
            return LiquidityRegime.HIGH
        elif score >= self.regime_thresholds[LiquidityRegime.NORMAL]:
            return LiquidityRegime.NORMAL
        elif score >= self.regime_thresholds[LiquidityRegime.LOW]:
            return LiquidityRegime.LOW
        else:
            return LiquidityRegime.VERY_LOW
    
    def _calculate_liquidity_risks(self, metrics: LiquidityMetrics) -> LiquidityMetrics:
        """Calculate liquidity risk measures"""
        
        # Basic liquidity risk (inverse of liquidity score)
        metrics.liquidity_risk = 1.0 - metrics.liquidity_score
        
        # Funding liquidity risk (simplified)
        # Higher risk when spreads are wide and depth is low
        spread_risk = min(1.0, metrics.bid_ask_spread_bps / 100.0)
        depth_risk = max(0, 1.0 - metrics.total_depth_shares / 10000)
        
        metrics.funding_liquidity_risk = (spread_risk + depth_risk) / 2.0
        
        return metrics
    
    def estimate_execution_cost(
        self,
        quantity: float,
        side: str = "buy",
        urgency: str = "normal"
    ) -> Dict[str, float]:
        """
        Estimate execution cost for given order
        
        Args:
            quantity: Order quantity
            side: Order side ("buy" or "sell")
            urgency: Execution urgency ("passive", "normal", "aggressive")
            
        Returns:
            Cost breakdown dictionary
        """
        
        if not self.current_metrics or not self.data_manager.current_order_book:
            return {'error': 'No current market data'}
        
        order_book = self.data_manager.current_order_book
        metrics = self.current_metrics
        
        # Base price impact
        base_impact_bps = order_book.calculate_price_impact(quantity, side)
        
        # Adjust for urgency
        urgency_multipliers = {
            'passive': 0.5,    # Patient execution
            'normal': 1.0,     # Normal execution
            'aggressive': 2.0  # Urgent execution
        }
        
        urgency_multiplier = urgency_multipliers.get(urgency, 1.0)
        adjusted_impact_bps = base_impact_bps * urgency_multiplier
        
        # Spread cost (half spread for normal orders)
        if urgency == "aggressive":
            spread_cost_bps = metrics.bid_ask_spread_bps  # Full spread
        elif urgency == "passive":
            spread_cost_bps = 0.0  # No spread cost for limit orders
        else:
            spread_cost_bps = metrics.bid_ask_spread_bps / 2.0  # Half spread
        
        # Timing risk (opportunity cost)
        timing_risk_bps = 0.0
        if urgency == "passive":
            # Higher timing risk for passive orders
            volatility_estimate = metrics.temporary_impact_bps / 10.0  # Rough estimate
            timing_risk_bps = volatility_estimate * 0.5
        
        # Total cost
        total_cost_bps = adjusted_impact_bps + spread_cost_bps + timing_risk_bps
        
        # Dollar costs
        notional_value = quantity * order_book.mid_price
        total_cost_dollars = (total_cost_bps / 10000) * notional_value
        
        return {
            'total_cost_bps': total_cost_bps,
            'total_cost_dollars': total_cost_dollars,
            'impact_cost_bps': adjusted_impact_bps,
            'spread_cost_bps': spread_cost_bps,
            'timing_risk_bps': timing_risk_bps,
            'notional_value': notional_value,
            'execution_probability': self._estimate_execution_probability(quantity, urgency)
        }
    
    def _estimate_execution_probability(self, quantity: float, urgency: str) -> float:
        """Estimate probability of successful execution"""
        
        if not self.current_metrics:
            return 0.5
        
        metrics = self.current_metrics
        
        # Base probability from liquidity score
        base_prob = metrics.liquidity_score
        
        # Adjust for quantity relative to market depth
        if metrics.total_depth_shares > 0:
            quantity_ratio = quantity / metrics.total_depth_shares
            
            if quantity_ratio > 1.0:
                # Order larger than available depth
                size_penalty = min(0.5, quantity_ratio - 1.0)
                base_prob -= size_penalty
            elif quantity_ratio > 0.5:
                # Large order relative to depth
                size_penalty = (quantity_ratio - 0.5) * 0.2
                base_prob -= size_penalty
        
        # Adjust for urgency
        if urgency == "aggressive":
            base_prob += 0.2  # Higher chance with aggressive orders
        elif urgency == "passive":
            base_prob -= 0.1  # Lower chance with passive orders
        
        return max(0.1, min(0.95, base_prob))
    
    def get_liquidity_alerts(self) -> List[Dict[str, Any]]:
        """Check for liquidity alerts"""
        
        alerts = []
        
        if not self.current_metrics:
            return alerts
        
        metrics = self.current_metrics
        
        # Low liquidity alert
        if metrics.liquidity_score < self.alert_thresholds['min_liquidity_score']:
            alerts.append({
                'type': 'LOW_LIQUIDITY',
                'severity': 'HIGH',
                'message': f"Liquidity score {metrics.liquidity_score:.2f} below threshold {self.alert_thresholds['min_liquidity_score']:.2f}",
                'timestamp': datetime.now()
            })
        
        # Wide spread alert
        if metrics.bid_ask_spread_bps > self.alert_thresholds['max_spread_bps']:
            alerts.append({
                'type': 'WIDE_SPREAD',
                'severity': 'MEDIUM',
                'message': f"Spread {metrics.bid_ask_spread_bps:.1f} bps above threshold {self.alert_thresholds['max_spread_bps']} bps",
                'timestamp': datetime.now()
            })
        
        # Low depth alert
        if metrics.total_depth_shares < self.alert_thresholds['min_depth_shares']:
            alerts.append({
                'type': 'LOW_DEPTH',
                'severity': 'MEDIUM',
                'message': f"Market depth {metrics.total_depth_shares:.0f} shares below threshold {self.alert_thresholds['min_depth_shares']} shares",
                'timestamp': datetime.now()
            })
        
        return alerts
    
    def generate_liquidity_report(self) -> Dict[str, Any]:
        """Generate comprehensive liquidity report"""
        
        if not self.current_metrics:
            return {'status': 'no_data'}
        
        metrics = self.current_metrics
        recent_metrics = list(self.data_manager.liquidity_metrics_history)[-50:]
        
        # Historical analysis
        if len(recent_metrics) > 10:
            recent_scores = [m.liquidity_score for m in recent_metrics]
            recent_spreads = [m.bid_ask_spread_bps for m in recent_metrics]
            
            score_trend = recent_scores[-1] - recent_scores[0] if len(recent_scores) > 1 else 0
            spread_trend = recent_spreads[-1] - recent_spreads[0] if len(recent_spreads) > 1 else 0
        else:
            score_trend = 0
            spread_trend = 0
        
        report = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'current_conditions': {
                'liquidity_score': metrics.liquidity_score,
                'liquidity_regime': metrics.liquidity_regime.value,
                'spread_bps': metrics.bid_ask_spread_bps,
                'depth_shares': metrics.total_depth_shares,
                'depth_dollars': metrics.depth_dollars,
                'kyle_lambda': metrics.kyle_lambda,
                'resilience_score': metrics.resilience_score
            },
            'risk_assessment': {
                'liquidity_risk': metrics.liquidity_risk,
                'funding_liquidity_risk': metrics.funding_liquidity_risk,
                'is_favorable_for_execution': metrics.is_favorable_for_execution
            },
            'trends': {
                'liquidity_score_trend': score_trend,
                'spread_trend_bps': spread_trend,
                'trend_direction': 'improving' if score_trend > 0.01 else 'deteriorating' if score_trend < -0.01 else 'stable'
            },
            'execution_guidance': {
                'recommended_urgency': self._recommend_execution_urgency(metrics),
                'max_recommended_size': self._recommend_max_order_size(metrics),
                'optimal_execution_style': self._recommend_execution_style(metrics)
            },
            'alerts': self.get_liquidity_alerts()
        }
        
        return report
    
    def _recommend_execution_urgency(self, metrics: LiquidityMetrics) -> str:
        """Recommend execution urgency based on conditions"""
        
        if metrics.liquidity_score > 0.7:
            return "normal"  # Good conditions, normal execution
        elif metrics.liquidity_score > 0.4:
            return "passive"  # Moderate conditions, be patient
        else:
            return "aggressive"  # Poor conditions, execute quickly
    
    def _recommend_max_order_size(self, metrics: LiquidityMetrics) -> float:
        """Recommend maximum order size"""
        
        # Conservative approach: don't exceed 25% of available depth
        return metrics.total_depth_shares * 0.25
    
    def _recommend_execution_style(self, metrics: LiquidityMetrics) -> str:
        """Recommend execution style"""
        
        if metrics.resilience_score > 0.6 and metrics.liquidity_score > 0.6:
            return "iceberg"  # Good conditions for hidden orders
        elif metrics.liquidity_score > 0.5:
            return "twap"     # Moderate conditions, time-weighted
        else:
            return "aggressive"  # Poor conditions, execute quickly


if __name__ == "__main__":
    import random
    
    # Example usage and testing
    print("Testing Liquidity Analyzer...")
    
    # Create analyzer
    analyzer = LiquidityAnalyzer("AAPL")
    
    # Simulate order book data
    print(f"\nSimulating order book updates...")
    
    base_price = 150.0
    
    for i in range(100):
        # Generate realistic order book
        mid_price = base_price + random.gauss(0, 0.5)
        spread = random.uniform(0.01, 0.05)
        
        # Bid side
        bid_prices = []
        bid_sizes = []
        for level in range(10):
            price = mid_price - spread/2 - level * 0.01
            size = random.randint(100, 2000) * (10 - level)  # Decreasing size
            bid_prices.append(price)
            bid_sizes.append(size)
        
        # Ask side
        ask_prices = []
        ask_sizes = []
        for level in range(10):
            price = mid_price + spread/2 + level * 0.01
            size = random.randint(100, 2000) * (10 - level)  # Decreasing size
            ask_prices.append(price)
            ask_sizes.append(size)
        
        # Update order book
        analyzer.data_manager.update_order_book(
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes
        )
        
        # Add some trades
        if random.random() < 0.7:  # 70% chance of trade
            trade_price = mid_price + random.gauss(0, 0.02)
            trade_size = random.randint(100, 1000)
            analyzer.data_manager.add_trade(trade_price, trade_size)
        
        # Add small random walk
        base_price += random.gauss(0, 0.01)
    
    # Analyze current liquidity
    print(f"Analyzing current liquidity...")
    
    metrics = analyzer.analyze_current_liquidity(detailed=True)
    
    print(f"\nCurrent Liquidity Metrics:")
    print(f"  Liquidity Score: {metrics.liquidity_score:.3f}")
    print(f"  Liquidity Regime: {metrics.liquidity_regime.value}")
    print(f"  Bid-Ask Spread: {metrics.bid_ask_spread_bps:.1f} bps")
    print(f"  Market Depth: {metrics.total_depth_shares:,.0f} shares (${metrics.depth_dollars:,.0f})")
    print(f"  Price Impact: {metrics.temporary_impact_bps:.1f} bps")
    print(f"  Kyle's Lambda: {metrics.kyle_lambda:.6f}")
    print(f"  Resilience Score: {metrics.resilience_score:.3f}")
    print(f"  Liquidity Risk: {metrics.liquidity_risk:.3f}")
    
    # Test execution cost estimation
    print(f"\nExecution Cost Estimates:")
    
    test_orders = [
        (5000, "buy", "passive"),
        (10000, "buy", "normal"), 
        (25000, "sell", "aggressive")
    ]
    
    for quantity, side, urgency in test_orders:
        cost_estimate = analyzer.estimate_execution_cost(quantity, side, urgency)
        
        print(f"  {quantity:,} shares {side} ({urgency}):")
        print(f"    Total Cost: {cost_estimate['total_cost_bps']:.1f} bps (${cost_estimate['total_cost_dollars']:,.0f})")
        print(f"    Impact: {cost_estimate['impact_cost_bps']:.1f} bps")
        print(f"    Spread: {cost_estimate['spread_cost_bps']:.1f} bps")
        print(f"    Execution Probability: {cost_estimate['execution_probability']:.1%}")
    
    # Generate liquidity report
    print(f"\nGenerating Liquidity Report...")
    
    report = analyzer.generate_liquidity_report()
    
    print(f"Liquidity Report for {report['symbol']}:")
    print(f"  Current Conditions:")
    conditions = report['current_conditions']
    for key, value in conditions.items():
        if isinstance(value, float):
            print(f"    {key.replace('_', ' ').title()}: {value:.3f}")
        else:
            print(f"    {key.replace('_', ' ').title()}: {value}")
    
    print(f"  Risk Assessment:")
    risk = report['risk_assessment']
    for key, value in risk.items():
        if isinstance(value, bool):
            print(f"    {key.replace('_', ' ').title()}: {'Yes' if value else 'No'}")
        else:
            print(f"    {key.replace('_', ' ').title()}: {value:.3f}")
    
    print(f"  Execution Guidance:")
    guidance = report['execution_guidance']
    for key, value in guidance.items():
        if isinstance(value, float):
            print(f"    {key.replace('_', ' ').title()}: {value:,.0f}")
        else:
            print(f"    {key.replace('_', ' ').title()}: {value}")
    
    # Check for alerts
    alerts = analyzer.get_liquidity_alerts()
    
    if alerts:
        print(f"\n  Active Alerts:")
        for alert in alerts:
            print(f"    {alert['type']} ({alert['severity']}): {alert['message']}")
    else:
        print(f"\n  No active alerts")
    
    print("\nLiquidity analysis testing completed!")