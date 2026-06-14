"""
Smart Order Routing for QUANTUM-FORGE
Implements intelligent order routing across multiple venues and dark pools.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from numba import jit, prange
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import time
from collections import deque
import heapq
warnings.filterwarnings('ignore')

class VenueType(Enum):
    """Types of trading venues."""
    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    ECN = "electronic_communication_network"
    CROSSING_NETWORK = "crossing_network"
    INTERNALIZER = "internalizer"
    RETAIL_NETWORK = "retail_network"

class VenueFeature(Enum):
    """Venue characteristics and features."""
    LOW_LATENCY = "low_latency"
    HIGH_LIQUIDITY = "high_liquidity"
    MINIMAL_IMPACT = "minimal_impact"
    COST_EFFECTIVE = "cost_effective"
    HIDDEN_LIQUIDITY = "hidden_liquidity"
    RETAIL_FLOW = "retail_flow"
    INSTITUTIONAL_FLOW = "institutional_flow"
    ANTI_GAMING = "anti_gaming"

class OrderType(Enum):
    """Enhanced order types for smart routing."""
    MARKET = "market"
    LIMIT = "limit"
    HIDDEN = "hidden"
    ICEBERG = "iceberg"
    PEGGED = "pegged"
    MIDPOINT = "midpoint"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    RESERVE = "reserve"

@dataclass
class VenueCharacteristics:
    """Trading venue characteristics."""
    venue_id: str
    venue_type: VenueType
    features: List[VenueFeature]
    average_latency: float  # microseconds
    tick_size: float
    min_size: int
    max_size: int
    maker_fee: float  # basis points
    taker_fee: float  # basis points
    market_share: float  # percentage of total volume
    typical_spread: float
    hidden_size_ratio: float  # ratio of hidden to displayed liquidity
    adverse_selection_cost: float
    fill_probability: float
    venue_priority: int  # Lower number = higher priority

@dataclass
class VenueLiquidity:
    """Real-time liquidity information for a venue."""
    venue_id: str
    timestamp: float
    bid_size: float
    ask_size: float
    bid_price: float
    ask_price: float
    last_trade_price: float
    volume_rate: float
    estimated_hidden_size: float
    liquidity_score: float
    volatility: float

@dataclass
class RoutingDecision:
    """Order routing decision."""
    venue_id: str
    quantity: float
    order_type: OrderType
    limit_price: Optional[float]
    expected_fill_rate: float
    expected_cost: float
    priority_score: float
    routing_reason: str

@dataclass
class RoutingResult:
    """Result of smart order routing."""
    total_quantity: float
    filled_quantity: float
    average_fill_price: float
    total_cost: float
    venue_allocations: List[RoutingDecision]
    execution_time: float
    success_rate: float
    cost_savings: float  # vs naive single-venue execution

class VenueModel:
    """Model for individual trading venue behavior."""
    
    def __init__(self, characteristics: VenueCharacteristics):
        """Initialize venue model."""
        self.characteristics = characteristics
        self.historical_performance = {
            'fill_rates': deque(maxlen=1000),
            'execution_costs': deque(maxlen=1000),
            'latencies': deque(maxlen=1000),
            'adverse_selection_events': deque(maxlen=1000)
        }
        
    def update_performance(self, fill_rate: float, cost: float, 
                         latency: float, adverse_selection: bool):
        """Update venue performance metrics."""
        self.historical_performance['fill_rates'].append(fill_rate)
        self.historical_performance['execution_costs'].append(cost)
        self.historical_performance['latencies'].append(latency)
        self.historical_performance['adverse_selection_events'].append(adverse_selection)
    
    def calculate_expected_fill_rate(self, quantity: float, 
                                   order_type: OrderType,
                                   market_conditions: Dict) -> float:
        """
        Calculate expected fill rate for order.
        
        Args:
            quantity: Order quantity
            order_type: Type of order
            market_conditions: Current market conditions
        
        Returns:
            Expected fill rate (0-1)
        """
        base_fill_rate = self.characteristics.fill_probability
        
        # Adjust for order size relative to venue capacity
        size_factor = min(1.0, self.characteristics.max_size / max(quantity, 1))
        
        # Adjust for order type
        if order_type == OrderType.MARKET:
            type_adjustment = 1.1  # Market orders fill more readily
        elif order_type == OrderType.HIDDEN:
            type_adjustment = 0.9  # Hidden orders may have lower fill rates
        elif order_type == OrderType.LIMIT:
            type_adjustment = 0.85  # Limit orders depend on market movement
        else:
            type_adjustment = 1.0
        
        # Adjust for market conditions
        volatility = market_conditions.get('volatility', 0.02)
        spread = market_conditions.get('spread', 0.001)
        
        condition_adjustment = 1.0 - 2 * volatility - 10 * spread
        condition_adjustment = max(0.3, condition_adjustment)
        
        # Venue-specific adjustments
        if VenueFeature.HIGH_LIQUIDITY in self.characteristics.features:
            liquidity_boost = 1.1
        elif VenueFeature.HIDDEN_LIQUIDITY in self.characteristics.features:
            liquidity_boost = 1.05
        else:
            liquidity_boost = 1.0
        
        expected_rate = (base_fill_rate * size_factor * type_adjustment * 
                        condition_adjustment * liquidity_boost)
        
        return min(1.0, max(0.1, expected_rate))
    
    def calculate_execution_cost(self, quantity: float, 
                               order_type: OrderType,
                               market_conditions: Dict,
                               is_aggressive: bool = False) -> float:
        """
        Calculate expected execution cost.
        
        Args:
            quantity: Order quantity
            order_type: Type of order
            market_conditions: Current market conditions
            is_aggressive: Whether order removes liquidity
        
        Returns:
            Expected cost in basis points
        """
        # Base fee
        if is_aggressive:
            base_cost = self.characteristics.taker_fee
        else:
            base_cost = -self.characteristics.maker_fee  # Rebate
        
        # Market impact cost
        market_share = self.characteristics.market_share / 100
        volume_rate = market_conditions.get('volume_rate', 1000)
        participation_rate = quantity / (volume_rate * market_share + 1e-6)
        
        # Square-root market impact
        impact_cost = 10 * np.sqrt(participation_rate)  # basis points
        
        # Adverse selection cost (for aggressive orders)
        if is_aggressive:
            adverse_cost = self.characteristics.adverse_selection_cost * participation_rate
        else:
            adverse_cost = 0.0
        
        # Spread cost (half-spread for aggressive orders)
        spread = market_conditions.get('spread', 0.001)
        if is_aggressive:
            spread_cost = spread * 10000 / 2  # Half spread in bp
        else:
            spread_cost = 0.0
        
        total_cost = base_cost + impact_cost + adverse_cost + spread_cost
        
        return total_cost
    
    def calculate_priority_score(self, quantity: float, urgency: float,
                               market_conditions: Dict) -> float:
        """
        Calculate venue priority score for order.
        
        Args:
            quantity: Order quantity
            urgency: Execution urgency (0-1)
            market_conditions: Market conditions
        
        Returns:
            Priority score (higher = better)
        """
        score = 0.0
        
        # Base priority from venue characteristics
        score += (10 - self.characteristics.venue_priority) * 10
        
        # Liquidity matching
        if quantity <= self.characteristics.max_size:
            score += 20
        
        # Feature bonuses based on conditions
        if urgency > 0.7 and VenueFeature.LOW_LATENCY in self.characteristics.features:
            score += 30
        
        if urgency < 0.4 and VenueFeature.COST_EFFECTIVE in self.characteristics.features:
            score += 25
        
        if quantity > 5000 and VenueFeature.HIDDEN_LIQUIDITY in self.characteristics.features:
            score += 20
        
        # Market share bonus
        score += self.characteristics.market_share * 2
        
        # Performance history bonus
        if len(self.historical_performance['fill_rates']) > 10:
            avg_fill_rate = np.mean(list(self.historical_performance['fill_rates'])[-50:])
            score += avg_fill_rate * 50
        
        return score

class LiquidityAggregator:
    """Aggregates liquidity information across venues."""
    
    def __init__(self):
        """Initialize liquidity aggregator."""
        self.venue_liquidity = {}
        self.consolidated_book = {'bids': [], 'asks': []}
        self.last_update = 0
        
    def update_venue_liquidity(self, venue_id: str, liquidity: VenueLiquidity):
        """Update liquidity for specific venue."""
        self.venue_liquidity[venue_id] = liquidity
        self.last_update = time.time()
        self._rebuild_consolidated_book()
    
    def _rebuild_consolidated_book(self):
        """Rebuild consolidated order book."""
        all_bids = []
        all_asks = []
        
        for venue_liquidity in self.venue_liquidity.values():
            if venue_liquidity.bid_price > 0:
                all_bids.append({
                    'price': venue_liquidity.bid_price,
                    'size': venue_liquidity.bid_size + venue_liquidity.estimated_hidden_size,
                    'venue': venue_liquidity.venue_id
                })
            
            if venue_liquidity.ask_price > 0:
                all_asks.append({
                    'price': venue_liquidity.ask_price,
                    'size': venue_liquidity.ask_size + venue_liquidity.estimated_hidden_size,
                    'venue': venue_liquidity.venue_id
                })
        
        # Sort bids descending, asks ascending
        self.consolidated_book['bids'] = sorted(all_bids, key=lambda x: x['price'], reverse=True)
        self.consolidated_book['asks'] = sorted(all_asks, key=lambda x: x['price'])
    
    def get_consolidated_liquidity(self, side: str, quantity: float) -> List[Dict]:
        """
        Get consolidated liquidity for specified quantity.
        
        Args:
            side: 'buy' or 'sell'
            quantity: Required quantity
        
        Returns:
            List of venue liquidity sources
        """
        if side == 'buy':
            levels = self.consolidated_book['asks']
        else:
            levels = self.consolidated_book['bids']
        
        liquidity_sources = []
        remaining_quantity = quantity
        
        for level in levels:
            if remaining_quantity <= 0:
                break
            
            available_quantity = min(level['size'], remaining_quantity)
            
            liquidity_sources.append({
                'venue': level['venue'],
                'price': level['price'],
                'quantity': available_quantity
            })
            
            remaining_quantity -= available_quantity
        
        return liquidity_sources
    
    def calculate_aggregate_metrics(self) -> Dict:
        """Calculate aggregate market metrics."""
        if not self.venue_liquidity:
            return {}
        
        # Best bid/ask across all venues
        best_bid = max((liq.bid_price for liq in self.venue_liquidity.values() 
                       if liq.bid_price > 0), default=0)
        best_ask = min((liq.ask_price for liq in self.venue_liquidity.values() 
                       if liq.ask_price > 0), default=float('inf'))
        
        # Aggregate volume
        total_volume = sum(liq.volume_rate for liq in self.venue_liquidity.values())
        
        # Weighted average spread
        spreads = []
        weights = []
        for liq in self.venue_liquidity.values():
            if liq.ask_price > liq.bid_price > 0:
                spread = (liq.ask_price - liq.bid_price) / liq.bid_price
                spreads.append(spread)
                weights.append(liq.volume_rate)
        
        if spreads:
            avg_spread = np.average(spreads, weights=weights)
        else:
            avg_spread = 0.0
        
        return {
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': best_ask - best_bid if best_ask != float('inf') else 0,
            'total_volume': total_volume,
            'avg_spread': avg_spread,
            'num_venues': len(self.venue_liquidity)
        }

class SmartOrderRouter:
    """Intelligent order routing engine."""
    
    def __init__(self):
        """Initialize smart order router."""
        self.venues = {}
        self.liquidity_aggregator = LiquidityAggregator()
        self.routing_history = []
        self.performance_tracker = {}
        
    def register_venue(self, venue_model: VenueModel):
        """Register trading venue."""
        self.venues[venue_model.characteristics.venue_id] = venue_model
        
    def route_order(self, symbol: str, side: str, quantity: float,
                   urgency: float = 0.5, cost_sensitivity: float = 0.5,
                   max_venues: int = 5,
                   market_conditions: Optional[Dict] = None) -> List[RoutingDecision]:
        """
        Route order across optimal venues.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Total order quantity
            urgency: Execution urgency (0-1)
            cost_sensitivity: Cost sensitivity (0-1)
            max_venues: Maximum number of venues to use
            market_conditions: Current market conditions
        
        Returns:
            List of RoutingDecision objects
        """
        if market_conditions is None:
            market_conditions = {
                'volatility': 0.02,
                'spread': 0.001,
                'volume_rate': 1000,
                'trend': 0.0
            }
        
        # Score all venues
        venue_scores = []
        
        for venue_id, venue_model in self.venues.items():
            # Calculate venue suitability
            fill_rate = venue_model.calculate_expected_fill_rate(
                quantity, OrderType.LIMIT, market_conditions
            )
            
            execution_cost = venue_model.calculate_execution_cost(
                quantity, OrderType.LIMIT, market_conditions, 
                is_aggressive=(urgency > 0.7)
            )
            
            priority_score = venue_model.calculate_priority_score(
                quantity, urgency, market_conditions
            )
            
            # Combined score
            cost_weight = cost_sensitivity
            urgency_weight = urgency
            fill_weight = 1.0 - cost_sensitivity
            
            combined_score = (
                priority_score * 0.3 +
                fill_rate * fill_weight * 50 +
                max(0, 50 - execution_cost) * cost_weight +
                (1.0 / max(venue_model.characteristics.average_latency, 1)) * urgency_weight * 100
            )
            
            venue_scores.append({
                'venue_id': venue_id,
                'venue_model': venue_model,
                'fill_rate': fill_rate,
                'execution_cost': execution_cost,
                'priority_score': priority_score,
                'combined_score': combined_score
            })
        
        # Sort by combined score
        venue_scores.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Allocate quantity across top venues
        routing_decisions = []
        remaining_quantity = quantity
        selected_venues = venue_scores[:max_venues]
        
        # Smart allocation algorithm
        if urgency > 0.8:
            # High urgency: concentrate in top venues
            allocation_weights = self._calculate_urgency_weights(selected_venues)
        elif cost_sensitivity > 0.8:
            # Cost sensitive: optimize for lowest cost
            allocation_weights = self._calculate_cost_weights(selected_venues)
        else:
            # Balanced: diversify across venues
            allocation_weights = self._calculate_balanced_weights(selected_venues)
        
        # Create routing decisions
        for i, venue_score in enumerate(selected_venues):
            if remaining_quantity <= 0:
                break
            
            venue_quantity = min(
                remaining_quantity,
                quantity * allocation_weights[i],
                venue_score['venue_model'].characteristics.max_size
            )
            
            if venue_quantity < venue_score['venue_model'].characteristics.min_size:
                continue
            
            # Determine order type
            order_type = self._select_order_type(
                venue_score['venue_model'], venue_quantity, urgency, market_conditions
            )
            
            # Calculate limit price if needed
            limit_price = None
            if order_type in [OrderType.LIMIT, OrderType.PEGGED, OrderType.MIDPOINT]:
                limit_price = self._calculate_limit_price(
                    side, market_conditions, venue_score['venue_model'], urgency
                )
            
            decision = RoutingDecision(
                venue_id=venue_score['venue_id'],
                quantity=venue_quantity,
                order_type=order_type,
                limit_price=limit_price,
                expected_fill_rate=venue_score['fill_rate'],
                expected_cost=venue_score['execution_cost'],
                priority_score=venue_score['combined_score'],
                routing_reason=self._generate_routing_reason(venue_score, urgency, cost_sensitivity)
            )
            
            routing_decisions.append(decision)
            remaining_quantity -= venue_quantity
        
        # Store routing decision
        self.routing_history.append({
            'timestamp': time.time(),
            'symbol': symbol,
            'side': side,
            'total_quantity': quantity,
            'decisions': routing_decisions,
            'market_conditions': market_conditions
        })
        
        return routing_decisions
    
    def _calculate_urgency_weights(self, venues: List[Dict]) -> np.ndarray:
        """Calculate venue weights for urgent execution."""
        scores = np.array([v['combined_score'] for v in venues])
        
        # Exponential weighting favoring top venues
        exp_scores = np.exp(scores / np.max(scores) * 2)
        weights = exp_scores / np.sum(exp_scores)
        
        # Ensure top venue gets majority
        weights[0] = max(weights[0], 0.4)
        weights = weights / np.sum(weights)
        
        return weights
    
    def _calculate_cost_weights(self, venues: List[Dict]) -> np.ndarray:
        """Calculate venue weights for cost optimization."""
        costs = np.array([v['execution_cost'] for v in venues])
        
        # Inverse cost weighting
        inverse_costs = 1.0 / (costs + 10)  # Add 10bp to avoid division by zero
        weights = inverse_costs / np.sum(inverse_costs)
        
        return weights
    
    def _calculate_balanced_weights(self, venues: List[Dict]) -> np.ndarray:
        """Calculate balanced venue weights."""
        n_venues = len(venues)
        
        # Start with equal weights
        weights = np.ones(n_venues) / n_venues
        
        # Adjust based on combined scores
        scores = np.array([v['combined_score'] for v in venues])
        score_weights = scores / np.sum(scores)
        
        # Blend equal and score-based weights
        weights = 0.5 * weights + 0.5 * score_weights
        
        return weights
    
    def _select_order_type(self, venue_model: VenueModel, quantity: float,
                         urgency: float, market_conditions: Dict) -> OrderType:
        """Select optimal order type for venue."""
        
        # Market orders for high urgency
        if urgency > 0.9:
            return OrderType.MARKET
        
        # Hidden orders for large sizes in dark pools
        if (venue_model.characteristics.venue_type == VenueType.DARK_POOL and
            quantity > 1000):
            return OrderType.HIDDEN
        
        # Iceberg for large visible orders
        if quantity > 5000 and venue_model.characteristics.venue_type == VenueType.EXCHANGE:
            return OrderType.ICEBERG
        
        # Midpoint orders in crossing networks
        if venue_model.characteristics.venue_type == VenueType.CROSSING_NETWORK:
            return OrderType.MIDPOINT
        
        # Default to limit orders
        return OrderType.LIMIT
    
    def _calculate_limit_price(self, side: str, market_conditions: Dict,
                             venue_model: VenueModel, urgency: float) -> float:
        """Calculate optimal limit price."""
        
        mid_price = market_conditions.get('mid_price', 100.0)
        spread = market_conditions.get('spread', 0.001)
        tick_size = venue_model.characteristics.tick_size
        
        # Aggressive pricing for urgent orders
        if urgency > 0.8:
            # Price at or through the market
            if side == 'buy':
                limit_price = mid_price + spread * 0.6
            else:
                limit_price = mid_price - spread * 0.6
        
        elif urgency > 0.5:
            # Mid-market pricing
            if side == 'buy':
                limit_price = mid_price + spread * 0.1
            else:
                limit_price = mid_price - spread * 0.1
        
        else:
            # Patient pricing
            if side == 'buy':
                limit_price = mid_price - spread * 0.3
            else:
                limit_price = mid_price + spread * 0.3
        
        # Round to tick size
        limit_price = round(limit_price / tick_size) * tick_size
        
        return limit_price
    
    def _generate_routing_reason(self, venue_score: Dict, urgency: float,
                               cost_sensitivity: float) -> str:
        """Generate human-readable routing reason."""
        
        venue_model = venue_score['venue_model']
        reasons = []
        
        if venue_score['combined_score'] > 80:
            reasons.append("high_venue_score")
        
        if venue_score['fill_rate'] > 0.9:
            reasons.append("high_fill_probability")
        
        if venue_score['execution_cost'] < 5:
            reasons.append("low_execution_cost")
        
        if VenueFeature.LOW_LATENCY in venue_model.characteristics.features and urgency > 0.7:
            reasons.append("low_latency_venue")
        
        if VenueFeature.HIDDEN_LIQUIDITY in venue_model.characteristics.features:
            reasons.append("hidden_liquidity_access")
        
        if not reasons:
            reasons.append("venue_diversification")
        
        return ", ".join(reasons)
    
    def optimize_routing_strategy(self, historical_performance: List[Dict]) -> Dict:
        """
        Optimize routing strategy based on historical performance.
        
        Args:
            historical_performance: Historical routing performance data
        
        Returns:
            Dictionary of optimized routing parameters
        """
        if len(historical_performance) < 20:
            return {'status': 'insufficient_data'}
        
        # Analyze performance by venue
        venue_performance = {}
        
        for performance in historical_performance:
            for decision in performance['routing_decisions']:
                venue_id = decision.venue_id
                
                if venue_id not in venue_performance:
                    venue_performance[venue_id] = {
                        'total_quantity': 0,
                        'total_cost': 0,
                        'fill_rates': [],
                        'success_count': 0
                    }
                
                venue_perf = venue_performance[venue_id]
                venue_perf['total_quantity'] += decision.quantity
                venue_perf['total_cost'] += decision.expected_cost * decision.quantity
                venue_perf['fill_rates'].append(decision.expected_fill_rate)
                
                if performance.get('success', False):
                    venue_perf['success_count'] += 1
        
        # Calculate venue rankings
        venue_rankings = {}
        
        for venue_id, perf in venue_performance.items():
            if perf['total_quantity'] > 0:
                avg_cost = perf['total_cost'] / perf['total_quantity']
                avg_fill_rate = np.mean(perf['fill_rates'])
                success_rate = perf['success_count'] / len(perf['fill_rates'])
                
                # Combined performance score
                performance_score = (
                    avg_fill_rate * 50 +
                    success_rate * 30 +
                    max(0, 20 - avg_cost) * 20
                )
                
                venue_rankings[venue_id] = {
                    'performance_score': performance_score,
                    'avg_cost': avg_cost,
                    'avg_fill_rate': avg_fill_rate,
                    'success_rate': success_rate
                }
        
        # Optimize allocation weights
        top_venues = sorted(venue_rankings.items(), 
                          key=lambda x: x[1]['performance_score'], 
                          reverse=True)[:5]
        
        optimized_weights = {}
        total_score = sum(venue[1]['performance_score'] for venue in top_venues)
        
        for venue_id, metrics in top_venues:
            weight = metrics['performance_score'] / total_score
            optimized_weights[venue_id] = weight
        
        return {
            'status': 'optimized',
            'venue_rankings': venue_rankings,
            'optimized_weights': optimized_weights,
            'top_venues': [venue[0] for venue in top_venues]
        }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Smart Order Routing System...")
    
    # Create sample venues
    venues_data = [
        {
            'venue_id': 'NYSE',
            'venue_type': VenueType.EXCHANGE,
            'features': [VenueFeature.HIGH_LIQUIDITY, VenueFeature.INSTITUTIONAL_FLOW],
            'average_latency': 100,
            'tick_size': 0.01,
            'min_size': 100,
            'max_size': 1000000,
            'maker_fee': -0.2,
            'taker_fee': 0.3,
            'market_share': 25.0,
            'typical_spread': 0.001,
            'hidden_size_ratio': 0.1,
            'adverse_selection_cost': 2.0,
            'fill_probability': 0.95,
            'venue_priority': 1
        },
        {
            'venue_id': 'DARK_POOL_1',
            'venue_type': VenueType.DARK_POOL,
            'features': [VenueFeature.HIDDEN_LIQUIDITY, VenueFeature.MINIMAL_IMPACT],
            'average_latency': 200,
            'tick_size': 0.01,
            'min_size': 100,
            'max_size': 500000,
            'maker_fee': 0.0,
            'taker_fee': 0.1,
            'market_share': 8.0,
            'typical_spread': 0.0005,
            'hidden_size_ratio': 0.9,
            'adverse_selection_cost': 0.5,
            'fill_probability': 0.85,
            'venue_priority': 3
        },
        {
            'venue_id': 'ECN_FAST',
            'venue_type': VenueType.ECN,
            'features': [VenueFeature.LOW_LATENCY, VenueFeature.COST_EFFECTIVE],
            'average_latency': 50,
            'tick_size': 0.001,
            'min_size': 1,
            'max_size': 100000,
            'maker_fee': -0.3,
            'taker_fee': 0.2,
            'market_share': 15.0,
            'typical_spread': 0.001,
            'hidden_size_ratio': 0.2,
            'adverse_selection_cost': 1.5,
            'fill_probability': 0.92,
            'venue_priority': 2
        },
        {
            'venue_id': 'RETAIL_NET',
            'venue_type': VenueType.RETAIL_NETWORK,
            'features': [VenueFeature.RETAIL_FLOW, VenueFeature.ANTI_GAMING],
            'average_latency': 300,
            'tick_size': 0.01,
            'min_size': 1,
            'max_size': 50000,
            'maker_fee': 0.0,
            'taker_fee': 0.0,
            'market_share': 5.0,
            'typical_spread': 0.002,
            'hidden_size_ratio': 0.3,
            'adverse_selection_cost': 0.1,
            'fill_probability': 0.88,
            'venue_priority': 4
        },
        {
            'venue_id': 'CROSSING_NET',
            'venue_type': VenueType.CROSSING_NETWORK,
            'features': [VenueFeature.COST_EFFECTIVE, VenueFeature.MINIMAL_IMPACT],
            'average_latency': 1000,
            'tick_size': 0.01,
            'min_size': 1000,
            'max_size': 2000000,
            'maker_fee': 0.0,
            'taker_fee': 0.05,
            'market_share': 3.0,
            'typical_spread': 0.0,
            'hidden_size_ratio': 1.0,
            'adverse_selection_cost': 0.0,
            'fill_probability': 0.75,
            'venue_priority': 5
        }
    ]
    
    # Initialize smart order router
    router = SmartOrderRouter()
    
    # Register venues
    for venue_data in venues_data:
        characteristics = VenueCharacteristics(**venue_data)
        venue_model = VenueModel(characteristics)
        router.register_venue(venue_model)
    
    print(f"Registered {len(router.venues)} trading venues")
    
    # Sample market conditions
    market_conditions = {
        'mid_price': 150.0,
        'volatility': 0.025,
        'spread': 0.02,
        'volume_rate': 2000,
        'trend': 0.001
    }
    
    print(f"Market conditions: Mid=${market_conditions['mid_price']:.2f}, "
          f"Vol={market_conditions['volatility']*100:.1f}%, "
          f"Spread=${market_conditions['spread']:.3f}")
    
    # Test different order scenarios
    test_scenarios = [
        {
            'name': 'Large Institutional Order (Low Urgency)',
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 50000,
            'urgency': 0.2,
            'cost_sensitivity': 0.8
        },
        {
            'name': 'Medium Urgent Order',
            'symbol': 'AAPL',
            'side': 'sell',
            'quantity': 10000,
            'urgency': 0.7,
            'cost_sensitivity': 0.5
        },
        {
            'name': 'Small High-Urgency Order',
            'symbol': 'AAPL',
            'side': 'buy',
            'quantity': 1000,
            'urgency': 0.9,
            'cost_sensitivity': 0.2
        },
        {
            'name': 'Large Cost-Sensitive Order',
            'symbol': 'AAPL',
            'side': 'sell',
            'quantity': 100000,
            'urgency': 0.3,
            'cost_sensitivity': 0.9
        }
    ]
    
    print(f"\nTesting {len(test_scenarios)} routing scenarios...")
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['name']} ---")
        print(f"Order: {scenario['side'].upper()} {scenario['quantity']:,} {scenario['symbol']}")
        print(f"Urgency: {scenario['urgency']:.1f}, Cost sensitivity: {scenario['cost_sensitivity']:.1f}")
        
        # Route the order
        routing_decisions = router.route_order(
            symbol=scenario['symbol'],
            side=scenario['side'],
            quantity=scenario['quantity'],
            urgency=scenario['urgency'],
            cost_sensitivity=scenario['cost_sensitivity'],
            max_venues=4,
            market_conditions=market_conditions
        )
        
        print(f"\nRouting decisions ({len(routing_decisions)} venues):")
        total_quantity = sum(d.quantity for d in routing_decisions)
        total_expected_cost = sum(d.expected_cost * d.quantity for d in routing_decisions)
        
        print(f"{'Venue':<15} {'Quantity':<10} {'%':<6} {'Order Type':<12} {'Fill Rate':<10} {'Cost (bp)':<10} {'Reason'}")
        print("-" * 85)
        
        for decision in routing_decisions:
            percentage = (decision.quantity / scenario['quantity']) * 100
            print(f"{decision.venue_id:<15} "
                  f"{decision.quantity:<10,.0f} "
                  f"{percentage:<6.1f} "
                  f"{decision.order_type.value:<12} "
                  f"{decision.expected_fill_rate:<10.2f} "
                  f"{decision.expected_cost:<10.1f} "
                  f"{decision.routing_reason}")
        
        print(f"\nSummary:")
        print(f"  Total allocated: {total_quantity:,.0f} ({(total_quantity/scenario['quantity'])*100:.1f}%)")
        print(f"  Weighted avg cost: {total_expected_cost/total_quantity:.1f} bp")
        print(f"  Avg fill rate: {np.mean([d.expected_fill_rate for d in routing_decisions]):.2f}")
    
    # Test liquidity aggregation
    print(f"\n--- Testing Liquidity Aggregation ---")
    
    # Create sample venue liquidity data
    sample_liquidity = [
        VenueLiquidity('NYSE', time.time(), 1000, 1500, 149.98, 150.02, 150.00, 2000, 200, 0.9, 0.02),
        VenueLiquidity('DARK_POOL_1', time.time(), 0, 0, 0, 0, 150.01, 500, 800, 0.8, 0.02),
        VenueLiquidity('ECN_FAST', time.time(), 800, 1200, 149.99, 150.01, 150.00, 1500, 300, 0.95, 0.02),
        VenueLiquidity('RETAIL_NET', time.time(), 500, 600, 149.97, 150.03, 150.00, 800, 150, 0.85, 0.02)
    ]
    
    # Update liquidity aggregator
    for liquidity in sample_liquidity:
        router.liquidity_aggregator.update_venue_liquidity(liquidity.venue_id, liquidity)
    
    # Get consolidated metrics
    aggregate_metrics = router.liquidity_aggregator.calculate_aggregate_metrics()
    
    print(f"Consolidated Market Data:")
    print(f"  Best Bid: ${aggregate_metrics['best_bid']:.4f}")
    print(f"  Best Ask: ${aggregate_metrics['best_ask']:.4f}")
    print(f"  Best Spread: ${aggregate_metrics['spread']:.4f}")
    print(f"  Total Volume Rate: {aggregate_metrics['total_volume']:,.0f}")
    print(f"  Average Spread: {aggregate_metrics['avg_spread']*100:.3f}%")
    print(f"  Active Venues: {aggregate_metrics['num_venues']}")
    
    # Test liquidity sourcing
    test_quantity = 5000
    liquidity_sources = router.liquidity_aggregator.get_consolidated_liquidity('buy', test_quantity)
    
    print(f"\nLiquidity sources for buying {test_quantity:,} shares:")
    print(f"{'Venue':<15} {'Price':<10} {'Quantity':<10}")
    print("-" * 35)
    
    total_sourced = 0
    for source in liquidity_sources:
        print(f"{source['venue']:<15} ${source['price']:<9.4f} {source['quantity']:<10,.0f}")
        total_sourced += source['quantity']
    
    print(f"Total sourced: {total_sourced:,.0f} ({(total_sourced/test_quantity)*100:.1f}%)")
    
    print("\nSmart Order Routing system test completed successfully!")