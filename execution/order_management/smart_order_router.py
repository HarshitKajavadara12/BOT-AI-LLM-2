"""
Smart Order Router (SOR)
Advanced order routing and venue selection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
from collections import defaultdict, deque
import warnings


class VenueType(Enum):
    """Venue type classification"""
    EXCHANGE = "EXCHANGE"
    DARK_POOL = "DARK_POOL"
    ECN = "ECN"
    MARKET_MAKER = "MARKET_MAKER"
    CROSS_NETWORK = "CROSS_NETWORK"


class RoutingStrategy(Enum):
    """Order routing strategies"""
    BEST_PRICE = "BEST_PRICE"
    MINIMAL_IMPACT = "MINIMAL_IMPACT"
    SPEED = "SPEED"
    HIDDEN_LIQUIDITY = "HIDDEN_LIQUIDITY"
    COST_REDUCTION = "COST_REDUCTION"
    SIZE_DISCOVERY = "SIZE_DISCOVERY"


@dataclass
class VenueData:
    """Venue market data and characteristics"""
    venue_id: str
    venue_name: str
    venue_type: VenueType
    
    # Market data
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_price: float = 0.0
    
    # Venue characteristics
    latency_ms: float = 5.0
    fill_rate: float = 0.95
    hidden_liquidity_factor: float = 2.0  # Multiplier for hidden size
    maker_rebate: float = 0.0  # Per share rebate for adding liquidity
    taker_fee: float = 0.003   # Per share fee for removing liquidity
    
    # Historical performance
    avg_fill_time_ms: float = 100.0
    rejection_rate: float = 0.02
    partial_fill_rate: float = 0.15
    
    # Timestamps
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def spread(self) -> float:
        """Current bid-ask spread"""
        return self.ask_price - self.bid_price if self.ask_price > self.bid_price else 0.0
    
    @property
    def mid_price(self) -> float:
        """Mid-market price"""
        return (self.bid_price + self.ask_price) / 2.0 if self.bid_price > 0 and self.ask_price > 0 else 0.0
    
    @property
    def is_crossed(self) -> bool:
        """Check if market is crossed"""
        return self.bid_price > self.ask_price


@dataclass
class RoutingDecision:
    """Routing decision output"""
    primary_venue: str
    backup_venues: List[str]
    quantity_allocation: Dict[str, float]  # venue_id -> quantity
    expected_fill_rate: float
    expected_cost: float
    routing_reason: str
    confidence_score: float = 0.0


class MarketDataAggregator:
    """
    Aggregates market data from multiple venues
    """
    
    def __init__(self):
        self.venue_data: Dict[str, VenueData] = {}
        self.consolidated_book = {}
        self.last_update = datetime.now()
        
        # Performance tracking
        self.update_latencies = deque(maxlen=1000)
        self.data_quality_scores = {}
    
    def add_venue(self, venue_data: VenueData) -> None:
        """Add venue to aggregator"""
        self.venue_data[venue_data.venue_id] = venue_data
        self.data_quality_scores[venue_data.venue_id] = 0.95  # Initial quality score
    
    def update_venue_data(self, venue_id: str, **kwargs) -> None:
        """Update venue market data"""
        
        if venue_id not in self.venue_data:
            return
        
        start_time = datetime.now()
        venue = self.venue_data[venue_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(venue, key):
                setattr(venue, key, value)
        
        venue.last_update = datetime.now()
        
        # Track update latency
        latency = (datetime.now() - start_time).total_seconds() * 1000
        self.update_latencies.append(latency)
        
        # Update consolidated book
        self._update_consolidated_book()
    
    def get_best_bid_ask(self, symbol: str) -> Tuple[float, float, float, float]:
        """
        Get best bid/ask across all venues
        
        Returns:
            (best_bid, best_ask, bid_size, ask_size)
        """
        
        best_bid = 0.0
        best_ask = float('inf')
        total_bid_size = 0.0
        total_ask_size = 0.0
        
        for venue in self.venue_data.values():
            if venue.bid_price > best_bid:
                best_bid = venue.bid_price
                total_bid_size = venue.bid_size
            elif venue.bid_price == best_bid:
                total_bid_size += venue.bid_size
            
            if venue.ask_price < best_ask and venue.ask_price > 0:
                best_ask = venue.ask_price
                total_ask_size = venue.ask_size
            elif venue.ask_price == best_ask:
                total_ask_size += venue.ask_size
        
        if best_ask == float('inf'):
            best_ask = 0.0
        
        return best_bid, best_ask, total_bid_size, total_ask_size
    
    def get_venue_ranking(self, side: str) -> List[Tuple[str, float]]:
        """
        Get venues ranked by price quality
        
        Args:
            side: 'BUY' or 'SELL'
        
        Returns:
            List of (venue_id, price) tuples, sorted by best price
        """
        
        venue_prices = []
        
        for venue_id, venue in self.venue_data.items():
            if side == 'BUY':
                price = venue.ask_price
            else:
                price = venue.bid_price
            
            if price > 0:
                venue_prices.append((venue_id, price))
        
        # Sort by best price
        if side == 'BUY':
            venue_prices.sort(key=lambda x: x[1])  # Lowest ask first
        else:
            venue_prices.sort(key=lambda x: x[1], reverse=True)  # Highest bid first
        
        return venue_prices
    
    def get_liquidity_estimate(self, venue_id: str, side: str) -> float:
        """Estimate available liquidity at venue"""
        
        if venue_id not in self.venue_data:
            return 0.0
        
        venue = self.venue_data[venue_id]
        
        if side == 'BUY':
            visible_size = venue.ask_size
        else:
            visible_size = venue.bid_size
        
        # Estimate hidden liquidity
        estimated_hidden = visible_size * venue.hidden_liquidity_factor
        
        return visible_size + estimated_hidden
    
    def _update_consolidated_book(self) -> None:
        """Update consolidated order book"""
        # Simplified implementation
        self.last_update = datetime.now()


class RoutingEngine:
    """
    Core routing engine with multiple strategies
    """
    
    def __init__(self, market_data_aggregator: MarketDataAggregator):
        self.market_data = market_data_aggregator
        self.routing_strategies = {
            RoutingStrategy.BEST_PRICE: self._route_best_price,
            RoutingStrategy.MINIMAL_IMPACT: self._route_minimal_impact,
            RoutingStrategy.SPEED: self._route_speed,
            RoutingStrategy.HIDDEN_LIQUIDITY: self._route_hidden_liquidity,
            RoutingStrategy.COST_REDUCTION: self._route_cost_reduction,
            RoutingStrategy.SIZE_DISCOVERY: self._route_size_discovery
        }
        
        # Strategy parameters
        self.strategy_weights = {
            RoutingStrategy.BEST_PRICE: 0.4,
            RoutingStrategy.MINIMAL_IMPACT: 0.3,
            RoutingStrategy.SPEED: 0.2,
            RoutingStrategy.COST_REDUCTION: 0.1
        }
        
        # Performance tracking
        self.routing_history = []
        self.venue_performance = defaultdict(lambda: {
            'fill_rate': 0.0,
            'avg_slippage': 0.0,
            'avg_latency': 0.0,
            'cost_savings': 0.0
        })
    
    def route_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
        strategy: RoutingStrategy = RoutingStrategy.BEST_PRICE,
        urgency: float = 0.5
    ) -> RoutingDecision:
        """
        Route order to optimal venue(s)
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            order_type: Order type
            price: Limit price (if applicable)
            strategy: Routing strategy
            urgency: Urgency factor (0-1)
        
        Returns:
            Routing decision
        """
        
        # Get routing strategy function
        routing_func = self.routing_strategies.get(strategy, self._route_best_price)
        
        # Execute routing strategy
        decision = routing_func(symbol, side, quantity, order_type, price, urgency)
        
        # Record routing decision
        self.routing_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'strategy': strategy,
            'decision': decision
        })
        
        return decision
    
    def _route_best_price(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float],
        urgency: float
    ) -> RoutingDecision:
        """Route to venue with best price"""
        
        venue_ranking = self.market_data.get_venue_ranking(side)
        
        if not venue_ranking:
            return self._default_routing_decision("No venues available")
        
        # Select best price venue
        primary_venue = venue_ranking[0][0]
        backup_venues = [v[0] for v in venue_ranking[1:3]]  # Top 3 backup venues
        
        # Calculate expected performance
        primary_venue_data = self.market_data.venue_data[primary_venue]
        expected_fill_rate = primary_venue_data.fill_rate
        
        if side == 'BUY':
            expected_cost = venue_ranking[0][1] * quantity
        else:
            expected_cost = venue_ranking[0][1] * quantity
        
        return RoutingDecision(
            primary_venue=primary_venue,
            backup_venues=backup_venues,
            quantity_allocation={primary_venue: quantity},
            expected_fill_rate=expected_fill_rate,
            expected_cost=expected_cost,
            routing_reason="Best price available",
            confidence_score=0.85
        )
    
    def _route_minimal_impact(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float],
        urgency: float
    ) -> RoutingDecision:
        """Route to minimize market impact"""
        
        # Find venues with best liquidity to size ratio
        venue_scores = []
        
        for venue_id, venue_data in self.market_data.venue_data.items():
            liquidity = self.market_data.get_liquidity_estimate(venue_id, side)
            
            if liquidity > 0:
                impact_score = quantity / liquidity  # Lower is better
                
                # Adjust for venue type (dark pools have lower impact)
                if venue_data.venue_type == VenueType.DARK_POOL:
                    impact_score *= 0.5
                
                venue_scores.append((venue_id, impact_score))
        
        # Sort by lowest impact
        venue_scores.sort(key=lambda x: x[1])
        
        if not venue_scores:
            return self._default_routing_decision("No suitable venues for impact minimization")
        
        # Allocate across multiple venues to minimize impact
        allocation = self._allocate_quantity_for_impact(quantity, venue_scores)
        
        primary_venue = venue_scores[0][0]
        backup_venues = [v[0] for v in venue_scores[1:3]]
        
        return RoutingDecision(
            primary_venue=primary_venue,
            backup_venues=backup_venues,
            quantity_allocation=allocation,
            expected_fill_rate=0.90,
            expected_cost=0.0,  # Would calculate based on allocation
            routing_reason="Minimal market impact strategy",
            confidence_score=0.80
        )
    
    def _route_speed(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float],
        urgency: float
    ) -> RoutingDecision:
        """Route for fastest execution"""
        
        # Rank venues by speed metrics
        venue_scores = []
        
        for venue_id, venue_data in self.market_data.venue_data.items():
            # Composite speed score (lower is better)
            speed_score = (
                venue_data.latency_ms * 0.4 +
                venue_data.avg_fill_time_ms * 0.4 +
                venue_data.rejection_rate * 1000 * 0.2  # Convert to ms penalty
            )
            
            venue_scores.append((venue_id, speed_score))
        
        # Sort by best speed
        venue_scores.sort(key=lambda x: x[1])
        
        if not venue_scores:
            return self._default_routing_decision("No venues available for speed routing")
        
        primary_venue = venue_scores[0][0]
        backup_venues = [v[0] for v in venue_scores[1:2]]  # Fewer backups for speed
        
        return RoutingDecision(
            primary_venue=primary_venue,
            backup_venues=backup_venues,
            quantity_allocation={primary_venue: quantity},
            expected_fill_rate=0.92,
            expected_cost=0.0,
            routing_reason="Speed-optimized routing",
            confidence_score=0.90
        )
    
    def _route_hidden_liquidity(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float],
        urgency: float
    ) -> RoutingDecision:
        """Route to access hidden liquidity"""
        
        # Prioritize dark pools and venues with hidden liquidity
        dark_venues = []
        
        for venue_id, venue_data in self.market_data.venue_data.items():
            if venue_data.venue_type == VenueType.DARK_POOL:
                hidden_liquidity = self.market_data.get_liquidity_estimate(venue_id, side)
                dark_venues.append((venue_id, hidden_liquidity))
        
        # Sort by available hidden liquidity
        dark_venues.sort(key=lambda x: x[1], reverse=True)
        
        if not dark_venues:
            # Fallback to venues with high hidden liquidity factors
            for venue_id, venue_data in self.market_data.venue_data.items():
                if venue_data.hidden_liquidity_factor > 1.5:
                    hidden_liquidity = self.market_data.get_liquidity_estimate(venue_id, side)
                    dark_venues.append((venue_id, hidden_liquidity))
        
        if not dark_venues:
            return self._default_routing_decision("No hidden liquidity venues available")
        
        primary_venue = dark_venues[0][0]
        backup_venues = [v[0] for v in dark_venues[1:3]]
        
        return RoutingDecision(
            primary_venue=primary_venue,
            backup_venues=backup_venues,
            quantity_allocation={primary_venue: quantity},
            expected_fill_rate=0.75,  # Lower fill rate but better prices
            expected_cost=0.0,
            routing_reason="Hidden liquidity access",
            confidence_score=0.70
        )
    
    def _route_cost_reduction(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float],
        urgency: float
    ) -> RoutingDecision:
        """Route to minimize total costs"""
        
        # Calculate total cost for each venue (fees + spread cost)
        venue_costs = []
        
        for venue_id, venue_data in self.market_data.venue_data.items():
            # Estimate execution price
            if side == 'BUY':
                execution_price = venue_data.ask_price
                fee_per_share = venue_data.taker_fee
            else:
                execution_price = venue_data.bid_price
                fee_per_share = -venue_data.maker_rebate  # Rebate reduces cost
            
            if execution_price > 0:
                spread_cost = abs(execution_price - venue_data.mid_price) * quantity
                total_fees = fee_per_share * quantity
                total_cost = spread_cost + total_fees
                
                venue_costs.append((venue_id, total_cost))
        
        # Sort by lowest total cost
        venue_costs.sort(key=lambda x: x[1])
        
        if not venue_costs:
            return self._default_routing_decision("No venues available for cost analysis")
        
        primary_venue = venue_costs[0][0]
        backup_venues = [v[0] for v in venue_costs[1:3]]
        
        return RoutingDecision(
            primary_venue=primary_venue,
            backup_venues=backup_venues,
            quantity_allocation={primary_venue: quantity},
            expected_fill_rate=0.88,
            expected_cost=venue_costs[0][1],
            routing_reason="Cost minimization strategy",
            confidence_score=0.85
        )
    
    def _route_size_discovery(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float],
        urgency: float
    ) -> RoutingDecision:
        """Route to discover large size opportunities"""
        
        # Look for venues with large size availability
        large_size_venues = []
        
        for venue_id, venue_data in self.market_data.venue_data.items():
            available_size = self.market_data.get_liquidity_estimate(venue_id, side)
            
            # Prioritize venues that can handle large orders
            if available_size >= quantity * 0.5:  # At least 50% of order size
                size_score = available_size / quantity
                large_size_venues.append((venue_id, size_score))
        
        # Sort by size capacity
        large_size_venues.sort(key=lambda x: x[1], reverse=True)
        
        if not large_size_venues:
            return self._default_routing_decision("No venues suitable for size discovery")
        
        # Allocate across top size venues
        allocation = {}
        remaining_qty = quantity
        
        for venue_id, _ in large_size_venues[:3]:  # Top 3 venues
            if remaining_qty <= 0:
                break
            
            venue_capacity = self.market_data.get_liquidity_estimate(venue_id, side)
            alloc_qty = min(remaining_qty, venue_capacity * 0.3)  # Don't exceed 30% of capacity
            
            if alloc_qty > 0:
                allocation[venue_id] = alloc_qty
                remaining_qty -= alloc_qty
        
        primary_venue = large_size_venues[0][0]
        backup_venues = [v[0] for v in large_size_venues[1:3]]
        
        return RoutingDecision(
            primary_venue=primary_venue,
            backup_venues=backup_venues,
            quantity_allocation=allocation,
            expected_fill_rate=0.80,
            expected_cost=0.0,
            routing_reason="Size discovery optimization",
            confidence_score=0.75
        )
    
    def _allocate_quantity_for_impact(
        self,
        total_quantity: float,
        venue_scores: List[Tuple[str, float]]
    ) -> Dict[str, float]:
        """Allocate quantity across venues to minimize impact"""
        
        allocation = {}
        remaining_qty = total_quantity
        
        # Allocate inversely proportional to impact scores
        total_inverse_score = sum(1.0 / score for _, score in venue_scores if score > 0)
        
        for venue_id, impact_score in venue_scores:
            if remaining_qty <= 0 or impact_score <= 0:
                break
            
            # Allocation proportional to inverse of impact score
            proportion = (1.0 / impact_score) / total_inverse_score
            alloc_qty = min(remaining_qty, total_quantity * proportion)
            
            if alloc_qty > 0:
                allocation[venue_id] = alloc_qty
                remaining_qty -= alloc_qty
        
        return allocation
    
    def _default_routing_decision(self, reason: str) -> RoutingDecision:
        """Create default routing decision when no suitable venues found"""
        
        return RoutingDecision(
            primary_venue="DEFAULT",
            backup_venues=[],
            quantity_allocation={"DEFAULT": 0},
            expected_fill_rate=0.0,
            expected_cost=0.0,
            routing_reason=reason,
            confidence_score=0.0
        )
    
    def update_venue_performance(
        self,
        venue_id: str,
        fill_rate: float,
        slippage: float,
        latency_ms: float,
        cost_savings: float
    ) -> None:
        """Update venue performance metrics"""
        
        perf = self.venue_performance[venue_id]
        
        # Exponential moving average
        alpha = 0.1
        perf['fill_rate'] = alpha * fill_rate + (1 - alpha) * perf['fill_rate']
        perf['avg_slippage'] = alpha * slippage + (1 - alpha) * perf['avg_slippage']
        perf['avg_latency'] = alpha * latency_ms + (1 - alpha) * perf['avg_latency']
        perf['cost_savings'] = alpha * cost_savings + (1 - alpha) * perf['cost_savings']


class SmartOrderRouter:
    """
    Main Smart Order Router class
    Combines market data aggregation with intelligent routing
    """
    
    def __init__(self):
        self.market_data = MarketDataAggregator()
        self.routing_engine = RoutingEngine(self.market_data)
        
        # Configuration
        self.default_strategy = RoutingStrategy.BEST_PRICE
        self.enable_venue_switching = True
        self.max_venues_per_order = 3
        
        # Performance tracking
        self.routing_stats = {
            'total_routes': 0,
            'successful_routes': 0,
            'average_confidence': 0.0,
            'venue_utilization': defaultdict(int)
        }
    
    def add_venue(self, venue_data: VenueData) -> None:
        """Add venue to router"""
        self.market_data.add_venue(venue_data)
    
    def update_market_data(self, venue_id: str, **market_data) -> None:
        """Update market data for venue"""
        self.market_data.update_venue_data(venue_id, **market_data)
    
    def route_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "LIMIT",
        price: Optional[float] = None,
        strategy: Optional[RoutingStrategy] = None,
        urgency: float = 0.5
    ) -> RoutingDecision:
        """
        Route order using smart routing logic
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            order_type: Order type
            price: Limit price
            strategy: Routing strategy (uses default if None)
            urgency: Urgency factor
        
        Returns:
            Routing decision
        """
        
        if strategy is None:
            strategy = self.default_strategy
        
        # Route order
        decision = self.routing_engine.route_order(
            symbol, side, quantity, order_type, price, strategy, urgency
        )
        
        # Update statistics
        self.routing_stats['total_routes'] += 1
        
        if decision.confidence_score > 0.5:
            self.routing_stats['successful_routes'] += 1
        
        # Update average confidence
        total_routes = self.routing_stats['total_routes']
        avg_conf = self.routing_stats['average_confidence']
        self.routing_stats['average_confidence'] = (
            (avg_conf * (total_routes - 1) + decision.confidence_score) / total_routes
        )
        
        # Update venue utilization
        self.routing_stats['venue_utilization'][decision.primary_venue] += 1
        
        return decision
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        
        success_rate = (
            self.routing_stats['successful_routes'] / 
            max(self.routing_stats['total_routes'], 1)
        )
        
        return {
            'total_routes': self.routing_stats['total_routes'],
            'success_rate': success_rate,
            'average_confidence': self.routing_stats['average_confidence'],
            'venue_utilization': dict(self.routing_stats['venue_utilization']),
            'venue_count': len(self.market_data.venue_data),
            'last_update': self.market_data.last_update
        }


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Smart Order Router...")
    
    # Create SOR
    sor = SmartOrderRouter()
    
    # Add test venues
    venues = [
        VenueData(
            venue_id="EXCH_A",
            venue_name="Exchange A",
            venue_type=VenueType.EXCHANGE,
            bid_price=100.25,
            ask_price=100.27,
            bid_size=5000,
            ask_size=3000,
            latency_ms=2.5,
            fill_rate=0.95,
            maker_rebate=0.001,
            taker_fee=0.003
        ),
        VenueData(
            venue_id="DARK_1",
            venue_name="Dark Pool 1",
            venue_type=VenueType.DARK_POOL,
            bid_price=100.26,
            ask_price=100.26,  # Mid-point pricing
            bid_size=2000,
            ask_size=2000,
            latency_ms=5.0,
            fill_rate=0.70,
            hidden_liquidity_factor=3.0,
            maker_rebate=0.0,
            taker_fee=0.001
        ),
        VenueData(
            venue_id="ECN_1",
            venue_name="ECN 1",
            venue_type=VenueType.ECN,
            bid_price=100.24,
            ask_price=100.28,
            bid_size=1500,
            ask_size=2500,
            latency_ms=1.8,
            fill_rate=0.92,
            maker_rebate=0.0015,
            taker_fee=0.0025
        )
    ]
    
    for venue in venues:
        sor.add_venue(venue)
    
    print(f"Added {len(venues)} venues to SOR")
    
    # Test different routing strategies
    strategies = [
        RoutingStrategy.BEST_PRICE,
        RoutingStrategy.MINIMAL_IMPACT,
        RoutingStrategy.SPEED,
        RoutingStrategy.HIDDEN_LIQUIDITY,
        RoutingStrategy.COST_REDUCTION
    ]
    
    print("\nTesting routing strategies...")
    
    for strategy in strategies:
        print(f"\n--- {strategy.value} Strategy ---")
        
        decision = sor.route_order(
            symbol="AAPL",
            side="BUY",
            quantity=10000,
            order_type="LIMIT",
            price=100.27,
            strategy=strategy,
            urgency=0.6
        )
        
        print(f"Primary Venue: {decision.primary_venue}")
        print(f"Backup Venues: {decision.backup_venues}")
        print(f"Quantity Allocation: {decision.quantity_allocation}")
        print(f"Expected Fill Rate: {decision.expected_fill_rate:.2%}")
        print(f"Routing Reason: {decision.routing_reason}")
        print(f"Confidence: {decision.confidence_score:.2f}")
    
    # Test market data updates
    print(f"\nUpdating market data...")
    
    # Simulate market data changes
    sor.update_market_data(
        "EXCH_A",
        bid_price=100.26,
        ask_price=100.28,
        bid_size=4000,
        ask_size=2000
    )
    
    # Route another order after market data update
    decision = sor.route_order(
        symbol="AAPL",
        side="SELL",
        quantity=5000,
        strategy=RoutingStrategy.BEST_PRICE
    )
    
    print(f"After market update - Primary Venue: {decision.primary_venue}")
    
    # Get routing statistics
    print(f"\nRouting Statistics:")
    stats = sor.get_routing_statistics()
    
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test venue ranking
    print(f"\nVenue Rankings:")
    
    buy_ranking = sor.market_data.get_venue_ranking("BUY")
    sell_ranking = sor.market_data.get_venue_ranking("SELL")
    
    print("Buy side (best ask first):")
    for venue_id, price in buy_ranking:
        print(f"  {venue_id}: {price:.4f}")
    
    print("Sell side (best bid first):")
    for venue_id, price in sell_ranking:
        print(f"  {venue_id}: {price:.4f}")
    
    # Test liquidity estimates
    print(f"\nLiquidity Estimates:")
    for venue_id in ["EXCH_A", "DARK_1", "ECN_1"]:
        buy_liquidity = sor.market_data.get_liquidity_estimate(venue_id, "BUY")
        sell_liquidity = sor.market_data.get_liquidity_estimate(venue_id, "SELL")
        print(f"  {venue_id}: Buy={buy_liquidity:.0f}, Sell={sell_liquidity:.0f}")
    
    print("\nSmart Order Router testing completed!")