"""
Venue Selector
Optimal venue selection for order execution
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
import heapq


class VenueType(Enum):
    """Types of trading venues"""
    EXCHANGE = "EXCHANGE"                    # Traditional exchanges (NYSE, NASDAQ)
    ECN = "ECN"                             # Electronic Communication Networks
    DARK_POOL = "DARK_POOL"                 # Dark pools
    INTERNALIZER = "INTERNALIZER"           # Retail wholesalers/internalizers
    ATS = "ATS"                             # Alternative Trading Systems
    CROSSING_NETWORK = "CROSSING_NETWORK"   # Crossing networks
    MARKET_MAKER = "MARKET_MAKER"           # Market maker destinations


class VenueCharacteristics(Enum):
    """Venue characteristics for selection"""
    LOW_COST = "LOW_COST"                   # Cost-focused venues
    HIGH_SPEED = "HIGH_SPEED"               # Speed-focused venues
    DARK_LIQUIDITY = "DARK_LIQUIDITY"       # Hidden liquidity venues
    INSTITUTIONAL = "INSTITUTIONAL"         # Institutional-focused venues
    RETAIL = "RETAIL"                       # Retail-focused venues
    BLOCK_FRIENDLY = "BLOCK_FRIENDLY"       # Large order friendly
    IOC_FRIENDLY = "IOC_FRIENDLY"           # Immediate-or-cancel friendly


@dataclass
class VenueMetrics:
    """Performance metrics for a trading venue"""
    venue_id: str
    venue_name: str
    venue_type: VenueType
    
    # Execution quality metrics
    fill_rate: float = 0.0                  # Order fill rate
    average_fill_size: float = 0.0          # Average fill size
    time_to_fill_seconds: float = 0.0       # Average time to fill
    
    # Cost metrics
    effective_spread_bps: float = 0.0       # Effective spread
    price_improvement_bps: float = 0.0      # Price improvement vs NBBO
    market_impact_bps: float = 0.0          # Market impact
    total_cost_bps: float = 0.0             # Total execution cost
    
    # Liquidity metrics
    displayed_size: float = 0.0             # Average displayed size
    hidden_size_estimate: float = 0.0       # Estimated hidden liquidity
    depth_at_touch: float = 0.0             # Liquidity at best price
    
    # Timing metrics
    response_time_ms: float = 0.0           # Order response time
    execution_speed_ms: float = 0.0         # Execution speed
    cancel_acknowledgment_ms: float = 0.0   # Cancel acknowledgment time
    
    # Reliability metrics
    uptime_percentage: float = 100.0        # Venue uptime
    reject_rate: float = 0.0                # Order reject rate
    error_rate: float = 0.0                 # Technical error rate
    
    # Market share and activity
    market_share_percentage: float = 0.0    # Venue market share
    volume_concentration: float = 0.0       # Volume concentration
    
    # Time-based statistics
    timestamp: datetime = field(default_factory=datetime.now)
    measurement_period_minutes: float = 60.0
    
    @property
    def execution_quality_score(self) -> float:
        """Overall execution quality score (0-1)"""
        fill_score = min(1.0, self.fill_rate)
        cost_score = max(0.0, 1.0 - self.total_cost_bps / 50.0)  # 50 bps = 0 score
        speed_score = max(0.0, 1.0 - self.time_to_fill_seconds / 10.0)  # 10s = 0 score
        reliability_score = (self.uptime_percentage / 100.0) * (1.0 - self.reject_rate)
        
        return (fill_score + cost_score + speed_score + reliability_score) / 4.0
    
    @property
    def is_reliable(self) -> bool:
        """Check if venue is reliable for execution"""
        return (
            self.uptime_percentage > 99.0 and
            self.reject_rate < 0.05 and
            self.error_rate < 0.02
        )
    
    @property
    def provides_price_improvement(self) -> bool:
        """Check if venue typically provides price improvement"""
        return self.price_improvement_bps > 0.5


@dataclass
class OrderContext:
    """Order context for venue selection"""
    
    # Order characteristics
    symbol: str = ""
    quantity: float = 0.0
    side: str = "buy"                       # "buy" or "sell"
    order_type: str = "limit"               # "market", "limit", "stop", etc.
    
    # Execution preferences
    urgency_level: float = 0.5              # 0-1 (0=patient, 1=urgent)
    cost_priority: float = 0.5              # 0-1 (0=speed focus, 1=cost focus)
    stealth_requirement: float = 0.0        # 0-1 (0=no stealth, 1=full stealth)
    
    # Size characteristics
    is_block_order: bool = False            # Large institutional order
    min_fill_size: Optional[float] = None   # Minimum acceptable fill size
    max_participation_rate: float = 0.20    # Maximum participation rate
    
    # Timing constraints
    time_in_force: str = "DAY"              # "IOC", "FOK", "DAY", "GTC"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Risk preferences
    allow_dark_pools: bool = True           # Allow dark pool execution
    allow_internalization: bool = True      # Allow internalization
    preferred_venue_types: List[VenueType] = field(default_factory=list)
    excluded_venues: List[str] = field(default_factory=list)
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value (requires current price)"""
        return 0.0  # Will be calculated by venue selector
    
    @property
    def execution_style(self) -> str:
        """Determine execution style from characteristics"""
        if self.urgency_level > 0.8:
            return "aggressive"
        elif self.stealth_requirement > 0.6:
            return "stealth"
        elif self.cost_priority > 0.7:
            return "cost_focused"
        else:
            return "balanced"


@dataclass
class VenueRecommendation:
    """Venue selection recommendation"""
    venue_id: str
    venue_name: str
    venue_type: VenueType
    
    # Recommendation scoring
    overall_score: float = 0.0              # Overall recommendation score
    confidence: float = 0.0                 # Confidence in recommendation
    
    # Component scores
    cost_score: float = 0.0                 # Cost effectiveness score
    speed_score: float = 0.0                # Execution speed score
    liquidity_score: float = 0.0            # Liquidity availability score
    reliability_score: float = 0.0          # Venue reliability score
    
    # Expected execution characteristics
    expected_fill_ratio: float = 0.0        # Expected fill ratio
    expected_cost_bps: float = 0.0          # Expected execution cost
    expected_time_seconds: float = 0.0      # Expected execution time
    
    # Allocation recommendation
    recommended_allocation: float = 0.0     # Recommended order allocation (0-1)
    max_recommended_size: float = 0.0       # Maximum recommended order size
    
    # Rationale
    selection_reasons: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    
    @property
    def recommendation_strength(self) -> str:
        """Get recommendation strength"""
        if self.overall_score > 0.8:
            return "STRONG"
        elif self.overall_score > 0.6:
            return "MODERATE" 
        elif self.overall_score > 0.4:
            return "WEAK"
        else:
            return "NOT_RECOMMENDED"


class VenueDataManager:
    """
    Manages venue performance data and metrics
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        
        # Venue data storage
        self.venue_metrics: Dict[str, VenueMetrics] = {}
        self.venue_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.execution_history: deque = deque(maxlen=max_history)
        
        # Real-time updates
        self.last_update_time: Dict[str, datetime] = {}
        self.data_staleness_threshold_minutes = 15
        
        # Thread safety
        self.data_lock = threading.Lock()
        
    def update_venue_metrics(self, venue_metrics: VenueMetrics) -> None:
        """Update metrics for a venue"""
        
        with self.data_lock:
            venue_id = venue_metrics.venue_id
            
            # Store current metrics
            self.venue_metrics[venue_id] = venue_metrics
            
            # Add to history
            self.venue_history[venue_id].append(venue_metrics)
            
            # Update timestamp
            self.last_update_time[venue_id] = datetime.now()
    
    def get_venue_metrics(self, venue_id: str) -> Optional[VenueMetrics]:
        """Get current metrics for a venue"""
        
        with self.data_lock:
            return self.venue_metrics.get(venue_id)
    
    def get_all_venues(self) -> List[VenueMetrics]:
        """Get metrics for all venues"""
        
        with self.data_lock:
            return list(self.venue_metrics.values())
    
    def get_venues_by_type(self, venue_type: VenueType) -> List[VenueMetrics]:
        """Get venues by type"""
        
        with self.data_lock:
            return [
                metrics for metrics in self.venue_metrics.values()
                if metrics.venue_type == venue_type
            ]
    
    def is_data_stale(self, venue_id: str) -> bool:
        """Check if venue data is stale"""
        
        if venue_id not in self.last_update_time:
            return True
        
        last_update = self.last_update_time[venue_id]
        staleness = datetime.now() - last_update
        
        return staleness.total_seconds() / 60 > self.data_staleness_threshold_minutes
    
    def get_venue_historical_performance(
        self, 
        venue_id: str, 
        lookback_periods: int = 50
    ) -> List[VenueMetrics]:
        """Get historical performance for a venue"""
        
        with self.data_lock:
            if venue_id not in self.venue_history:
                return []
            
            history = list(self.venue_history[venue_id])
            return history[-lookback_periods:] if history else []
    
    def calculate_venue_trends(self, venue_id: str) -> Dict[str, float]:
        """Calculate performance trends for a venue"""
        
        history = self.get_venue_historical_performance(venue_id, 20)
        
        if len(history) < 5:
            return {'insufficient_data': True}
        
        # Calculate trends (simple linear)
        recent_metrics = history[-5:]
        older_metrics = history[:5]
        
        def avg_metric(metrics_list, attr):
            values = [getattr(m, attr) for m in metrics_list]
            return sum(values) / len(values) if values else 0.0
        
        trends = {
            'fill_rate_trend': avg_metric(recent_metrics, 'fill_rate') - avg_metric(older_metrics, 'fill_rate'),
            'cost_trend_bps': avg_metric(recent_metrics, 'total_cost_bps') - avg_metric(older_metrics, 'total_cost_bps'),
            'speed_trend_ms': avg_metric(recent_metrics, 'execution_speed_ms') - avg_metric(older_metrics, 'execution_speed_ms'),
            'reliability_trend': avg_metric(recent_metrics, 'uptime_percentage') - avg_metric(older_metrics, 'uptime_percentage')
        }
        
        return trends


class VenueSelector:
    """
    Main venue selection engine
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.data_manager = VenueDataManager()
        
        # Selection parameters
        self.selection_weights = {
            'cost': 0.30,           # Cost importance
            'speed': 0.25,          # Speed importance
            'liquidity': 0.25,      # Liquidity importance
            'reliability': 0.20     # Reliability importance
        }
        
        # Venue scoring parameters
        self.cost_threshold_bps = 30.0      # Above this = poor cost score
        self.speed_threshold_ms = 100.0     # Above this = poor speed score
        self.min_fill_rate = 0.80           # Minimum acceptable fill rate
        self.min_uptime = 99.0              # Minimum acceptable uptime
        
        # Default venue universe (would be loaded from configuration)
        self._initialize_venue_universe()
        
    def _initialize_venue_universe(self) -> None:
        """Initialize venue universe with sample venues"""
        
        # Sample venue configurations (in practice, loaded from database)
        sample_venues = [
            VenueMetrics(
                venue_id="NYSE", venue_name="New York Stock Exchange", venue_type=VenueType.EXCHANGE,
                fill_rate=0.92, average_fill_size=800, time_to_fill_seconds=2.5,
                effective_spread_bps=8.5, price_improvement_bps=0.2, total_cost_bps=12.0,
                displayed_size=15000, depth_at_touch=25000, response_time_ms=45.0,
                uptime_percentage=99.8, reject_rate=0.02, market_share_percentage=18.5
            ),
            VenueMetrics(
                venue_id="NASDAQ", venue_name="NASDAQ", venue_type=VenueType.EXCHANGE,
                fill_rate=0.94, average_fill_size=650, time_to_fill_seconds=2.2,
                effective_spread_bps=7.8, price_improvement_bps=0.3, total_cost_bps=11.2,
                displayed_size=18000, depth_at_touch=28000, response_time_ms=38.0,
                uptime_percentage=99.9, reject_rate=0.015, market_share_percentage=22.1
            ),
            VenueMetrics(
                venue_id="ARCA", venue_name="NYSE Arca", venue_type=VenueType.ECN,
                fill_rate=0.89, average_fill_size=1200, time_to_fill_seconds=1.8,
                effective_spread_bps=6.2, price_improvement_bps=0.8, total_cost_bps=9.5,
                displayed_size=12000, depth_at_touch=20000, response_time_ms=25.0,
                uptime_percentage=99.7, reject_rate=0.025, market_share_percentage=8.9
            ),
            VenueMetrics(
                venue_id="DARKPOOL_1", venue_name="Institutional Dark Pool", venue_type=VenueType.DARK_POOL,
                fill_rate=0.75, average_fill_size=2500, time_to_fill_seconds=8.5,
                effective_spread_bps=3.2, price_improvement_bps=2.1, total_cost_bps=7.8,
                displayed_size=0, hidden_size_estimate=50000, response_time_ms=120.0,
                uptime_percentage=99.5, reject_rate=0.05, market_share_percentage=5.2
            ),
            VenueMetrics(
                venue_id="IEX", venue_name="Investors Exchange", venue_type=VenueType.EXCHANGE,
                fill_rate=0.88, average_fill_size=900, time_to_fill_seconds=3.2,
                effective_spread_bps=9.1, price_improvement_bps=1.2, total_cost_bps=13.5,
                displayed_size=8000, depth_at_touch=15000, response_time_ms=85.0,
                uptime_percentage=99.6, reject_rate=0.03, market_share_percentage=2.8
            ),
            VenueMetrics(
                venue_id="CITADEL_MM", venue_name="Citadel Securities", venue_type=VenueType.INTERNALIZER,
                fill_rate=0.96, average_fill_size=450, time_to_fill_seconds=0.8,
                effective_spread_bps=4.5, price_improvement_bps=1.8, total_cost_bps=8.2,
                displayed_size=5000, depth_at_touch=10000, response_time_ms=15.0,
                uptime_percentage=99.9, reject_rate=0.01, market_share_percentage=12.5
            )
        ]
        
        # Add sample venues to data manager
        for venue in sample_venues:
            self.data_manager.update_venue_metrics(venue)
    
    def select_optimal_venues(
        self,
        order_context: OrderContext,
        max_venues: int = 5
    ) -> List[VenueRecommendation]:
        """
        Select optimal venues for order execution
        
        Args:
            order_context: Order characteristics and requirements
            max_venues: Maximum number of venues to recommend
            
        Returns:
            List of venue recommendations ranked by suitability
        """
        
        # Get available venues
        available_venues = self._filter_available_venues(order_context)
        
        if not available_venues:
            return []
        
        # Score each venue
        venue_scores = []
        
        for venue in available_venues:
            score = self._score_venue(venue, order_context)
            
            if score.overall_score > 0.1:  # Minimum viable score
                venue_scores.append(score)
        
        # Sort by overall score (descending)
        venue_scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Return top venues
        return venue_scores[:max_venues]
    
    def _filter_available_venues(self, order_context: OrderContext) -> List[VenueMetrics]:
        """Filter venues based on order context requirements"""
        
        all_venues = self.data_manager.get_all_venues()
        available_venues = []
        
        for venue in all_venues:
            
            # Skip excluded venues
            if venue.venue_id in order_context.excluded_venues:
                continue
            
            # Check venue type preferences
            if order_context.preferred_venue_types:
                if venue.venue_type not in order_context.preferred_venue_types:
                    continue
            
            # Check dark pool allowance
            if venue.venue_type == VenueType.DARK_POOL and not order_context.allow_dark_pools:
                continue
            
            # Check internalization allowance
            if venue.venue_type == VenueType.INTERNALIZER and not order_context.allow_internalization:
                continue
            
            # Check basic reliability requirements
            if not venue.is_reliable:
                continue
            
            # Check if data is stale
            if self.data_manager.is_data_stale(venue.venue_id):
                continue
            
            available_venues.append(venue)
        
        return available_venues
    
    def _score_venue(self, venue: VenueMetrics, order_context: OrderContext) -> VenueRecommendation:
        """Score a venue for the given order context"""
        
        # Calculate component scores
        cost_score = self._calculate_cost_score(venue, order_context)
        speed_score = self._calculate_speed_score(venue, order_context)
        liquidity_score = self._calculate_liquidity_score(venue, order_context)
        reliability_score = self._calculate_reliability_score(venue, order_context)
        
        # Adjust weights based on order context
        adjusted_weights = self._adjust_weights_for_context(order_context)
        
        # Calculate overall score
        overall_score = (
            adjusted_weights['cost'] * cost_score +
            adjusted_weights['speed'] * speed_score +
            adjusted_weights['liquidity'] * liquidity_score +
            adjusted_weights['reliability'] * reliability_score
        )
        
        # Calculate expected execution characteristics
        expected_fill_ratio = self._estimate_fill_ratio(venue, order_context)
        expected_cost_bps = self._estimate_execution_cost(venue, order_context)
        expected_time_seconds = self._estimate_execution_time(venue, order_context)
        
        # Calculate allocation recommendation
        recommended_allocation = self._calculate_allocation_recommendation(
            venue, order_context, overall_score
        )
        
        # Generate selection rationale
        selection_reasons = self._generate_selection_reasons(venue, order_context, {
            'cost': cost_score, 'speed': speed_score,
            'liquidity': liquidity_score, 'reliability': reliability_score
        })
        
        risk_factors = self._identify_risk_factors(venue, order_context)
        
        # Calculate confidence based on data quality and venue characteristics
        confidence = self._calculate_confidence(venue, order_context)
        
        return VenueRecommendation(
            venue_id=venue.venue_id,
            venue_name=venue.venue_name,
            venue_type=venue.venue_type,
            overall_score=overall_score,
            confidence=confidence,
            cost_score=cost_score,
            speed_score=speed_score,
            liquidity_score=liquidity_score,
            reliability_score=reliability_score,
            expected_fill_ratio=expected_fill_ratio,
            expected_cost_bps=expected_cost_bps,
            expected_time_seconds=expected_time_seconds,
            recommended_allocation=recommended_allocation,
            max_recommended_size=self._calculate_max_order_size(venue, order_context),
            selection_reasons=selection_reasons,
            risk_factors=risk_factors
        )
    
    def _calculate_cost_score(self, venue: VenueMetrics, order_context: OrderContext) -> float:
        """Calculate cost effectiveness score for venue"""
        
        # Base cost score (inverse of total cost)
        if venue.total_cost_bps <= 0:
            cost_score = 1.0
        else:
            cost_score = max(0.0, 1.0 - venue.total_cost_bps / self.cost_threshold_bps)
        
        # Price improvement bonus
        if venue.price_improvement_bps > 0:
            improvement_bonus = min(0.3, venue.price_improvement_bps / 10.0)  # Up to 30% bonus
            cost_score = min(1.0, cost_score + improvement_bonus)
        
        # Adjust for order characteristics
        if order_context.cost_priority > 0.7:
            # Cost-sensitive order
            cost_score *= 1.2  # Boost cost importance
        
        if order_context.is_block_order and venue.venue_type == VenueType.DARK_POOL:
            # Dark pools often better for large orders
            cost_score *= 1.1
        
        return max(0.0, min(1.0, cost_score))
    
    def _calculate_speed_score(self, venue: VenueMetrics, order_context: OrderContext) -> float:
        """Calculate execution speed score for venue"""
        
        # Base speed score (inverse of execution time)
        if venue.execution_speed_ms <= 0:
            speed_score = 1.0
        else:
            speed_score = max(0.0, 1.0 - venue.execution_speed_ms / self.speed_threshold_ms)
        
        # Response time factor
        if venue.response_time_ms > 0:
            response_factor = max(0.5, 1.0 - venue.response_time_ms / 200.0)  # 200ms threshold
            speed_score *= response_factor
        
        # Adjust for order urgency
        if order_context.urgency_level > 0.8:
            # Urgent order - speed is critical
            speed_score *= 1.3
        elif order_context.urgency_level < 0.3:
            # Patient order - speed less important
            speed_score *= 0.8
        
        # IOC/FOK orders need fast venues
        if order_context.time_in_force in ["IOC", "FOK"]:
            speed_score *= 1.2
        
        return max(0.0, min(1.0, speed_score))
    
    def _calculate_liquidity_score(self, venue: VenueMetrics, order_context: OrderContext) -> float:
        """Calculate liquidity availability score for venue"""
        
        # Base liquidity from displayed + hidden size
        total_liquidity = venue.depth_at_touch + venue.hidden_size_estimate
        
        if total_liquidity <= 0:
            return 0.0
        
        # Score based on order size relative to available liquidity
        if order_context.quantity > 0:
            liquidity_ratio = total_liquidity / order_context.quantity
            
            if liquidity_ratio >= 5.0:
                liquidity_score = 1.0  # Excellent liquidity
            elif liquidity_ratio >= 2.0:
                liquidity_score = 0.8  # Good liquidity
            elif liquidity_ratio >= 1.0:
                liquidity_score = 0.6  # Adequate liquidity
            else:
                liquidity_score = 0.3 * liquidity_ratio  # Poor liquidity
        else:
            liquidity_score = 0.7  # Default score
        
        # Fill rate adjustment
        liquidity_score *= venue.fill_rate
        
        # Dark pool bonus for stealth requirements
        if order_context.stealth_requirement > 0.5 and venue.venue_type == VenueType.DARK_POOL:
            liquidity_score *= (1.0 + 0.3 * order_context.stealth_requirement)
        
        # Block order considerations
        if order_context.is_block_order:
            if venue.average_fill_size > 1000:  # Good for large orders
                liquidity_score *= 1.2
        
        return max(0.0, min(1.0, liquidity_score))
    
    def _calculate_reliability_score(self, venue: VenueMetrics, order_context: OrderContext) -> float:
        """Calculate venue reliability score"""
        
        # Base reliability from uptime and error rates
        uptime_score = venue.uptime_percentage / 100.0
        error_penalty = min(0.5, venue.reject_rate + venue.error_rate)
        
        reliability_score = uptime_score * (1.0 - error_penalty)
        
        # Market share consideration (higher market share = more reliable)
        if venue.market_share_percentage > 10.0:
            reliability_score *= 1.1
        elif venue.market_share_percentage < 2.0:
            reliability_score *= 0.9
        
        # Historical trend consideration
        trends = self.data_manager.calculate_venue_trends(venue.venue_id)
        
        if not trends.get('insufficient_data', False):
            if trends.get('reliability_trend', 0) > 0.5:
                reliability_score *= 1.05  # Improving reliability
            elif trends.get('reliability_trend', 0) < -0.5:
                reliability_score *= 0.95  # Declining reliability
        
        return max(0.0, min(1.0, reliability_score))
    
    def _adjust_weights_for_context(self, order_context: OrderContext) -> Dict[str, float]:
        """Adjust scoring weights based on order context"""
        
        weights = self.selection_weights.copy()
        
        # Adjust based on cost priority
        if order_context.cost_priority > 0.7:
            weights['cost'] += 0.2
            weights['speed'] -= 0.1
            weights['liquidity'] -= 0.1
        elif order_context.cost_priority < 0.3:
            weights['cost'] -= 0.1
            weights['speed'] += 0.15
            weights['liquidity'] -= 0.05
        
        # Adjust based on urgency
        if order_context.urgency_level > 0.8:
            weights['speed'] += 0.2
            weights['cost'] -= 0.1
            weights['reliability'] -= 0.1
        elif order_context.urgency_level < 0.3:
            weights['speed'] -= 0.1
            weights['cost'] += 0.05
            weights['liquidity'] += 0.05
        
        # Adjust for block orders
        if order_context.is_block_order:
            weights['liquidity'] += 0.15
            weights['speed'] -= 0.1
            weights['cost'] -= 0.05
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _estimate_fill_ratio(self, venue: VenueMetrics, order_context: OrderContext) -> float:
        """Estimate expected fill ratio for the order"""
        
        base_fill_ratio = venue.fill_rate
        
        # Adjust for order size vs available liquidity
        total_liquidity = venue.depth_at_touch + venue.hidden_size_estimate
        
        if total_liquidity > 0 and order_context.quantity > 0:
            liquidity_ratio = total_liquidity / order_context.quantity
            
            if liquidity_ratio < 0.5:
                base_fill_ratio *= 0.6  # Likely partial fill
            elif liquidity_ratio < 1.0:
                base_fill_ratio *= 0.8  # Possible partial fill
        
        # Time in force adjustments
        if order_context.time_in_force == "IOC":
            base_fill_ratio *= 0.7  # IOC orders have lower fill rates
        elif order_context.time_in_force == "FOK":
            if order_context.quantity > venue.depth_at_touch:
                base_fill_ratio = 0.1  # FOK unlikely to fill if larger than depth
        
        return max(0.0, min(1.0, base_fill_ratio))
    
    def _estimate_execution_cost(self, venue: VenueMetrics, order_context: OrderContext) -> float:
        """Estimate execution cost in basis points"""
        
        base_cost = venue.total_cost_bps
        
        # Adjust for order characteristics
        if order_context.is_block_order:
            # Large orders typically have higher impact
            base_cost *= 1.3
        
        if order_context.urgency_level > 0.8:
            # Urgent orders pay higher costs
            base_cost *= 1.2
        elif order_context.urgency_level < 0.3:
            # Patient orders can achieve better costs
            base_cost *= 0.8
        
        # Dark pool cost advantage
        if venue.venue_type == VenueType.DARK_POOL and order_context.stealth_requirement > 0.5:
            base_cost *= 0.7  # Lower cost for hidden execution
        
        return base_cost
    
    def _estimate_execution_time(self, venue: VenueMetrics, order_context: OrderContext) -> float:
        """Estimate execution time in seconds"""
        
        base_time = venue.time_to_fill_seconds
        
        # Adjust for order size
        if order_context.is_block_order:
            base_time *= 2.5  # Large orders take longer
        
        # Adjust for venue characteristics
        if venue.venue_type == VenueType.DARK_POOL:
            base_time *= 1.8  # Dark pools typically slower
        elif venue.venue_type == VenueType.INTERNALIZER:
            base_time *= 0.6  # Internalizers often faster
        
        # Time in force adjustments
        if order_context.time_in_force == "IOC":
            base_time = min(base_time, 1.0)  # IOC orders are immediate
        
        return base_time
    
    def _calculate_allocation_recommendation(
        self,
        venue: VenueMetrics,
        order_context: OrderContext,
        overall_score: float
    ) -> float:
        """Calculate recommended order allocation to venue"""
        
        # Base allocation from overall score
        base_allocation = overall_score
        
        # Adjust for liquidity constraints
        total_liquidity = venue.depth_at_touch + venue.hidden_size_estimate
        
        if total_liquidity > 0 and order_context.quantity > 0:
            max_safe_allocation = min(1.0, total_liquidity / order_context.quantity * 0.5)
            base_allocation = min(base_allocation, max_safe_allocation)
        
        # Venue type considerations
        if venue.venue_type == VenueType.DARK_POOL:
            # Don't allocate too much to single dark pool
            base_allocation = min(base_allocation, 0.4)
        elif venue.venue_type == VenueType.EXCHANGE:
            # Exchanges can handle larger allocations
            base_allocation = min(base_allocation, 0.7)
        
        return max(0.0, min(1.0, base_allocation))
    
    def _calculate_max_order_size(self, venue: VenueMetrics, order_context: OrderContext) -> float:
        """Calculate maximum recommended order size for venue"""
        
        # Base on available liquidity
        total_liquidity = venue.depth_at_touch + venue.hidden_size_estimate
        
        # Conservative approach: don't exceed 50% of available liquidity
        max_size = total_liquidity * 0.5
        
        # Venue-specific adjustments
        if venue.venue_type == VenueType.DARK_POOL:
            # Dark pools can handle larger sizes
            max_size = total_liquidity * 0.8
        elif venue.venue_type == VenueType.ECN:
            # ECNs prefer smaller sizes
            max_size = total_liquidity * 0.3
        
        # Historical performance consideration
        if venue.average_fill_size > 0:
            # Don't exceed 3x typical fill size
            max_size = min(max_size, venue.average_fill_size * 3)
        
        return max(100.0, max_size)  # Minimum 100 shares
    
    def _generate_selection_reasons(
        self,
        venue: VenueMetrics,
        order_context: OrderContext,
        component_scores: Dict[str, float]
    ) -> List[str]:
        """Generate human-readable selection reasons"""
        
        reasons = []
        
        # Cost reasons
        if component_scores['cost'] > 0.8:
            if venue.price_improvement_bps > 1.0:
                reasons.append(f"Excellent cost profile with {venue.price_improvement_bps:.1f} bps price improvement")
            else:
                reasons.append(f"Low execution cost ({venue.total_cost_bps:.1f} bps)")
        
        # Speed reasons
        if component_scores['speed'] > 0.8:
            reasons.append(f"Fast execution (avg {venue.execution_speed_ms:.0f}ms)")
        
        # Liquidity reasons
        if component_scores['liquidity'] > 0.8:
            if venue.venue_type == VenueType.DARK_POOL:
                reasons.append(f"Deep hidden liquidity (~{venue.hidden_size_estimate:,.0f} shares)")
            else:
                reasons.append(f"Strong displayed liquidity ({venue.depth_at_touch:,.0f} shares at touch)")
        
        # Reliability reasons
        if component_scores['reliability'] > 0.9:
            reasons.append(f"Highly reliable ({venue.uptime_percentage:.1f}% uptime)")
        
        # Context-specific reasons
        if order_context.stealth_requirement > 0.5 and venue.venue_type == VenueType.DARK_POOL:
            reasons.append("Provides execution stealth for sensitive orders")
        
        if order_context.is_block_order and venue.average_fill_size > 1500:
            reasons.append("Well-suited for large institutional orders")
        
        if order_context.urgency_level > 0.8 and venue.response_time_ms < 50:
            reasons.append("Rapid response time for urgent execution")
        
        return reasons
    
    def _identify_risk_factors(self, venue: VenueMetrics, order_context: OrderContext) -> List[str]:
        """Identify potential risk factors"""
        
        risks = []
        
        # Performance risks
        if venue.fill_rate < 0.85:
            risks.append(f"Lower fill rate ({venue.fill_rate:.1%}) may result in partial execution")
        
        if venue.reject_rate > 0.05:
            risks.append(f"Higher reject rate ({venue.reject_rate:.1%}) increases execution uncertainty")
        
        # Liquidity risks
        total_liquidity = venue.depth_at_touch + venue.hidden_size_estimate
        if order_context.quantity > 0 and total_liquidity < order_context.quantity:
            risks.append("Order size exceeds estimated available liquidity")
        
        # Market structure risks
        if venue.venue_type == VenueType.DARK_POOL:
            risks.append("Dark pool execution subject to information leakage risk")
        
        if venue.venue_type == VenueType.INTERNALIZER:
            risks.append("Internalization may not reflect true market conditions")
        
        # Data quality risks
        if self.data_manager.is_data_stale(venue.venue_id):
            risks.append("Venue performance data may be stale")
        
        # Market share risks
        if venue.market_share_percentage < 1.0:
            risks.append("Low market share venue may have limited liquidity")
        
        return risks
    
    def _calculate_confidence(self, venue: VenueMetrics, order_context: OrderContext) -> float:
        """Calculate confidence in venue recommendation"""
        
        confidence = 0.8  # Base confidence
        
        # Data quality factors
        if not self.data_manager.is_data_stale(venue.venue_id):
            confidence += 0.1
        else:
            confidence -= 0.2
        
        # Historical data availability
        history = self.data_manager.get_venue_historical_performance(venue.venue_id)
        if len(history) > 20:
            confidence += 0.1
        elif len(history) < 5:
            confidence -= 0.15
        
        # Venue characteristics alignment
        if venue.venue_type in order_context.preferred_venue_types:
            confidence += 0.05
        
        # Market share consideration
        if venue.market_share_percentage > 5.0:
            confidence += 0.05
        elif venue.market_share_percentage < 1.0:
            confidence -= 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def generate_venue_selection_report(
        self,
        order_context: OrderContext,
        max_venues: int = 10
    ) -> Dict[str, Any]:
        """Generate comprehensive venue selection report"""
        
        # Get venue recommendations
        recommendations = self.select_optimal_venues(order_context, max_venues)
        
        # Calculate total recommended allocation
        total_allocation = sum(rec.recommended_allocation for rec in recommendations)
        
        # Venue type distribution
        venue_type_dist = defaultdict(int)
        for rec in recommendations:
            venue_type_dist[rec.venue_type.value] += 1
        
        # Cost and performance estimates
        if recommendations:
            weighted_avg_cost = sum(
                rec.expected_cost_bps * rec.recommended_allocation 
                for rec in recommendations
            ) / max(total_allocation, 0.01)
            
            weighted_avg_time = sum(
                rec.expected_time_seconds * rec.recommended_allocation
                for rec in recommendations
            ) / max(total_allocation, 0.01)
            
            weighted_avg_fill = sum(
                rec.expected_fill_ratio * rec.recommended_allocation
                for rec in recommendations
            ) / max(total_allocation, 0.01)
        else:
            weighted_avg_cost = 0.0
            weighted_avg_time = 0.0
            weighted_avg_fill = 0.0
        
        report = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'order_summary': {
                'quantity': order_context.quantity,
                'side': order_context.side,
                'execution_style': order_context.execution_style,
                'urgency_level': order_context.urgency_level,
                'cost_priority': order_context.cost_priority,
                'stealth_requirement': order_context.stealth_requirement
            },
            'venue_recommendations': [
                {
                    'venue_name': rec.venue_name,
                    'venue_type': rec.venue_type.value,
                    'overall_score': rec.overall_score,
                    'confidence': rec.confidence,
                    'recommended_allocation': rec.recommended_allocation,
                    'expected_cost_bps': rec.expected_cost_bps,
                    'expected_time_seconds': rec.expected_time_seconds,
                    'expected_fill_ratio': rec.expected_fill_ratio,
                    'recommendation_strength': rec.recommendation_strength,
                    'selection_reasons': rec.selection_reasons,
                    'risk_factors': rec.risk_factors
                }
                for rec in recommendations
            ],
            'execution_summary': {
                'total_venues_recommended': len(recommendations),
                'total_allocation_coverage': total_allocation,
                'weighted_average_cost_bps': weighted_avg_cost,
                'weighted_average_time_seconds': weighted_avg_time,
                'weighted_average_fill_ratio': weighted_avg_fill
            },
            'venue_distribution': dict(venue_type_dist),
            'risk_assessment': {
                'primary_risks': self._assess_primary_risks(recommendations, order_context),
                'mitigation_suggestions': self._suggest_risk_mitigations(recommendations, order_context)
            }
        }
        
        return report
    
    def _assess_primary_risks(
        self,
        recommendations: List[VenueRecommendation],
        order_context: OrderContext
    ) -> List[str]:
        """Assess primary execution risks"""
        
        risks = []
        
        if not recommendations:
            risks.append("No suitable venues found for execution")
            return risks
        
        total_allocation = sum(rec.recommended_allocation for rec in recommendations)
        
        if total_allocation < 0.8:
            risks.append("Low venue allocation coverage - execution may be difficult")
        
        avg_fill_ratio = sum(rec.expected_fill_ratio for rec in recommendations) / len(recommendations)
        if avg_fill_ratio < 0.8:
            risks.append("Low expected fill ratios - partial execution likely")
        
        # Check venue concentration
        if len(recommendations) == 1:
            risks.append("Single venue dependence - no execution diversification")
        
        # Check for high-cost venues
        high_cost_venues = [rec for rec in recommendations if rec.expected_cost_bps > 25.0]
        if len(high_cost_venues) > len(recommendations) / 2:
            risks.append("Multiple high-cost venues recommended - execution may be expensive")
        
        return risks
    
    def _suggest_risk_mitigations(
        self,
        recommendations: List[VenueRecommendation],
        order_context: OrderContext
    ) -> List[str]:
        """Suggest risk mitigation strategies"""
        
        suggestions = []
        
        if not recommendations:
            suggestions.append("Consider relaxing venue selection criteria or splitting order")
            return suggestions
        
        # Diversification suggestions
        if len(recommendations) < 3:
            suggestions.append("Consider using additional venues for better diversification")
        
        # Size suggestions
        total_max_size = sum(rec.max_recommended_size for rec in recommendations)
        if order_context.quantity > total_max_size:
            suggestions.append("Consider breaking order into smaller pieces over time")
        
        # Timing suggestions
        urgent_venues = [rec for rec in recommendations if rec.expected_time_seconds < 5.0]
        if order_context.urgency_level > 0.8 and not urgent_venues:
            suggestions.append("Consider using market orders or more aggressive execution styles")
        
        # Cost optimization suggestions
        if order_context.cost_priority > 0.7:
            suggestions.append("Consider using dark pools or longer execution timeframes for cost savings")
        
        return suggestions


if __name__ == "__main__":
    import random
    
    # Example usage and testing
    print("Testing Venue Selector...")
    
    # Create venue selector
    selector = VenueSelector("AAPL")
    
    # Test different order scenarios
    test_orders = [
        {
            'name': 'Small Urgent Order',
            'context': OrderContext(
                symbol="AAPL",
                quantity=2000,
                side="buy",
                urgency_level=0.9,
                cost_priority=0.3,
                time_in_force="IOC"
            )
        },
        {
            'name': 'Large Cost-Focused Order',
            'context': OrderContext(
                symbol="AAPL",
                quantity=50000,
                side="sell",
                urgency_level=0.2,
                cost_priority=0.9,
                stealth_requirement=0.7,
                is_block_order=True,
                allow_dark_pools=True
            )
        },
        {
            'name': 'Medium Balanced Order',
            'context': OrderContext(
                symbol="AAPL",
                quantity=15000,
                side="buy",
                urgency_level=0.5,
                cost_priority=0.5,
                preferred_venue_types=[VenueType.EXCHANGE, VenueType.ECN]
            )
        }
    ]
    
    print(f"\nVenue Selection Results:")
    
    for test_case in test_orders:
        print(f"\n{test_case['name']}:")
        context = test_case['context']
        
        print(f"  Order: {context.quantity:,} shares {context.side}")
        print(f"  Urgency: {context.urgency_level:.1f}, Cost Priority: {context.cost_priority:.1f}")
        print(f"  Stealth Requirement: {context.stealth_requirement:.1f}")
        
        # Get venue recommendations
        recommendations = selector.select_optimal_venues(context, max_venues=5)
        
        if recommendations:
            print(f"  Top Venue Recommendations:")
            
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"    {i}. {rec.venue_name} ({rec.venue_type.value})")
                print(f"       Score: {rec.overall_score:.2f} (Confidence: {rec.confidence:.2f})")
                print(f"       Allocation: {rec.recommended_allocation:.1%}")
                print(f"       Expected Cost: {rec.expected_cost_bps:.1f} bps")
                print(f"       Expected Fill: {rec.expected_fill_ratio:.1%}")
                print(f"       Expected Time: {rec.expected_time_seconds:.1f}s")
                print(f"       Strength: {rec.recommendation_strength}")
                
                if rec.selection_reasons:
                    print(f"       Reasons: {', '.join(rec.selection_reasons[:2])}")
                
                if rec.risk_factors:
                    print(f"       Risks: {rec.risk_factors[0]}")
        else:
            print(f"  No suitable venues found")
    
    # Generate comprehensive report
    print(f"\nGenerating Comprehensive Venue Selection Report...")
    
    sample_order = OrderContext(
        symbol="AAPL",
        quantity=25000,
        side="buy",
        urgency_level=0.6,
        cost_priority=0.7,
        stealth_requirement=0.4,
        allow_dark_pools=True
    )
    
    report = selector.generate_venue_selection_report(sample_order, max_venues=8)
    
    print(f"Venue Selection Report:")
    print(f"  Symbol: {report['symbol']}")
    print(f"  Order: {report['order_summary']['quantity']:,} shares {report['order_summary']['side']}")
    
    print(f"  Execution Summary:")
    summary = report['execution_summary']
    print(f"    Venues Recommended: {summary['total_venues_recommended']}")
    print(f"    Allocation Coverage: {summary['total_allocation_coverage']:.1%}")
    print(f"    Weighted Avg Cost: {summary['weighted_average_cost_bps']:.1f} bps")
    print(f"    Weighted Avg Time: {summary['weighted_average_time_seconds']:.1f}s")
    print(f"    Weighted Avg Fill: {summary['weighted_average_fill_ratio']:.1%}")
    
    print(f"  Venue Distribution:")
    for venue_type, count in report['venue_distribution'].items():
        print(f"    {venue_type}: {count}")
    
    if report['risk_assessment']['primary_risks']:
        print(f"  Primary Risks:")
        for risk in report['risk_assessment']['primary_risks']:
            print(f"    - {risk}")
    
    if report['risk_assessment']['mitigation_suggestions']:
        print(f"  Risk Mitigation Suggestions:")
        for suggestion in report['risk_assessment']['mitigation_suggestions']:
            print(f"    - {suggestion}")
    
    print(f"  Top 3 Venue Recommendations:")
    for i, rec in enumerate(report['venue_recommendations'][:3], 1):
        print(f"    {i}. {rec['venue_name']} ({rec['venue_type']})")
        print(f"       Score: {rec['overall_score']:.2f}, Allocation: {rec['recommended_allocation']:.1%}")
        print(f"       Cost: {rec['expected_cost_bps']:.1f} bps, Fill: {rec['expected_fill_ratio']:.1%}")
    
    print("\nVenue selector testing completed!")