"""
Cost Estimator
Comprehensive execution cost prediction and analysis
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
import math


class CostComponent(Enum):
    """Types of execution cost components"""
    MARKET_IMPACT = "MARKET_IMPACT"           # Price impact from trading
    SPREAD_COST = "SPREAD_COST"               # Bid-ask spread cost
    TIMING_RISK = "TIMING_RISK"               # Opportunity cost/timing risk
    COMMISSION = "COMMISSION"                 # Brokerage commissions
    FEES = "FEES"                            # Exchange and regulatory fees
    SLIPPAGE = "SLIPPAGE"                    # Execution slippage
    DELAY_COST = "DELAY_COST"                # Cost of execution delays


class ExecutionStyle(Enum):
    """Execution style for cost estimation"""
    MARKET = "MARKET"                        # Market orders
    LIMIT = "LIMIT"                          # Limit orders
    ICEBERG = "ICEBERG"                      # Iceberg orders
    TWAP = "TWAP"                           # Time-weighted average price
    VWAP = "VWAP"                           # Volume-weighted average price
    IMPLEMENTATION_SHORTFALL = "IS"          # Implementation shortfall
    ADAPTIVE = "ADAPTIVE"                    # Adaptive execution


class TimeHorizon(Enum):
    """Execution time horizon"""
    IMMEDIATE = "IMMEDIATE"                  # Immediate execution (seconds)
    SHORT_TERM = "SHORT_TERM"               # Short-term (minutes to hour)
    MEDIUM_TERM = "MEDIUM_TERM"             # Medium-term (hours)
    LONG_TERM = "LONG_TERM"                 # Long-term (days)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Individual cost components (in basis points)
    market_impact_bps: float = 0.0
    spread_cost_bps: float = 0.0
    timing_risk_bps: float = 0.0
    commission_bps: float = 0.0
    fees_bps: float = 0.0
    slippage_bps: float = 0.0
    delay_cost_bps: float = 0.0
    
    # Total costs
    total_cost_bps: float = 0.0
    total_cost_dollars: float = 0.0
    
    # Risk measures
    cost_variance_bps: float = 0.0           # Cost uncertainty
    value_at_risk_bps: float = 0.0           # 95% VaR
    expected_shortfall_bps: float = 0.0      # Expected shortfall
    
    # Execution characteristics
    expected_fill_ratio: float = 1.0         # Expected fill ratio
    expected_duration_minutes: float = 0.0   # Expected execution time
    liquidity_consumption_ratio: float = 0.0 # % of available liquidity
    
    # Confidence intervals
    cost_confidence_95_lower_bps: float = 0.0
    cost_confidence_95_upper_bps: float = 0.0
    
    def __post_init__(self):
        """Calculate total cost"""
        self.total_cost_bps = (
            self.market_impact_bps + self.spread_cost_bps + 
            self.timing_risk_bps + self.commission_bps + 
            self.fees_bps + self.slippage_bps + self.delay_cost_bps
        )
    
    @property
    def cost_efficiency_score(self) -> float:
        """Calculate cost efficiency score (0-1, higher is better)"""
        # Penalize high costs and high uncertainty
        base_score = max(0, 1.0 - self.total_cost_bps / 100.0)  # 100 bps = 0 score
        uncertainty_penalty = min(0.5, self.cost_variance_bps / 50.0)  # Up to 50% penalty
        
        return max(0.0, base_score - uncertainty_penalty)
    
    @property 
    def risk_adjusted_cost_bps(self) -> float:
        """Risk-adjusted cost including uncertainty"""
        return self.total_cost_bps + self.cost_variance_bps


@dataclass
class MarketConditions:
    """Market conditions for cost estimation"""
    
    # Price and volatility
    current_price: float = 0.0
    volatility_annual: float = 0.30
    volatility_intraday: float = 0.02
    
    # Liquidity measures
    bid_ask_spread_bps: float = 10.0
    market_depth_shares: float = 10000
    average_daily_volume: float = 1000000
    
    # Market microstructure
    average_trade_size: float = 500
    trades_per_minute: float = 2.0
    order_flow_imbalance: float = 0.0        # -1 to +1
    
    # Regime indicators
    is_market_open: bool = True
    is_volatile_period: bool = False
    is_illiquid_period: bool = False
    news_impact_level: float = 0.0           # 0-1 scale
    
    # Historical context
    recent_price_trend: float = 0.0          # Recent price momentum
    volume_ratio_vs_average: float = 1.0     # Current vs average volume
    
    @property
    def liquidity_score(self) -> float:
        """Overall liquidity score (0-1)"""
        spread_score = max(0, 1.0 - self.bid_ask_spread_bps / 100.0)
        depth_score = min(1.0, self.market_depth_shares / 50000)
        volume_score = min(1.0, self.volume_ratio_vs_average)
        
        return (spread_score + depth_score + volume_score) / 3.0
    
    @property
    def market_impact_multiplier(self) -> float:
        """Market impact multiplier based on conditions"""
        base_multiplier = 1.0
        
        # Volatility adjustment
        vol_adjustment = 1.0 + (self.volatility_annual - 0.30) / 0.30
        
        # Liquidity adjustment  
        liquidity_adjustment = 2.0 - self.liquidity_score
        
        # News/event adjustment
        news_adjustment = 1.0 + self.news_impact_level
        
        return base_multiplier * vol_adjustment * liquidity_adjustment * news_adjustment


@dataclass
class OrderCharacteristics:
    """Order characteristics for cost estimation"""
    
    # Basic order details
    quantity: float = 0.0
    side: str = "buy"                        # "buy" or "sell"
    order_type: str = "market"               # "market", "limit", "stop", etc.
    
    # Execution preferences
    execution_style: ExecutionStyle = ExecutionStyle.MARKET
    time_horizon: TimeHorizon = TimeHorizon.IMMEDIATE
    urgency_level: float = 0.5               # 0-1 scale (0=patient, 1=urgent)
    
    # Size characteristics
    is_block_trade: bool = False             # Large institutional trade
    iceberg_display_size: Optional[float] = None  # For iceberg orders
    participation_rate_limit: float = 0.20   # Max participation rate
    
    # Risk preferences
    risk_tolerance: float = 0.5              # 0-1 scale
    max_acceptable_cost_bps: Optional[float] = None
    
    # Timing constraints
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    allow_overnight: bool = False
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value (requires price)"""
        return 0.0  # Will be calculated by cost estimator
    
    @property
    def is_large_order(self) -> bool:
        """Check if this is considered a large order"""
        return self.is_block_trade or self.quantity > 50000


class CostModel(ABC):
    """Abstract base class for cost models"""
    
    @abstractmethod
    def estimate_cost(
        self,
        order: OrderCharacteristics,
        market_conditions: MarketConditions
    ) -> CostBreakdown:
        """Estimate execution cost"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name"""
        pass


class LinearCostModel(CostModel):
    """Linear cost model (simple baseline)"""
    
    def __init__(self):
        # Model parameters (calibrated to typical equity markets)
        self.temporary_impact_coefficient = 0.5    # bps per sqrt(% ADV)
        self.permanent_impact_coefficient = 0.3    # bps per sqrt(% ADV)
        self.spread_capture_rate = 0.5             # Fraction of spread captured
        self.commission_per_share = 0.005          # $0.005 per share
        
    def estimate_cost(
        self,
        order: OrderCharacteristics,
        market_conditions: MarketConditions
    ) -> CostBreakdown:
        """Estimate cost using linear model"""
        
        # Calculate participation rate
        if market_conditions.average_daily_volume > 0:
            participation_rate = order.quantity / market_conditions.average_daily_volume
        else:
            participation_rate = 0.01
        
        participation_pct = participation_rate * 100
        
        # Market impact (temporary + permanent)
        temp_impact = self.temporary_impact_coefficient * math.sqrt(participation_pct)
        perm_impact = self.permanent_impact_coefficient * math.sqrt(participation_pct)
        total_impact = temp_impact + perm_impact
        
        # Apply market conditions multiplier
        total_impact *= market_conditions.market_impact_multiplier
        
        # Spread cost
        spread_cost = market_conditions.bid_ask_spread_bps * self.spread_capture_rate
        
        # Adjust spread cost based on execution style
        if order.execution_style == ExecutionStyle.LIMIT:
            spread_cost *= 0.2  # Lower spread cost for limit orders
        elif order.execution_style == ExecutionStyle.MARKET:
            spread_cost *= 1.0  # Full spread cost
        elif order.execution_style in [ExecutionStyle.TWAP, ExecutionStyle.VWAP]:
            spread_cost *= 0.6  # Moderate spread cost
        
        # Commission
        notional_value = order.quantity * market_conditions.current_price
        commission_dollars = order.quantity * self.commission_per_share
        commission_bps = (commission_dollars / notional_value) * 10000 if notional_value > 0 else 0
        
        # Timing risk (simplified)
        timing_risk = 0.0
        if order.time_horizon != TimeHorizon.IMMEDIATE:
            vol_contribution = market_conditions.volatility_intraday * 10000  # Convert to bps
            
            time_multipliers = {
                TimeHorizon.SHORT_TERM: 0.5,
                TimeHorizon.MEDIUM_TERM: 1.0,
                TimeHorizon.LONG_TERM: 2.0
            }
            
            timing_risk = vol_contribution * time_multipliers.get(order.time_horizon, 1.0)
        
        # Cost variance (simplified uncertainty measure)
        cost_variance = total_impact * 0.3  # 30% of impact as uncertainty
        
        return CostBreakdown(
            market_impact_bps=total_impact,
            spread_cost_bps=spread_cost,
            timing_risk_bps=timing_risk,
            commission_bps=commission_bps,
            fees_bps=1.0,  # Fixed fee estimate
            cost_variance_bps=cost_variance,
            expected_fill_ratio=self._estimate_fill_ratio(order, market_conditions),
            expected_duration_minutes=self._estimate_duration(order, market_conditions),
            total_cost_dollars=((total_impact + spread_cost + timing_risk + commission_bps + 1.0) / 10000) * notional_value
        )
    
    def _estimate_fill_ratio(self, order: OrderCharacteristics, conditions: MarketConditions) -> float:
        """Estimate expected fill ratio"""
        base_fill_ratio = 0.95
        
        # Adjust for order size vs market depth
        if conditions.market_depth_shares > 0:
            size_ratio = order.quantity / conditions.market_depth_shares
            if size_ratio > 1.0:
                base_fill_ratio *= (1.0 / size_ratio) ** 0.5
        
        # Adjust for execution style
        if order.execution_style == ExecutionStyle.LIMIT:
            base_fill_ratio *= 0.8  # Lower fill ratio for limit orders
        elif order.execution_style == ExecutionStyle.MARKET:
            base_fill_ratio = 0.98  # High fill ratio for market orders
        
        return max(0.1, min(1.0, base_fill_ratio))
    
    def _estimate_duration(self, order: OrderCharacteristics, conditions: MarketConditions) -> float:
        """Estimate execution duration in minutes"""
        
        if order.execution_style == ExecutionStyle.MARKET:
            return 1.0  # Nearly immediate
        
        # Base duration from participation rate
        participation_rate = min(order.participation_rate_limit, 0.20)
        if conditions.average_daily_volume > 0:
            trading_hours_per_day = 6.5  # US market hours
            volume_per_minute = conditions.average_daily_volume / (trading_hours_per_day * 60)
            order_volume_per_minute = volume_per_minute * participation_rate
            
            if order_volume_per_minute > 0:
                estimated_minutes = order.quantity / order_volume_per_minute
            else:
                estimated_minutes = 60.0  # Default 1 hour
        else:
            estimated_minutes = 60.0
        
        # Adjust for execution style
        style_multipliers = {
            ExecutionStyle.TWAP: 1.0,
            ExecutionStyle.VWAP: 0.8,
            ExecutionStyle.ICEBERG: 1.2,
            ExecutionStyle.ADAPTIVE: 0.9
        }
        
        multiplier = style_multipliers.get(order.execution_style, 1.0)
        return max(5.0, estimated_minutes * multiplier)  # Minimum 5 minutes
    
    def get_model_name(self) -> str:
        return "LinearCostModel"


class AlmgrenChrissCostModel(CostModel):
    """Almgren-Chriss cost model implementation"""
    
    def __init__(self):
        # Model parameters
        self.eta = 2.5e-6      # Permanent impact parameter
        self.gamma = 2.5e-7    # Temporary impact parameter  
        self.sigma = 0.30      # Volatility
        self.lambda_risk = 2e-6 # Risk aversion parameter
        
    def estimate_cost(
        self,
        order: OrderCharacteristics,
        market_conditions: MarketConditions
    ) -> CostBreakdown:
        """Estimate cost using Almgren-Chriss model"""
        
        # Model inputs
        X = order.quantity  # Total shares to trade
        T = self._get_time_horizon_hours(order.time_horizon)  # Time horizon in hours
        sigma = market_conditions.volatility_annual / math.sqrt(252 * 24)  # Hourly volatility
        
        # Calculate optimal trajectory
        kappa = math.sqrt(self.lambda_risk * sigma**2 / self.gamma)
        
        if T > 0:
            # Optimal trading rate
            if kappa * T < 0.01:  # Small kappa*T approximation
                trading_rate = X / T
            else:
                sinh_kt = math.sinh(kappa * T)
                cosh_kt = math.cosh(kappa * T)
                trading_rate = X * kappa * cosh_kt / sinh_kt
        else:
            trading_rate = X  # Immediate execution
        
        # Cost components
        # Permanent impact cost
        permanent_cost_bps = self.eta * X * market_conditions.current_price / 10000 * 10000
        
        # Temporary impact cost
        if T > 0:
            avg_trading_rate = X / T
            temporary_cost_bps = self.gamma * avg_trading_rate * market_conditions.current_price / 10000 * 10000
        else:
            temporary_cost_bps = self.gamma * X * market_conditions.current_price / 10000 * 10000
        
        # Risk cost (timing risk)
        if T > 0:
            risk_cost_bps = 0.5 * self.lambda_risk * sigma**2 * T * X**2 / (market_conditions.current_price * X) * 10000
        else:
            risk_cost_bps = 0.0
        
        # Spread cost (not in original AC model, but added for completeness)
        spread_cost_bps = market_conditions.bid_ask_spread_bps * 0.5
        
        # Commission
        notional_value = order.quantity * market_conditions.current_price
        commission_bps = (order.quantity * 0.005 / notional_value) * 10000 if notional_value > 0 else 0
        
        # Cost variance (based on Almgren-Chriss risk measure)
        if T > 0:
            cost_variance_bps = math.sqrt(self.lambda_risk * sigma**2 * T * X) / market_conditions.current_price * 10000
        else:
            cost_variance_bps = 0.0
        
        return CostBreakdown(
            market_impact_bps=permanent_cost_bps + temporary_cost_bps,
            spread_cost_bps=spread_cost_bps,
            timing_risk_bps=risk_cost_bps,
            commission_bps=commission_bps,
            fees_bps=1.0,
            cost_variance_bps=cost_variance_bps,
            expected_fill_ratio=0.95,
            expected_duration_minutes=T * 60 if T > 0 else 1.0,
            total_cost_dollars=((permanent_cost_bps + temporary_cost_bps + spread_cost_bps + risk_cost_bps + commission_bps + 1.0) / 10000) * notional_value
        )
    
    def _get_time_horizon_hours(self, horizon: TimeHorizon) -> float:
        """Convert time horizon enum to hours"""
        mapping = {
            TimeHorizon.IMMEDIATE: 0.0,
            TimeHorizon.SHORT_TERM: 0.5,
            TimeHorizon.MEDIUM_TERM: 2.0,
            TimeHorizon.LONG_TERM: 6.0
        }
        return mapping.get(horizon, 1.0)
    
    def get_model_name(self) -> str:
        return "AlmgrenChrissCostModel"


class BertsimasLoCostModel(CostModel):
    """Bertsimas-Lo cost model with adaptive features"""
    
    def __init__(self):
        # Model parameters
        self.alpha = 0.6       # Impact decay parameter
        self.beta = 0.4        # Impact intensity parameter
        self.delta = 1.2       # Size scaling parameter
        
    def estimate_cost(
        self,
        order: OrderCharacteristics,
        market_conditions: MarketConditions
    ) -> CostBreakdown:
        """Estimate cost using Bertsimas-Lo model"""
        
        # Calculate participation rate
        participation_rate = order.quantity / market_conditions.average_daily_volume if market_conditions.average_daily_volume > 0 else 0.01
        
        # Market impact using power law
        base_impact = self.beta * (participation_rate ** self.delta)
        
        # Adjust for volatility and liquidity
        vol_adjustment = (market_conditions.volatility_annual / 0.30) ** 0.8
        liquidity_adjustment = (2.0 - market_conditions.liquidity_score)
        
        market_impact_bps = base_impact * vol_adjustment * liquidity_adjustment * 10000
        
        # Temporary impact decay
        if order.time_horizon != TimeHorizon.IMMEDIATE:
            time_hours = self._get_time_horizon_hours(order.time_horizon)
            decay_factor = math.exp(-self.alpha * time_hours)
            temporary_impact = market_impact_bps * decay_factor
            permanent_impact = market_impact_bps * (1 - decay_factor)
        else:
            temporary_impact = market_impact_bps * 0.7
            permanent_impact = market_impact_bps * 0.3
        
        total_impact = temporary_impact + permanent_impact
        
        # Other cost components
        spread_cost_bps = market_conditions.bid_ask_spread_bps * 0.5
        
        # Timing risk
        timing_risk_bps = 0.0
        if order.time_horizon != TimeHorizon.IMMEDIATE:
            vol_bps = market_conditions.volatility_intraday * 10000
            time_multiplier = {
                TimeHorizon.SHORT_TERM: 0.3,
                TimeHorizon.MEDIUM_TERM: 0.7,
                TimeHorizon.LONG_TERM: 1.2
            }.get(order.time_horizon, 0.5)
            
            timing_risk_bps = vol_bps * time_multiplier
        
        # Commission and fees
        notional_value = order.quantity * market_conditions.current_price
        commission_bps = (order.quantity * 0.005 / notional_value) * 10000 if notional_value > 0 else 0
        
        # Cost uncertainty
        cost_variance_bps = total_impact * 0.25  # 25% of impact as uncertainty
        
        return CostBreakdown(
            market_impact_bps=total_impact,
            spread_cost_bps=spread_cost_bps,
            timing_risk_bps=timing_risk_bps,
            commission_bps=commission_bps,
            fees_bps=1.5,
            cost_variance_bps=cost_variance_bps,
            expected_fill_ratio=0.93,
            expected_duration_minutes=self._get_time_horizon_hours(order.time_horizon) * 60,
            total_cost_dollars=((total_impact + spread_cost_bps + timing_risk_bps + commission_bps + 1.5) / 10000) * notional_value
        )
    
    def _get_time_horizon_hours(self, horizon: TimeHorizon) -> float:
        """Convert time horizon enum to hours"""
        mapping = {
            TimeHorizon.IMMEDIATE: 0.0,
            TimeHorizon.SHORT_TERM: 0.5,
            TimeHorizon.MEDIUM_TERM: 2.0,
            TimeHorizon.LONG_TERM: 6.0
        }
        return mapping.get(horizon, 1.0)
    
    def get_model_name(self) -> str:
        return "BertsimasLoCostModel"


class EnsembleCostModel(CostModel):
    """Ensemble model combining multiple cost models"""
    
    def __init__(self):
        self.models = [
            LinearCostModel(),
            AlmgrenChrissCostModel(), 
            BertsimasLoCostModel()
        ]
        
        # Model weights (can be calibrated)
        self.weights = [0.4, 0.35, 0.25]
        
    def estimate_cost(
        self,
        order: OrderCharacteristics,
        market_conditions: MarketConditions
    ) -> CostBreakdown:
        """Estimate cost using ensemble of models"""
        
        # Get estimates from all models
        estimates = []
        for model in self.models:
            try:
                estimate = model.estimate_cost(order, market_conditions)
                estimates.append(estimate)
            except Exception as e:
                warnings.warn(f"Model {model.get_model_name()} failed: {e}")
                continue
        
        if not estimates:
            # Fallback to simple estimate
            return self._fallback_estimate(order, market_conditions)
        
        # Weighted combination
        weighted_estimate = self._combine_estimates(estimates, self.weights[:len(estimates)])
        
        # Add ensemble-specific adjustments
        weighted_estimate.cost_variance_bps *= 1.1  # Slightly higher uncertainty
        
        return weighted_estimate
    
    def _combine_estimates(self, estimates: List[CostBreakdown], weights: List[float]) -> CostBreakdown:
        """Combine multiple cost estimates"""
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(estimates)] * len(estimates)
        
        # Weighted average of all components
        combined = CostBreakdown()
        
        for i, (estimate, weight) in enumerate(zip(estimates, normalized_weights)):
            combined.market_impact_bps += estimate.market_impact_bps * weight
            combined.spread_cost_bps += estimate.spread_cost_bps * weight
            combined.timing_risk_bps += estimate.timing_risk_bps * weight
            combined.commission_bps += estimate.commission_bps * weight
            combined.fees_bps += estimate.fees_bps * weight
            combined.slippage_bps += estimate.slippage_bps * weight
            combined.delay_cost_bps += estimate.delay_cost_bps * weight
            combined.cost_variance_bps += estimate.cost_variance_bps * weight
            combined.expected_fill_ratio += estimate.expected_fill_ratio * weight
            combined.expected_duration_minutes += estimate.expected_duration_minutes * weight
            combined.total_cost_dollars += estimate.total_cost_dollars * weight
        
        # Calculate variance across models (model uncertainty)
        impact_values = [e.market_impact_bps for e in estimates]
        model_uncertainty = np.std(impact_values) if len(impact_values) > 1 else 0.0
        combined.cost_variance_bps = max(combined.cost_variance_bps, model_uncertainty)
        
        return combined
    
    def _fallback_estimate(self, order: OrderCharacteristics, conditions: MarketConditions) -> CostBreakdown:
        """Simple fallback estimate"""
        
        # Very basic cost estimate
        impact_bps = 10.0  # 10 bps base impact
        spread_bps = conditions.bid_ask_spread_bps * 0.5
        notional = order.quantity * conditions.current_price
        commission_bps = (order.quantity * 0.005 / notional) * 10000 if notional > 0 else 0
        
        return CostBreakdown(
            market_impact_bps=impact_bps,
            spread_cost_bps=spread_bps,
            commission_bps=commission_bps,
            fees_bps=1.0,
            cost_variance_bps=5.0,
            expected_fill_ratio=0.9,
            expected_duration_minutes=30.0,
            total_cost_dollars=((impact_bps + spread_bps + commission_bps + 1.0) / 10000) * notional
        )
    
    def get_model_name(self) -> str:
        return "EnsembleCostModel"


class ExecutionCostEstimator:
    """
    Main execution cost estimation engine
    """
    
    def __init__(self, default_model: str = "ensemble"):
        
        # Available models
        self.models = {
            "linear": LinearCostModel(),
            "almgren_chriss": AlmgrenChrissCostModel(),
            "bertsimas_lo": BertsimasLoCostModel(),
            "ensemble": EnsembleCostModel()
        }
        
        self.default_model = default_model
        
        # Cost estimation history
        self.estimation_history: List[Tuple[OrderCharacteristics, MarketConditions, CostBreakdown]] = []
        
        # Model performance tracking
        self.model_performance = defaultdict(list)
        
    def estimate_execution_cost(
        self,
        order: OrderCharacteristics,
        market_conditions: MarketConditions,
        model_name: Optional[str] = None
    ) -> CostBreakdown:
        """
        Estimate execution cost for given order and market conditions
        
        Args:
            order: Order characteristics
            market_conditions: Current market conditions
            model_name: Specific model to use (optional)
            
        Returns:
            Detailed cost breakdown
        """
        
        # Select model
        model_to_use = model_name or self.default_model
        
        if model_to_use not in self.models:
            raise ValueError(f"Unknown model: {model_to_use}")
        
        model = self.models[model_to_use]
        
        # Estimate cost
        cost_breakdown = model.estimate_cost(order, market_conditions)
        
        # Add confidence intervals (simplified)
        self._add_confidence_intervals(cost_breakdown)
        
        # Store in history
        self.estimation_history.append((order, market_conditions, cost_breakdown))
        
        return cost_breakdown
    
    def _add_confidence_intervals(self, cost_breakdown: CostBreakdown) -> None:
        """Add confidence intervals to cost breakdown"""
        
        # Simple confidence interval based on cost variance
        std_dev = cost_breakdown.cost_variance_bps
        
        # 95% confidence interval (±1.96 standard deviations)
        cost_breakdown.cost_confidence_95_lower_bps = max(0, cost_breakdown.total_cost_bps - 1.96 * std_dev)
        cost_breakdown.cost_confidence_95_upper_bps = cost_breakdown.total_cost_bps + 1.96 * std_dev
        
        # Value at Risk (95th percentile)
        cost_breakdown.value_at_risk_bps = cost_breakdown.total_cost_bps + 1.65 * std_dev
        
        # Expected shortfall (average of worst 5%)
        cost_breakdown.expected_shortfall_bps = cost_breakdown.total_cost_bps + 2.0 * std_dev
    
    def compare_execution_strategies(
        self,
        order: OrderCharacteristics,
        market_conditions: MarketConditions
    ) -> Dict[str, CostBreakdown]:
        """Compare costs across different execution strategies"""
        
        strategies = [
            ExecutionStyle.MARKET,
            ExecutionStyle.LIMIT,
            ExecutionStyle.TWAP,
            ExecutionStyle.VWAP,
            ExecutionStyle.ICEBERG
        ]
        
        results = {}
        
        for strategy in strategies:
            # Create modified order with different strategy
            modified_order = OrderCharacteristics(
                quantity=order.quantity,
                side=order.side,
                execution_style=strategy,
                time_horizon=order.time_horizon,
                urgency_level=order.urgency_level,
                participation_rate_limit=order.participation_rate_limit,
                risk_tolerance=order.risk_tolerance
            )
            
            # Estimate cost
            cost_estimate = self.estimate_execution_cost(modified_order, market_conditions)
            results[strategy.value] = cost_estimate
        
        return results
    
    def optimize_execution_parameters(
        self,
        order: OrderCharacteristics,
        market_conditions: MarketConditions
    ) -> Tuple[OrderCharacteristics, CostBreakdown]:
        """Optimize execution parameters for lowest cost"""
        
        best_cost = float('inf')
        best_order = order
        best_breakdown = None
        
        # Test different time horizons
        time_horizons = [TimeHorizon.IMMEDIATE, TimeHorizon.SHORT_TERM, TimeHorizon.MEDIUM_TERM]
        
        # Test different participation rates
        participation_rates = [0.05, 0.10, 0.15, 0.20]
        
        # Test different execution styles
        execution_styles = [ExecutionStyle.MARKET, ExecutionStyle.TWAP, ExecutionStyle.VWAP]
        
        for time_horizon in time_horizons:
            for participation_rate in participation_rates:
                for execution_style in execution_styles:
                    
                    # Create test order
                    test_order = OrderCharacteristics(
                        quantity=order.quantity,
                        side=order.side,
                        execution_style=execution_style,
                        time_horizon=time_horizon,
                        participation_rate_limit=participation_rate,
                        urgency_level=order.urgency_level,
                        risk_tolerance=order.risk_tolerance
                    )
                    
                    # Estimate cost
                    cost_breakdown = self.estimate_execution_cost(test_order, market_conditions)
                    
                    # Check if this is better (risk-adjusted cost)
                    risk_adjusted_cost = cost_breakdown.risk_adjusted_cost_bps
                    
                    if risk_adjusted_cost < best_cost:
                        best_cost = risk_adjusted_cost
                        best_order = test_order
                        best_breakdown = cost_breakdown
        
        return best_order, best_breakdown
    
    def get_cost_attribution(self, cost_breakdown: CostBreakdown) -> Dict[str, float]:
        """Get cost attribution by component"""
        
        total_cost = cost_breakdown.total_cost_bps
        
        if total_cost == 0:
            return {}
        
        attribution = {
            'Market Impact': (cost_breakdown.market_impact_bps / total_cost) * 100,
            'Spread Cost': (cost_breakdown.spread_cost_bps / total_cost) * 100,
            'Timing Risk': (cost_breakdown.timing_risk_bps / total_cost) * 100,
            'Commission': (cost_breakdown.commission_bps / total_cost) * 100,
            'Fees': (cost_breakdown.fees_bps / total_cost) * 100,
            'Slippage': (cost_breakdown.slippage_bps / total_cost) * 100,
            'Delay Cost': (cost_breakdown.delay_cost_bps / total_cost) * 100
        }
        
        return attribution
    
    def generate_cost_report(
        self,
        order: OrderCharacteristics,
        market_conditions: MarketConditions
    ) -> Dict[str, Any]:
        """Generate comprehensive cost analysis report"""
        
        # Get base cost estimate
        base_estimate = self.estimate_execution_cost(order, market_conditions)
        
        # Compare strategies
        strategy_comparison = self.compare_execution_strategies(order, market_conditions)
        
        # Find optimal parameters
        optimal_order, optimal_cost = self.optimize_execution_parameters(order, market_conditions)
        
        # Cost attribution
        cost_attribution = self.get_cost_attribution(base_estimate)
        
        # Risk analysis
        risk_metrics = {
            'cost_variance_bps': base_estimate.cost_variance_bps,
            'value_at_risk_bps': base_estimate.value_at_risk_bps,
            'expected_shortfall_bps': base_estimate.expected_shortfall_bps,
            'cost_efficiency_score': base_estimate.cost_efficiency_score,
            'confidence_interval_95': [
                base_estimate.cost_confidence_95_lower_bps,
                base_estimate.cost_confidence_95_upper_bps
            ]
        }
        
        # Execution characteristics
        execution_summary = {
            'expected_duration_minutes': base_estimate.expected_duration_minutes,
            'expected_fill_ratio': base_estimate.expected_fill_ratio,
            'liquidity_consumption_ratio': base_estimate.liquidity_consumption_ratio
        }
        
        report = {
            'timestamp': datetime.now(),
            'order_summary': {
                'quantity': order.quantity,
                'side': order.side,
                'execution_style': order.execution_style.value,
                'time_horizon': order.time_horizon.value,
                'notional_value': order.quantity * market_conditions.current_price
            },
            'base_estimate': {
                'total_cost_bps': base_estimate.total_cost_bps,
                'total_cost_dollars': base_estimate.total_cost_dollars,
                'model_used': self.default_model
            },
            'cost_breakdown': cost_attribution,
            'strategy_comparison': {
                strategy: {
                    'total_cost_bps': breakdown.total_cost_bps,
                    'cost_efficiency_score': breakdown.cost_efficiency_score
                }
                for strategy, breakdown in strategy_comparison.items()
            },
            'optimal_execution': {
                'execution_style': optimal_order.execution_style.value,
                'time_horizon': optimal_order.time_horizon.value,
                'participation_rate': optimal_order.participation_rate_limit,
                'estimated_cost_bps': optimal_cost.total_cost_bps,
                'cost_savings_bps': base_estimate.total_cost_bps - optimal_cost.total_cost_bps
            },
            'risk_analysis': risk_metrics,
            'execution_characteristics': execution_summary,
            'market_conditions_summary': {
                'liquidity_score': market_conditions.liquidity_score,
                'spread_bps': market_conditions.bid_ask_spread_bps,
                'volatility': market_conditions.volatility_annual,
                'market_impact_multiplier': market_conditions.market_impact_multiplier
            }
        }
        
        return report


if __name__ == "__main__":
    import random
    
    # Example usage and testing
    print("Testing Execution Cost Estimator...")
    
    # Create cost estimator
    estimator = ExecutionCostEstimator("ensemble")
    
    # Sample market conditions
    market_conditions = MarketConditions(
        current_price=150.0,
        volatility_annual=0.35,
        volatility_intraday=0.025,
        bid_ask_spread_bps=8.0,
        market_depth_shares=25000,
        average_daily_volume=2000000,
        average_trade_size=800,
        trades_per_minute=3.5,
        is_market_open=True,
        volume_ratio_vs_average=1.2
    )
    
    # Test different order scenarios
    test_orders = [
        {
            'name': 'Small Market Order',
            'order': OrderCharacteristics(
                quantity=1000,
                side='buy',
                execution_style=ExecutionStyle.MARKET,
                time_horizon=TimeHorizon.IMMEDIATE,
                urgency_level=0.8
            )
        },
        {
            'name': 'Large TWAP Order',
            'order': OrderCharacteristics(
                quantity=50000,
                side='sell',
                execution_style=ExecutionStyle.TWAP,
                time_horizon=TimeHorizon.MEDIUM_TERM,
                participation_rate_limit=0.15,
                urgency_level=0.3
            )
        },
        {
            'name': 'Block Trade',
            'order': OrderCharacteristics(
                quantity=100000,
                side='buy',
                execution_style=ExecutionStyle.VWAP,
                time_horizon=TimeHorizon.LONG_TERM,
                is_block_trade=True,
                participation_rate_limit=0.20,
                urgency_level=0.4
            )
        }
    ]
    
    print(f"\nCost Estimation Results:")
    print(f"Market Conditions:")
    print(f"  Price: ${market_conditions.current_price:.2f}")
    print(f"  Volatility: {market_conditions.volatility_annual:.1%}")
    print(f"  Spread: {market_conditions.bid_ask_spread_bps:.1f} bps")
    print(f"  Liquidity Score: {market_conditions.liquidity_score:.2f}")
    
    for test_case in test_orders:
        print(f"\n{test_case['name']}:")
        order = test_case['order']
        
        print(f"  Order: {order.quantity:,} shares {order.side}")
        print(f"  Style: {order.execution_style.value}")
        print(f"  Horizon: {order.time_horizon.value}")
        
        # Estimate cost
        cost_breakdown = estimator.estimate_execution_cost(order, market_conditions)
        
        print(f"  Cost Breakdown:")
        print(f"    Total Cost: {cost_breakdown.total_cost_bps:.1f} bps (${cost_breakdown.total_cost_dollars:,.0f})")
        print(f"    Market Impact: {cost_breakdown.market_impact_bps:.1f} bps")
        print(f"    Spread Cost: {cost_breakdown.spread_cost_bps:.1f} bps")
        print(f"    Timing Risk: {cost_breakdown.timing_risk_bps:.1f} bps")
        print(f"    Commission: {cost_breakdown.commission_bps:.1f} bps")
        print(f"    Cost Uncertainty: ±{cost_breakdown.cost_variance_bps:.1f} bps")
        print(f"    VaR (95%): {cost_breakdown.value_at_risk_bps:.1f} bps")
        print(f"    Expected Fill: {cost_breakdown.expected_fill_ratio:.1%}")
        print(f"    Expected Duration: {cost_breakdown.expected_duration_minutes:.0f} minutes")
        print(f"    Cost Efficiency Score: {cost_breakdown.cost_efficiency_score:.2f}")
    
    # Test strategy comparison
    print(f"\nStrategy Comparison for 25,000 share buy order:")
    
    comparison_order = OrderCharacteristics(
        quantity=25000,
        side='buy',
        time_horizon=TimeHorizon.MEDIUM_TERM,
        participation_rate_limit=0.15
    )
    
    strategy_comparison = estimator.compare_execution_strategies(comparison_order, market_conditions)
    
    for strategy, cost_breakdown in strategy_comparison.items():
        print(f"  {strategy}: {cost_breakdown.total_cost_bps:.1f} bps (efficiency: {cost_breakdown.cost_efficiency_score:.2f})")
    
    # Test optimization
    print(f"\nExecution Optimization:")
    
    original_order = OrderCharacteristics(
        quantity=25000,
        side='buy',
        execution_style=ExecutionStyle.MARKET,
        time_horizon=TimeHorizon.IMMEDIATE,
        urgency_level=0.7
    )
    
    original_cost = estimator.estimate_execution_cost(original_order, market_conditions)
    optimal_order, optimal_cost = estimator.optimize_execution_parameters(original_order, market_conditions)
    
    print(f"  Original Strategy: {original_order.execution_style.value}")
    print(f"    Cost: {original_cost.total_cost_bps:.1f} bps")
    
    print(f"  Optimal Strategy: {optimal_order.execution_style.value}")
    print(f"    Time Horizon: {optimal_order.time_horizon.value}")
    print(f"    Participation Rate: {optimal_order.participation_rate_limit:.1%}")
    print(f"    Cost: {optimal_cost.total_cost_bps:.1f} bps")
    print(f"    Savings: {original_cost.total_cost_bps - optimal_cost.total_cost_bps:.1f} bps")
    
    # Generate comprehensive report
    print(f"\nGenerating comprehensive cost report...")
    
    report = estimator.generate_cost_report(comparison_order, market_conditions)
    
    print(f"Cost Analysis Report:")
    print(f"  Order: {report['order_summary']['quantity']:,} shares {report['order_summary']['side']}")
    print(f"  Notional: ${report['order_summary']['notional_value']:,.0f}")
    print(f"  Base Cost: {report['base_estimate']['total_cost_bps']:.1f} bps")
    
    print(f"  Cost Attribution:")
    for component, percentage in report['cost_breakdown'].items():
        if percentage > 0.1:  # Only show significant components
            print(f"    {component}: {percentage:.1f}%")
    
    print(f"  Optimal Execution:")
    opt = report['optimal_execution'] 
    print(f"    Strategy: {opt['execution_style']}")
    print(f"    Cost: {opt['estimated_cost_bps']:.1f} bps")
    print(f"    Savings: {opt['cost_savings_bps']:.1f} bps")
    
    print(f"  Risk Metrics:")
    risk = report['risk_analysis']
    print(f"    Cost Uncertainty: ±{risk['cost_variance_bps']:.1f} bps")
    print(f"    95% Confidence: [{risk['confidence_interval_95'][0]:.1f}, {risk['confidence_interval_95'][1]:.1f}] bps")
    print(f"    VaR (95%): {risk['value_at_risk_bps']:.1f} bps")
    print(f"    Efficiency Score: {risk['cost_efficiency_score']:.2f}")
    
    print("\nExecution cost estimation testing completed!")