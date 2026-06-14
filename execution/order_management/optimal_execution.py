"""
Optimal Execution Strategies for QUANTUM-FORGE
Implements advanced execution algorithms including TWAP, VWAP, implementation shortfall, and adaptive strategies.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
from numba import jit, prange
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import time
from collections import deque
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class ExecutionStyle(Enum):
    """Execution algorithm styles."""
    TWAP = "time_weighted_average_price"
    VWAP = "volume_weighted_average_price"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    MARKET_ON_CLOSE = "market_on_close"
    PARTICIPATION_RATE = "participation_rate"
    ADAPTIVE = "adaptive"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"

class OrderType(Enum):
    """Order types for execution."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    ICEBERG = "iceberg"
    HIDDEN = "hidden"
    PEG = "pegged"

class MarketCondition(Enum):
    """Market condition states."""
    LIQUID = "liquid"
    ILLIQUID = "illiquid"
    VOLATILE = "volatile"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    STRESSED = "stressed"

@dataclass
class ExecutionOrder:
    """Individual execution order specification."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    time_in_force: str = "DAY"
    execution_style: ExecutionStyle = ExecutionStyle.TWAP
    urgency: float = 0.5  # 0 = patient, 1 = urgent
    risk_aversion: float = 0.5  # 0 = risk-seeking, 1 = risk-averse
    max_participation_rate: float = 0.1  # Maximum market participation
    target_completion_time: Optional[int] = None  # Minutes
    price_limit: Optional[float] = None
    
@dataclass
class ExecutionSlice:
    """Individual slice of execution."""
    timestamp: float
    quantity: float
    price: Optional[float]
    order_type: OrderType
    expected_cost: float
    market_impact: float
    timing_risk: float

@dataclass
class ExecutionResult:
    """Results of execution strategy."""
    total_quantity: float
    average_price: float
    total_cost: float
    market_impact_cost: float
    timing_risk_cost: float
    implementation_shortfall: float
    vwap_performance: float
    slices: List[ExecutionSlice]
    completion_time: float
    success_rate: float

class MarketImpactModel:
    """Advanced market impact modeling."""
    
    def __init__(self, permanent_impact_coeff: float = 0.1,
                 temporary_impact_coeff: float = 0.5,
                 nonlinear_exponent: float = 0.6):
        """
        Initialize market impact model.
        
        Args:
            permanent_impact_coeff: Permanent impact coefficient
            temporary_impact_coeff: Temporary impact coefficient  
            nonlinear_exponent: Nonlinearity in impact function
        """
        self.permanent_coeff = permanent_impact_coeff
        self.temporary_coeff = temporary_impact_coeff
        self.nonlinear_exp = nonlinear_exponent
        
    def calculate_impact(self, quantity: float, volume_rate: float,
                        volatility: float, spread: float,
                        participation_rate: float) -> Tuple[float, float]:
        """
        Calculate market impact costs.
        
        Args:
            quantity: Order quantity
            volume_rate: Market volume rate
            volatility: Price volatility
            spread: Bid-ask spread
            participation_rate: Order participation rate
        
        Returns:
            Tuple of (permanent_impact, temporary_impact)
        """
        # Normalize by market conditions
        normalized_quantity = quantity / max(volume_rate, 1e-6)
        
        # Permanent impact (square-root law with modifications)
        permanent_impact = (self.permanent_coeff * volatility * 
                          np.power(normalized_quantity, self.nonlinear_exp))
        
        # Temporary impact (linear in participation rate)
        temporary_impact = (self.temporary_coeff * spread * 
                          participation_rate +
                          0.1 * volatility * np.power(participation_rate, 1.5))
        
        return permanent_impact, temporary_impact
    
    def estimate_total_cost(self, execution_schedule: List[ExecutionSlice],
                          market_data: Dict) -> float:
        """
        Estimate total execution cost for a schedule.
        
        Args:
            execution_schedule: List of execution slices
            market_data: Market data dictionary
        
        Returns:
            Total estimated cost
        """
        total_cost = 0.0
        cumulative_quantity = 0.0
        
        volume_rate = market_data.get('volume_rate', 1000)
        volatility = market_data.get('volatility', 0.02)
        spread = market_data.get('spread', 0.001)
        
        for slice_info in execution_schedule:
            participation_rate = slice_info.quantity / volume_rate
            
            permanent, temporary = self.calculate_impact(
                slice_info.quantity, volume_rate, volatility, 
                spread, participation_rate
            )
            
            # Permanent impact affects all remaining shares
            remaining_quantity = sum(s.quantity for s in execution_schedule 
                                   if s.timestamp >= slice_info.timestamp)
            
            slice_cost = (temporary * slice_info.quantity + 
                         permanent * remaining_quantity)
            
            total_cost += slice_cost
            cumulative_quantity += slice_info.quantity
        
        return total_cost

class TWAPStrategy:
    """Time-Weighted Average Price execution strategy."""
    
    def __init__(self, impact_model: MarketImpactModel):
        """Initialize TWAP strategy."""
        self.impact_model = impact_model
        
    def generate_schedule(self, order: ExecutionOrder, 
                         execution_horizon: int,
                         market_data: Dict) -> List[ExecutionSlice]:
        """
        Generate TWAP execution schedule.
        
        Args:
            order: ExecutionOrder object
            execution_horizon: Execution time horizon in minutes
            market_data: Market data dictionary
        
        Returns:
            List of ExecutionSlice objects
        """
        # Determine number of slices
        target_slices = min(execution_horizon, 50)  # Maximum 50 slices
        slice_duration = execution_horizon / target_slices
        
        # Equal quantity per slice
        quantity_per_slice = order.quantity / target_slices
        
        slices = []
        current_time = 0.0
        
        volume_rate = market_data.get('volume_rate', 1000)
        volatility = market_data.get('volatility', 0.02)
        spread = market_data.get('spread', 0.001)
        
        for i in range(target_slices):
            # Adjust final slice for rounding
            if i == target_slices - 1:
                quantity_per_slice = order.quantity - sum(s.quantity for s in slices)
            
            # Calculate participation rate
            participation_rate = min(
                quantity_per_slice / (volume_rate * slice_duration / 60),
                order.max_participation_rate
            )
            
            # Calculate impact costs
            permanent, temporary = self.impact_model.calculate_impact(
                quantity_per_slice, volume_rate, volatility, 
                spread, participation_rate
            )
            
            # Determine order type based on urgency
            if order.urgency > 0.8:
                order_type = OrderType.MARKET
            elif order.urgency < 0.3:
                order_type = OrderType.LIMIT
            else:
                order_type = OrderType.LIMIT  # Default to limit orders
            
            slice_obj = ExecutionSlice(
                timestamp=current_time,
                quantity=quantity_per_slice,
                price=order.price,
                order_type=order_type,
                expected_cost=permanent + temporary,
                market_impact=permanent + temporary,
                timing_risk=0.0  # TWAP has minimal timing risk
            )
            
            slices.append(slice_obj)
            current_time += slice_duration
        
        return slices

class VWAPStrategy:
    """Volume-Weighted Average Price execution strategy."""
    
    def __init__(self, impact_model: MarketImpactModel):
        """Initialize VWAP strategy."""
        self.impact_model = impact_model
        
    def generate_schedule(self, order: ExecutionOrder,
                         execution_horizon: int,
                         volume_profile: np.ndarray,
                         market_data: Dict) -> List[ExecutionSlice]:
        """
        Generate VWAP execution schedule.
        
        Args:
            order: ExecutionOrder object
            execution_horizon: Execution horizon in minutes
            volume_profile: Expected volume profile over horizon
            market_data: Market data dictionary
        
        Returns:
            List of ExecutionSlice objects
        """
        # Normalize volume profile
        total_expected_volume = np.sum(volume_profile)
        volume_weights = volume_profile / total_expected_volume
        
        # Allocate quantity based on volume weights
        slice_quantities = order.quantity * volume_weights
        
        slices = []
        current_time = 0.0
        time_step = execution_horizon / len(volume_profile)
        
        volatility = market_data.get('volatility', 0.02)
        spread = market_data.get('spread', 0.001)
        
        for i, (quantity, expected_volume) in enumerate(zip(slice_quantities, volume_profile)):
            if quantity <= 0:
                current_time += time_step
                continue
            
            # Calculate participation rate
            participation_rate = min(
                quantity / max(expected_volume, 1e-6),
                order.max_participation_rate
            )
            
            # Calculate impact costs
            permanent, temporary = self.impact_model.calculate_impact(
                quantity, expected_volume, volatility, 
                spread, participation_rate
            )
            
            # Order type selection
            if participation_rate > 0.15:  # High participation
                order_type = OrderType.ICEBERG  # Hide large orders
            elif order.urgency > 0.7:
                order_type = OrderType.MARKET
            else:
                order_type = OrderType.LIMIT
            
            slice_obj = ExecutionSlice(
                timestamp=current_time,
                quantity=quantity,
                price=order.price,
                order_type=order_type,
                expected_cost=permanent + temporary,
                market_impact=permanent + temporary,
                timing_risk=volatility * np.sqrt(time_step / 252)  # Daily vol scaled
            )
            
            slices.append(slice_obj)
            current_time += time_step
        
        return slices

class ImplementationShortfallStrategy:
    """Implementation Shortfall (Almgren-Chriss) execution strategy."""
    
    def __init__(self, impact_model: MarketImpactModel):
        """Initialize Implementation Shortfall strategy."""
        self.impact_model = impact_model
        
    def generate_schedule(self, order: ExecutionOrder,
                         execution_horizon: int,
                         market_data: Dict) -> List[ExecutionSlice]:
        """
        Generate optimal IS execution schedule.
        
        Args:
            order: ExecutionOrder object
            execution_horizon: Execution horizon in minutes
            market_data: Market data dictionary
        
        Returns:
            List of ExecutionSlice objects
        """
        # Market parameters
        volatility = market_data.get('volatility', 0.02)
        spread = market_data.get('spread', 0.001)
        volume_rate = market_data.get('volume_rate', 1000)
        
        # Model parameters
        gamma = self.impact_model.permanent_coeff  # Permanent impact
        eta = self.impact_model.temporary_coeff    # Temporary impact
        sigma = volatility
        
        # Risk aversion parameter (scaled by order risk aversion)
        lambda_risk = order.risk_aversion * 1e-6  # Risk aversion coefficient
        
        # Optimal trading trajectory (Almgren-Chriss solution)
        T = execution_horizon / (252 * 6.5 * 60)  # Convert to years
        n_steps = min(execution_horizon, 100)
        dt = T / n_steps
        
        # Calculate optimal kappa
        kappa = np.sqrt(lambda_risk * sigma**2 / eta)
        
        # Optimal trajectory
        slices = []
        current_time = 0.0
        time_step = execution_horizon / n_steps
        
        for i in range(n_steps):
            t = i * dt
            time_remaining = T - t
            
            # Optimal trading rate (exponential decay)
            if time_remaining > 1e-6:
                sinh_term = np.sinh(kappa * time_remaining)
                cosh_term = np.cosh(kappa * time_remaining)
                
                # Remaining quantity to trade
                remaining_fraction = sinh_term / np.sinh(kappa * T)
                
                # Trading rate
                trading_rate = (kappa * cosh_term / sinh_term * 
                              order.quantity * remaining_fraction)
                
                quantity_this_step = trading_rate * dt * (252 * 6.5 * 60)  # Convert back
                quantity_this_step = min(quantity_this_step, 
                                       order.quantity - sum(s.quantity for s in slices))
            else:
                # Final slice - trade remaining quantity
                quantity_this_step = order.quantity - sum(s.quantity for s in slices)
            
            if quantity_this_step <= 0:
                current_time += time_step
                continue
            
            # Calculate participation rate
            participation_rate = min(
                quantity_this_step / (volume_rate * time_step / 60),
                order.max_participation_rate
            )
            
            # Calculate costs
            permanent, temporary = self.impact_model.calculate_impact(
                quantity_this_step, volume_rate, volatility,
                spread, participation_rate
            )
            
            # Timing risk
            timing_risk = sigma * np.sqrt(dt)
            
            # Order type based on urgency and quantity
            if participation_rate > 0.2:
                order_type = OrderType.ICEBERG
            elif order.urgency > 0.8:
                order_type = OrderType.MARKET
            else:
                order_type = OrderType.LIMIT
            
            slice_obj = ExecutionSlice(
                timestamp=current_time,
                quantity=quantity_this_step,
                price=order.price,
                order_type=order_type,
                expected_cost=permanent + temporary,
                market_impact=permanent + temporary,
                timing_risk=timing_risk
            )
            
            slices.append(slice_obj)
            current_time += time_step
        
        return slices

class AdaptiveStrategy:
    """Adaptive execution strategy that adjusts to market conditions."""
    
    def __init__(self, impact_model: MarketImpactModel):
        """Initialize adaptive strategy."""
        self.impact_model = impact_model
        self.twap_strategy = TWAPStrategy(impact_model)
        self.vwap_strategy = VWAPStrategy(impact_model)
        self.is_strategy = ImplementationShortfallStrategy(impact_model)
        
    def classify_market_condition(self, market_data: Dict) -> MarketCondition:
        """
        Classify current market condition.
        
        Args:
            market_data: Market data dictionary
        
        Returns:
            MarketCondition enum
        """
        volatility = market_data.get('volatility', 0.02)
        volume = market_data.get('volume_rate', 1000)
        spread = market_data.get('spread', 0.001)
        price_trend = market_data.get('price_trend', 0.0)
        
        # Classification logic
        if spread > 0.005:  # Wide spreads
            return MarketCondition.ILLIQUID
        elif volatility > 0.05:  # High volatility
            return MarketCondition.VOLATILE
        elif abs(price_trend) > 0.02:  # Strong trend
            return MarketCondition.TRENDING
        elif volatility < 0.01 and volume > 2000:  # Low vol, high volume
            return MarketCondition.LIQUID
        elif volatility > 0.03 and volume < 500:  # High vol, low volume
            return MarketCondition.STRESSED
        else:
            return MarketCondition.MEAN_REVERTING
    
    def select_strategy(self, order: ExecutionOrder, 
                       market_condition: MarketCondition) -> ExecutionStyle:
        """
        Select optimal execution strategy based on conditions.
        
        Args:
            order: ExecutionOrder object
            market_condition: Current market condition
        
        Returns:
            Selected ExecutionStyle
        """
        # Strategy selection matrix
        if market_condition == MarketCondition.LIQUID:
            if order.urgency > 0.7:
                return ExecutionStyle.AGGRESSIVE
            elif order.risk_aversion > 0.7:
                return ExecutionStyle.IMPLEMENTATION_SHORTFALL
            else:
                return ExecutionStyle.VWAP
                
        elif market_condition == MarketCondition.ILLIQUID:
            if order.urgency > 0.8:
                return ExecutionStyle.MARKET_ON_CLOSE
            else:
                return ExecutionStyle.PASSIVE
                
        elif market_condition == MarketCondition.VOLATILE:
            if order.risk_aversion > 0.6:
                return ExecutionStyle.IMPLEMENTATION_SHORTFALL
            else:
                return ExecutionStyle.PARTICIPATION_RATE
                
        elif market_condition == MarketCondition.TRENDING:
            if order.urgency > 0.6:
                return ExecutionStyle.AGGRESSIVE
            else:
                return ExecutionStyle.TWAP
                
        elif market_condition == MarketCondition.STRESSED:
            return ExecutionStyle.PASSIVE
            
        else:  # MEAN_REVERTING
            return ExecutionStyle.VWAP
    
    def generate_schedule(self, order: ExecutionOrder,
                         execution_horizon: int,
                         market_data: Dict,
                         volume_profile: Optional[np.ndarray] = None) -> List[ExecutionSlice]:
        """
        Generate adaptive execution schedule.
        
        Args:
            order: ExecutionOrder object
            execution_horizon: Execution horizon in minutes
            market_data: Market data dictionary
            volume_profile: Volume profile (optional)
        
        Returns:
            List of ExecutionSlice objects
        """
        # Classify market condition
        market_condition = self.classify_market_condition(market_data)
        
        # Select appropriate strategy
        selected_style = self.select_strategy(order, market_condition)
        
        # Generate base schedule
        if selected_style in [ExecutionStyle.VWAP] and volume_profile is not None:
            base_schedule = self.vwap_strategy.generate_schedule(
                order, execution_horizon, volume_profile, market_data
            )
        elif selected_style == ExecutionStyle.IMPLEMENTATION_SHORTFALL:
            base_schedule = self.is_strategy.generate_schedule(
                order, execution_horizon, market_data
            )
        else:  # Default to TWAP with modifications
            base_schedule = self.twap_strategy.generate_schedule(
                order, execution_horizon, market_data
            )
        
        # Apply adaptive modifications
        modified_schedule = self._apply_adaptive_modifications(
            base_schedule, market_condition, order, market_data
        )
        
        return modified_schedule
    
    def _apply_adaptive_modifications(self, schedule: List[ExecutionSlice],
                                    market_condition: MarketCondition,
                                    order: ExecutionOrder,
                                    market_data: Dict) -> List[ExecutionSlice]:
        """Apply adaptive modifications to base schedule."""
        
        modified_schedule = []
        
        for slice_info in schedule:
            modified_slice = ExecutionSlice(
                timestamp=slice_info.timestamp,
                quantity=slice_info.quantity,
                price=slice_info.price,
                order_type=slice_info.order_type,
                expected_cost=slice_info.expected_cost,
                market_impact=slice_info.market_impact,
                timing_risk=slice_info.timing_risk
            )
            
            # Modify based on market condition
            if market_condition == MarketCondition.VOLATILE:
                # Reduce quantity per slice, increase frequency
                modified_slice.quantity *= 0.7
                modified_slice.order_type = OrderType.LIMIT
                
            elif market_condition == MarketCondition.ILLIQUID:
                # Use iceberg orders, reduce participation
                if slice_info.quantity > order.quantity * 0.05:
                    modified_slice.order_type = OrderType.ICEBERG
                
            elif market_condition == MarketCondition.TRENDING:
                # Front-load or back-load based on trend direction
                trend = market_data.get('price_trend', 0.0)
                if (order.side == 'buy' and trend > 0) or (order.side == 'sell' and trend < 0):
                    # Accelerate execution
                    if slice_info.timestamp < len(schedule) * 0.3:
                        modified_slice.quantity *= 1.2
                    
            elif market_condition == MarketCondition.STRESSED:
                # Ultra-conservative approach
                modified_slice.quantity *= 0.5
                modified_slice.order_type = OrderType.LIMIT
            
            modified_schedule.append(modified_slice)
        
        # Ensure total quantity matches
        total_modified = sum(s.quantity for s in modified_schedule)
        if total_modified != order.quantity:
            # Proportionally adjust
            adjustment_factor = order.quantity / total_modified
            for slice_info in modified_schedule:
                slice_info.quantity *= adjustment_factor
        
        return modified_schedule

class ExecutionEngine:
    """Main execution engine coordinating all strategies."""
    
    def __init__(self, impact_model: Optional[MarketImpactModel] = None):
        """Initialize execution engine."""
        if impact_model is None:
            impact_model = MarketImpactModel()
        
        self.impact_model = impact_model
        self.twap_strategy = TWAPStrategy(impact_model)
        self.vwap_strategy = VWAPStrategy(impact_model)
        self.is_strategy = ImplementationShortfallStrategy(impact_model)
        self.adaptive_strategy = AdaptiveStrategy(impact_model)
        
        self.execution_history = []
        
    def execute_order(self, order: ExecutionOrder,
                     execution_horizon: int,
                     market_data: Dict,
                     volume_profile: Optional[np.ndarray] = None,
                     real_time_execution: bool = False) -> ExecutionResult:
        """
        Execute order using specified strategy.
        
        Args:
            order: ExecutionOrder object
            execution_horizon: Execution horizon in minutes
            market_data: Market data dictionary
            volume_profile: Volume profile for VWAP
            real_time_execution: Whether to simulate real-time execution
        
        Returns:
            ExecutionResult object
        """
        start_time = time.time()
        
        # Generate execution schedule
        if order.execution_style == ExecutionStyle.TWAP:
            schedule = self.twap_strategy.generate_schedule(
                order, execution_horizon, market_data
            )
        elif order.execution_style == ExecutionStyle.VWAP and volume_profile is not None:
            schedule = self.vwap_strategy.generate_schedule(
                order, execution_horizon, volume_profile, market_data
            )
        elif order.execution_style == ExecutionStyle.IMPLEMENTATION_SHORTFALL:
            schedule = self.is_strategy.generate_schedule(
                order, execution_horizon, market_data
            )
        elif order.execution_style == ExecutionStyle.ADAPTIVE:
            schedule = self.adaptive_strategy.generate_schedule(
                order, execution_horizon, market_data, volume_profile
            )
        else:
            # Default to TWAP
            schedule = self.twap_strategy.generate_schedule(
                order, execution_horizon, market_data
            )
        
        # Simulate execution
        execution_results = self._simulate_execution(schedule, order, market_data, real_time_execution)
        
        # Store in history
        self.execution_history.append({
            'order': order,
            'schedule': schedule,
            'results': execution_results,
            'timestamp': start_time
        })
        
        return execution_results
    
    def _simulate_execution(self, schedule: List[ExecutionSlice],
                          order: ExecutionOrder,
                          market_data: Dict,
                          real_time: bool = False) -> ExecutionResult:
        """Simulate execution of schedule."""
        
        executed_slices = []
        total_quantity = 0.0
        total_cost = 0.0
        total_market_impact = 0.0
        total_timing_risk = 0.0
        
        base_price = market_data.get('current_price', 100.0)
        volatility = market_data.get('volatility', 0.02)
        
        successful_slices = 0
        
        for slice_info in schedule:
            if real_time:
                # Simulate real-time delays
                time.sleep(0.001)  # 1ms delay per slice
            
            # Deterministic price movement: small deterministic drift based on volatility and optional trend
            minute_std = volatility * np.sqrt(1/252/6.5/60)
            drift = market_data.get('price_trend', 0.0) * 1e-4
            # Use a small, reproducible fraction of the minute standard deviation as the price change
            price_change = minute_std * 0.1 + drift
            execution_price = base_price * (1 + price_change)
            
            # Add market impact
            impact_adjustment = slice_info.market_impact
            if order.side == 'buy':
                execution_price *= (1 + impact_adjustment)
            else:
                execution_price *= (1 - impact_adjustment)
            
            # Simulate execution success (depends on order type and market conditions)
            success_probability = self._calculate_fill_probability(slice_info, market_data)
            # Deterministic execution outcome: treat slices with success_probability >= 0.5 as full fills
            execution_successful = float(success_probability) >= 0.5

            if execution_successful:
                executed_quantity = slice_info.quantity
                successful_slices += 1
            else:
                executed_quantity = slice_info.quantity * 0.7  # Partial fill (deterministic)
            
            # Record execution
            executed_slice = ExecutionSlice(
                timestamp=slice_info.timestamp,
                quantity=executed_quantity,
                price=execution_price,
                order_type=slice_info.order_type,
                expected_cost=slice_info.expected_cost,
                market_impact=slice_info.market_impact,
                timing_risk=slice_info.timing_risk
            )
            
            executed_slices.append(executed_slice)
            
            # Update totals
            total_quantity += executed_quantity
            total_cost += executed_quantity * execution_price
            total_market_impact += executed_quantity * slice_info.market_impact * base_price
            total_timing_risk += slice_info.timing_risk
            
            # Update base price for next slice
            base_price = execution_price
        
        # Calculate performance metrics
        average_price = total_cost / total_quantity if total_quantity > 0 else 0.0
        
        # Implementation shortfall vs arrival price
        arrival_price = market_data.get('current_price', 100.0)
        implementation_shortfall = (average_price - arrival_price) / arrival_price
        if order.side == 'sell':
            implementation_shortfall *= -1
        
        # VWAP performance (simplified)
        benchmark_vwap = market_data.get('benchmark_vwap', arrival_price)
        vwap_performance = (average_price - benchmark_vwap) / benchmark_vwap
        if order.side == 'sell':
            vwap_performance *= -1
        
        completion_time = schedule[-1].timestamp if schedule else 0.0
        success_rate = successful_slices / len(schedule) if schedule else 0.0
        
        return ExecutionResult(
            total_quantity=total_quantity,
            average_price=average_price,
            total_cost=total_cost,
            market_impact_cost=total_market_impact,
            timing_risk_cost=total_timing_risk * base_price * total_quantity,
            implementation_shortfall=implementation_shortfall,
            vwap_performance=vwap_performance,
            slices=executed_slices,
            completion_time=completion_time,
            success_rate=success_rate
        )
    
    def _calculate_fill_probability(self, slice_info: ExecutionSlice, 
                                  market_data: Dict) -> float:
        """Calculate probability of order fill."""
        
        base_probability = 0.95  # Base fill probability
        
        # Adjust based on order type
        if slice_info.order_type == OrderType.MARKET:
            return 0.99  # Market orders almost always fill
        elif slice_info.order_type == OrderType.LIMIT:
            # Limit orders depend on market conditions
            volatility = market_data.get('volatility', 0.02)
            spread = market_data.get('spread', 0.001)
            
            # Higher volatility and spread reduce fill probability
            adjustment = -2 * volatility - 5 * spread
            return max(0.3, base_probability + adjustment)
        elif slice_info.order_type == OrderType.ICEBERG:
            return 0.85  # Iceberg orders may have reduced fill rates
        
        return base_probability
    
    def optimize_execution_parameters(self, order: ExecutionOrder,
                                    market_data: Dict,
                                    objective: str = 'implementation_shortfall') -> Dict:
        """
        Optimize execution parameters for given objective.
        
        Args:
            order: ExecutionOrder object
            market_data: Market data dictionary
            objective: Optimization objective
        
        Returns:
            Dictionary of optimal parameters
        """
        
        def objective_function(params):
            """Objective function for optimization."""
            urgency, risk_aversion, max_participation = params
            
            # Create modified order
            modified_order = ExecutionOrder(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                order_type=order.order_type,
                price=order.price,
                urgency=max(0.1, min(0.9, urgency)),
                risk_aversion=max(0.1, min(0.9, risk_aversion)),
                max_participation_rate=max(0.01, min(0.5, max_participation)),
                execution_style=ExecutionStyle.ADAPTIVE
            )
            
            # Execute with modified parameters
            result = self.execute_order(
                modified_order, 60, market_data  # 1-hour horizon
            )
            
            # Return objective value
            if objective == 'implementation_shortfall':
                return abs(result.implementation_shortfall)
            elif objective == 'market_impact':
                return result.market_impact_cost
            elif objective == 'total_cost':
                return result.total_cost
            else:
                return result.implementation_shortfall + result.market_impact_cost
        
        # Optimization bounds
        bounds = [(0.1, 0.9), (0.1, 0.9), (0.01, 0.3)]
        
        # Initial guess
        x0 = [order.urgency, order.risk_aversion, order.max_participation_rate]
        
        try:
            result = optimize.minimize(
                objective_function,
                x0,
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                optimal_urgency, optimal_risk_aversion, optimal_participation = result.x
                
                return {
                    'success': True,
                    'optimal_urgency': optimal_urgency,
                    'optimal_risk_aversion': optimal_risk_aversion,
                    'optimal_participation_rate': optimal_participation,
                    'expected_cost': result.fun,
                    'optimization_iterations': result.nit
                }
        except:
            pass
        
        return {
            'success': False,
            'optimal_urgency': order.urgency,
            'optimal_risk_aversion': order.risk_aversion,
            'optimal_participation_rate': order.max_participation_rate
        }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Optimal Execution Strategies...")
    
    # Initialize components
    impact_model = MarketImpactModel(
        permanent_impact_coeff=0.1,
        temporary_impact_coeff=0.5,
        nonlinear_exponent=0.6
    )
    
    execution_engine = ExecutionEngine(impact_model)
    
    # Sample market data
    market_data = {
        'current_price': 100.0,
        'volatility': 0.02,
        'volume_rate': 1000,
        'spread': 0.001,
        'price_trend': 0.005,
        'benchmark_vwap': 100.2
    }
    
    # Create sample execution order
    order = ExecutionOrder(
        symbol="AAPL",
        side="buy",
        quantity=10000,
        order_type=OrderType.LIMIT,
        price=100.0,
        execution_style=ExecutionStyle.ADAPTIVE,
        urgency=0.5,
        risk_aversion=0.6,
        max_participation_rate=0.1,
        target_completion_time=60
    )
    
    print(f"Order: {order.side.upper()} {order.quantity:,} shares of {order.symbol}")
    print(f"Execution style: {order.execution_style.value}")
    print(f"Urgency: {order.urgency:.2f}, Risk aversion: {order.risk_aversion:.2f}")
    
    # Test TWAP strategy
    print("\nTesting TWAP Strategy...")
    twap_order = ExecutionOrder(
        symbol=order.symbol, side=order.side, quantity=order.quantity,
        order_type=order.order_type, execution_style=ExecutionStyle.TWAP,
        urgency=order.urgency, risk_aversion=order.risk_aversion
    )
    
    twap_result = execution_engine.execute_order(
        twap_order, 60, market_data
    )
    
    print(f"TWAP Results:")
    print(f"  Total quantity executed: {twap_result.total_quantity:,.0f}")
    print(f"  Average price: ${twap_result.average_price:.4f}")
    print(f"  Total cost: ${twap_result.total_cost:,.2f}")
    print(f"  Implementation shortfall: {twap_result.implementation_shortfall*100:.3f}%")
    print(f"  Market impact cost: ${twap_result.market_impact_cost:.2f}")
    print(f"  Success rate: {twap_result.success_rate*100:.1f}%")
    print(f"  Number of slices: {len(twap_result.slices)}")
    
    # Test VWAP strategy with volume profile
    print("\nTesting VWAP Strategy...")
    
    # Generate realistic intraday volume profile
    hours = np.arange(0, 6.5, 0.25)  # Trading hours in 15-min intervals
    volume_profile = 1000 * (
        0.3 * np.exp(-((hours - 0.5)**2) / 0.5) +  # Opening spike
        0.5 * np.exp(-((hours - 6)**2) / 0.3) +    # Closing spike
        0.2  # Base volume
    )
    
    vwap_order = ExecutionOrder(
        symbol=order.symbol, side=order.side, quantity=order.quantity,
        order_type=order.order_type, execution_style=ExecutionStyle.VWAP,
        urgency=order.urgency, risk_aversion=order.risk_aversion
    )
    
    vwap_result = execution_engine.execute_order(
        vwap_order, 390, market_data, volume_profile  # Full trading day
    )
    
    print(f"VWAP Results:")
    print(f"  Total quantity executed: {vwap_result.total_quantity:,.0f}")
    print(f"  Average price: ${vwap_result.average_price:.4f}")
    print(f"  VWAP performance: {vwap_result.vwap_performance*100:.3f}%")
    print(f"  Market impact cost: ${vwap_result.market_impact_cost:.2f}")
    print(f"  Success rate: {vwap_result.success_rate*100:.1f}%")
    print(f"  Number of slices: {len(vwap_result.slices)}")
    
    # Test Implementation Shortfall strategy
    print("\nTesting Implementation Shortfall Strategy...")
    
    is_order = ExecutionOrder(
        symbol=order.symbol, side=order.side, quantity=order.quantity,
        order_type=order.order_type, execution_style=ExecutionStyle.IMPLEMENTATION_SHORTFALL,
        urgency=0.3, risk_aversion=0.8  # More risk-averse
    )
    
    is_result = execution_engine.execute_order(
        is_order, 120, market_data  # 2-hour horizon
    )
    
    print(f"Implementation Shortfall Results:")
    print(f"  Total quantity executed: {is_result.total_quantity:,.0f}")
    print(f"  Average price: ${is_result.average_price:.4f}")
    print(f"  Implementation shortfall: {is_result.implementation_shortfall*100:.3f}%")
    print(f"  Market impact cost: ${is_result.market_impact_cost:.2f}")
    print(f"  Timing risk cost: ${is_result.timing_risk_cost:.2f}")
    print(f"  Success rate: {is_result.success_rate*100:.1f}%")
    print(f"  Number of slices: {len(is_result.slices)}")
    
    # Test Adaptive strategy
    print("\nTesting Adaptive Strategy...")
    
    adaptive_result = execution_engine.execute_order(
        order, 90, market_data, volume_profile
    )
    
    print(f"Adaptive Strategy Results:")
    print(f"  Total quantity executed: {adaptive_result.total_quantity:,.0f}")
    print(f"  Average price: ${adaptive_result.average_price:.4f}")
    print(f"  Implementation shortfall: {adaptive_result.implementation_shortfall*100:.3f}%")
    print(f"  Market impact cost: ${adaptive_result.market_impact_cost:.2f}")
    print(f"  Success rate: {adaptive_result.success_rate*100:.1f}%")
    print(f"  Number of slices: {len(adaptive_result.slices)}")
    
    # Test parameter optimization
    print("\nTesting Parameter Optimization...")
    
    optimization_result = execution_engine.optimize_execution_parameters(
        order, market_data, objective='implementation_shortfall'
    )
    
    if optimization_result['success']:
        print(f"Optimization successful!")
        print(f"  Optimal urgency: {optimization_result['optimal_urgency']:.3f}")
        print(f"  Optimal risk aversion: {optimization_result['optimal_risk_aversion']:.3f}")
        print(f"  Optimal participation rate: {optimization_result['optimal_participation_rate']:.3f}")
        print(f"  Expected cost improvement: ${optimization_result['expected_cost']:.4f}")
        print(f"  Optimization iterations: {optimization_result['optimization_iterations']}")
    else:
        print("Optimization failed - using default parameters")
    
    # Performance comparison
    print("\nStrategy Performance Comparison:")
    print(f"{'Strategy':<20} {'Shortfall %':<12} {'Impact $':<10} {'Success %':<10}")
    print("-" * 55)
    print(f"{'TWAP':<20} {twap_result.implementation_shortfall*100:<12.3f} {twap_result.market_impact_cost:<10.2f} {twap_result.success_rate*100:<10.1f}")
    print(f"{'VWAP':<20} {vwap_result.implementation_shortfall*100:<12.3f} {vwap_result.market_impact_cost:<10.2f} {vwap_result.success_rate*100:<10.1f}")
    print(f"{'Impl. Shortfall':<20} {is_result.implementation_shortfall*100:<12.3f} {is_result.market_impact_cost:<10.2f} {is_result.success_rate*100:<10.1f}")
    print(f"{'Adaptive':<20} {adaptive_result.implementation_shortfall*100:<12.3f} {adaptive_result.market_impact_cost:<10.2f} {adaptive_result.success_rate*100:<10.1f}")
    
    print("\nOptimal execution strategies testing completed successfully!")