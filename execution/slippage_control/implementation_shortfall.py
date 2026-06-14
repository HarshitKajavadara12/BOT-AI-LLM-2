"""
Implementation Shortfall
Optimal execution strategy based on Implementation Shortfall framework
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
import warnings


class ExecutionStyle(Enum):
    """Execution style preferences"""
    AGGRESSIVE = "AGGRESSIVE"     # Minimize timing risk, accept higher impact
    PASSIVE = "PASSIVE"          # Minimize market impact, accept timing risk
    BALANCED = "BALANCED"        # Balance between impact and timing risk
    OPPORTUNISTIC = "OPPORTUNISTIC"  # Wait for favorable conditions
    ICEBERG = "ICEBERG"          # Large hidden orders with small visible size


class RiskPreference(Enum):
    """Risk tolerance for execution"""
    RISK_AVERSE = "RISK_AVERSE"     # Minimize execution risk
    RISK_NEUTRAL = "RISK_NEUTRAL"   # Balanced risk approach
    RISK_SEEKING = "RISK_SEEKING"   # Accept higher risk for better prices


@dataclass
class ExecutionSchedule:
    """Optimal execution schedule"""
    total_quantity: float
    time_horizon_minutes: float
    
    # Schedule parameters
    slice_count: int = 10
    slice_intervals_minutes: List[float] = field(default_factory=list)
    slice_quantities: List[float] = field(default_factory=list)
    slice_participation_rates: List[float] = field(default_factory=list)
    
    # Execution style adjustments
    front_loaded_ratio: float = 0.0  # Front-load execution (0=uniform, 1=all upfront)
    back_loaded_ratio: float = 0.0   # Back-load execution
    
    # Risk controls
    max_slice_size: Optional[float] = None
    min_slice_size: Optional[float] = None
    adaptive_adjustment: bool = True
    
    def __post_init__(self):
        """Initialize schedule if not provided"""
        if not self.slice_intervals_minutes:
            self.slice_intervals_minutes = [
                self.time_horizon_minutes / self.slice_count
            ] * self.slice_count
        
        if not self.slice_quantities:
            self._generate_uniform_schedule()
    
    def _generate_uniform_schedule(self) -> None:
        """Generate uniform execution schedule"""
        base_quantity = self.total_quantity / self.slice_count
        
        self.slice_quantities = []
        
        for i in range(self.slice_count):
            # Apply front/back loading
            if self.front_loaded_ratio > 0:
                # More quantity in early slices
                weight = (self.slice_count - i) / self.slice_count
                adjustment = 1 + (weight - 0.5) * self.front_loaded_ratio
            elif self.back_loaded_ratio > 0:
                # More quantity in later slices
                weight = (i + 1) / self.slice_count
                adjustment = 1 + (weight - 0.5) * self.back_loaded_ratio
            else:
                adjustment = 1.0
            
            slice_qty = base_quantity * adjustment
            
            # Apply size constraints
            if self.max_slice_size:
                slice_qty = min(slice_qty, self.max_slice_size)
            if self.min_slice_size:
                slice_qty = max(slice_qty, self.min_slice_size)
            
            self.slice_quantities.append(slice_qty)
        
        # Normalize to ensure total matches
        total_scheduled = sum(self.slice_quantities)
        if total_scheduled > 0:
            scale_factor = self.total_quantity / total_scheduled
            self.slice_quantities = [qty * scale_factor for qty in self.slice_quantities]
    
    @property
    def total_scheduled_quantity(self) -> float:
        """Total quantity in schedule"""
        return sum(self.slice_quantities)
    
    @property
    def average_slice_size(self) -> float:
        """Average slice size"""
        return self.total_quantity / self.slice_count if self.slice_count > 0 else 0.0


@dataclass
class ShortfallComponents:
    """Implementation shortfall cost components"""
    
    # Timing costs (opportunity cost)
    timing_cost_bps: float = 0.0
    timing_cost_dollars: float = 0.0
    
    # Impact costs (market impact)
    impact_cost_bps: float = 0.0
    impact_cost_dollars: float = 0.0
    
    # Spread costs (bid-ask spread)
    spread_cost_bps: float = 0.0
    spread_cost_dollars: float = 0.0
    
    # Delay costs (execution delays)
    delay_cost_bps: float = 0.0
    delay_cost_dollars: float = 0.0
    
    # Commission and fees
    commission_dollars: float = 0.0
    
    # Total implementation shortfall
    total_shortfall_bps: float = 0.0
    total_shortfall_dollars: float = 0.0
    
    def __post_init__(self):
        """Calculate totals"""
        self.total_shortfall_bps = (
            self.timing_cost_bps + self.impact_cost_bps + 
            self.spread_cost_bps + self.delay_cost_bps
        )
        
        self.total_shortfall_dollars = (
            self.timing_cost_dollars + self.impact_cost_dollars +
            self.spread_cost_dollars + self.delay_cost_dollars + 
            self.commission_dollars
        )
    
    @property
    def cost_breakdown(self) -> Dict[str, float]:
        """Cost breakdown in basis points"""
        return {
            'timing_bps': self.timing_cost_bps,
            'impact_bps': self.impact_cost_bps,
            'spread_bps': self.spread_cost_bps,
            'delay_bps': self.delay_cost_bps,
            'total_bps': self.total_shortfall_bps
        }


@dataclass
class ExecutionState:
    """Current execution state"""
    order_id: str
    original_quantity: float
    executed_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    
    # Execution progress
    slices_completed: int = 0
    slices_remaining: int = 0
    current_slice_id: Optional[str] = None
    
    # Performance tracking
    volume_weighted_price: float = 0.0
    total_execution_cost: float = 0.0
    realized_shortfall: Optional[ShortfallComponents] = None
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    expected_completion_time: Optional[datetime] = None
    actual_completion_time: Optional[datetime] = None
    
    # Market context
    decision_price: float = 0.0  # Price when execution decision was made
    current_market_price: float = 0.0
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.original_quantity - self.executed_quantity
    
    @property
    def execution_progress(self) -> float:
        """Execution progress as percentage"""
        if self.original_quantity > 0:
            return self.executed_quantity / self.original_quantity
        return 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if execution is complete"""
        return self.remaining_quantity <= 0.01  # Allow small rounding errors
    
    @property
    def elapsed_time_minutes(self) -> float:
        """Elapsed time in minutes"""
        return (datetime.now() - self.start_time).total_seconds() / 60.0


class ImplementationShortfallOptimizer:
    """
    Implementation Shortfall optimization engine
    """
    
    def __init__(self, name: str = "IS_Optimizer"):
        self.name = name
        
        # Model parameters
        self.risk_aversion_lambda = 2.5e-7  # Risk aversion parameter
        self.permanent_impact_eta = 1e-6    # Permanent impact coefficient
        self.temporary_impact_gamma = 2.5e-7 # Temporary impact coefficient
        self.volatility_sigma = 0.30        # Price volatility (annual)
        
        # Execution constraints
        self.max_participation_rate = 0.20   # Max 20% of volume
        self.min_participation_rate = 0.01   # Min 1% of volume
        self.max_slice_count = 50            # Maximum number of slices
        self.min_slice_count = 2             # Minimum number of slices
        
        # Optimization cache
        self.optimization_cache: Dict[str, ExecutionSchedule] = {}
        
        # Performance tracking
        self.optimization_count = 0
        self.cache_hits = 0
    
    def optimize_execution_schedule(
        self,
        total_quantity: float,
        time_horizon_minutes: float,
        current_price: float,
        daily_volume_shares: float,
        volatility: Optional[float] = None,
        execution_style: ExecutionStyle = ExecutionStyle.BALANCED,
        risk_preference: RiskPreference = RiskPreference.RISK_NEUTRAL
    ) -> ExecutionSchedule:
        """
        Optimize execution schedule using Implementation Shortfall framework
        
        Args:
            total_quantity: Total shares to execute
            time_horizon_minutes: Time horizon for execution
            current_price: Current stock price
            daily_volume_shares: Average daily volume in shares
            volatility: Stock volatility (annual)
            execution_style: Execution style preference
            risk_preference: Risk tolerance
        
        Returns:
            Optimal execution schedule
        """
        
        # Use provided volatility or default
        sigma = volatility if volatility is not None else self.volatility_sigma
        
        # Create cache key
        cache_key = self._create_cache_key(
            total_quantity, time_horizon_minutes, current_price,
            daily_volume_shares, sigma, execution_style, risk_preference
        )
        
        # Check cache
        if cache_key in self.optimization_cache:
            self.cache_hits += 1
            return self.optimization_cache[cache_key]
        
        # Perform optimization
        schedule = self._optimize_schedule(
            total_quantity=total_quantity,
            time_horizon_minutes=time_horizon_minutes,
            current_price=current_price,
            daily_volume_shares=daily_volume_shares,
            volatility=sigma,
            execution_style=execution_style,
            risk_preference=risk_preference
        )
        
        # Cache result
        self.optimization_cache[cache_key] = schedule
        self.optimization_count += 1
        
        return schedule
    
    def _optimize_schedule(
        self,
        total_quantity: float,
        time_horizon_minutes: float,
        current_price: float,
        daily_volume_shares: float,
        volatility: float,
        execution_style: ExecutionStyle,
        risk_preference: RiskPreference
    ) -> ExecutionSchedule:
        """Perform the actual schedule optimization"""
        
        # Convert time to years for calculations
        T = time_horizon_minutes / (252 * 24 * 60)  # Trading year
        
        # Adjust risk aversion based on preference
        risk_multipliers = {
            RiskPreference.RISK_AVERSE: 2.0,
            RiskPreference.RISK_NEUTRAL: 1.0,
            RiskPreference.RISK_SEEKING: 0.5
        }
        
        lambda_adj = self.risk_aversion_lambda * risk_multipliers.get(risk_preference, 1.0)
        
        # Calculate optimal number of slices
        if execution_style == ExecutionStyle.AGGRESSIVE:
            # Fewer, larger slices to minimize timing risk
            n_slices = max(2, min(10, int(time_horizon_minutes / 15)))  # Every 15 minutes
        elif execution_style == ExecutionStyle.PASSIVE:
            # More, smaller slices to minimize impact
            n_slices = max(10, min(self.max_slice_count, int(time_horizon_minutes / 5)))  # Every 5 minutes
        elif execution_style == ExecutionStyle.ICEBERG:
            # Many small slices
            n_slices = max(20, min(self.max_slice_count, int(time_horizon_minutes / 2)))  # Every 2 minutes
        else:  # BALANCED or OPPORTUNISTIC
            # Moderate number of slices
            n_slices = max(5, min(25, int(time_horizon_minutes / 10)))  # Every 10 minutes
        
        # Ensure reasonable bounds
        n_slices = max(self.min_slice_count, min(self.max_slice_count, n_slices))
        
        # Calculate slice intervals
        if execution_style == ExecutionStyle.OPPORTUNISTIC:
            # Non-uniform intervals (more time for later slices)
            intervals = self._generate_opportunistic_intervals(time_horizon_minutes, n_slices)
        else:
            # Uniform intervals
            intervals = [time_horizon_minutes / n_slices] * n_slices
        
        # Calculate optimal trading trajectory using Almgren-Chriss model
        slice_quantities = self._calculate_optimal_trajectory(
            total_quantity=total_quantity,
            n_slices=n_slices,
            time_horizon=T,
            current_price=current_price,
            daily_volume=daily_volume_shares,
            volatility=volatility,
            lambda_risk=lambda_adj,
            execution_style=execution_style
        )
        
        # Calculate participation rates
        participation_rates = []
        for i, (qty, interval) in enumerate(zip(slice_quantities, intervals)):
            # Estimate volume during slice interval
            expected_volume_during_slice = daily_volume_shares * (interval / (24 * 60))
            
            if expected_volume_during_slice > 0:
                participation_rate = qty / expected_volume_during_slice
                # Apply constraints
                participation_rate = max(
                    self.min_participation_rate,
                    min(self.max_participation_rate, participation_rate)
                )
            else:
                participation_rate = self.max_participation_rate
            
            participation_rates.append(participation_rate)
        
        # Apply execution style adjustments
        front_loaded_ratio = 0.0
        back_loaded_ratio = 0.0
        
        if execution_style == ExecutionStyle.AGGRESSIVE:
            front_loaded_ratio = 0.3  # Front-load 30%
        elif execution_style == ExecutionStyle.OPPORTUNISTIC:
            back_loaded_ratio = 0.2   # Back-load 20%
        
        # Create execution schedule
        schedule = ExecutionSchedule(
            total_quantity=total_quantity,
            time_horizon_minutes=time_horizon_minutes,
            slice_count=n_slices,
            slice_intervals_minutes=intervals,
            slice_quantities=slice_quantities,
            slice_participation_rates=participation_rates,
            front_loaded_ratio=front_loaded_ratio,
            back_loaded_ratio=back_loaded_ratio,
            adaptive_adjustment=execution_style != ExecutionStyle.ICEBERG
        )
        
        return schedule
    
    def _calculate_optimal_trajectory(
        self,
        total_quantity: float,
        n_slices: int,
        time_horizon: float,
        current_price: float,
        daily_volume: float,
        volatility: float,
        lambda_risk: float,
        execution_style: ExecutionStyle
    ) -> List[float]:
        """Calculate optimal trading trajectory"""
        
        # Time intervals
        dt = time_horizon / n_slices
        
        # Model parameters
        eta = self.permanent_impact_eta
        gamma = self.temporary_impact_gamma
        sigma = volatility
        
        # Calculate kappa (decay rate)
        kappa = np.sqrt(lambda_risk * sigma**2 / gamma)
        
        # For simplicity, use uniform distribution with style adjustments
        if execution_style == ExecutionStyle.AGGRESSIVE:
            # Front-loaded exponential decay
            weights = np.exp(-0.5 * np.arange(n_slices))
        elif execution_style == ExecutionStyle.PASSIVE:
            # Back-loaded (reverse exponential)
            weights = np.exp(-0.5 * np.arange(n_slices)[::-1])
        elif execution_style == ExecutionStyle.ICEBERG:
            # Uniform small slices
            weights = np.ones(n_slices)
        else:  # BALANCED or OPPORTUNISTIC
            # Slightly front-loaded
            weights = np.exp(-0.2 * np.arange(n_slices))
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate slice quantities
        slice_quantities = [total_quantity * weight for weight in weights]
        
        return slice_quantities
    
    def _generate_opportunistic_intervals(
        self,
        total_time_minutes: float,
        n_slices: int
    ) -> List[float]:
        """Generate non-uniform intervals for opportunistic execution"""
        
        # Start with equal intervals
        base_interval = total_time_minutes / n_slices
        
        # Make later intervals longer (more time to wait for opportunities)
        intervals = []
        remaining_time = total_time_minutes
        
        for i in range(n_slices):
            if i < n_slices - 1:
                # Earlier slices get shorter intervals
                weight = 0.5 + 0.5 * (i / n_slices)  # 0.5 to 1.0
                interval = base_interval * weight
                intervals.append(min(interval, remaining_time))
                remaining_time -= interval
            else:
                # Last slice gets all remaining time
                intervals.append(max(remaining_time, 0))
        
        return intervals
    
    def calculate_expected_shortfall(
        self,
        schedule: ExecutionSchedule,
        current_price: float,
        daily_volume_shares: float,
        volatility: Optional[float] = None
    ) -> ShortfallComponents:
        """
        Calculate expected implementation shortfall for a schedule
        
        Args:
            schedule: Execution schedule
            current_price: Current stock price
            daily_volume_shares: Daily volume in shares
            volatility: Stock volatility
        
        Returns:
            Expected shortfall components
        """
        
        sigma = volatility if volatility is not None else self.volatility_sigma
        
        # Calculate timing cost (opportunity cost of not trading immediately)
        timing_cost_bps = 0.5 * sigma * np.sqrt(
            schedule.time_horizon_minutes / (252 * 24 * 60)
        ) * 10000
        
        timing_cost_dollars = (
            timing_cost_bps / 10000
        ) * schedule.total_quantity * current_price
        
        # Calculate impact cost
        total_impact_cost_bps = 0.0
        
        for i, qty in enumerate(schedule.slice_quantities):
            # Participation rate for this slice
            participation_rate = schedule.slice_participation_rates[i] if schedule.slice_participation_rates else 0.1
            
            # Slice impact (simplified linear model)
            slice_impact_bps = self.permanent_impact_eta * (qty / daily_volume_shares) * 10000
            slice_impact_bps += self.temporary_impact_gamma * participation_rate * 10000
            
            total_impact_cost_bps += slice_impact_bps * (qty / schedule.total_quantity)
        
        impact_cost_dollars = (
            total_impact_cost_bps / 10000
        ) * schedule.total_quantity * current_price
        
        # Spread cost (half spread per share)
        spread_cost_bps = 5.0  # Assume 5 bps half spread
        spread_cost_dollars = (
            spread_cost_bps / 10000
        ) * schedule.total_quantity * current_price
        
        # Delay cost (minimal for optimized schedule)
        delay_cost_bps = 1.0  # Assume 1 bps delay cost
        delay_cost_dollars = (
            delay_cost_bps / 10000
        ) * schedule.total_quantity * current_price
        
        # Commission (simplified)
        commission_dollars = schedule.total_quantity * 0.005  # 0.5 cents per share
        
        return ShortfallComponents(
            timing_cost_bps=timing_cost_bps,
            timing_cost_dollars=timing_cost_dollars,
            impact_cost_bps=total_impact_cost_bps,
            impact_cost_dollars=impact_cost_dollars,
            spread_cost_bps=spread_cost_bps,
            spread_cost_dollars=spread_cost_dollars,
            delay_cost_bps=delay_cost_bps,
            delay_cost_dollars=delay_cost_dollars,
            commission_dollars=commission_dollars
        )
    
    def update_schedule_realtime(
        self,
        original_schedule: ExecutionSchedule,
        execution_state: ExecutionState,
        current_market_conditions: Dict[str, Any]
    ) -> ExecutionSchedule:
        """
        Update execution schedule based on real-time conditions
        
        Args:
            original_schedule: Original execution schedule
            execution_state: Current execution state
            current_market_conditions: Current market data
        
        Returns:
            Updated execution schedule
        """
        
        if not original_schedule.adaptive_adjustment:
            return original_schedule  # No adaptation for non-adaptive strategies
        
        # Calculate remaining execution parameters
        remaining_quantity = execution_state.remaining_quantity
        remaining_time = None
        
        if execution_state.expected_completion_time:
            remaining_time = (
                execution_state.expected_completion_time - datetime.now()
            ).total_seconds() / 60.0
        else:
            # Estimate based on original schedule
            elapsed_ratio = execution_state.execution_progress
            if elapsed_ratio > 0:
                total_estimated_time = execution_state.elapsed_time_minutes / elapsed_ratio
                remaining_time = total_estimated_time - execution_state.elapsed_time_minutes
            else:
                remaining_time = original_schedule.time_horizon_minutes
        
        remaining_time = max(1.0, remaining_time)  # At least 1 minute
        
        # Extract current market conditions
        current_price = current_market_conditions.get('price', execution_state.current_market_price)
        daily_volume = current_market_conditions.get('daily_volume', 1000000)  # Default
        volatility = current_market_conditions.get('volatility', self.volatility_sigma)
        
        # Determine if conditions have changed significantly
        price_change = abs(current_price - execution_state.decision_price) / execution_state.decision_price
        
        # Adapt execution style based on conditions
        if price_change > 0.02:  # 2% price change
            # Market moving against us - become more aggressive
            execution_style = ExecutionStyle.AGGRESSIVE
            risk_preference = RiskPreference.RISK_AVERSE
        elif volatility > self.volatility_sigma * 1.5:
            # High volatility - be more careful
            execution_style = ExecutionStyle.PASSIVE
            risk_preference = RiskPreference.RISK_AVERSE
        else:
            # Normal conditions - maintain balance
            execution_style = ExecutionStyle.BALANCED
            risk_preference = RiskPreference.RISK_NEUTRAL
        
        # Re-optimize for remaining quantity and time
        updated_schedule = self.optimize_execution_schedule(
            total_quantity=remaining_quantity,
            time_horizon_minutes=remaining_time,
            current_price=current_price,
            daily_volume_shares=daily_volume,
            volatility=volatility,
            execution_style=execution_style,
            risk_preference=risk_preference
        )
        
        return updated_schedule
    
    def _create_cache_key(self, *args) -> str:
        """Create cache key for optimization parameters"""
        # Create a hash of the parameters for caching
        return str(hash(args))
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization performance statistics"""
        
        cache_hit_rate = (
            self.cache_hits / max(1, self.optimization_count + self.cache_hits)
        )
        
        return {
            'total_optimizations': self.optimization_count,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'cached_schedules': len(self.optimization_cache)
        }
    
    def clear_cache(self) -> None:
        """Clear optimization cache"""
        self.optimization_cache.clear()
        self.cache_hits = 0


class ShortfallAnalyzer:
    """
    Implementation Shortfall analysis and reporting
    """
    
    def __init__(self):
        self.execution_history: List[Tuple[ExecutionState, ShortfallComponents]] = []
        
    def analyze_execution_performance(
        self,
        execution_state: ExecutionState,
        original_schedule: ExecutionSchedule,
        realized_prices: List[float],
        execution_times: List[datetime]
    ) -> ShortfallComponents:
        """
        Analyze actual execution performance vs. benchmark
        
        Args:
            execution_state: Final execution state
            original_schedule: Original execution schedule
            realized_prices: Actual execution prices
            execution_times: Actual execution times
        
        Returns:
            Realized implementation shortfall
        """
        
        if not realized_prices or not execution_times:
            raise ValueError("Need realized prices and times for analysis")
        
        # Calculate volume-weighted average price
        total_quantity = execution_state.executed_quantity
        vwap = execution_state.volume_weighted_price
        
        # Benchmark price (decision price)
        benchmark_price = execution_state.decision_price
        
        # Calculate timing cost (price moved while we were executing)
        current_price = execution_state.current_market_price
        price_movement = current_price - benchmark_price
        
        timing_cost_bps = (price_movement / benchmark_price) * 10000
        timing_cost_dollars = price_movement * total_quantity
        
        # Calculate impact cost (our execution vs. benchmark)
        impact_cost_per_share = vwap - benchmark_price
        impact_cost_bps = (impact_cost_per_share / benchmark_price) * 10000
        impact_cost_dollars = impact_cost_per_share * total_quantity
        
        # Spread cost (estimated)
        estimated_spread_cost_bps = 3.0  # Assume 3 bps average
        spread_cost_dollars = (estimated_spread_cost_bps / 10000) * total_quantity * benchmark_price
        
        # Delay cost (time delays in execution)
        planned_duration = original_schedule.time_horizon_minutes
        actual_duration = execution_state.elapsed_time_minutes
        delay_ratio = max(0, (actual_duration - planned_duration) / planned_duration)
        
        delay_cost_bps = delay_ratio * 5.0  # Up to 5 bps for significant delays
        delay_cost_dollars = (delay_cost_bps / 10000) * total_quantity * benchmark_price
        
        # Commission (from execution state)
        commission_dollars = execution_state.total_execution_cost
        
        shortfall = ShortfallComponents(
            timing_cost_bps=timing_cost_bps,
            timing_cost_dollars=timing_cost_dollars,
            impact_cost_bps=impact_cost_bps,
            impact_cost_dollars=impact_cost_dollars,
            spread_cost_bps=estimated_spread_cost_bps,
            spread_cost_dollars=spread_cost_dollars,
            delay_cost_bps=delay_cost_bps,
            delay_cost_dollars=delay_cost_dollars,
            commission_dollars=commission_dollars
        )
        
        # Store in history
        self.execution_history.append((execution_state, shortfall))
        
        return shortfall
    
    def generate_performance_report(
        self,
        lookback_periods: int = 30
    ) -> Dict[str, Any]:
        """Generate performance report for recent executions"""
        
        if not self.execution_history:
            return {'status': 'no_data'}
        
        # Get recent executions
        recent_executions = self.execution_history[-lookback_periods:]
        
        # Calculate statistics
        shortfalls = [shortfall for _, shortfall in recent_executions]
        
        timing_costs = [s.timing_cost_bps for s in shortfalls]
        impact_costs = [s.impact_cost_bps for s in shortfalls]
        total_costs = [s.total_shortfall_bps for s in shortfalls]
        
        import numpy as np
        
        report = {
            'period_summary': {
                'executions_analyzed': len(recent_executions),
                'average_timing_cost_bps': np.mean(timing_costs),
                'average_impact_cost_bps': np.mean(impact_costs),
                'average_total_cost_bps': np.mean(total_costs),
                'std_total_cost_bps': np.std(total_costs),
                'best_execution_bps': np.min(total_costs),
                'worst_execution_bps': np.max(total_costs)
            },
            'cost_breakdown': {
                'timing_cost_percentage': np.mean(timing_costs) / np.mean(total_costs) * 100,
                'impact_cost_percentage': np.mean(impact_costs) / np.mean(total_costs) * 100,
                'spread_cost_percentage': np.mean([s.spread_cost_bps for s in shortfalls]) / np.mean(total_costs) * 100
            },
            'performance_metrics': {
                'execution_success_rate': len([s for s in shortfalls if s.total_shortfall_bps < 50]) / len(shortfalls),
                'average_execution_time_minutes': np.mean([
                    state.elapsed_time_minutes for state, _ in recent_executions
                ]),
                'completion_rate': len([
                    state for state, _ in recent_executions if state.is_complete
                ]) / len(recent_executions)
            }
        }
        
        return report


if __name__ == "__main__":
    import random
    
    # Example usage and testing
    print("Testing Implementation Shortfall Optimizer...")
    
    # Create optimizer
    optimizer = ImplementationShortfallOptimizer("TestOptimizer")
    
    # Test parameters
    test_cases = [
        {
            'name': 'Large Passive Order',
            'quantity': 100_000,
            'time_horizon': 180,  # 3 hours
            'price': 150.0,
            'daily_volume': 5_000_000,
            'style': ExecutionStyle.PASSIVE,
            'risk': RiskPreference.RISK_AVERSE
        },
        {
            'name': 'Urgent Aggressive Order',
            'quantity': 25_000,
            'time_horizon': 30,   # 30 minutes
            'price': 150.0,
            'daily_volume': 5_000_000,
            'style': ExecutionStyle.AGGRESSIVE,
            'risk': RiskPreference.RISK_SEEKING
        },
        {
            'name': 'Balanced Medium Order',
            'quantity': 50_000,
            'time_horizon': 90,   # 1.5 hours
            'price': 150.0,
            'daily_volume': 5_000_000,
            'style': ExecutionStyle.BALANCED,
            'risk': RiskPreference.RISK_NEUTRAL
        },
        {
            'name': 'Iceberg Hidden Order',
            'quantity': 200_000,
            'time_horizon': 300,  # 5 hours
            'price': 150.0,
            'daily_volume': 5_000_000,
            'style': ExecutionStyle.ICEBERG,
            'risk': RiskPreference.RISK_NEUTRAL
        }
    ]
    
    print(f"\nOptimizing Execution Schedules:")
    
    schedules = {}
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(f"  Quantity: {test_case['quantity']:,} shares")
        print(f"  Time Horizon: {test_case['time_horizon']} minutes")
        print(f"  Style: {test_case['style'].value}")
        
        schedule = optimizer.optimize_execution_schedule(
            total_quantity=test_case['quantity'],
            time_horizon_minutes=test_case['time_horizon'],
            current_price=test_case['price'],
            daily_volume_shares=test_case['daily_volume'],
            execution_style=test_case['style'],
            risk_preference=test_case['risk']
        )
        
        schedules[test_case['name']] = schedule
        
        print(f"  Optimized Schedule:")
        print(f"    Slices: {schedule.slice_count}")
        print(f"    Average Slice Size: {schedule.average_slice_size:,.0f} shares")
        print(f"    Front-loaded: {schedule.front_loaded_ratio:.1%}")
        print(f"    Back-loaded: {schedule.back_loaded_ratio:.1%}")
        
        if schedule.slice_participation_rates:
            avg_participation = sum(schedule.slice_participation_rates) / len(schedule.slice_participation_rates)
            print(f"    Average Participation Rate: {avg_participation:.1%}")
        
        # Calculate expected shortfall
        expected_shortfall = optimizer.calculate_expected_shortfall(
            schedule=schedule,
            current_price=test_case['price'],
            daily_volume_shares=test_case['daily_volume']
        )
        
        print(f"  Expected Implementation Shortfall:")
        print(f"    Total: {expected_shortfall.total_shortfall_bps:.1f} bps (${expected_shortfall.total_shortfall_dollars:,.0f})")
        print(f"    Timing Cost: {expected_shortfall.timing_cost_bps:.1f} bps")
        print(f"    Impact Cost: {expected_shortfall.impact_cost_bps:.1f} bps")
        print(f"    Spread Cost: {expected_shortfall.spread_cost_bps:.1f} bps")
    
    # Test adaptive schedule updates
    print(f"\nTesting Adaptive Schedule Updates:")
    
    # Take the balanced order for testing
    original_schedule = schedules['Balanced Medium Order']
    
    # Simulate execution state after 50% completion
    execution_state = ExecutionState(
        order_id="TEST_001",
        original_quantity=50_000,
        executed_quantity=25_000,
        remaining_quantity=25_000,
        slices_completed=5,
        slices_remaining=5,
        volume_weighted_price=150.25,
        decision_price=150.0,
        current_market_price=150.50,  # Price moved up
        start_time=datetime.now() - timedelta(minutes=45),
        expected_completion_time=datetime.now() + timedelta(minutes=45)
    )
    
    # Market conditions changed
    changed_market_conditions = {
        'price': 150.50,
        'daily_volume': 4_000_000,  # Lower volume
        'volatility': 0.35          # Higher volatility
    }
    
    print(f"Original Schedule: {original_schedule.slice_count} slices")
    
    updated_schedule = optimizer.update_schedule_realtime(
        original_schedule=original_schedule,
        execution_state=execution_state,
        current_market_conditions=changed_market_conditions
    )
    
    print(f"Updated Schedule: {updated_schedule.slice_count} slices")
    print(f"Adaptation triggered due to:")
    print(f"  Price change: {((150.50 - 150.0) / 150.0) * 100:.1f}%")
    print(f"  Volume decrease: {((4_000_000 - 5_000_000) / 5_000_000) * 100:.1f}%")
    print(f"  Volatility increase: {((0.35 - 0.30) / 0.30) * 100:.1f}%")
    
    # Test performance analysis
    print(f"\nTesting Performance Analysis:")
    
    analyzer = ShortfallAnalyzer()
    
    # Simulate some completed executions
    for i in range(10):
        # Create mock execution state
        mock_state = ExecutionState(
            order_id=f"MOCK_{i:03d}",
            original_quantity=random.randint(10_000, 100_000),
            executed_quantity=random.randint(9_000, 100_000),
            volume_weighted_price=150.0 + random.uniform(-2, 2),
            decision_price=150.0,
            current_market_price=150.0 + random.uniform(-1, 1),
            start_time=datetime.now() - timedelta(hours=random.uniform(1, 6))
        )
        
        mock_state.remaining_quantity = 0  # Completed
        
        # Mock realized prices and times
        realized_prices = [150.0 + random.uniform(-1, 1) for _ in range(5)]
        execution_times = [datetime.now() - timedelta(minutes=random.randint(10, 300)) for _ in range(5)]
        
        # Analyze performance
        shortfall = analyzer.analyze_execution_performance(
            execution_state=mock_state,
            original_schedule=original_schedule,
            realized_prices=realized_prices,
            execution_times=execution_times
        )
    
    # Generate performance report
    performance_report = analyzer.generate_performance_report()
    
    print(f"Performance Report:")
    print(f"  Executions Analyzed: {performance_report['period_summary']['executions_analyzed']}")
    print(f"  Average Total Cost: {performance_report['period_summary']['average_total_cost_bps']:.1f} bps")
    print(f"  Cost Standard Deviation: {performance_report['period_summary']['std_total_cost_bps']:.1f} bps")
    print(f"  Best Execution: {performance_report['period_summary']['best_execution_bps']:.1f} bps")
    print(f"  Worst Execution: {performance_report['period_summary']['worst_execution_bps']:.1f} bps")
    
    print(f"\n  Cost Breakdown:")
    breakdown = performance_report['cost_breakdown']
    print(f"    Timing Cost: {breakdown['timing_cost_percentage']:.1f}%")
    print(f"    Impact Cost: {breakdown['impact_cost_percentage']:.1f}%")
    print(f"    Spread Cost: {breakdown['spread_cost_percentage']:.1f}%")
    
    print(f"\n  Performance Metrics:")
    metrics = performance_report['performance_metrics']
    print(f"    Success Rate (< 50 bps): {metrics['execution_success_rate']:.1%}")
    print(f"    Average Execution Time: {metrics['average_execution_time_minutes']:.1f} minutes")
    print(f"    Completion Rate: {metrics['completion_rate']:.1%}")
    
    # Test optimizer statistics
    print(f"\nOptimizer Statistics:")
    opt_stats = optimizer.get_optimization_statistics()
    
    for key, value in opt_stats.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.1%}" if 'rate' in key else f"  {key.replace('_', ' ').title()}: {value:.0f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print("\nImplementation Shortfall optimization testing completed!")