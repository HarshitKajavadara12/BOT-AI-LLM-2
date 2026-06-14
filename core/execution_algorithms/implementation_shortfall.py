"""
Implementation Shortfall Algorithm
Minimizes implementation shortfall (market impact + timing risk)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from abc import ABC, abstractmethod
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import math


@dataclass
class ISParameters:
    """Implementation Shortfall algorithm parameters"""
    total_quantity: float
    duration_minutes: int
    risk_aversion: float = 0.5  # Risk aversion parameter (0 = risk-neutral, 1 = very risk-averse)
    volatility_estimate: float = 0.02  # Daily volatility estimate
    spread_cost: float = 0.0005  # Half-spread as fraction of price
    market_impact_coeff: float = 0.1  # Market impact coefficient
    temporary_impact_coeff: float = 0.05  # Temporary impact coefficient
    participation_limit: float = 0.3  # Maximum participation rate
    min_order_size: float = 100
    max_order_size: float = 25000
    price_prediction_alpha: float = 0.1  # Price prediction decay factor


@dataclass
class RiskModel:
    """Risk model for implementation shortfall"""
    price_volatility: float
    volume_volatility: float = 0.3
    correlation_price_volume: float = -0.2
    half_life_minutes: float = 30.0  # Half-life of temporary impact


@dataclass
class MarketImpactModel:
    """Market impact model parameters"""
    permanent_impact_coeff: float = 0.1
    temporary_impact_coeff: float = 0.05
    impact_decay_rate: float = 0.1  # Decay rate of temporary impact
    nonlinear_exponent: float = 0.6  # Nonlinearity in market impact
    
    
@dataclass
class ISState:
    """Implementation Shortfall algorithm state"""
    remaining_quantity: float
    executed_quantity: float
    weighted_price: float
    total_cost: float
    start_time: datetime
    start_price: float
    elapsed_minutes: float = 0.0
    implementation_shortfall: float = 0.0
    market_impact_cost: float = 0.0
    timing_cost: float = 0.0
    opportunity_cost: float = 0.0


class ImplementationShortfallAlgorithm:
    """
    Implementation Shortfall Algorithm (Almgren-Chriss model)
    Optimizes trade-off between market impact and timing risk
    """
    
    def __init__(
        self,
        parameters: ISParameters,
        risk_model: Optional[RiskModel] = None,
        impact_model: Optional[MarketImpactModel] = None
    ):
        self.params = parameters
        self.risk_model = risk_model or RiskModel(price_volatility=parameters.volatility_estimate)
        self.impact_model = impact_model or MarketImpactModel()
        
        # Algorithm state
        self.state = None
        self.is_active = False
        self.side = None
        self.symbol = None
        
        # Market data and predictions
        self.market_data_history = deque(maxlen=500)
        self.price_predictions = deque(maxlen=100)
        self.volume_predictions = deque(maxlen=100)
        
        # Execution schedule
        self.optimal_schedule = None
        self.execution_times = []
        self.execution_history = []
        
        # Performance tracking
        self.cost_components = {
            'market_impact': [],
            'timing_cost': [],
            'spread_cost': [],
            'opportunity_cost': []
        }
        
    def initialize_order(
        self,
        side: str,
        symbol: str,
        market_data: 'MarketData'
    ) -> None:
        """
        Initialize Implementation Shortfall order
        
        Args:
            side: Order side ('buy' or 'sell')
            symbol: Trading symbol
            market_data: Current market data
        """
        
        self.side = side
        self.symbol = symbol
        self.is_active = True
        
        # Initialize state
        self.state = ISState(
            remaining_quantity=self.params.total_quantity,
            executed_quantity=0.0,
            weighted_price=0.0,
            total_cost=0.0,
            start_time=market_data.timestamp,
            start_price=market_data.last_price,
            elapsed_minutes=0.0
        )
        
        # Calculate optimal execution schedule
        self._calculate_optimal_schedule(market_data)
        
        print(f"IS Algorithm initialized: {self.params.total_quantity} shares")
        print(f"Risk aversion: {self.params.risk_aversion}")
        print(f"Optimal schedule calculated with {len(self.execution_times)} intervals")
    
    def update_market_data(self, market_data: 'MarketData') -> None:
        """Update with new market data and recalculate if needed"""
        
        self.market_data_history.append(market_data)
        
        if self.state:
            # Update elapsed time
            elapsed = market_data.timestamp - self.state.start_time
            self.state.elapsed_minutes = elapsed.total_seconds() / 60.0
            
            # Update cost components
            self._update_cost_analysis(market_data)
        
        # Update price predictions
        self._update_price_predictions(market_data)
        
        # Recalculate schedule if significant market changes
        if self._should_recalculate_schedule(market_data):
            self._calculate_optimal_schedule(market_data)
    
    def should_place_order(self, market_data: 'MarketData') -> bool:
        """Determine if order should be placed based on optimal schedule"""
        
        if not self.is_active or not self.state:
            return False
        
        if self.state.remaining_quantity <= 0:
            self.complete_order()
            return False
        
        # Check if it's time according to optimal schedule
        current_time_index = self._get_current_time_index()
        
        if current_time_index >= len(self.execution_times):
            # End of schedule - place remaining quantity
            return True
        
        # Check if enough time has passed since last execution
        target_time = self.execution_times[current_time_index]
        
        return self.state.elapsed_minutes >= target_time
    
    def calculate_order_size(self, market_data: 'MarketData') -> float:
        """Calculate optimal order size based on schedule and market conditions"""
        
        current_time_index = self._get_current_time_index()
        
        if self.optimal_schedule is None or current_time_index >= len(self.optimal_schedule):
            # Fallback: equal distribution of remaining quantity
            remaining_time = self.params.duration_minutes - self.state.elapsed_minutes
            remaining_intervals = max(1, remaining_time / 5)  # 5-minute intervals
            base_size = self.state.remaining_quantity / remaining_intervals
        else:
            # Use optimal schedule
            base_size = self.optimal_schedule[current_time_index]
            base_size = min(base_size, self.state.remaining_quantity)
        
        # Adjust for market conditions
        adjusted_size = self._adjust_size_for_market_conditions(base_size, market_data)
        
        # Apply constraints
        final_size = max(adjusted_size, self.params.min_order_size)
        final_size = min(final_size, self.params.max_order_size)
        final_size = min(final_size, self.state.remaining_quantity)
        
        return final_size
    
    def calculate_limit_price(self, market_data: 'MarketData', order_size: float) -> float:
        """Calculate optimal limit price minimizing implementation shortfall"""
        
        # Base price (mid-market)
        mid_price = (market_data.bid_price + market_data.ask_price) / 2
        
        # Estimate market impact
        permanent_impact = self._calculate_permanent_impact(order_size, market_data)
        temporary_impact = self._calculate_temporary_impact(order_size, market_data)
        
        # Price prediction
        predicted_drift = self._get_price_drift_prediction(market_data)
        
        if self.side == 'buy':
            # For buy orders, start from ask and adjust
            base_price = market_data.ask_price
            
            # Adjust for predicted price movement
            price_adjustment = predicted_drift * self._get_prediction_weight()
            
            # Account for impact
            impact_adjustment = (permanent_impact + temporary_impact) * 0.5
            
            limit_price = base_price + price_adjustment - impact_adjustment
            
        else:  # sell orders
            base_price = market_data.bid_price
            price_adjustment = predicted_drift * self._get_prediction_weight()
            impact_adjustment = (permanent_impact + temporary_impact) * 0.5
            
            limit_price = base_price + price_adjustment + impact_adjustment
        
        # Ensure reasonable limit relative to mid-price
        max_deviation = mid_price * 0.005  # 50 bps
        limit_price = max(min(limit_price, mid_price + max_deviation), 
                         mid_price - max_deviation)
        
        return limit_price
    
    def process_execution(
        self,
        executed_quantity: float,
        execution_price: float,
        execution_time: datetime
    ) -> None:
        """Process execution and update IS tracking"""
        
        if not self.state:
            return
        
        # Update basic state
        self.state.executed_quantity += executed_quantity
        self.state.remaining_quantity -= executed_quantity
        
        # Update weighted price
        new_cost = executed_quantity * execution_price
        self.state.total_cost += new_cost
        
        if self.state.executed_quantity > 0:
            self.state.weighted_price = self.state.total_cost / self.state.executed_quantity
        
        # Calculate implementation shortfall components
        arrival_price = self.state.start_price
        
        # Market impact cost
        mid_price = execution_price  # Simplified
        impact_cost = abs(execution_price - mid_price) * executed_quantity
        
        # Timing cost (difference from arrival price)
        timing_cost = (execution_price - arrival_price) * executed_quantity
        if self.side == 'sell':
            timing_cost = -timing_cost
        
        # Update cumulative costs
        self.state.market_impact_cost += impact_cost
        self.state.timing_cost += timing_cost
        
        # Total implementation shortfall
        total_cost = self.state.total_cost
        benchmark_cost = self.state.executed_quantity * arrival_price
        
        self.state.implementation_shortfall = (total_cost - benchmark_cost) / benchmark_cost
        
        if self.side == 'sell':
            self.state.implementation_shortfall = -self.state.implementation_shortfall
        
        # Record execution
        execution_record = {
            'timestamp': execution_time,
            'quantity': executed_quantity,
            'price': execution_price,
            'side': self.side,
            'cumulative_quantity': self.state.executed_quantity,
            'weighted_price': self.state.weighted_price,
            'implementation_shortfall': self.state.implementation_shortfall,
            'market_impact_cost': impact_cost,
            'timing_cost': timing_cost
        }
        
        self.execution_history.append(execution_record)
        
        # Check completion
        if self.state.remaining_quantity <= 0:
            self.complete_order()
        
        print(f"IS Execution: {executed_quantity} @ {execution_price:.4f}")
        print(f"  Implementation Shortfall: {self.state.implementation_shortfall:.4f}")
        print(f"  Market Impact: {impact_cost:.2f}, Timing: {timing_cost:.2f}")
    
    def complete_order(self) -> None:
        """Complete IS order and calculate final metrics"""
        
        self.is_active = False
        
        if self.state and self.state.executed_quantity > 0:
            # Calculate opportunity cost for unexecuted quantity
            if self.state.remaining_quantity > 0 and len(self.market_data_history) > 0:
                current_price = self.market_data_history[-1].last_price
                arrival_price = self.state.start_price
                
                opportunity_cost = (current_price - arrival_price) * self.state.remaining_quantity
                if self.side == 'sell':
                    opportunity_cost = -opportunity_cost
                
                self.state.opportunity_cost = opportunity_cost
            
            # Final metrics
            metrics = self.get_performance_metrics()
            
            print(f"IS Algorithm completed:")
            print(f"  Executed: {self.state.executed_quantity}/{self.params.total_quantity}")
            print(f"  Implementation Shortfall: {metrics['implementation_shortfall']:.4f}")
            print(f"  Market Impact: {metrics['market_impact_bps']:.1f} bps")
            print(f"  Timing Cost: {metrics['timing_cost_bps']:.1f} bps")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive IS performance metrics"""
        
        if not self.state or self.state.executed_quantity == 0:
            return {}
        
        # Basic metrics
        fill_rate = self.state.executed_quantity / self.params.total_quantity
        
        # Implementation shortfall (total)
        is_total = self.state.implementation_shortfall
        
        # Cost breakdown in basis points
        arrival_price = self.state.start_price
        total_value = self.state.executed_quantity * arrival_price
        
        market_impact_bps = (self.state.market_impact_cost / total_value) * 10000 if total_value > 0 else 0
        timing_cost_bps = (self.state.timing_cost / total_value) * 10000 if total_value > 0 else 0
        opportunity_cost_bps = (self.state.opportunity_cost / total_value) * 10000 if total_value > 0 else 0
        
        # Risk-adjusted performance
        predicted_cost = self._calculate_predicted_cost()
        risk_adjusted_performance = (predicted_cost - abs(is_total)) / max(predicted_cost, 0.0001)
        
        return {
            'fill_rate': fill_rate,
            'implementation_shortfall': is_total,
            'market_impact_bps': market_impact_bps,
            'timing_cost_bps': timing_cost_bps,
            'opportunity_cost_bps': opportunity_cost_bps,
            'total_cost_bps': market_impact_bps + timing_cost_bps + opportunity_cost_bps,
            'risk_adjusted_performance': risk_adjusted_performance,
            'weighted_price': self.state.weighted_price,
            'arrival_price': arrival_price,
            'executed_quantity': self.state.executed_quantity,
            'remaining_quantity': self.state.remaining_quantity
        }
    
    def _calculate_optimal_schedule(self, market_data: 'MarketData') -> None:
        """Calculate optimal execution schedule using Almgren-Chriss model"""
        
        # Parameters
        T = self.params.duration_minutes / (60 * 24)  # Convert to fraction of day
        X = self.params.total_quantity
        sigma = self.risk_model.price_volatility
        lambda_param = self.params.risk_aversion
        eta = self.impact_model.temporary_impact_coeff
        gamma = self.impact_model.permanent_impact_coeff
        
        # Number of time intervals
        N = max(1, self.params.duration_minutes // 5)  # 5-minute intervals
        tau = T / N
        
        # Calculate optimal schedule parameters
        kappa_hat = lambda_param * sigma * sigma / eta
        kappa = np.sqrt(kappa_hat * tau)
        
        # Optimal trajectory (continuous approximation)
        times = np.linspace(0, T, N + 1)
        
        if kappa > 1e-8:  # Avoid division by zero
            trajectory = []
            for t in times:
                remaining_time = T - t
                if remaining_time > 1e-8:
                    x_t = X * np.sinh(kappa * remaining_time) / np.sinh(kappa * T)
                else:
                    x_t = 0
                trajectory.append(x_t)
        else:
            # Linear trajectory when kappa is very small
            trajectory = [X * (T - t) / T for t in times]
        
        # Convert trajectory to trading schedule (differences)
        schedule = []
        for i in range(N):
            trade_size = trajectory[i] - trajectory[i + 1]
            schedule.append(max(trade_size, 0))
        
        # Normalize to ensure total equals target quantity
        total_scheduled = sum(schedule)
        if total_scheduled > 0:
            schedule = [size * X / total_scheduled for size in schedule]
        
        self.optimal_schedule = schedule
        self.execution_times = [i * 5 for i in range(N)]  # 5-minute intervals
        
        print(f"Optimal schedule: {len(schedule)} intervals")
        print(f"First 5 sizes: {[f'{s:.0f}' for s in schedule[:5]]}")
    
    def _get_current_time_index(self) -> int:
        """Get current time index in the execution schedule"""
        
        if not self.execution_times:
            return 0
        
        elapsed = self.state.elapsed_minutes if self.state else 0
        
        # Find the appropriate time index
        for i, target_time in enumerate(self.execution_times):
            if elapsed < target_time:
                return max(0, i - 1)
        
        return len(self.execution_times) - 1
    
    def _adjust_size_for_market_conditions(
        self,
        base_size: float,
        market_data: 'MarketData'
    ) -> float:
        """Adjust order size based on current market conditions"""
        
        # Volatility adjustment
        current_vol = getattr(market_data, 'volatility', self.risk_model.price_volatility)
        expected_vol = self.risk_model.price_volatility
        
        vol_ratio = current_vol / expected_vol if expected_vol > 0 else 1.0
        
        # Reduce size in high volatility
        if vol_ratio > 1.5:
            vol_adjustment = 0.7
        elif vol_ratio > 1.2:
            vol_adjustment = 0.85
        elif vol_ratio < 0.8:
            vol_adjustment = 1.15
        else:
            vol_adjustment = 1.0
        
        # Liquidity adjustment based on spread
        spread = market_data.ask_price - market_data.bid_price
        mid_price = (market_data.ask_price + market_data.bid_price) / 2
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
        
        if spread_bps > 20:  # Wide spread
            liquidity_adjustment = 0.8
        elif spread_bps > 10:
            liquidity_adjustment = 0.9
        else:
            liquidity_adjustment = 1.0
        
        # Volume adjustment
        volume_adjustment = 1.0
        if hasattr(market_data, 'volume') and market_data.volume > 0:
            # Increase size when volume is high
            if len(self.market_data_history) > 10:
                avg_volume = np.mean([md.volume for md in list(self.market_data_history)[-10:]])
                volume_ratio = market_data.volume / avg_volume
                
                if volume_ratio > 1.5:
                    volume_adjustment = 1.2
                elif volume_ratio < 0.7:
                    volume_adjustment = 0.8
        
        # Apply all adjustments
        adjusted_size = base_size * vol_adjustment * liquidity_adjustment * volume_adjustment
        
        # Participation rate constraint
        if hasattr(market_data, 'volume') and market_data.volume > 0:
            max_participation_size = market_data.volume * self.params.participation_limit
            adjusted_size = min(adjusted_size, max_participation_size)
        
        return adjusted_size
    
    def _calculate_permanent_impact(self, quantity: float, market_data: 'MarketData') -> float:
        """Calculate permanent market impact"""
        
        # Simple square-root model
        volume = getattr(market_data, 'volume', 10000)
        if volume <= 0:
            volume = 10000
        
        participation_rate = quantity / volume
        impact_coeff = self.impact_model.permanent_impact_coeff
        
        # Nonlinear impact
        impact_ratio = impact_coeff * (participation_rate ** self.impact_model.nonlinear_exponent)
        
        return impact_ratio * market_data.last_price
    
    def _calculate_temporary_impact(self, quantity: float, market_data: 'MarketData') -> float:
        """Calculate temporary market impact"""
        
        volume = getattr(market_data, 'volume', 10000)
        if volume <= 0:
            volume = 10000
        
        participation_rate = quantity / volume
        impact_coeff = self.impact_model.temporary_impact_coeff
        
        impact_ratio = impact_coeff * (participation_rate ** self.impact_model.nonlinear_exponent)
        
        return impact_ratio * market_data.last_price
    
    def _update_price_predictions(self, market_data: 'MarketData') -> None:
        """Update price movement predictions"""
        
        if len(self.market_data_history) < 2:
            return
        
        # Simple momentum-based prediction
        recent_prices = [md.last_price for md in list(self.market_data_history)[-10:]]
        
        if len(recent_prices) >= 2:
            # Calculate short-term trend
            price_changes = np.diff(recent_prices)
            avg_change = np.mean(price_changes)
            
            # Exponentially weighted prediction
            alpha = self.params.price_prediction_alpha
            
            if self.price_predictions:
                last_prediction = self.price_predictions[-1]
                new_prediction = alpha * avg_change + (1 - alpha) * last_prediction
            else:
                new_prediction = avg_change
            
            self.price_predictions.append(new_prediction)
    
    def _get_price_drift_prediction(self, market_data: 'MarketData') -> float:
        """Get predicted price drift"""
        
        if not self.price_predictions:
            return 0.0
        
        # Use latest prediction
        return self.price_predictions[-1]
    
    def _get_prediction_weight(self) -> float:
        """Get weight for price predictions based on confidence"""
        
        if len(self.price_predictions) < 5:
            return 0.1  # Low confidence with few predictions
        
        # Calculate prediction stability
        recent_predictions = list(self.price_predictions)[-5:]
        prediction_std = np.std(recent_predictions)
        
        # Lower weight for unstable predictions
        if prediction_std > 0.001:  # High variance
            return 0.1
        elif prediction_std > 0.0005:
            return 0.3
        else:
            return 0.5  # Medium confidence
    
    def _should_recalculate_schedule(self, market_data: 'MarketData') -> bool:
        """Determine if schedule should be recalculated"""
        
        if len(self.market_data_history) < 10:
            return False
        
        # Check for significant volatility change
        recent_prices = [md.last_price for md in list(self.market_data_history)[-10:]]
        recent_vol = np.std(np.diff(recent_prices)) * np.sqrt(252 * 24 * 60)  # Annualized
        
        vol_change_ratio = recent_vol / self.risk_model.price_volatility
        
        # Recalculate if volatility changed significantly
        if vol_change_ratio > 1.5 or vol_change_ratio < 0.7:
            self.risk_model.price_volatility = recent_vol
            return True
        
        return False
    
    def _update_cost_analysis(self, market_data: 'MarketData') -> None:
        """Update real-time cost analysis"""
        
        if not self.state or self.state.executed_quantity == 0:
            return
        
        # Update opportunity cost for remaining quantity
        current_price = market_data.last_price
        arrival_price = self.state.start_price
        
        unrealized_pnl = (current_price - arrival_price) * self.state.remaining_quantity
        if self.side == 'sell':
            unrealized_pnl = -unrealized_pnl
        
        self.state.opportunity_cost = -unrealized_pnl  # Cost is negative PnL
    
    def _calculate_predicted_cost(self) -> float:
        """Calculate predicted implementation shortfall cost"""
        
        # Simplified Almgren-Chriss cost prediction
        X = self.params.total_quantity
        T = self.params.duration_minutes / (60 * 24)
        sigma = self.risk_model.price_volatility
        lambda_param = self.params.risk_aversion
        eta = self.impact_model.temporary_impact_coeff
        gamma = self.impact_model.permanent_impact_coeff
        
        # Predicted cost components
        market_impact_cost = gamma * X  # Permanent impact
        timing_risk_cost = 0.5 * lambda_param * sigma * sigma * T * X  # Timing risk
        
        total_predicted_cost = (market_impact_cost + timing_risk_cost) / (X * self.state.start_price)
        
        return total_predicted_cost


# Import MarketData from twap_algorithm for consistency
from .twap_algorithm import MarketData


