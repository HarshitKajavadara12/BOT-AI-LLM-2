"""
TWAP Algorithm Implementation
Time-Weighted Average Price execution algorithm
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from abc import ABC, abstractmethod
import warnings
from dataclasses import dataclass
import time
from datetime import datetime, timedelta


@dataclass
class TWAPParameters:
    """Parameters for TWAP algorithm"""
    total_quantity: float
    duration_minutes: int
    participation_rate: float = 0.1  # Maximum participation rate
    min_order_size: float = 100
    max_order_size: float = 10000
    price_tolerance: float = 0.001  # 10 bps
    urgency_factor: float = 1.0
    risk_aversion: float = 0.5


@dataclass
class MarketData:
    """Market data snapshot"""
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    volatility: float = 0.0


@dataclass
class OrderState:
    """Current order state"""
    remaining_quantity: float
    executed_quantity: float
    vwap: float
    total_cost: float
    start_time: datetime
    elapsed_minutes: float


class TWAPAlgorithm:
    """
    Time-Weighted Average Price Algorithm
    Executes large orders by breaking them into smaller chunks over time
    """
    
    def __init__(self, parameters: TWAPParameters):
        self.params = parameters
        self.order_state = None
        self.execution_history = []
        self.market_data_history = []
        
        # Algorithm state
        self.is_active = False
        self.next_order_time = None
        self.adaptive_participation = parameters.participation_rate
        
        # Performance tracking
        self.slippage_estimates = []
        self.timing_performance = []
    
    def initialize_order(
        self,
        side: str,  # 'buy' or 'sell'
        symbol: str,
        market_data: MarketData
    ) -> None:
        """
        Initialize TWAP order execution
        
        Args:
            side: Order side ('buy' or 'sell')
            symbol: Trading symbol
            market_data: Current market data
        """
        
        self.side = side
        self.symbol = symbol
        self.is_active = True
        
        # Initialize order state
        self.order_state = OrderState(
            remaining_quantity=self.params.total_quantity,
            executed_quantity=0.0,
            vwap=0.0,
            total_cost=0.0,
            start_time=market_data.timestamp,
            elapsed_minutes=0.0
        )
        
        # Calculate time intervals
        self.total_intervals = max(1, self.params.duration_minutes)
        self.interval_duration = self.params.duration_minutes / self.total_intervals
        
        # Initialize next order time
        self.next_order_time = market_data.timestamp + timedelta(
            minutes=self.interval_duration
        )
        
        print(f"TWAP initialized: {self.params.total_quantity} shares over {self.params.duration_minutes} minutes")
    
    def update_market_data(self, market_data: MarketData) -> None:
        """Update with new market data"""
        self.market_data_history.append(market_data)
        
        if self.order_state:
            # Update elapsed time
            elapsed = market_data.timestamp - self.order_state.start_time
            self.order_state.elapsed_minutes = elapsed.total_seconds() / 60.0
    
    def should_place_order(self, market_data: MarketData) -> bool:
        """
        Determine if an order should be placed
        
        Args:
            market_data: Current market data
        
        Returns:
            True if order should be placed
        """
        
        if not self.is_active or not self.order_state:
            return False
        
        # Check if it's time for next order
        if market_data.timestamp < self.next_order_time:
            return False
        
        # Check if there's remaining quantity
        if self.order_state.remaining_quantity <= 0:
            self.complete_order()
            return False
        
        # Check market conditions
        if not self._check_market_conditions(market_data):
            return False
        
        return True
    
    def calculate_order_size(self, market_data: MarketData) -> float:
        """
        Calculate the size of the next order
        
        Args:
            market_data: Current market data
        
        Returns:
            Order size
        """
        
        # Base size: remaining quantity divided by remaining time intervals
        remaining_time = self.params.duration_minutes - self.order_state.elapsed_minutes
        remaining_intervals = max(1, remaining_time / self.interval_duration)
        
        base_size = self.order_state.remaining_quantity / remaining_intervals
        
        # Adjust for participation rate
        market_volume = market_data.volume if market_data.volume > 0 else 1000
        max_participation_size = market_volume * self.adaptive_participation
        
        # Urgency adjustment
        urgency_multiplier = self._calculate_urgency_multiplier()
        adjusted_size = base_size * urgency_multiplier
        
        # Apply constraints
        order_size = min(
            adjusted_size,
            max_participation_size,
            self.params.max_order_size,
            self.order_state.remaining_quantity
        )
        
        order_size = max(order_size, self.params.min_order_size)
        
        # Don't exceed remaining quantity
        order_size = min(order_size, self.order_state.remaining_quantity)
        
        return order_size
    
    def calculate_limit_price(self, market_data: MarketData, order_size: float) -> float:
        """
        Calculate limit price for the order
        
        Args:
            market_data: Current market data
            order_size: Size of the order
        
        Returns:
            Limit price
        """
        
        # Reference price
        if self.side == 'buy':
            reference_price = market_data.ask_price
            # Adjust for market impact
            market_impact = self._estimate_market_impact(order_size, market_data)
            limit_price = reference_price + market_impact
        else:
            reference_price = market_data.bid_price
            market_impact = self._estimate_market_impact(order_size, market_data)
            limit_price = reference_price - market_impact
        
        # Apply price tolerance
        tolerance = reference_price * self.params.price_tolerance
        
        if self.side == 'buy':
            limit_price = min(limit_price, reference_price + tolerance)
        else:
            limit_price = max(limit_price, reference_price - tolerance)
        
        return limit_price
    
    def process_execution(
        self,
        executed_quantity: float,
        execution_price: float,
        execution_time: datetime
    ) -> None:
        """
        Process an execution
        
        Args:
            executed_quantity: Quantity executed
            execution_price: Execution price
            execution_time: Execution timestamp
        """
        
        if not self.order_state:
            return
        
        # Update order state
        self.order_state.executed_quantity += executed_quantity
        self.order_state.remaining_quantity -= executed_quantity
        
        # Update VWAP
        new_cost = executed_quantity * execution_price
        self.order_state.total_cost += new_cost
        
        if self.order_state.executed_quantity > 0:
            self.order_state.vwap = self.order_state.total_cost / self.order_state.executed_quantity
        
        # Record execution
        execution_record = {
            'timestamp': execution_time,
            'quantity': executed_quantity,
            'price': execution_price,
            'side': self.side,
            'cumulative_quantity': self.order_state.executed_quantity,
            'vwap': self.order_state.vwap
        }
        
        self.execution_history.append(execution_record)
        
        # Update next order time
        if self.order_state.remaining_quantity > 0:
            self.next_order_time = execution_time + timedelta(
                minutes=self.interval_duration
            )
        else:
            self.complete_order()
        
        print(f"Execution: {executed_quantity} @ {execution_price:.4f}, "
              f"Remaining: {self.order_state.remaining_quantity}, "
              f"VWAP: {self.order_state.vwap:.4f}")
    
    def complete_order(self) -> None:
        """Complete the TWAP order"""
        self.is_active = False
        
        if self.order_state and self.order_state.executed_quantity > 0:
            print(f"TWAP completed: {self.order_state.executed_quantity} shares "
                  f"@ VWAP {self.order_state.vwap:.4f}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary of performance metrics
        """
        
        if not self.order_state or self.order_state.executed_quantity == 0:
            return {}
        
        # Calculate metrics
        fill_rate = self.order_state.executed_quantity / self.params.total_quantity
        
        # Implementation shortfall (simplified)
        if self.market_data_history:
            arrival_price = self.market_data_history[0].last_price
            implementation_shortfall = (self.order_state.vwap - arrival_price) / arrival_price
            
            if self.side == 'sell':
                implementation_shortfall = -implementation_shortfall
        else:
            implementation_shortfall = 0.0
        
        # Timing performance
        target_duration = self.params.duration_minutes
        actual_duration = self.order_state.elapsed_minutes
        timing_ratio = actual_duration / target_duration if target_duration > 0 else 1.0
        
        return {
            'fill_rate': fill_rate,
            'vwap': self.order_state.vwap,
            'implementation_shortfall': implementation_shortfall,
            'timing_ratio': timing_ratio,
            'total_executed': self.order_state.executed_quantity,
            'remaining_quantity': self.order_state.remaining_quantity
        }
    
    def _check_market_conditions(self, market_data: MarketData) -> bool:
        """Check if market conditions are suitable for trading"""
        
        # Check spread
        spread = market_data.ask_price - market_data.bid_price
        mid_price = (market_data.ask_price + market_data.bid_price) / 2
        spread_bps = (spread / mid_price) * 10000
        
        # Don't trade if spread is too wide
        if spread_bps > 50:  # 5 bps
            return False
        
        # Check volatility if available
        if market_data.volatility > 0 and market_data.volatility > 0.05:  # 5% volatility
            # Reduce participation in high volatility
            self.adaptive_participation = min(
                self.params.participation_rate * 0.5,
                self.adaptive_participation
            )
        
        return True
    
    def _calculate_urgency_multiplier(self) -> float:
        """Calculate urgency multiplier based on progress"""
        
        # Urgency increases as we fall behind schedule
        progress_ratio = self.order_state.executed_quantity / self.params.total_quantity
        time_ratio = self.order_state.elapsed_minutes / self.params.duration_minutes
        
        if time_ratio > progress_ratio and time_ratio > 0.5:
            # Behind schedule
            urgency = 1.0 + (time_ratio - progress_ratio) * self.params.urgency_factor
        else:
            urgency = 1.0
        
        return min(urgency, 2.0)  # Cap at 2x
    
    def _estimate_market_impact(self, order_size: float, market_data: MarketData) -> float:
        """
        Estimate market impact of order
        
        Args:
            order_size: Size of the order
            market_data: Current market data
        
        Returns:
            Estimated market impact in price units
        """
        
        # Simple market impact model
        # Impact = alpha * (order_size / avg_volume)^beta * volatility
        
        avg_volume = market_data.volume if market_data.volume > 0 else 10000
        volume_ratio = order_size / avg_volume
        
        # Market impact parameters (simplified)
        alpha = 0.1
        beta = 0.6
        
        # Base impact
        base_impact = alpha * (volume_ratio ** beta)
        
        # Adjust for volatility
        volatility = market_data.volatility if market_data.volatility > 0 else 0.01
        impact = base_impact * volatility * market_data.last_price
        
        return impact


class AdaptiveTWAP(TWAPAlgorithm):
    """
    Adaptive TWAP that adjusts to market conditions
    """
    
    def __init__(self, parameters: TWAPParameters):
        super().__init__(parameters)
        self.volume_profile = None
        self.volatility_adjustment = 1.0
        self.liquidity_adjustment = 1.0
    
    def set_volume_profile(self, volume_profile: List[float]) -> None:
        """
        Set intraday volume profile for adaptive scheduling
        
        Args:
            volume_profile: List of volume percentages by time interval
        """
        self.volume_profile = np.array(volume_profile)
        self.volume_profile = self.volume_profile / np.sum(self.volume_profile)
    
    def calculate_order_size(self, market_data: MarketData) -> float:
        """
        Calculate order size with adaptive adjustments
        
        Args:
            market_data: Current market data
        
        Returns:
            Adaptive order size
        """
        
        base_size = super().calculate_order_size(market_data)
        
        # Volume profile adjustment
        if self.volume_profile is not None:
            time_progress = self.order_state.elapsed_minutes / self.params.duration_minutes
            profile_index = min(
                int(time_progress * len(self.volume_profile)),
                len(self.volume_profile) - 1
            )
            volume_multiplier = self.volume_profile[profile_index] * len(self.volume_profile)
            base_size *= volume_multiplier
        
        # Volatility adjustment
        if market_data.volatility > 0.02:  # High volatility
            self.volatility_adjustment = 0.8  # Reduce size
        elif market_data.volatility < 0.01:  # Low volatility
            self.volatility_adjustment = 1.2  # Increase size
        else:
            self.volatility_adjustment = 1.0
        
        base_size *= self.volatility_adjustment
        
        # Liquidity adjustment
        spread = market_data.ask_price - market_data.bid_price
        mid_price = (market_data.ask_price + market_data.bid_price) / 2
        spread_bps = (spread / mid_price) * 10000
        
        if spread_bps > 10:  # Wide spread
            self.liquidity_adjustment = 0.7
        else:
            self.liquidity_adjustment = 1.0
        
        base_size *= self.liquidity_adjustment
        
        # Apply constraints
        return max(
            min(base_size, self.params.max_order_size, self.order_state.remaining_quantity),
            self.params.min_order_size
        )


class TWAPScheduler:
    """
    Scheduler for TWAP algorithm execution
    """
    
    def __init__(self):
        self.active_algorithms = {}
        self.completed_algorithms = []
    
    def add_twap_order(
        self,
        order_id: str,
        parameters: TWAPParameters,
        side: str,
        symbol: str,
        adaptive: bool = False
    ) -> None:
        """
        Add TWAP order to scheduler
        
        Args:
            order_id: Unique order identifier
            parameters: TWAP parameters
            side: Order side
            symbol: Trading symbol
            adaptive: Whether to use adaptive TWAP
        """
        
        if adaptive:
            algorithm = AdaptiveTWAP(parameters)
        else:
            algorithm = TWAPAlgorithm(parameters)
        
        self.active_algorithms[order_id] = {
            'algorithm': algorithm,
            'side': side,
            'symbol': symbol,
            'created_at': datetime.now()
        }
    
    def process_market_data(self, market_data: MarketData, symbol: str) -> List[Dict]:
        """
        Process market data for all active algorithms
        
        Args:
            market_data: Market data update
            symbol: Symbol for the market data
        
        Returns:
            List of order recommendations
        """
        
        recommendations = []
        
        for order_id, order_info in list(self.active_algorithms.items()):
            if order_info['symbol'] != symbol:
                continue
            
            algorithm = order_info['algorithm']
            
            # Initialize if needed
            if not algorithm.is_active and algorithm.order_state is None:
                algorithm.initialize_order(
                    order_info['side'],
                    symbol,
                    market_data
                )
            
            # Update market data
            algorithm.update_market_data(market_data)
            
            # Check if order should be placed
            if algorithm.should_place_order(market_data):
                order_size = algorithm.calculate_order_size(market_data)
                limit_price = algorithm.calculate_limit_price(market_data, order_size)
                
                recommendation = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': order_info['side'],
                    'quantity': order_size,
                    'limit_price': limit_price,
                    'order_type': 'limit',
                    'algorithm': 'TWAP'
                }
                
                recommendations.append(recommendation)
            
            # Check if algorithm is completed
            if not algorithm.is_active:
                self.completed_algorithms.append(self.active_algorithms.pop(order_id))
        
        return recommendations
    
    def process_execution(
        self,
        order_id: str,
        executed_quantity: float,
        execution_price: float,
        execution_time: datetime
    ) -> None:
        """
        Process execution for specific order
        
        Args:
            order_id: Order identifier
            executed_quantity: Executed quantity
            execution_price: Execution price
            execution_time: Execution time
        """
        
        if order_id in self.active_algorithms:
            algorithm = self.active_algorithms[order_id]['algorithm']
            algorithm.process_execution(executed_quantity, execution_price, execution_time)
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of specific order"""
        
        if order_id in self.active_algorithms:
            algorithm = self.active_algorithms[order_id]['algorithm']
            return algorithm.get_performance_metrics()
        
        return None
    
    def get_all_statuses(self) -> Dict[str, Dict]:
        """Get status of all active orders"""
        
        statuses = {}
        for order_id, order_info in self.active_algorithms.items():
            algorithm = order_info['algorithm']
            statuses[order_id] = algorithm.get_performance_metrics()
        
        return statuses


