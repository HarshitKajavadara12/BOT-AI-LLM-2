"""
VWAP Algorithm Implementation
Volume-Weighted Average Price execution algorithm
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any
from abc import ABC, abstractmethod
import warnings
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
from collections import deque


@dataclass
class VWAPParameters:
    """Parameters for VWAP algorithm"""
    total_quantity: float
    duration_minutes: int
    max_participation_rate: float = 0.25
    min_participation_rate: float = 0.05
    target_participation_rate: float = 0.15
    min_order_size: float = 100
    max_order_size: float = 20000
    price_tolerance: float = 0.0015  # 15 bps
    volume_imbalance_threshold: float = 0.3
    aggressive_fill_threshold: float = 0.8  # When to become more aggressive


@dataclass
class VolumeProfile:
    """Historical volume profile"""
    time_buckets: List[int]  # Minutes from start
    volume_percentages: List[float]  # Volume percentage in each bucket
    bucket_duration: int = 5  # Minutes per bucket


@dataclass
class VWAPState:
    """VWAP algorithm state"""
    remaining_quantity: float
    executed_quantity: float
    volume_weighted_price: float
    total_cost: float
    start_time: datetime
    elapsed_minutes: float
    current_vwap_benchmark: float = 0.0
    volume_participation: float = 0.0


class VWAPAlgorithm:
    """
    Volume-Weighted Average Price Algorithm
    Tracks historical VWAP and executes orders to minimize tracking error
    """
    
    def __init__(self, parameters: VWAPParameters):
        self.params = parameters
        self.state = None
        self.volume_profile = None
        
        # Market data tracking
        self.market_data_history = deque(maxlen=1000)
        self.volume_history = deque(maxlen=100)
        self.trade_history = []
        
        # VWAP tracking
        self.vwap_calculator = VWAPCalculator()
        self.current_market_vwap = 0.0
        
        # Algorithm state
        self.is_active = False
        self.last_order_time = None
        self.adaptive_participation = parameters.target_participation_rate
        
        # Performance metrics
        self.slippage_history = []
        self.participation_history = []
        
    def set_volume_profile(self, volume_profile: VolumeProfile) -> None:
        """Set historical volume profile for better scheduling"""
        self.volume_profile = volume_profile
    
    def initialize_order(
        self,
        side: str,
        symbol: str,
        market_data: 'MarketData'
    ) -> None:
        """
        Initialize VWAP order execution
        
        Args:
            side: Order side ('buy' or 'sell')
            symbol: Trading symbol
            market_data: Current market data
        """
        
        self.side = side
        self.symbol = symbol
        self.is_active = True
        
        # Initialize state
        self.state = VWAPState(
            remaining_quantity=self.params.total_quantity,
            executed_quantity=0.0,
            volume_weighted_price=0.0,
            total_cost=0.0,
            start_time=market_data.timestamp,
            elapsed_minutes=0.0,
            current_vwap_benchmark=market_data.last_price
        )
        
        # Initialize VWAP calculator
        self.vwap_calculator.reset(market_data.timestamp)
        self.last_order_time = market_data.timestamp
        
        print(f"VWAP initialized: {self.params.total_quantity} shares, "
              f"target participation: {self.params.target_participation_rate:.1%}")
    
    def update_market_data(self, market_data: 'MarketData') -> None:
        """
        Update with new market data and calculate current VWAP
        
        Args:
            market_data: New market data
        """
        
        self.market_data_history.append(market_data)
        
        if self.state:
            # Update elapsed time
            elapsed = market_data.timestamp - self.state.start_time
            self.state.elapsed_minutes = elapsed.total_seconds() / 60.0
        
        # Update VWAP calculator
        self.vwap_calculator.update(
            market_data.last_price,
            market_data.volume,
            market_data.timestamp
        )
        
        self.current_market_vwap = self.vwap_calculator.get_vwap()
        
        # Track volume
        if market_data.volume > 0:
            self.volume_history.append(market_data.volume)
    
    def should_place_order(self, market_data: 'MarketData') -> bool:
        """
        Determine if order should be placed based on VWAP strategy
        
        Args:
            market_data: Current market data
        
        Returns:
            True if order should be placed
        """
        
        if not self.is_active or not self.state:
            return False
        
        # Check remaining quantity
        if self.state.remaining_quantity <= 0:
            self.complete_order()
            return False
        
        # Time-based check - don't place orders too frequently
        if self.last_order_time:
            time_since_last = market_data.timestamp - self.last_order_time
            if time_since_last.total_seconds() < 10:  # 10 seconds minimum
                return False
        
        # Volume-based check
        if not self._has_sufficient_volume(market_data):
            return False
        
        # Market condition check
        if not self._check_market_conditions(market_data):
            return False
        
        # VWAP deviation check
        if self._should_be_more_aggressive(market_data):
            return True
        
        # Volume participation check
        target_volume = self._calculate_target_volume(market_data)
        
        return target_volume >= self.params.min_order_size
    
    def calculate_order_size(self, market_data: 'MarketData') -> float:
        """
        Calculate optimal order size based on VWAP strategy
        
        Args:
            market_data: Current market data
        
        Returns:
            Order size
        """
        
        # Base calculation using volume participation
        target_volume = self._calculate_target_volume(market_data)
        
        # Adjust for VWAP tracking error
        vwap_adjustment = self._calculate_vwap_adjustment(market_data)
        adjusted_size = target_volume * vwap_adjustment
        
        # Progress-based urgency adjustment
        urgency_multiplier = self._calculate_urgency_multiplier()
        final_size = adjusted_size * urgency_multiplier
        
        # Apply constraints
        final_size = max(final_size, self.params.min_order_size)
        final_size = min(final_size, self.params.max_order_size)
        final_size = min(final_size, self.state.remaining_quantity)
        
        return final_size
    
    def calculate_limit_price(self, market_data: 'MarketData', order_size: float) -> float:
        """
        Calculate limit price based on VWAP and market conditions
        
        Args:
            market_data: Current market data
            order_size: Order size
        
        Returns:
            Limit price
        """
        
        # Start with current VWAP as reference
        vwap_reference = self.current_market_vwap
        
        if self.side == 'buy':
            # For buy orders, consider ask price and market impact
            base_price = market_data.ask_price
            
            # If current price is below VWAP, we can be more aggressive
            if market_data.last_price < vwap_reference:
                price_adjustment = (vwap_reference - market_data.last_price) * 0.5
                limit_price = base_price + price_adjustment
            else:
                # Price above VWAP, be more conservative
                limit_price = min(base_price, vwap_reference + self._get_price_tolerance())
        
        else:  # sell orders
            base_price = market_data.bid_price
            
            if market_data.last_price > vwap_reference:
                price_adjustment = (market_data.last_price - vwap_reference) * 0.5
                limit_price = base_price - price_adjustment
            else:
                limit_price = max(base_price, vwap_reference - self._get_price_tolerance())
        
        # Market impact adjustment
        market_impact = self._estimate_market_impact(order_size, market_data)
        
        if self.side == 'buy':
            limit_price += market_impact
        else:
            limit_price -= market_impact
        
        return limit_price
    
    def process_execution(
        self,
        executed_quantity: float,
        execution_price: float,
        execution_time: datetime
    ) -> None:
        """
        Process execution and update VWAP tracking
        
        Args:
            executed_quantity: Executed quantity
            execution_price: Execution price
            execution_time: Execution timestamp
        """
        
        if not self.state:
            return
        
        # Update state
        self.state.executed_quantity += executed_quantity
        self.state.remaining_quantity -= executed_quantity
        
        # Update volume-weighted price
        new_cost = executed_quantity * execution_price
        self.state.total_cost += new_cost
        
        if self.state.executed_quantity > 0:
            self.state.volume_weighted_price = (
                self.state.total_cost / self.state.executed_quantity
            )
        
        # Calculate slippage vs VWAP
        vwap_at_execution = self.current_market_vwap
        if vwap_at_execution > 0:
            slippage = (execution_price - vwap_at_execution) / vwap_at_execution
            if self.side == 'sell':
                slippage = -slippage
            self.slippage_history.append(slippage)
        
        # Record trade
        trade_record = {
            'timestamp': execution_time,
            'quantity': executed_quantity,
            'price': execution_price,
            'side': self.side,
            'cumulative_quantity': self.state.executed_quantity,
            'algo_vwap': self.state.volume_weighted_price,
            'market_vwap': vwap_at_execution,
            'slippage_vs_vwap': slippage if 'slippage' in locals() else 0.0
        }
        
        self.trade_history.append(trade_record)
        self.last_order_time = execution_time
        
        # Check completion
        if self.state.remaining_quantity <= 0:
            self.complete_order()
        
        print(f"VWAP Execution: {executed_quantity} @ {execution_price:.4f}, "
              f"Algo VWAP: {self.state.volume_weighted_price:.4f}, "
              f"Market VWAP: {vwap_at_execution:.4f}")
    
    def complete_order(self) -> None:
        """Complete VWAP order and calculate final metrics"""
        self.is_active = False
        
        if self.state and self.state.executed_quantity > 0:
            final_metrics = self.get_performance_metrics()
            print(f"VWAP completed: {self.state.executed_quantity} shares")
            print(f"Algorithm VWAP: {self.state.volume_weighted_price:.4f}")
            print(f"Market VWAP: {self.current_market_vwap:.4f}")
            print(f"VWAP tracking error: {final_metrics.get('vwap_tracking_error', 0):.4f}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        if not self.state or self.state.executed_quantity == 0:
            return {}
        
        # Basic metrics
        fill_rate = self.state.executed_quantity / self.params.total_quantity
        
        # VWAP tracking error
        vwap_tracking_error = 0.0
        if self.current_market_vwap > 0:
            vwap_tracking_error = (
                (self.state.volume_weighted_price - self.current_market_vwap) /
                self.current_market_vwap
            )
        
        # Average slippage
        avg_slippage = np.mean(self.slippage_history) if self.slippage_history else 0.0
        
        # Participation rate
        total_market_volume = sum(self.volume_history) if self.volume_history else 1
        participation_rate = self.state.executed_quantity / total_market_volume
        
        # Implementation shortfall vs VWAP
        implementation_shortfall = vwap_tracking_error
        
        return {
            'fill_rate': fill_rate,
            'algorithm_vwap': self.state.volume_weighted_price,
            'market_vwap': self.current_market_vwap,
            'vwap_tracking_error': vwap_tracking_error,
            'average_slippage': avg_slippage,
            'participation_rate': participation_rate,
            'implementation_shortfall': implementation_shortfall,
            'total_executed': self.state.executed_quantity,
            'remaining_quantity': self.state.remaining_quantity
        }
    
    def _calculate_target_volume(self, market_data: 'MarketData') -> float:
        """Calculate target volume based on market volume and profile"""
        
        current_volume = market_data.volume if market_data.volume > 0 else 1000
        
        # Base participation
        base_target = current_volume * self.adaptive_participation
        
        # Volume profile adjustment
        if self.volume_profile:
            profile_multiplier = self._get_volume_profile_multiplier()
            base_target *= profile_multiplier
        
        return base_target
    
    def _calculate_vwap_adjustment(self, market_data: 'MarketData') -> float:
        """Calculate adjustment factor based on VWAP deviation"""
        
        if self.current_market_vwap == 0:
            return 1.0
        
        current_price = market_data.last_price
        vwap_deviation = (current_price - self.current_market_vwap) / self.current_market_vwap
        
        # If we're buying and price is below VWAP, be more aggressive
        # If we're selling and price is above VWAP, be more aggressive
        if self.side == 'buy':
            if vwap_deviation < -0.001:  # Price 10 bps below VWAP
                return 1.5  # Increase size
            elif vwap_deviation > 0.001:  # Price above VWAP
                return 0.8  # Reduce size
        else:  # sell
            if vwap_deviation > 0.001:  # Price above VWAP
                return 1.5  # Increase size
            elif vwap_deviation < -0.001:  # Price below VWAP
                return 0.8  # Reduce size
        
        return 1.0
    
    def _calculate_urgency_multiplier(self) -> float:
        """Calculate urgency based on time progress vs execution progress"""
        
        time_progress = self.state.elapsed_minutes / self.params.duration_minutes
        execution_progress = self.state.executed_quantity / self.params.total_quantity
        
        # If we're behind schedule, increase urgency
        if time_progress > execution_progress and time_progress > 0.3:
            urgency = 1.0 + (time_progress - execution_progress) * 2.0
            return min(urgency, 2.5)  # Cap at 2.5x
        
        return 1.0
    
    def _should_be_more_aggressive(self, market_data: 'MarketData') -> bool:
        """Determine if we should be more aggressive based on conditions"""
        
        # Check if we're significantly behind VWAP target
        time_progress = self.state.elapsed_minutes / self.params.duration_minutes
        
        if time_progress > self.params.aggressive_fill_threshold:
            return True
        
        # Check VWAP deviation
        if self.current_market_vwap > 0:
            price_deviation = abs(
                (market_data.last_price - self.current_market_vwap) / self.current_market_vwap
            )
            
            if price_deviation > 0.005:  # 50 bps deviation
                return True
        
        return False
    
    def _has_sufficient_volume(self, market_data: 'MarketData') -> bool:
        """Check if there's sufficient volume to participate"""
        
        min_volume_threshold = self.params.min_order_size / self.params.max_participation_rate
        return market_data.volume >= min_volume_threshold
    
    def _check_market_conditions(self, market_data: 'MarketData') -> bool:
        """Check if market conditions are suitable"""
        
        # Check spread
        spread = market_data.ask_price - market_data.bid_price
        mid_price = (market_data.ask_price + market_data.bid_price) / 2
        spread_bps = (spread / mid_price) * 10000
        
        if spread_bps > 100:  # 1% spread
            return False
        
        # Check for volume imbalance
        if hasattr(market_data, 'bid_size') and hasattr(market_data, 'ask_size'):
            total_size = market_data.bid_size + market_data.ask_size
            if total_size > 0:
                imbalance = abs(market_data.bid_size - market_data.ask_size) / total_size
                if imbalance > self.params.volume_imbalance_threshold:
                    # Adjust participation in imbalanced markets
                    self.adaptive_participation *= 0.8
        
        return True
    
    def _get_volume_profile_multiplier(self) -> float:
        """Get volume profile multiplier for current time"""
        
        if not self.volume_profile:
            return 1.0
        
        elapsed_minutes = int(self.state.elapsed_minutes)
        
        # Find appropriate bucket
        for i, bucket_time in enumerate(self.volume_profile.time_buckets):
            if elapsed_minutes <= bucket_time:
                return self.volume_profile.volume_percentages[i] * len(self.volume_profile.time_buckets)
        
        # Default to last bucket
        return self.volume_profile.volume_percentages[-1] * len(self.volume_profile.time_buckets)
    
    def _get_price_tolerance(self) -> float:
        """Get current price tolerance"""
        
        # Increase tolerance as we approach deadline
        time_progress = self.state.elapsed_minutes / self.params.duration_minutes
        base_tolerance = self.params.price_tolerance
        
        if time_progress > 0.8:
            return base_tolerance * 2.0
        elif time_progress > 0.6:
            return base_tolerance * 1.5
        
        return base_tolerance
    
    def _estimate_market_impact(self, order_size: float, market_data: 'MarketData') -> float:
        """Estimate market impact of order"""
        
        # Simple square-root market impact model
        avg_volume = np.mean(self.volume_history) if self.volume_history else market_data.volume
        
        if avg_volume <= 0:
            return 0.0
        
        volume_ratio = order_size / avg_volume
        volatility = getattr(market_data, 'volatility', 0.01)
        
        # Market impact = alpha * sqrt(volume_ratio) * volatility * price
        alpha = 0.1
        impact_ratio = alpha * np.sqrt(volume_ratio) * volatility
        
        return impact_ratio * market_data.last_price


class VWAPCalculator:
    """
    Real-time VWAP calculation engine
    """
    
    def __init__(self, window_minutes: int = 390):  # Full trading day
        self.window_minutes = window_minutes
        self.reset()
    
    def reset(self, start_time: Optional[datetime] = None) -> None:
        """Reset VWAP calculation"""
        self.start_time = start_time or datetime.now()
        self.price_volume_sum = 0.0
        self.volume_sum = 0.0
        self.data_points = deque(maxlen=self.window_minutes * 10)  # 6-second intervals
    
    def update(self, price: float, volume: float, timestamp: datetime) -> None:
        """
        Update VWAP with new price and volume
        
        Args:
            price: Trade price
            volume: Trade volume
            timestamp: Trade timestamp
        """
        
        if volume > 0:
            pv = price * volume
            
            # Add new data point
            self.data_points.append({
                'timestamp': timestamp,
                'price_volume': pv,
                'volume': volume
            })
            
            # Remove old data points outside window
            cutoff_time = timestamp - timedelta(minutes=self.window_minutes)
            
            while (self.data_points and 
                   self.data_points[0]['timestamp'] < cutoff_time):
                old_point = self.data_points.popleft()
                self.price_volume_sum -= old_point['price_volume']
                self.volume_sum -= old_point['volume']
            
            # Add new data
            self.price_volume_sum += pv
            self.volume_sum += volume
    
    def get_vwap(self) -> float:
        """Get current VWAP"""
        if self.volume_sum > 0:
            return self.price_volume_sum / self.volume_sum
        return 0.0
    
    def get_volume(self) -> float:
        """Get total volume in window"""
        return self.volume_sum


# Import MarketData from twap_algorithm for consistency
from .twap_algorithm import MarketData

