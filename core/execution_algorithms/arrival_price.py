"""
Arrival Price Algorithm
Balances market impact and timing risk relative to arrival price
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import math


@dataclass
class ArrivalPriceParameters:
    """Arrival Price algorithm parameters"""
    total_quantity: float
    duration_minutes: int
    target_percentage: float = 0.0  # Target as percentage of ADTV
    urgency: float = 0.5  # Urgency factor (0 = patient, 1 = aggressive)
    risk_aversion: float = 0.3  # Risk aversion parameter
    max_participation_rate: float = 0.4
    min_participation_rate: float = 0.02
    min_order_size: float = 100
    max_order_size: float = 50000
    price_limit_tolerance: float = 0.003  # 30 bps from arrival price
    market_on_close_pct: float = 0.1  # Percentage to MOC if incomplete


@dataclass
class ArrivalPriceState:
    """Arrival Price algorithm state"""
    remaining_quantity: float
    executed_quantity: float
    arrival_price: float
    current_vwap: float
    total_cost: float
    start_time: datetime
    elapsed_minutes: float = 0.0
    arrival_price_performance: float = 0.0  # Performance vs arrival price
    market_impact_estimate: float = 0.0
    aggressive_fills: int = 0  # Count of aggressive fills


class ArrivalPriceAlgorithm:
    """
    Arrival Price Algorithm
    Aims to minimize cost relative to the price at order arrival time
    Balances market impact against timing risk
    """
    
    def __init__(self, parameters: ArrivalPriceParameters):
        self.params = parameters
        self.state = None
        self.is_active = False
        self.side = None
        self.symbol = None
        
        # Market tracking
        self.market_data_history = deque(maxlen=300)
        self.execution_history = []
        
        # Dynamic parameters
        self.current_urgency = parameters.urgency
        self.current_participation_rate = (parameters.max_participation_rate + 
                                         parameters.min_participation_rate) / 2
        
        # Market analysis
        self.volatility_estimate = 0.02
        self.trend_estimate = 0.0
        self.liquidity_estimate = 1.0
        
        # Performance tracking
        self.impact_measurements = []
        self.timing_costs = []
        
    def initialize_order(
        self,
        side: str,
        symbol: str,
        market_data: 'MarketData'
    ) -> None:
        """
        Initialize Arrival Price order
        
        Args:
            side: Order side ('buy' or 'sell')
            symbol: Trading symbol
            market_data: Market data at arrival
        """
        
        self.side = side
        self.symbol = symbol
        self.is_active = True
        
        # Initialize state with arrival price
        self.state = ArrivalPriceState(
            remaining_quantity=self.params.total_quantity,
            executed_quantity=0.0,
            arrival_price=market_data.last_price,
            current_vwap=0.0,
            total_cost=0.0,
            start_time=market_data.timestamp,
            elapsed_minutes=0.0
        )
        
        # Initialize market estimates
        self._initialize_market_estimates(market_data)
        
        print(f"Arrival Price Algorithm initialized:")
        print(f"  Quantity: {self.params.total_quantity}")
        print(f"  Arrival Price: {self.state.arrival_price:.4f}")
        print(f"  Duration: {self.params.duration_minutes} minutes")
        print(f"  Urgency: {self.params.urgency}")
    
    def update_market_data(self, market_data: 'MarketData') -> None:
        """Update with new market data and adjust strategy"""
        
        self.market_data_history.append(market_data)
        
        if self.state:
            # Update elapsed time
            elapsed = market_data.timestamp - self.state.start_time
            self.state.elapsed_minutes = elapsed.total_seconds() / 60.0
            
            # Update current VWAP
            self._update_vwap(market_data)
            
            # Update market estimates
            self._update_market_estimates(market_data)
            
            # Adjust strategy parameters
            self._adjust_strategy_parameters(market_data)
    
    def should_place_order(self, market_data: 'MarketData') -> bool:
        """Determine if order should be placed"""
        
        if not self.is_active or not self.state:
            return False
        
        if self.state.remaining_quantity <= 0:
            self.complete_order()
            return False
        
        # Time-based urgency
        time_progress = self.state.elapsed_minutes / self.params.duration_minutes
        
        # Always place order if near end of time window
        if time_progress > 0.9:
            return True
        
        # Check market conditions
        if not self._check_market_conditions(market_data):
            return False
        
        # Price level check
        if not self._check_price_level(market_data):
            return False
        
        # Participation rate check
        target_size = self._calculate_base_order_size(market_data)
        
        return target_size >= self.params.min_order_size
    
    def calculate_order_size(self, market_data: 'MarketData') -> float:
        """Calculate order size based on arrival price strategy"""
        
        # Base size calculation
        base_size = self._calculate_base_order_size(market_data)
        
        # Urgency adjustment
        urgency_multiplier = self._calculate_urgency_multiplier(market_data)
        adjusted_size = base_size * urgency_multiplier
        
        # Market condition adjustment
        market_multiplier = self._calculate_market_multiplier(market_data)
        final_size = adjusted_size * market_multiplier
        
        # Apply constraints
        final_size = max(final_size, self.params.min_order_size)
        final_size = min(final_size, self.params.max_order_size)
        final_size = min(final_size, self.state.remaining_quantity)
        
        return final_size
    
    def calculate_limit_price(self, market_data: 'MarketData', order_size: float) -> float:
        """Calculate limit price relative to arrival price"""
        
        arrival_price = self.state.arrival_price
        current_price = market_data.last_price
        
        # Base limit calculation
        if self.side == 'buy':
            # For buy orders, start from ask
            base_limit = market_data.ask_price
            
            # Adjust based on arrival price performance
            price_drift = current_price - arrival_price
            
            # If price has moved up significantly, be more aggressive
            if price_drift > arrival_price * 0.002:  # 20 bps
                aggressiveness = min(self.current_urgency * 1.5, 1.0)
                price_adjustment = price_drift * aggressiveness * 0.5
                base_limit += price_adjustment
            
        else:  # sell orders
            base_limit = market_data.bid_price
            
            price_drift = current_price - arrival_price
            
            # If price has moved down, be more aggressive
            if price_drift < -arrival_price * 0.002:
                aggressiveness = min(self.current_urgency * 1.5, 1.0)
                price_adjustment = abs(price_drift) * aggressiveness * 0.5
                base_limit -= price_adjustment
        
        # Market impact adjustment
        estimated_impact = self._estimate_market_impact(order_size, market_data)
        
        if self.side == 'buy':
            limit_price = base_limit + estimated_impact * 0.5
        else:
            limit_price = base_limit - estimated_impact * 0.5
        
        # Ensure limit is within tolerance of arrival price
        max_deviation = arrival_price * self.params.price_limit_tolerance
        
        if self.side == 'buy':
            max_limit = arrival_price + max_deviation
            limit_price = min(limit_price, max_limit)
        else:
            min_limit = arrival_price - max_deviation
            limit_price = max(limit_price, min_limit)
        
        return limit_price
    
    def process_execution(
        self,
        executed_quantity: float,
        execution_price: float,
        execution_time: datetime
    ) -> None:
        """Process execution and update arrival price tracking"""
        
        if not self.state:
            return
        
        # Update basic state
        self.state.executed_quantity += executed_quantity
        self.state.remaining_quantity -= executed_quantity
        
        # Update cost tracking
        execution_cost = executed_quantity * execution_price
        self.state.total_cost += execution_cost
        
        # Update VWAP
        if self.state.executed_quantity > 0:
            self.state.current_vwap = self.state.total_cost / self.state.executed_quantity
        
        # Calculate arrival price performance
        arrival_cost = self.state.executed_quantity * self.state.arrival_price
        cost_difference = self.state.total_cost - arrival_cost
        
        self.state.arrival_price_performance = cost_difference / arrival_cost if arrival_cost > 0 else 0
        
        # Adjust sign for sell orders
        if self.side == 'sell':
            self.state.arrival_price_performance = -self.state.arrival_price_performance
        
        # Estimate market impact
        mid_price = (execution_price + self.state.arrival_price) / 2
        impact_estimate = abs(execution_price - mid_price)
        self.impact_measurements.append(impact_estimate)
        
        # Update average impact estimate
        if self.impact_measurements:
            self.state.market_impact_estimate = np.mean(self.impact_measurements[-10:])
        
        # Check for aggressive fill
        spread = 0.01  # Simplified spread estimate
        if abs(execution_price - self.state.arrival_price) > spread:
            self.state.aggressive_fills += 1
        
        # Record execution
        execution_record = {
            'timestamp': execution_time,
            'quantity': executed_quantity,
            'price': execution_price,
            'side': self.side,
            'cumulative_quantity': self.state.executed_quantity,
            'current_vwap': self.state.current_vwap,
            'arrival_price': self.state.arrival_price,
            'arrival_performance': self.state.arrival_price_performance,
            'estimated_impact': impact_estimate
        }
        
        self.execution_history.append(execution_record)
        
        # Check completion
        if self.state.remaining_quantity <= 0:
            self.complete_order()
        
        print(f"AP Execution: {executed_quantity} @ {execution_price:.4f}")
        print(f"  Arrival Performance: {self.state.arrival_price_performance:.4f}")
        print(f"  VWAP: {self.state.current_vwap:.4f} vs Arrival: {self.state.arrival_price:.4f}")
    
    def complete_order(self) -> None:
        """Complete arrival price order"""
        
        self.is_active = False
        
        if self.state and self.state.executed_quantity > 0:
            metrics = self.get_performance_metrics()
            
            print(f"Arrival Price Algorithm completed:")
            print(f"  Fill Rate: {metrics['fill_rate']:.2%}")
            print(f"  Arrival Price Performance: {metrics['arrival_price_performance']:.4f}")
            print(f"  Average Impact: {metrics['average_market_impact']:.4f}")
            
            # Handle remaining quantity
            if self.state.remaining_quantity > 0:
                remaining_pct = self.state.remaining_quantity / self.params.total_quantity
                print(f"  Remaining quantity: {self.state.remaining_quantity} ({remaining_pct:.1%})")
                
                if remaining_pct >= self.params.market_on_close_pct:
                    print("  -> Recommend Market-on-Close for remainder")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive arrival price performance metrics"""
        
        if not self.state or self.state.executed_quantity == 0:
            return {}
        
        # Basic metrics
        fill_rate = self.state.executed_quantity / self.params.total_quantity
        
        # Arrival price performance (already calculated)
        arrival_performance = self.state.arrival_price_performance
        
        # Convert to basis points
        arrival_performance_bps = arrival_performance * 10000
        
        # Market impact analysis
        avg_impact = np.mean(self.impact_measurements) if self.impact_measurements else 0.0
        avg_impact_bps = (avg_impact / self.state.arrival_price) * 10000 if self.state.arrival_price > 0 else 0
        
        # Execution efficiency
        aggressive_fill_rate = self.state.aggressive_fills / len(self.execution_history) if self.execution_history else 0
        
        # Price improvement (negative arrival performance is good for buys)
        if self.side == 'buy':
            price_improvement = -arrival_performance_bps  # Negative cost is improvement
        else:
            price_improvement = arrival_performance_bps   # Positive revenue is improvement
        
        # Risk metrics
        execution_prices = [exec['price'] for exec in self.execution_history]
        price_volatility = np.std(execution_prices) if len(execution_prices) > 1 else 0.0
        
        # Timing analysis
        if len(self.execution_history) > 1:
            execution_times = [exec['timestamp'] for exec in self.execution_history]
            time_spans = [(execution_times[i] - execution_times[i-1]).total_seconds() / 60 
                         for i in range(1, len(execution_times))]
            avg_time_between_executions = np.mean(time_spans)
        else:
            avg_time_between_executions = 0.0
        
        return {
            'fill_rate': fill_rate,
            'arrival_price_performance': arrival_performance,
            'arrival_performance_bps': arrival_performance_bps,
            'price_improvement_bps': price_improvement,
            'average_market_impact': avg_impact,
            'market_impact_bps': avg_impact_bps,
            'aggressive_fill_rate': aggressive_fill_rate,
            'execution_price_volatility': price_volatility,
            'avg_time_between_executions': avg_time_between_executions,
            'current_vwap': self.state.current_vwap,
            'arrival_price': self.state.arrival_price,
            'executed_quantity': self.state.executed_quantity,
            'remaining_quantity': self.state.remaining_quantity
        }
    
    def _initialize_market_estimates(self, market_data: 'MarketData') -> None:
        """Initialize market condition estimates"""
        
        # Initial volatility estimate
        self.volatility_estimate = getattr(market_data, 'volatility', 0.02)
        
        # Initial liquidity estimate based on spread
        spread = market_data.ask_price - market_data.bid_price
        mid_price = (market_data.ask_price + market_data.bid_price) / 2
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
        
        # Lower spread indicates higher liquidity
        self.liquidity_estimate = max(0.1, 1.0 - spread_bps / 100.0)
        
        print(f"Initial estimates: Vol={self.volatility_estimate:.3f}, Liquidity={self.liquidity_estimate:.2f}")
    
    def _update_vwap(self, market_data: 'MarketData') -> None:
        """Update volume-weighted average price tracking"""
        
        # This is a simplified VWAP - in practice, would use more sophisticated calculation
        if hasattr(market_data, 'volume') and market_data.volume > 0:
            # Update with market VWAP (simplified)
            pass
    
    def _update_market_estimates(self, market_data: 'MarketData') -> None:
        """Update market condition estimates based on recent data"""
        
        if len(self.market_data_history) < 5:
            return
        
        # Update volatility estimate
        recent_prices = [md.last_price for md in list(self.market_data_history)[-20:]]
        if len(recent_prices) > 2:
            price_changes = np.diff(recent_prices)
            recent_vol = np.std(price_changes) * np.sqrt(252 * 24 * 60)  # Annualized
            
            # Exponential smoothing
            alpha = 0.1
            self.volatility_estimate = alpha * recent_vol + (1 - alpha) * self.volatility_estimate
        
        # Update trend estimate
        if len(recent_prices) >= 10:
            # Simple linear trend
            x = np.arange(len(recent_prices))
            slope, _ = np.polyfit(x, recent_prices, 1)
            self.trend_estimate = slope / recent_prices[-1] if recent_prices[-1] > 0 else 0
        
        # Update liquidity estimate
        recent_spreads = []
        for md in list(self.market_data_history)[-10:]:
            spread = md.ask_price - md.bid_price
            mid = (md.ask_price + md.bid_price) / 2
            if mid > 0:
                recent_spreads.append(spread / mid)
        
        if recent_spreads:
            avg_spread_pct = np.mean(recent_spreads)
            self.liquidity_estimate = max(0.1, 1.0 - avg_spread_pct * 100)
    
    def _adjust_strategy_parameters(self, market_data: 'MarketData') -> None:
        """Adjust strategy parameters based on market conditions and progress"""
        
        time_progress = self.state.elapsed_minutes / self.params.duration_minutes
        execution_progress = self.state.executed_quantity / self.params.total_quantity
        
        # Increase urgency if behind schedule
        if time_progress > execution_progress and time_progress > 0.3:
            urgency_boost = (time_progress - execution_progress) * 2.0
            self.current_urgency = min(1.0, self.params.urgency + urgency_boost)
        else:
            self.current_urgency = self.params.urgency
        
        # Adjust participation rate based on market conditions
        base_participation = (self.params.max_participation_rate + 
                            self.params.min_participation_rate) / 2
        
        # Increase participation in high liquidity
        liquidity_adj = self.liquidity_estimate
        
        # Decrease participation in high volatility
        vol_adj = 1.0 / (1.0 + self.volatility_estimate * 10)
        
        self.current_participation_rate = base_participation * liquidity_adj * vol_adj
        self.current_participation_rate = max(
            self.params.min_participation_rate,
            min(self.params.max_participation_rate, self.current_participation_rate)
        )
    
    def _calculate_base_order_size(self, market_data: 'MarketData') -> float:
        """Calculate base order size"""
        
        # Time-based distribution
        remaining_time = self.params.duration_minutes - self.state.elapsed_minutes
        if remaining_time <= 0:
            return self.state.remaining_quantity
        
        # Base size assuming equal distribution over remaining time
        remaining_intervals = max(1, remaining_time / 5)  # 5-minute intervals
        time_based_size = self.state.remaining_quantity / remaining_intervals
        
        # Volume-based size
        market_volume = getattr(market_data, 'volume', 10000)
        volume_based_size = market_volume * self.current_participation_rate
        
        # Use the smaller of the two
        base_size = min(time_based_size, volume_based_size)
        
        return base_size
    
    def _calculate_urgency_multiplier(self, market_data: 'MarketData') -> float:
        """Calculate urgency multiplier for order size"""
        
        # Base urgency multiplier
        urgency_mult = 1.0 + self.current_urgency
        
        # Price-based urgency
        current_price = market_data.last_price
        arrival_price = self.state.arrival_price
        price_move = (current_price - arrival_price) / arrival_price
        
        # Increase urgency if price is moving against us
        if self.side == 'buy' and price_move > 0.001:  # Price moving up
            price_urgency = 1.0 + min(price_move * 5, 0.5)
        elif self.side == 'sell' and price_move < -0.001:  # Price moving down
            price_urgency = 1.0 + min(abs(price_move) * 5, 0.5)
        else:
            price_urgency = 1.0
        
        return urgency_mult * price_urgency
    
    def _calculate_market_multiplier(self, market_data: 'MarketData') -> float:
        """Calculate market condition multiplier"""
        
        # Liquidity multiplier
        liquidity_mult = 0.5 + 1.5 * self.liquidity_estimate
        
        # Volatility multiplier (reduce size in high vol)
        vol_mult = 1.0 / (1.0 + self.volatility_estimate * 5)
        
        # Trend multiplier
        if abs(self.trend_estimate) > 0.0001:  # Significant trend
            if (self.side == 'buy' and self.trend_estimate > 0) or \
               (self.side == 'sell' and self.trend_estimate < 0):
                # Trend is favorable
                trend_mult = 1.2
            else:
                # Trend is unfavorable
                trend_mult = 0.8
        else:
            trend_mult = 1.0
        
        return liquidity_mult * vol_mult * trend_mult
    
    def _check_market_conditions(self, market_data: 'MarketData') -> bool:
        """Check if market conditions are suitable for trading"""
        
        # Check spread
        spread = market_data.ask_price - market_data.bid_price
        mid_price = (market_data.ask_price + market_data.bid_price) / 2
        spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
        
        # Don't trade if spread is extremely wide
        if spread_bps > 200:  # 2%
            return False
        
        # Check for reasonable volume
        if hasattr(market_data, 'volume') and market_data.volume > 0:
            if market_data.volume < 100:  # Very low volume
                return False
        
        return True
    
    def _check_price_level(self, market_data: 'MarketData') -> bool:
        """Check if current price level is acceptable relative to arrival price"""
        
        current_price = market_data.last_price
        arrival_price = self.state.arrival_price
        
        price_deviation = abs(current_price - arrival_price) / arrival_price
        
        # Always trade if near end of time
        time_progress = self.state.elapsed_minutes / self.params.duration_minutes
        if time_progress > 0.85:
            return True
        
        # Check price tolerance
        max_deviation = self.params.price_limit_tolerance
        
        # Adjust tolerance based on urgency and time
        adjusted_tolerance = max_deviation * (1 + self.current_urgency) * (1 + time_progress)
        
        return price_deviation <= adjusted_tolerance
    
    def _estimate_market_impact(self, order_size: float, market_data: 'MarketData') -> float:
        """Estimate market impact of order size"""
        
        # Use historical impact measurements if available
        if self.impact_measurements:
            base_impact = np.mean(self.impact_measurements[-5:])
        else:
            # Fallback estimate
            base_impact = 0.001 * self.state.arrival_price
        
        # Scale by order size (square root law)
        if hasattr(market_data, 'volume') and market_data.volume > 0:
            size_ratio = order_size / market_data.volume
            impact_scaling = np.sqrt(max(size_ratio, 0.01))
        else:
            impact_scaling = 1.0
        
        estimated_impact = base_impact * impact_scaling
        
        return estimated_impact


# Import MarketData from twap_algorithm for consistency
from .twap_algorithm import MarketData


if __name__ == "__main__":
    # Use real-time crypto market data (via RealTimeDataCache) to drive deterministic testing
    from data.ingestion.realtime_data_cache import RealTimeDataCache
    import time

    print("Running Arrival Price algorithm with real-time crypto data")

    # Configuration
    SYMBOL = 'BTCUSDT'  # Change to desired trading pair
    params = ArrivalPriceParameters(
        total_quantity=75000,
        duration_minutes=240,  # 4 hours
        urgency=0.4,
        risk_aversion=0.25,
        max_participation_rate=0.35,
        price_limit_tolerance=0.004,  # 40 bps
        market_on_close_pct=0.15
    )

    ap_algo = ArrivalPriceAlgorithm(params)

    # Start real-time cache
    cache = RealTimeDataCache([SYMBOL])
    cache.start(enable_websocket=True)

    # Allow cache to warm up
    time.sleep(1)

    # Fetch recent OHLCV (minute or hourly depending on ingestion capability)
    try:
        ohlcv = cache.get_ohlcv_dataframe(SYMBOL, interval='1m', days=1)
    except Exception:
        ohlcv = cache.get_ohlcv_dataframe(SYMBOL, interval='1h', days=1)

    if ohlcv is None or ohlcv.empty:
        raise RuntimeError("No historical data available for symbol: %s" % SYMBOL)

    # Iterate deterministically over historical snapshots to drive the algorithm
    for idx, row in ohlcv.iterrows():
        last_price = float(row.get('close', row.get('last', 0)))

        # Try to get order book for sizes/prices
        ob = cache.get_order_book(SYMBOL) or {}
        bids = ob.get('bids') if isinstance(ob, dict) else None
        asks = ob.get('asks') if isinstance(ob, dict) else None

        if bids and len(bids) > 0:
            bid_price, bid_size = float(bids[0][0]), float(bids[0][1])
        else:
            bid_price = last_price * (1 - 0.0006)
            bid_size = float(row.get('volume', 1000)) * 0.01

        if asks and len(asks) > 0:
            ask_price, ask_size = float(asks[0][0]), float(asks[0][1])
        else:
            ask_price = last_price * (1 + 0.0006)
            ask_size = float(row.get('volume', 1000)) * 0.01

        market_data = MarketData(
            timestamp=pd.to_datetime(idx).to_pydatetime() if not isinstance(idx, pd.Timestamp) else idx.to_pydatetime(),
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            last_price=last_price,
            volume=float(row.get('volume', 0)),
            volatility=cache.get_volatility(SYMBOL, days=1)
        )

        # Initialize on first snapshot
        if not ap_algo.is_active:
            ap_algo.initialize_order('buy', SYMBOL, market_data)

        ap_algo.update_market_data(market_data)

        if ap_algo.should_place_order(market_data):
            order_size = ap_algo.calculate_order_size(market_data)
            limit_price = ap_algo.calculate_limit_price(market_data, order_size)

            # Deterministic execution logic based on order book depth
            if limit_price >= ask_price:
                executed_qty = min(order_size, ask_size)
                execution_price = min(limit_price, ask_price)
            elif limit_price >= last_price:
                executed_qty = min(order_size, max(1.0, ask_size * 0.5))
                execution_price = limit_price
            else:
                executed_qty = min(order_size, max(1.0, ask_size * 0.1))
                execution_price = limit_price

            if executed_qty > 0:
                ap_algo.process_execution(executed_qty, execution_price, market_data.timestamp)

        # Stop early if completed
        if not ap_algo.is_active:
            break

    # Stop real-time cache
    cache.stop()

    # Results
    print("\nArrival Price Algorithm Results:")
    metrics = ap_algo.get_performance_metrics()
    for key, value in metrics.items():
        if 'bps' in key:
            print(f"  {key}: {value:.2f} bps")
        elif 'rate' in key or 'performance' in key:
            print(f"  {key}: {value:.4f}")
        elif 'price' in key and 'volatility' not in key:
            print(f"  {key}: {value:.4f}")
        else:
            try:
                print(f"  {key}: {value}")
            except Exception:
                pass

    print("\nArrival Price algorithm run completed with real data")