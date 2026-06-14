"""
Advanced Order Book Dynamics Engine for QUANTUM-FORGE
Implements sophisticated models for limit order book behavior and dynamics.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson, expon
from scipy.optimize import minimize
from collections import deque, OrderedDict
from typing import Dict, List, Tuple, Optional, Union
import heapq
from numba import jit, prange
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')
import time

class OrderType(Enum):
    """Order types in the limit order book."""
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    HIDDEN = "hidden"
    ICEBERG = "iceberg"

class OrderSide(Enum):
    """Order sides."""
    BID = "bid"
    ASK = "ask"

@dataclass
class Order:
    """Individual order representation."""
    order_id: str
    timestamp: float
    side: OrderSide
    order_type: OrderType
    price: float
    quantity: int
    remaining_quantity: int
    trader_id: str
    hidden_quantity: int = 0
    iceberg_peak: int = 0
    
    def __post_init__(self):
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity

@dataclass
class Trade:
    """Trade execution record."""
    timestamp: float
    price: float
    quantity: int
    aggressor_side: OrderSide
    buyer_id: str
    seller_id: str
    trade_id: str

class PriceLevel:
    """Price level in the order book."""
    
    def __init__(self, price: float):
        self.price = price
        self.orders: OrderedDict[str, Order] = OrderedDict()
        self.total_quantity = 0
        self.visible_quantity = 0
        self.hidden_quantity = 0
    
    def add_order(self, order: Order):
        """Add order to this price level."""
        self.orders[order.order_id] = order
        self.total_quantity += order.remaining_quantity
        
        if order.order_type == OrderType.HIDDEN:
            self.hidden_quantity += order.remaining_quantity
        else:
            self.visible_quantity += order.remaining_quantity
    
    def remove_order(self, order_id: str) -> Optional[Order]:
        """Remove order from this price level."""
        if order_id in self.orders:
            order = self.orders.pop(order_id)
            self.total_quantity -= order.remaining_quantity
            
            if order.order_type == OrderType.HIDDEN:
                self.hidden_quantity -= order.remaining_quantity
            else:
                self.visible_quantity -= order.remaining_quantity
            
            return order
        return None
    
    def reduce_quantity(self, order_id: str, quantity: int) -> bool:
        """Reduce quantity of a specific order."""
        if order_id in self.orders:
            order = self.orders[order_id]
            reduction = min(quantity, order.remaining_quantity)
            
            order.remaining_quantity -= reduction
            self.total_quantity -= reduction
            
            if order.order_type == OrderType.HIDDEN:
                self.hidden_quantity -= reduction
            else:
                self.visible_quantity -= reduction
            
            # Remove order if fully filled
            if order.remaining_quantity == 0:
                self.orders.pop(order_id)
            
            return True
        return False
    
    def is_empty(self) -> bool:
        """Check if price level is empty."""
        return len(self.orders) == 0

class LimitOrderBook:
    """Advanced limit order book implementation with microstructure features."""
    
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bids: Dict[float, PriceLevel] = {}  # Price -> PriceLevel
        self.asks: Dict[float, PriceLevel] = {}  # Price -> PriceLevel
        self.bid_prices = []  # Max heap (negative values)
        self.ask_prices = []  # Min heap
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.order_counter = 0
        self.trade_counter = 0
        
        # Market microstructure metrics
        self.bid_ask_spread_history = deque(maxlen=10000)
        self.depth_history = deque(maxlen=10000)
        self.trade_history = deque(maxlen=10000)
        self.order_flow_imbalance = deque(maxlen=1000)
        
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        while self.bid_prices and (-self.bid_prices[0] not in self.bids or self.bids[-self.bid_prices[0]].is_empty()):
            heapq.heappop(self.bid_prices)
        
        return -self.bid_prices[0] if self.bid_prices else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        while self.ask_prices and (self.ask_prices[0] not in self.asks or self.asks[self.ask_prices[0]].is_empty()):
            heapq.heappop(self.ask_prices)
        
        return self.ask_prices[0] if self.ask_prices else None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid is not None and best_ask is not None:
            return best_ask - best_bid
        return None
    
    def add_order(self, side: OrderSide, order_type: OrderType, price: float, 
                 quantity: int, trader_id: str, timestamp: float = None,
                 hidden_quantity: int = 0, iceberg_peak: int = 0) -> str:
        """Add order to the book."""
        if timestamp is None:
            # Use wall-clock time for deterministic timestamping instead of random
            timestamp = time.time()
        
        self.order_counter += 1
        order_id = f"order_{self.order_counter}"
        
        # Create order
        order = Order(
            order_id=order_id,
            timestamp=timestamp,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
            remaining_quantity=quantity,
            trader_id=trader_id,
            hidden_quantity=hidden_quantity,
            iceberg_peak=iceberg_peak
        )
        
        self.orders[order_id] = order
        
        # Handle market orders
        if order_type == OrderType.MARKET:
            self._execute_market_order(order)
        else:
            # Add limit order to book
            self._add_limit_order(order)
        
        # Update microstructure metrics
        self._update_metrics()
        
        return order_id
    
    def _add_limit_order(self, order: Order):
        """Add limit order to appropriate side of the book."""
        if order.side == OrderSide.BID:
            if order.price not in self.bids:
                self.bids[order.price] = PriceLevel(order.price)
                heapq.heappush(self.bid_prices, -order.price)
            
            self.bids[order.price].add_order(order)
            
        else:  # ASK
            if order.price not in self.asks:
                self.asks[order.price] = PriceLevel(order.price)
                heapq.heappush(self.ask_prices, order.price)
            
            self.asks[order.price].add_order(order)
    
    def _execute_market_order(self, order: Order):
        """Execute market order against the book."""
        remaining_qty = order.remaining_quantity
        
        if order.side == OrderSide.BID:
            # Buy market order - match against asks
            while remaining_qty > 0 and self.ask_prices:
                best_ask_price = self.get_best_ask()
                if best_ask_price is None:
                    break
                
                price_level = self.asks[best_ask_price]
                matched_qty = min(remaining_qty, price_level.visible_quantity)
                
                if matched_qty > 0:
                    self._execute_trade(order, price_level, matched_qty, best_ask_price)
                    remaining_qty -= matched_qty
                
                # Remove empty price level
                if price_level.is_empty():
                    del self.asks[best_ask_price]
        
        else:  # ASK
            # Sell market order - match against bids
            while remaining_qty > 0 and self.bid_prices:
                best_bid_price = self.get_best_bid()
                if best_bid_price is None:
                    break
                
                price_level = self.bids[best_bid_price]
                matched_qty = min(remaining_qty, price_level.visible_quantity)
                
                if matched_qty > 0:
                    self._execute_trade(order, price_level, matched_qty, best_bid_price)
                    remaining_qty -= matched_qty
                
                # Remove empty price level
                if price_level.is_empty():
                    del self.bids[best_bid_price]
        
        order.remaining_quantity = remaining_qty
    
    def _execute_trade(self, incoming_order: Order, price_level: PriceLevel, 
                      quantity: int, trade_price: float):
        """Execute trade between orders."""
        executed_qty = 0
        
        # Match against orders in price level (FIFO)
        for order_id, resting_order in list(price_level.orders.items()):
            if executed_qty >= quantity:
                break
            
            trade_qty = min(quantity - executed_qty, resting_order.remaining_quantity)
            
            # Create trade record
            self.trade_counter += 1
            trade = Trade(
                timestamp=max(incoming_order.timestamp, resting_order.timestamp),
                price=trade_price,
                quantity=trade_qty,
                aggressor_side=incoming_order.side,
                buyer_id=incoming_order.trader_id if incoming_order.side == OrderSide.BID else resting_order.trader_id,
                seller_id=resting_order.trader_id if incoming_order.side == OrderSide.BID else incoming_order.trader_id,
                trade_id=f"trade_{self.trade_counter}"
            )
            
            self.trades.append(trade)
            self.trade_history.append(trade)
            
            # Update order quantities
            price_level.reduce_quantity(order_id, trade_qty)
            executed_qty += trade_qty
            
            # Handle iceberg orders
            if resting_order.order_type == OrderType.ICEBERG and resting_order.remaining_quantity == 0:
                if resting_order.hidden_quantity > 0:
                    # Replenish iceberg order
                    replenish_qty = min(resting_order.iceberg_peak, resting_order.hidden_quantity)
                    resting_order.remaining_quantity = replenish_qty
                    resting_order.hidden_quantity -= replenish_qty
                    price_level.add_order(resting_order)
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        # Remove from appropriate side
        if order.side == OrderSide.BID and order.price in self.bids:
            self.bids[order.price].remove_order(order_id)
            if self.bids[order.price].is_empty():
                del self.bids[order.price]
        
        elif order.side == OrderSide.ASK and order.price in self.asks:
            self.asks[order.price].remove_order(order_id)
            if self.asks[order.price].is_empty():
                del self.asks[order.price]
        
        # Remove from orders dict
        del self.orders[order_id]
        
        self._update_metrics()
        return True
    
    def modify_order(self, order_id: str, new_price: Optional[float] = None, 
                    new_quantity: Optional[int] = None) -> bool:
        """Modify an existing order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        # Cancel existing order
        self.cancel_order(order_id)
        
        # Add modified order
        price = new_price if new_price is not None else order.price
        quantity = new_quantity if new_quantity is not None else order.quantity
        
        self.add_order(
            side=order.side,
            order_type=order.order_type,
            price=price,
            quantity=quantity,
            trader_id=order.trader_id,
            timestamp=order.timestamp
        )
        
        return True
    
    def get_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, int]]]:
        """Get market depth for specified number of levels."""
        bids = []
        asks = []
        
        # Get bid levels
        bid_prices_sorted = sorted(self.bids.keys(), reverse=True)
        for i, price in enumerate(bid_prices_sorted[:levels]):
            if price in self.bids and not self.bids[price].is_empty():
                bids.append((price, self.bids[price].visible_quantity))
        
        # Get ask levels
        ask_prices_sorted = sorted(self.asks.keys())
        for i, price in enumerate(ask_prices_sorted[:levels]):
            if price in self.asks and not self.asks[price].is_empty():
                asks.append((price, self.asks[price].visible_quantity))
        
        return {'bids': bids, 'asks': asks}
    
    def _update_metrics(self):
        """Update microstructure metrics."""
        # Bid-ask spread
        spread = self.get_spread()
        if spread is not None:
            self.bid_ask_spread_history.append(spread)
        
        # Market depth
        depth = self.get_depth(5)
        total_bid_qty = sum(qty for _, qty in depth['bids'])
        total_ask_qty = sum(qty for _, qty in depth['asks'])
        self.depth_history.append((total_bid_qty, total_ask_qty))
        
        # Order flow imbalance
        if len(self.depth_history) > 0:
            bid_qty, ask_qty = self.depth_history[-1]
            if bid_qty + ask_qty > 0:
                imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
                self.order_flow_imbalance.append(imbalance)

class OrderBookAnalyzer:
    """Advanced analysis of order book dynamics."""
    
    @staticmethod
    def calculate_resilience(lob: LimitOrderBook, price_impact: float, 
                           time_horizon: int = 100) -> float:
        """
        Calculate order book resilience after a price impact.
        
        Args:
            lob: Limit order book
            price_impact: Size of price impact
            time_horizon: Time steps to measure recovery
        
        Returns:
            Resilience measure (0-1, 1 = full recovery)
        """
        initial_mid = lob.get_mid_price()
        if initial_mid is None:
            return 0.0
        
        # Simulate price impact (simplified)
        impacted_price = initial_mid * (1 + price_impact)
        
        # Measure recovery over time (would need actual time series data)
        # This is a simplified placeholder
        recovery_rate = np.exp(-0.1 * time_horizon)  # Exponential recovery model
        resilience = min(1.0, recovery_rate)
        
        return resilience
    
    @staticmethod
    def estimate_market_impact(lob: LimitOrderBook, side: OrderSide, 
                             quantity: int) -> float:
        """
        Estimate market impact of a large order.
        
        Args:
            lob: Limit order book
            side: Order side
            quantity: Order quantity
        
        Returns:
            Expected price impact (relative)
        """
        if side == OrderSide.BID:
            # Buying - impact on ask side
            levels = sorted(lob.asks.keys())
            remaining_qty = quantity
            total_cost = 0.0
            
            for price in levels:
                if remaining_qty <= 0:
                    break
                
                available_qty = lob.asks[price].visible_quantity
                trade_qty = min(remaining_qty, available_qty)
                
                total_cost += trade_qty * price
                remaining_qty -= trade_qty
            
            if quantity > remaining_qty:
                avg_price = total_cost / (quantity - remaining_qty)
                initial_price = lob.get_best_ask()
                
                return (avg_price - initial_price) / initial_price if initial_price else 0.0
        
        else:  # ASK
            # Selling - impact on bid side
            levels = sorted(lob.bids.keys(), reverse=True)
            remaining_qty = quantity
            total_proceeds = 0.0
            
            for price in levels:
                if remaining_qty <= 0:
                    break
                
                available_qty = lob.bids[price].visible_quantity
                trade_qty = min(remaining_qty, available_qty)
                
                total_proceeds += trade_qty * price
                remaining_qty -= trade_qty
            
            if quantity > remaining_qty:
                avg_price = total_proceeds / (quantity - remaining_qty)
                initial_price = lob.get_best_bid()
                
                return (initial_price - avg_price) / initial_price if initial_price else 0.0
        
        return 0.0
    
    @staticmethod
    def calculate_order_flow_imbalance(lob: LimitOrderBook, lookback: int = 100) -> float:
        """Calculate order flow imbalance over recent period."""
        if len(lob.order_flow_imbalance) == 0:
            return 0.0
        
        recent_imbalances = list(lob.order_flow_imbalance)[-lookback:]
        return np.mean(recent_imbalances)
    
    @staticmethod
    def estimate_information_content(trade_history: List[Trade], 
                                   window: int = 100) -> float:
        """
        Estimate information content of trades using PIN model concepts.
        
        Args:
            trade_history: List of recent trades
            window: Analysis window
        
        Returns:
            Information content estimate
        """
        if len(trade_history) < window:
            return 0.0
        
        recent_trades = trade_history[-window:]
        
        # Calculate buy/sell trade imbalance
        buy_volume = sum(t.quantity for t in recent_trades if t.aggressor_side == OrderSide.BID)
        sell_volume = sum(t.quantity for t in recent_trades if t.aggressor_side == OrderSide.ASK)
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0
        
        # Simple information content proxy
        imbalance = abs(buy_volume - sell_volume) / total_volume
        return min(1.0, imbalance)

class HawkesProcessOrderBook:
    """Order book model using Hawkes process for order arrivals."""
    
    def __init__(self, base_intensity: float = 1.0, decay_rate: float = 2.0, 
                 excitation: float = 0.5):
        """
        Initialize Hawkes process parameters.
        
        Args:
            base_intensity: Background arrival rate
            decay_rate: Exponential decay rate
            excitation: Self-excitation parameter
        """
        self.base_intensity = base_intensity
        self.decay_rate = decay_rate
        self.excitation = excitation
        self.event_times = []
        self.current_intensity = base_intensity
    
    def update_intensity(self, current_time: float):
        """Update intensity based on recent events."""
        self.current_intensity = self.base_intensity
        
        for event_time in self.event_times:
            if current_time > event_time:
                self.current_intensity += self.excitation * np.exp(-self.decay_rate * (current_time - event_time))
    
    def simulate_next_arrival(self, current_time: float) -> float:
        """Simulate next order arrival time."""
        self.update_intensity(current_time)
        # Deterministic inter-arrival: use mean inter-arrival (1/intensity)
        # This removes stochastic sampling and keeps timing driven by model intensity.
        inter_arrival = 1.0 / max(self.current_intensity, 1e-12)
        next_time = current_time + inter_arrival
        
        # Add to event history
        self.event_times.append(next_time)
        if len(self.event_times) > 1000:  # Keep only recent events
            self.event_times.pop(0)
        
        return next_time

# Example usage and testing
if __name__ == "__main__":
    print("Testing Order Book Dynamics Engine...")
    
    # Create limit order book
    lob = LimitOrderBook(tick_size=0.01)
    
    # Add some orders
    print("\nAdding initial orders...")
    lob.add_order(OrderSide.BID, OrderType.LIMIT, 99.95, 100, "trader_1")
    lob.add_order(OrderSide.BID, OrderType.LIMIT, 99.90, 200, "trader_2")
    lob.add_order(OrderSide.ASK, OrderType.LIMIT, 100.05, 150, "trader_3")
    lob.add_order(OrderSide.ASK, OrderType.LIMIT, 100.10, 100, "trader_4")
    
    # Display market state
    print(f"Best bid: {lob.get_best_bid()}")
    print(f"Best ask: {lob.get_best_ask()}")
    print(f"Mid price: {lob.get_mid_price():.4f}")
    print(f"Spread: {lob.get_spread():.4f}")
    
    # Execute market order
    print("\nExecuting market buy order for 75 shares...")
    lob.add_order(OrderSide.BID, OrderType.MARKET, 0, 75, "trader_5")
    
    print(f"New best ask: {lob.get_best_ask()}")
    print(f"New mid price: {lob.get_mid_price():.4f}")
    print(f"Number of trades: {len(lob.trades)}")
    
    # Analyze market impact
    analyzer = OrderBookAnalyzer()
    market_impact = analyzer.estimate_market_impact(lob, OrderSide.BID, 200)
    print(f"Estimated market impact for 200 share buy: {market_impact:.4f}")
    
    # Get market depth
    depth = lob.get_depth(3)
    print(f"\nMarket depth (3 levels):")
    print(f"Bids: {depth['bids']}")
    print(f"Asks: {depth['asks']}")
    
    # Test Hawkes process
    print("\nTesting Hawkes process order arrivals...")
    hawkes = HawkesProcessOrderBook(base_intensity=5.0, decay_rate=2.0, excitation=0.3)
    
    current_time = 0.0
    for i in range(10):
        next_arrival = hawkes.simulate_next_arrival(current_time)
        print(f"Next order arrival at time: {next_arrival:.4f}")
        current_time = next_arrival
    
    print("\nOrder book dynamics engine test completed successfully!")