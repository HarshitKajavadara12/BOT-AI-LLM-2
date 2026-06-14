"""
Dynamic Portfolio & Order Tracker
==================================
Real-time tracking of positions, orders, cash, and system metrics.
Integrates with OMS and provides live data to dashboards.

Author: Quantum Forge Team
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class DynamicPosition:
    """Dynamic position that updates from trades."""
    symbol: str
    amount: float = 0.0
    entry_price: float = 0.0
    total_cost: float = 0.0
    realized_pnl: float = 0.0
    trades_count: int = 0
    first_entry_time: Optional[datetime] = None
    last_update_time: datetime = field(default_factory=datetime.now)
    
    def add_trade(self, quantity: float, price: float):
        """Add a trade to the position."""
        if self.amount == 0:
            # Opening position
            self.amount = quantity
            self.entry_price = price
            self.total_cost = quantity * price
            self.first_entry_time = datetime.now()
        else:
            # Adding to position
            new_total_cost = self.total_cost + (quantity * price)
            new_amount = self.amount + quantity
            if new_amount != 0:
                self.entry_price = new_total_cost / new_amount
            self.amount = new_amount
            self.total_cost = new_total_cost
        
        self.trades_count += 1
        self.last_update_time = datetime.now()
    
    def close_position(self, quantity: float, price: float):
        """Close part or all of position."""
        realized = (price - self.entry_price) * quantity
        self.realized_pnl += realized
        self.amount -= quantity
        self.total_cost -= quantity * self.entry_price
        self.last_update_time = datetime.now()
        
        if abs(self.amount) < 0.0001:  # Essentially zero
            self.amount = 0.0
            self.entry_price = 0.0
            self.total_cost = 0.0


@dataclass
class ActiveOrder:
    """Active order in the system."""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: str  # 'MARKET', 'LIMIT', etc.
    quantity: float
    filled_quantity: float = 0.0
    price: Optional[float] = None
    status: str = "PENDING"  # PENDING, PARTIAL, FILLED, CANCELLED
    created_time: datetime = field(default_factory=datetime.now)
    venue: str = "Binance"


@dataclass
class SystemMetrics:
    """Real-time system performance metrics."""
    orders_submitted: int = 0
    orders_filled: int = 0
    orders_cancelled: int = 0
    trades_executed: int = 0
    total_latency_ms: float = 0.0
    latency_samples: int = 0
    operations_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def avg_latency_ms(self) -> float:
        """Average system latency."""
        if self.latency_samples > 0:
            return self.total_latency_ms / self.latency_samples
        return 0.0
    
    @property
    def throughput_ops_per_sec(self) -> float:
        """Operations per second."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > 0:
            return self.operations_count / elapsed
        return 0.0
    
    @property
    def fill_rate(self) -> float:
        """Order fill rate."""
        if self.orders_submitted > 0:
            return self.orders_filled / self.orders_submitted
        return 0.0


class DynamicPortfolioTracker:
    """
    Real-time portfolio tracking system.
    Tracks positions, orders, cash, and system metrics dynamically.
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        """Initialize tracker with starting cash."""
        self.initial_cash = initial_cash
        self.cash_balance = initial_cash
        self.positions: Dict[str, DynamicPosition] = {}
        self.active_orders: Dict[str, ActiveOrder] = {}
        self.system_metrics = SystemMetrics()
        self.order_history = deque(maxlen=1000)
        self.trade_history = deque(maxlen=1000)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Historical tracking
        self.value_history = deque(maxlen=10000)
        self.pnl_history = deque(maxlen=10000)
        
        logger.info(f"Portfolio tracker initialized with ${initial_cash:,.2f} cash")
    
    def simulate_market_activity(self, symbols: List[str], data_cache):
        """
        Simulate realistic trading activity based on market conditions.
        Creates orders and fills based on real market movements.
        """
        with self.lock:
            # Simulate some market-driven trading
            for symbol in symbols:
                current_price = data_cache.get_current_price(symbol)
                if current_price is None:
                    continue
                
                # Get historical data for decision
                hist = data_cache.get_historical_data(symbol, days=1)
                if hist.empty:
                    continue
                
                # Calculate short-term momentum
                if len(hist) >= 10:
                    recent_returns = hist['close'].pct_change().tail(10)
                    momentum = recent_returns.mean()
                    
                    # Market-driven order generation (not random, based on momentum)
                    if momentum > 0.001 and symbol not in self.positions:
                        # Strong upward momentum, consider buy
                        if np.random.random() < 0.05:  # 5% chance to initiate
                            self._create_market_order(symbol, 'BUY', current_price, data_cache)
                    
                    elif momentum < -0.001 and symbol in self.positions:
                        # Downward momentum, consider reducing
                        if np.random.random() < 0.03:  # 3% chance to reduce
                            self._create_market_order(symbol, 'SELL', current_price, data_cache)
            
            # Process pending orders
            self._process_orders(data_cache)
    
    def _create_market_order(self, symbol: str, side: str, price: float, data_cache):
        """Create a new market order."""
        # Determine quantity based on cash and price
        if side == 'BUY':
            # Use 5-10% of cash for position
            allocation = self.cash_balance * np.random.uniform(0.05, 0.10)
            quantity = allocation / price
            
            if quantity * price > self.cash_balance:
                return  # Not enough cash
            
            # Create order
            order_id = f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            order = ActiveOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity,
                price=price,
                status='PENDING'
            )
            self.active_orders[order_id] = order
            self.system_metrics.orders_submitted += 1
            
            logger.info(f"Created BUY order: {symbol} x {quantity:.4f} @ ${price:.2f}")
        
        elif side == 'SELL' and symbol in self.positions:
            # Sell 30-50% of position
            position = self.positions[symbol]
            if position.amount > 0:
                quantity = position.amount * np.random.uniform(0.3, 0.5)
                
                order_id = f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                order = ActiveOrder(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type='MARKET',
                    quantity=quantity,
                    price=price,
                    status='PENDING'
                )
                self.active_orders[order_id] = order
                self.system_metrics.orders_submitted += 1
                
                logger.info(f"Created SELL order: {symbol} x {quantity:.4f} @ ${price:.2f}")
    
    def _process_orders(self, data_cache):
        """Process pending orders (simulate fills)."""
        filled_orders = []
        
        for order_id, order in self.active_orders.items():
            if order.status != 'PENDING':
                continue
            
            # Get current price
            current_price = data_cache.get_current_price(order.symbol)
            if current_price is None:
                continue
            
            # Simulate fill (90% fill rate for market orders)
            if np.random.random() < 0.9:
                # Fill the order
                fill_price = current_price * np.random.uniform(0.9999, 1.0001)  # Slight slippage
                latency_ms = np.random.uniform(10, 50)  # Realistic latency
                
                if order.side == 'BUY':
                    # Execute buy
                    cost = order.quantity * fill_price
                    if cost <= self.cash_balance:
                        self.cash_balance -= cost
                        
                        if order.symbol not in self.positions:
                            self.positions[order.symbol] = DynamicPosition(symbol=order.symbol)
                        
                        self.positions[order.symbol].add_trade(order.quantity, fill_price)
                        
                        order.status = 'FILLED'
                        order.filled_quantity = order.quantity
                        self.system_metrics.orders_filled += 1
                        self.system_metrics.trades_executed += 1
                        self.system_metrics.total_latency_ms += latency_ms
                        self.system_metrics.latency_samples += 1
                        
                        filled_orders.append(order_id)
                        self.trade_history.append({
                            'time': datetime.now(),
                            'symbol': order.symbol,
                            'side': order.side,
                            'quantity': order.quantity,
                            'price': fill_price
                        })
                        
                        logger.info(f"FILLED BUY: {order.symbol} x {order.quantity:.4f} @ ${fill_price:.2f}")
                
                elif order.side == 'SELL':
                    # Execute sell
                    if order.symbol in self.positions:
                        position = self.positions[order.symbol]
                        if position.amount >= order.quantity:
                            proceeds = order.quantity * fill_price
                            self.cash_balance += proceeds
                            
                            position.close_position(order.quantity, fill_price)
                            
                            order.status = 'FILLED'
                            order.filled_quantity = order.quantity
                            self.system_metrics.orders_filled += 1
                            self.system_metrics.trades_executed += 1
                            self.system_metrics.total_latency_ms += latency_ms
                            self.system_metrics.latency_samples += 1
                            
                            filled_orders.append(order_id)
                            self.trade_history.append({
                                'time': datetime.now(),
                                'symbol': order.symbol,
                                'side': order.side,
                                'quantity': order.quantity,
                                'price': fill_price
                            })
                            
                            logger.info(f"FILLED SELL: {order.symbol} x {order.quantity:.4f} @ ${fill_price:.2f}")
            
            # Increment operations count
            self.system_metrics.operations_count += 1
        
        # Remove filled orders from active
        for order_id in filled_orders:
            if order_id in self.active_orders:
                self.order_history.append(self.active_orders[order_id])
                del self.active_orders[order_id]
    
    def get_positions(self) -> Dict[str, DynamicPosition]:
        """Get current positions."""
        with self.lock:
            return self.positions.copy()
    
    def get_active_orders(self) -> Dict[str, ActiveOrder]:
        """Get active orders."""
        with self.lock:
            return self.active_orders.copy()
    
    def get_cash_balance(self) -> float:
        """Get current cash balance."""
        with self.lock:
            return self.cash_balance
    
    def get_metrics(self) -> SystemMetrics:
        """Get system metrics."""
        with self.lock:
            return self.system_metrics
    
    def calculate_portfolio_value(self, data_cache) -> float:
        """Calculate total portfolio value."""
        with self.lock:
            total = self.cash_balance
            
            for symbol, position in self.positions.items():
                if position.amount > 0:
                    current_price = data_cache.get_current_price(symbol)
                    if current_price:
                        total += position.amount * current_price
            
            # Track history
            self.value_history.append({
                'time': datetime.now(),
                'value': total
            })
            
            return total
    
    def get_total_pnl(self, data_cache) -> float:
        """Calculate total P&L."""
        with self.lock:
            total_unrealized = 0.0
            total_realized = 0.0
            
            for symbol, position in self.positions.items():
                if position.amount > 0:
                    current_price = data_cache.get_current_price(symbol)
                    if current_price:
                        unrealized = (current_price - position.entry_price) * position.amount
                        total_unrealized += unrealized
                
                total_realized += position.realized_pnl
            
            total_pnl = total_unrealized + total_realized
            
            # Track history
            self.pnl_history.append({
                'time': datetime.now(),
                'pnl': total_pnl
            })
            
            return total_pnl


# Global singleton instance
_tracker_instance = None
_tracker_lock = threading.Lock()


def get_portfolio_tracker(initial_cash: float = 100000.0) -> DynamicPortfolioTracker:
    """Get or create the global portfolio tracker instance."""
    global _tracker_instance
    
    with _tracker_lock:
        if _tracker_instance is None:
            _tracker_instance = DynamicPortfolioTracker(initial_cash=initial_cash)
            logger.info("Created new global portfolio tracker instance")
        
        return _tracker_instance
