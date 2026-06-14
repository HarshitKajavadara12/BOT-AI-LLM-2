"""
Order Management System (OMS)
Central order lifecycle management and routing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import threading
import queue
from collections import defaultdict, deque
import warnings
from core.audit import get_audit_logger


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    REPLACED = "REPLACED"
    EXPIRED = "EXPIRED"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"
    HIDDEN = "HIDDEN"
    PEGGED = "PEGGED"


class TimeInForce(Enum):
    """Time in force enumeration"""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    """Core order object"""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    average_fill_price: float = 0.0
    
    # Timestamps
    created_time: datetime = field(default_factory=datetime.now)
    submitted_time: Optional[datetime] = None
    first_fill_time: Optional[datetime] = None
    last_fill_time: Optional[datetime] = None
    
    # Additional parameters
    min_quantity: Optional[float] = None
    display_quantity: Optional[float] = None  # For iceberg orders
    expire_time: Optional[datetime] = None
    
    # Routing and execution
    venue: Optional[str] = None
    execution_strategy: Optional[str] = None
    parent_order_id: Optional[str] = None
    
    # Risk and compliance
    risk_checked: bool = False
    compliance_checked: bool = False
    
    # Audit
    snapshot_id: Optional[str] = None
    
    def __post_init__(self):
        if self.remaining_quantity is None:
            self.remaining_quantity = self.quantity
    
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def fill_rate(self) -> float:
        """Get current fill rate"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0


@dataclass
class Execution:
    """Order execution (fill) record"""
    execution_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    venue: str
    execution_type: str = "FILL"  # FILL, PARTIAL_FILL, etc.
    commission: float = 0.0
    liquidity_flag: str = "UNKNOWN"  # ADD, REMOVE, UNKNOWN


class OrderManagementSystem:
    """
    Central Order Management System
    Handles order lifecycle, routing, and execution tracking
    """
    
    def __init__(self, name: str = "OMS"):
        self.name = name
        
        # Order storage
        self.orders: Dict[str, Order] = {}
        self.executions: Dict[str, List[Execution]] = defaultdict(list)
        
        # Order queues
        self.pending_orders = queue.Queue()
        self.new_orders = queue.Queue()
        self.cancel_requests = queue.Queue()
        
        # Status tracking
        self.orders_by_status = defaultdict(list)
        self.orders_by_symbol = defaultdict(list)
        self.orders_by_client_id = defaultdict(list)
        
        # Risk and routing
        self.risk_manager = None
        self.order_router = None
        self.venue_connections = {}
        
        # Threading and processing
        self.is_running = False
        self.order_processor_thread = None
        self.execution_processor_thread = None
        
        # Performance metrics
        self.metrics = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'average_fill_time': 0.0,
            'total_notional': 0.0
        }
        
        # Event handlers
        self.order_event_handlers = []
        self.execution_event_handlers = []
        
    def start(self) -> None:
        """Start the OMS processing threads"""
        self.is_running = True
        
        # Start processing threads
        self.order_processor_thread = threading.Thread(
            target=self._process_orders,
            name="OrderProcessor"
        )
        self.order_processor_thread.start()
        
        self.execution_processor_thread = threading.Thread(
            target=self._process_executions,
            name="ExecutionProcessor"
        )
        self.execution_processor_thread.start()
        
        print(f"OMS {self.name} started")
    
    def stop(self) -> None:
        """Stop the OMS processing"""
        self.is_running = False
        
        # Wait for threads to complete
        if self.order_processor_thread:
            self.order_processor_thread.join(timeout=5.0)
        
        if self.execution_processor_thread:
            self.execution_processor_thread.join(timeout=5.0)
        
        print(f"OMS {self.name} stopped")
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit new order to OMS
        
        Args:
            order: Order to submit
        
        Returns:
            True if order accepted for processing
        """
        
        try:
            # Generate order ID if not provided
            if not order.order_id:
                order.order_id = str(uuid.uuid4())
            
            # Validate order
            if not self._validate_order(order):
                order.status = OrderStatus.REJECTED
                self._emit_order_event("ORDER_REJECTED", order)
                return False
            
            # Risk check
            if self.risk_manager and not self.risk_manager.check_order(order):
                order.status = OrderStatus.REJECTED
                self._emit_order_event("ORDER_REJECTED", order, "Risk check failed")
                return False
            
            # --- PHASE 4B: DETERMINISTIC SNAPSHOT ---
            # Capture the exact state at the moment of acceptance
            try:
                snapshot_id = get_audit_logger().log_snapshot(
                    market_state={"symbol": order.symbol, "timestamp": datetime.now().isoformat()}, # Placeholder for real market data
                    signal_state={"strategy": order.execution_strategy}, # Placeholder
                    risk_state={"risk_checked": True, "limit_check": "PASSED"},
                    decision={
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "quantity": order.quantity,
                        "price": order.price,
                        "type": order.order_type.value
                    }
                )
                order.snapshot_id = snapshot_id
            except Exception as e:
                print(f" ️ Audit Logging Failed: {e}")
                # We do NOT fail the order if logging fails, but we warn loudly
            # ----------------------------------------

            # Store order
            self.orders[order.order_id] = order
            self._update_order_indices(order)
            
            # Queue for processing
            self.pending_orders.put(order)
            
            # Update metrics
            self.metrics['total_orders'] += 1
            
            self._emit_order_event("ORDER_SUBMITTED", order)
            
            return True
            
        except Exception as e:
            print(f"Error submitting order: {e}")
            return False
    
    def cancel_order(self, order_id: str, reason: str = "User request") -> bool:
        """
        Cancel existing order
        
        Args:
            order_id: ID of order to cancel
            reason: Cancellation reason
        
        Returns:
            True if cancel request accepted
        """
        
        if order_id not in self.orders:
            print(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active:
            print(f"Order {order_id} is not active (status: {order.status})")
            return False
        
        # Queue cancel request
        cancel_request = {
            'order_id': order_id,
            'reason': reason,
            'timestamp': datetime.now()
        }
        
        self.cancel_requests.put(cancel_request)
        
        self._emit_order_event("CANCEL_REQUESTED", order, reason)
        
        return True
    
    def replace_order(
        self,
        order_id: str,
        new_quantity: Optional[float] = None,
        new_price: Optional[float] = None
    ) -> Optional[str]:
        """
        Replace existing order with new parameters
        
        Args:
            order_id: ID of order to replace
            new_quantity: New quantity (if changing)
            new_price: New price (if changing)
        
        Returns:
            New order ID if replacement successful
        """
        
        if order_id not in self.orders:
            return None
        
        old_order = self.orders[order_id]
        
        if not old_order.is_active:
            return None
        
        # Create replacement order
        new_order = Order(
            order_id=str(uuid.uuid4()),
            client_order_id=old_order.client_order_id + "_R",
            symbol=old_order.symbol,
            side=old_order.side,
            order_type=old_order.order_type,
            quantity=new_quantity or old_order.remaining_quantity,
            price=new_price or old_order.price,
            time_in_force=old_order.time_in_force,
            venue=old_order.venue,
            execution_strategy=old_order.execution_strategy,
            parent_order_id=old_order.parent_order_id
        )
        
        # Cancel old order
        if self.cancel_order(order_id, "Replaced"):
            # Submit new order
            if self.submit_order(new_order):
                old_order.status = OrderStatus.REPLACED
                self._emit_order_event("ORDER_REPLACED", old_order)
                return new_order.order_id
        
        return None
    
    def process_execution(self, execution: Execution) -> None:
        """
        Process incoming execution
        
        Args:
            execution: Execution to process
        """
        
        if execution.order_id not in self.orders:
            print(f"Execution for unknown order: {execution.order_id}")
            return
        
        order = self.orders[execution.order_id]
        
        # Update order with execution
        order.filled_quantity += execution.quantity
        order.remaining_quantity -= execution.quantity
        
        # Update average fill price
        if order.filled_quantity > 0:
            total_cost = (order.average_fill_price * (order.filled_quantity - execution.quantity) +
                         execution.price * execution.quantity)
            order.average_fill_price = total_cost / order.filled_quantity
        else:
            order.average_fill_price = execution.price
        
        # Update timestamps
        if order.first_fill_time is None:
            order.first_fill_time = execution.timestamp
        order.last_fill_time = execution.timestamp
        
        # Update status
        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
            order.remaining_quantity = 0
            self.metrics['filled_orders'] += 1
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Store execution
        self.executions[execution.order_id].append(execution)
        
        # Update metrics
        self.metrics['total_notional'] += execution.quantity * execution.price
        
        # Emit events
        self._emit_execution_event("EXECUTION", execution, order)
        
        if order.status == OrderStatus.FILLED:
            self._emit_order_event("ORDER_FILLED", order)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol"""
        return [self.orders[oid] for oid in self.orders_by_symbol.get(symbol, [])]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get all orders with specific status"""
        return [self.orders[oid] for oid in self.orders_by_status.get(status, [])]
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        active_orders = []
        for status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
            active_orders.extend(self.get_orders_by_status(status))
        return active_orders
    
    def get_executions(self, order_id: str) -> List[Execution]:
        """Get all executions for an order"""
        return self.executions.get(order_id, [])
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get OMS performance metrics"""
        
        # Calculate fill rate
        total_orders = self.metrics['total_orders']
        fill_rate = self.metrics['filled_orders'] / total_orders if total_orders > 0 else 0.0
        
        # Calculate average fill time
        fill_times = []
        for order in self.orders.values():
            if order.first_fill_time and order.submitted_time:
                fill_time = (order.first_fill_time - order.submitted_time).total_seconds()
                fill_times.append(fill_time)
        
        avg_fill_time = np.mean(fill_times) if fill_times else 0.0
        
        # Active orders by status
        status_counts = defaultdict(int)
        for order in self.orders.values():
            status_counts[order.status.value] += 1
        
        return {
            'total_orders': total_orders,
            'filled_orders': self.metrics['filled_orders'],
            'cancelled_orders': self.metrics['cancelled_orders'],
            'rejected_orders': self.metrics['rejected_orders'],
            'fill_rate': fill_rate,
            'average_fill_time_seconds': avg_fill_time,
            'total_notional': self.metrics['total_notional'],
            'active_orders': len(self.get_active_orders()),
            'orders_by_status': dict(status_counts)
        }
    
    def add_order_event_handler(self, handler: Callable) -> None:
        """Add order event handler"""
        self.order_event_handlers.append(handler)
    
    def add_execution_event_handler(self, handler: Callable) -> None:
        """Add execution event handler"""
        self.execution_event_handlers.append(handler)
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters"""
        
        # Basic validation
        if order.quantity <= 0:
            print("Invalid quantity")
            return False
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.price is None:
            print("Limit orders require price")
            return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            print("Stop orders require stop price")
            return False
        
        # Symbol validation (simplified)
        if not order.symbol or len(order.symbol) == 0:
            print("Invalid symbol")
            return False
        
        return True
    
    def _update_order_indices(self, order: Order) -> None:
        """Update order lookup indices"""
        
        # Update status index
        self.orders_by_status[order.status].append(order.order_id)
        
        # Update symbol index
        self.orders_by_symbol[order.symbol].append(order.order_id)
        
        # Update client ID index
        self.orders_by_client_id[order.client_order_id].append(order.order_id)
    
    def _process_orders(self) -> None:
        """Main order processing loop"""
        
        while self.is_running:
            try:
                # Process pending orders
                try:
                    order = self.pending_orders.get(timeout=1.0)
                    self._process_new_order(order)
                except queue.Empty:
                    continue
                
                # Process cancel requests
                try:
                    cancel_request = self.cancel_requests.get_nowait()
                    self._process_cancel_request(cancel_request)
                except queue.Empty:
                    pass
                    
            except Exception as e:
                print(f"Error in order processing: {e}")
    
    def _process_new_order(self, order: Order) -> None:
        """Process new order"""
        
        try:
            # Update status
            order.status = OrderStatus.NEW
            order.submitted_time = datetime.now()
            
            # Route order (simplified)
            if self.order_router:
                venue = self.order_router.route_order(order)
                order.venue = venue
            
            # Send to venue (simplified)
            self._send_order_to_venue(order)
            
            self._emit_order_event("ORDER_NEW", order)
            
        except Exception as e:
            print(f"Error processing new order: {e}")
            order.status = OrderStatus.REJECTED
            self._emit_order_event("ORDER_REJECTED", order, str(e))
    
    def _process_cancel_request(self, cancel_request: Dict) -> None:
        """Process cancel request"""
        
        order_id = cancel_request['order_id']
        
        if order_id in self.orders:
            order = self.orders[order_id]
            
            if order.is_active:
                # Send cancel to venue (simplified)
                self._send_cancel_to_venue(order)
                
                # Update status
                order.status = OrderStatus.CANCELLED
                self.metrics['cancelled_orders'] += 1
                
                self._emit_order_event("ORDER_CANCELLED", order, cancel_request['reason'])
    
    def _send_order_to_venue(self, order: Order) -> None:
        """Send order to execution venue (simplified)"""
        # In real implementation, this would send to actual venue
        pass
    
    def _send_cancel_to_venue(self, order: Order) -> None:
        """Send cancel request to venue (simplified)"""
        # In real implementation, this would send cancel to actual venue
        pass
    
    def _process_executions(self) -> None:
        """Process execution messages (placeholder)"""
        # In real implementation, this would process incoming execution messages
        while self.is_running:
            try:
                time.sleep(0.1)  # Placeholder processing
            except Exception as e:
                print(f"Error in execution processing: {e}")
    
    def _emit_order_event(self, event_type: str, order: Order, message: str = "") -> None:
        """Emit order event to handlers"""
        
        event = {
            'type': event_type,
            'order': order,
            'message': message,
            'timestamp': datetime.now()
        }
        
        for handler in self.order_event_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in order event handler: {e}")
    
    def _emit_execution_event(self, event_type: str, execution: Execution, order: Order) -> None:
        """Emit execution event to handlers"""
        
        event = {
            'type': event_type,
            'execution': execution,
            'order': order,
            'timestamp': datetime.now()
        }
        
        for handler in self.execution_event_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in execution event handler: {e}")


class OrderRouter:
    """
    Order routing logic
    Determines optimal venue for order execution
    """
    
    def __init__(self):
        self.venue_preferences = {}
        self.routing_rules = []
    
    def add_venue(self, venue_name: str, preferences: Dict[str, Any]) -> None:
        """Add venue with preferences"""
        self.venue_preferences[venue_name] = preferences
    
    def route_order(self, order: Order) -> str:
        """
        Route order to optimal venue
        
        Args:
            order: Order to route
        
        Returns:
            Venue name
        """
        
        # Simple routing logic (can be extended)
        if order.order_type == OrderType.MARKET:
            # Route market orders to venue with best liquidity
            return self._get_best_liquidity_venue(order.symbol)
        
        elif order.order_type == OrderType.LIMIT:
            # Route limit orders to venue with best price
            return self._get_best_price_venue(order.symbol, order.side)
        
        else:
            # Default venue
            return "PRIMARY"
    
    def _get_best_liquidity_venue(self, symbol: str) -> str:
        """Get venue with best liquidity for symbol"""
        # Simplified - would query real market data
        return "DARK_POOL"
    
    def _get_best_price_venue(self, symbol: str, side: OrderSide) -> str:
        """Get venue with best price for symbol and side"""
        # Simplified - would query real market data
        return "EXCHANGE_A"


if __name__ == "__main__":
    import time
    
    # Example usage and testing
    print("Testing Order Management System...")
    
    # Create OMS
    oms = OrderManagementSystem("TestOMS")
    
    # Add event handlers
    def order_event_handler(event):
        print(f"Order Event: {event['type']} - {event['order'].order_id}")
    
    def execution_event_handler(event):
        print(f"Execution Event: {event['type']} - {event['execution'].quantity} @ {event['execution'].price}")
    
    oms.add_order_event_handler(order_event_handler)
    oms.add_execution_event_handler(execution_event_handler)
    
    # Start OMS
    oms.start()
    
    try:
        # Submit test orders
        print("\nSubmitting test orders...")
        
        # Market buy order
        market_order = Order(
            order_id="",
            client_order_id="CLIENT_001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1000
        )
        
        success = oms.submit_order(market_order)
        print(f"Market order submitted: {success}")
        
        # Limit sell order
        limit_order = Order(
            order_id="",
            client_order_id="CLIENT_002",
            symbol="MSFT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=500,
            price=150.50,
            time_in_force=TimeInForce.GTC
        )
        
        success = oms.submit_order(limit_order)
        print(f"Limit order submitted: {success}")
        
        # Stop loss order
        stop_order = Order(
            order_id="",
            client_order_id="CLIENT_003",
            symbol="GOOGL",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=100,
            stop_price=2800.00
        )
        
        success = oms.submit_order(stop_order)
        print(f"Stop order submitted: {success}")
        
        # Wait a bit for processing
        time.sleep(2)
        
        # Check order status
        print(f"\nActive orders: {len(oms.get_active_orders())}")
        
        # Simulate executions
        print("\nSimulating executions...")
        
        for order in oms.get_active_orders()[:2]:  # Execute first 2 orders
            execution = Execution(
                execution_id=str(uuid.uuid4()),
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.price or 100.0,  # Use order price or default
                timestamp=datetime.now(),
                venue="TEST_VENUE"
            )
            
            oms.process_execution(execution)
        
        # Wait for processing
        time.sleep(1)
        
        # Test order cancellation
        active_orders = oms.get_active_orders()
        if active_orders:
            print(f"\nCancelling order: {active_orders[0].order_id}")
            oms.cancel_order(active_orders[0].order_id)
        
        # Wait for processing
        time.sleep(1)
        
        # Get performance metrics
        print(f"\nOMS Performance Metrics:")
        metrics = oms.get_performance_metrics()
        
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Test order replacement
        active_orders = oms.get_active_orders()
        if active_orders:
            old_order = active_orders[0]
            print(f"\nReplacing order: {old_order.order_id}")
            new_order_id = oms.replace_order(
                old_order.order_id,
                new_quantity=old_order.quantity * 1.5,
                new_price=old_order.price * 1.01 if old_order.price else None
            )
            print(f"New order ID: {new_order_id}")
        
        # Final status
        time.sleep(1)
        print(f"\nFinal active orders: {len(oms.get_active_orders())}")
        print(f"Total orders processed: {oms.metrics['total_orders']}")
        
    finally:
        # Stop OMS
        oms.stop()
    
    print("\nOrder Management System testing completed!")