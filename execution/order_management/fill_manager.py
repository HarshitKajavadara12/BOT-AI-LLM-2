"""
Fill Manager
Execution and fill processing with sophisticated fill matching
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import uuid
from collections import defaultdict, deque
import warnings


class FillType(Enum):
    """Fill type classification"""
    FULL = "FULL"           # Complete fill
    PARTIAL = "PARTIAL"     # Partial fill
    CANCEL = "CANCEL"       # Order cancelled
    REJECT = "REJECT"       # Order rejected
    EXPIRE = "EXPIRE"       # Order expired


class LiquidityType(Enum):
    """Liquidity provision type"""
    MAKER = "MAKER"         # Added liquidity
    TAKER = "TAKER"         # Removed liquidity
    UNKNOWN = "UNKNOWN"     # Liquidity type unknown


@dataclass
class Fill:
    """Individual fill record"""
    fill_id: str
    order_id: str
    client_order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    
    # Fill details
    fill_quantity: float
    fill_price: float
    leaves_quantity: float  # Remaining unfilled quantity
    
    # Execution details
    venue: str
    execution_id: str
    liquidity_type: LiquidityType = LiquidityType.UNKNOWN
    
    # Fees and costs
    commission: float = 0.0
    sec_fee: float = 0.0
    nscc_fee: float = 0.0
    other_fees: float = 0.0
    
    # Timestamps
    execution_time: datetime = field(default_factory=datetime.now)
    report_time: datetime = field(default_factory=datetime.now)
    
    # Additional data
    contra_broker: Optional[str] = None
    settlement_date: Optional[datetime] = None
    trade_reference: Optional[str] = None
    
    @property
    def total_fees(self) -> float:
        """Total fees for this fill"""
        return self.commission + self.sec_fee + self.nscc_fee + self.other_fees
    
    @property
    def gross_proceeds(self) -> float:
        """Gross proceeds (before fees)"""
        return self.fill_quantity * self.fill_price
    
    @property
    def net_proceeds(self) -> float:
        """Net proceeds (after fees)"""
        if self.side == 'SELL':
            return self.gross_proceeds - self.total_fees
        else:  # BUY
            return -(self.gross_proceeds + self.total_fees)


@dataclass
class OrderFillSummary:
    """Summary of fills for an order"""
    order_id: str
    total_filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    total_fees: float = 0.0
    fill_count: int = 0
    
    # Timing
    first_fill_time: Optional[datetime] = None
    last_fill_time: Optional[datetime] = None
    
    # Performance
    total_slippage: float = 0.0
    price_improvement: float = 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled"""
        return self.fill_count > 0  # Simplified check


@dataclass
class VenueFillStatistics:
    """Fill statistics by venue"""
    venue: str
    total_fills: int = 0
    total_quantity: float = 0.0
    total_notional: float = 0.0
    
    # Performance metrics
    average_fill_size: float = 0.0
    average_fill_time_ms: float = 0.0
    fill_rate: float = 0.0
    
    # Quality metrics
    price_improvement: float = 0.0
    slippage: float = 0.0
    
    # Liquidity metrics
    maker_percentage: float = 0.0
    taker_percentage: float = 0.0
    
    last_update: datetime = field(default_factory=datetime.now)


class FillMatcher:
    """
    Matches incoming fills with orders
    """
    
    def __init__(self):
        self.pending_orders = {}  # order_id -> order_info
        self.fill_history = {}    # order_id -> list of fills
        self.matching_rules = []
        
        # Matching statistics
        self.match_stats = {
            'successful_matches': 0,
            'failed_matches': 0,
            'orphaned_fills': 0,
            'duplicate_fills': 0
        }
    
    def add_pending_order(
        self,
        order_id: str,
        client_order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None
    ) -> None:
        """Add order to pending list for fill matching"""
        
        self.pending_orders[order_id] = {
            'client_order_id': client_order_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'filled_quantity': 0.0,
            'remaining_quantity': quantity,
            'created_time': datetime.now()
        }
        
        self.fill_history[order_id] = []
    
    def match_fill(self, fill: Fill) -> Tuple[bool, str]:
        """
        Match fill to pending order
        
        Args:
            fill: Fill to match
        
        Returns:
            (success, message)
        """
        
        # Check if order exists
        if fill.order_id not in self.pending_orders:
            # Try to match by client order ID
            matched_order_id = self._match_by_client_id(fill.client_order_id)
            
            if matched_order_id:
                fill.order_id = matched_order_id
            else:
                self.match_stats['orphaned_fills'] += 1
                return False, f"No matching order found for fill {fill.fill_id}"
        
        order = self.pending_orders[fill.order_id]
        
        # Validate fill details
        validation_result = self._validate_fill(fill, order)
        if not validation_result[0]:
            self.match_stats['failed_matches'] += 1
            return validation_result
        
        # Check for duplicate fills
        if self._is_duplicate_fill(fill):
            self.match_stats['duplicate_fills'] += 1
            return False, f"Duplicate fill detected: {fill.fill_id}"
        
        # Update order with fill
        order['filled_quantity'] += fill.fill_quantity
        order['remaining_quantity'] -= fill.fill_quantity
        
        # Ensure remaining quantity doesn't go negative
        order['remaining_quantity'] = max(0, order['remaining_quantity'])
        
        # Store fill
        self.fill_history[fill.order_id].append(fill)
        
        # Update statistics
        self.match_stats['successful_matches'] += 1
        
        return True, "Fill matched successfully"
    
    def get_order_fills(self, order_id: str) -> List[Fill]:
        """Get all fills for an order"""
        return self.fill_history.get(order_id, [])
    
    def get_order_fill_summary(self, order_id: str) -> Optional[OrderFillSummary]:
        """Get fill summary for an order"""
        
        if order_id not in self.fill_history:
            return None
        
        fills = self.fill_history[order_id]
        
        if not fills:
            return None
        
        # Calculate summary
        total_quantity = sum(f.fill_quantity for f in fills)
        total_notional = sum(f.gross_proceeds for f in fills)
        total_fees = sum(f.total_fees for f in fills)
        
        avg_price = total_notional / total_quantity if total_quantity > 0 else 0.0
        
        summary = OrderFillSummary(
            order_id=order_id,
            total_filled_quantity=total_quantity,
            average_fill_price=avg_price,
            total_fees=total_fees,
            fill_count=len(fills),
            first_fill_time=fills[0].execution_time,
            last_fill_time=fills[-1].execution_time
        )
        
        # Calculate performance metrics
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            
            if order['price']:  # Limit order
                if order['side'] == 'BUY':
                    summary.price_improvement = max(0, order['price'] - avg_price)
                    summary.total_slippage = max(0, avg_price - order['price'])
                else:  # SELL
                    summary.price_improvement = max(0, avg_price - order['price'])
                    summary.total_slippage = max(0, order['price'] - avg_price)
        
        return summary
    
    def _match_by_client_id(self, client_order_id: str) -> Optional[str]:
        """Match fill by client order ID"""
        
        for order_id, order in self.pending_orders.items():
            if order['client_order_id'] == client_order_id:
                return order_id
        
        return None
    
    def _validate_fill(self, fill: Fill, order: Dict) -> Tuple[bool, str]:
        """Validate fill against order"""
        
        # Check symbol match
        if fill.symbol != order['symbol']:
            return False, "Symbol mismatch"
        
        # Check side match
        if fill.side != order['side']:
            return False, "Side mismatch"
        
        # Check quantity doesn't exceed remaining
        if fill.fill_quantity > order['remaining_quantity'] + 1e-6:  # Small tolerance
            return False, "Fill quantity exceeds remaining quantity"
        
        # Check price reasonableness (if limit order)
        if order['price']:
            if order['side'] == 'BUY' and fill.fill_price > order['price'] * 1.01:  # 1% tolerance
                return False, "Fill price too high for buy limit order"
            elif order['side'] == 'SELL' and fill.fill_price < order['price'] * 0.99:
                return False, "Fill price too low for sell limit order"
        
        return True, "Fill validation passed"
    
    def _is_duplicate_fill(self, fill: Fill) -> bool:
        """Check if fill is a duplicate"""
        
        if fill.order_id not in self.fill_history:
            return False
        
        existing_fills = self.fill_history[fill.order_id]
        
        for existing_fill in existing_fills:
            if (existing_fill.fill_id == fill.fill_id or
                (existing_fill.execution_id == fill.execution_id and
                 existing_fill.execution_time == fill.execution_time and
                 existing_fill.fill_quantity == fill.fill_quantity and
                 existing_fill.fill_price == fill.fill_price)):
                return True
        
        return False


class FillManager:
    """
    Main fill management system
    Processes and manages execution fills
    """
    
    def __init__(self, name: str = "FillManager"):
        self.name = name
        
        # Core components
        self.fill_matcher = FillMatcher()
        
        # Fill storage and tracking
        self.all_fills: Dict[str, Fill] = {}
        self.fills_by_order: Dict[str, List[Fill]] = defaultdict(list)
        self.fills_by_symbol: Dict[str, List[Fill]] = defaultdict(list)
        self.fills_by_venue: Dict[str, List[Fill]] = defaultdict(list)
        
        # Processing queues
        self.incoming_fills = queue.Queue()
        self.processed_fills = queue.Queue()
        
        # Statistics and analytics
        self.venue_statistics: Dict[str, VenueFillStatistics] = {}
        self.daily_statistics = defaultdict(float)
        
        # Event handling
        self.fill_event_handlers = []
        self.order_complete_handlers = []
        
        # Threading
        self.is_running = False
        self.processing_thread = None
        self.lock = threading.RLock()
        
        # Performance metrics
        self.metrics = {
            'total_fills_processed': 0,
            'total_notional_processed': 0.0,
            'processing_errors': 0,
            'average_processing_time_ms': 0.0,
            'last_fill_time': None
        }
    
    def start(self) -> None:
        """Start fill manager"""
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="FillProcessingLoop"
        )
        self.processing_thread.start()
        
        print(f"Fill Manager {self.name} started")
    
    def stop(self) -> None:
        """Stop fill manager"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        print(f"Fill Manager {self.name} stopped")
    
    def add_order(
        self,
        order_id: str,
        client_order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None
    ) -> None:
        """Add order for fill matching"""
        
        with self.lock:
            self.fill_matcher.add_pending_order(
                order_id, client_order_id, symbol, side, quantity, price
            )
    
    def process_fill(self, fill: Fill) -> bool:
        """
        Queue fill for processing
        
        Args:
            fill: Fill to process
        
        Returns:
            True if queued successfully
        """
        
        try:
            self.incoming_fills.put(fill, timeout=1.0)
            return True
        except queue.Full:
            print("Fill processing queue is full")
            return False
    
    def create_fill_from_execution(
        self,
        order_id: str,
        client_order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        venue: str,
        execution_id: str,
        execution_time: Optional[datetime] = None,
        leaves_quantity: Optional[float] = None,
        **kwargs
    ) -> Fill:
        """
        Create fill object from execution data
        
        Args:
            order_id: Order identifier
            client_order_id: Client order identifier
            symbol: Trading symbol
            side: Order side
            quantity: Fill quantity
            price: Fill price
            venue: Execution venue
            execution_id: Execution identifier
            execution_time: Execution timestamp
            leaves_quantity: Remaining quantity
            **kwargs: Additional fill parameters
        
        Returns:
            Fill object
        """
        
        if execution_time is None:
            execution_time = datetime.now()
        
        if leaves_quantity is None:
            leaves_quantity = 0.0  # Assume complete fill if not specified
        
        fill = Fill(
            fill_id=str(uuid.uuid4()),
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            fill_quantity=quantity,
            fill_price=price,
            leaves_quantity=leaves_quantity,
            venue=venue,
            execution_id=execution_id,
            execution_time=execution_time,
            **kwargs
        )
        
        return fill
    
    def get_fills_for_order(self, order_id: str) -> List[Fill]:
        """Get all fills for an order"""
        with self.lock:
            return self.fills_by_order.get(order_id, []).copy()
    
    def get_order_fill_summary(self, order_id: str) -> Optional[OrderFillSummary]:
        """Get fill summary for an order"""
        with self.lock:
            return self.fill_matcher.get_order_fill_summary(order_id)
    
    def get_fills_by_symbol(self, symbol: str) -> List[Fill]:
        """Get all fills for a symbol"""
        with self.lock:
            return self.fills_by_symbol.get(symbol, []).copy()
    
    def get_fills_by_venue(self, venue: str) -> List[Fill]:
        """Get all fills for a venue"""
        with self.lock:
            return self.fills_by_venue.get(venue, []).copy()
    
    def get_venue_statistics(self, venue: str) -> Optional[VenueFillStatistics]:
        """Get statistics for a venue"""
        with self.lock:
            return self.venue_statistics.get(venue)
    
    def get_daily_statistics(self) -> Dict[str, float]:
        """Get daily fill statistics"""
        with self.lock:
            return dict(self.daily_statistics)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get fill manager performance metrics"""
        
        with self.lock:
            total_fills = self.metrics['total_fills_processed']
            
            return {
                'total_fills_processed': total_fills,
                'total_notional_processed': self.metrics['total_notional_processed'],
                'processing_errors': self.metrics['processing_errors'],
                'average_processing_time_ms': self.metrics['average_processing_time_ms'],
                'last_fill_time': self.metrics['last_fill_time'],
                'match_statistics': self.fill_matcher.match_stats.copy(),
                'venue_count': len(self.venue_statistics),
                'unique_symbols': len(self.fills_by_symbol)
            }
    
    def add_fill_event_handler(self, handler: Callable) -> None:
        """Add fill event handler"""
        self.fill_event_handlers.append(handler)
    
    def add_order_complete_handler(self, handler: Callable) -> None:
        """Add order complete event handler"""
        self.order_complete_handlers.append(handler)
    
    def _processing_loop(self) -> None:
        """Main fill processing loop"""
        
        while self.is_running:
            try:
                # Get next fill to process
                try:
                    fill = self.incoming_fills.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the fill
                self._process_single_fill(fill)
                
            except Exception as e:
                print(f"Error in fill processing loop: {e}")
                self.metrics['processing_errors'] += 1
    
    def _process_single_fill(self, fill: Fill) -> None:
        """Process a single fill"""
        
        start_time = datetime.now()
        
        with self.lock:
            try:
                # Match fill to order
                match_success, match_message = self.fill_matcher.match_fill(fill)
                
                if not match_success:
                    print(f"Fill matching failed: {match_message}")
                    return
                
                # Store fill
                self.all_fills[fill.fill_id] = fill
                self.fills_by_order[fill.order_id].append(fill)
                self.fills_by_symbol[fill.symbol].append(fill)
                self.fills_by_venue[fill.venue].append(fill)
                
                # Update venue statistics
                self._update_venue_statistics(fill)
                
                # Update daily statistics
                self._update_daily_statistics(fill)
                
                # Update metrics
                self.metrics['total_fills_processed'] += 1
                self.metrics['total_notional_processed'] += fill.gross_proceeds
                self.metrics['last_fill_time'] = fill.execution_time
                
                # Calculate processing time
                processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                # Update average processing time
                total_fills = self.metrics['total_fills_processed']
                current_avg = self.metrics['average_processing_time_ms']
                
                self.metrics['average_processing_time_ms'] = (
                    (current_avg * (total_fills - 1) + processing_time_ms) / total_fills
                )
                
                # Emit fill event
                self._emit_fill_event("FILL_PROCESSED", fill)
                
                # Check if order is complete
                order_summary = self.fill_matcher.get_order_fill_summary(fill.order_id)
                
                if order_summary and fill.leaves_quantity == 0:
                    self._emit_order_complete_event(fill.order_id, order_summary)
                
                # Put in processed queue
                try:
                    self.processed_fills.put(fill, timeout=0.1)
                except queue.Full:
                    pass  # Processed queue full, continue
                
            except Exception as e:
                print(f"Error processing fill {fill.fill_id}: {e}")
                self.metrics['processing_errors'] += 1
    
    def _update_venue_statistics(self, fill: Fill) -> None:
        """Update venue-level statistics"""
        
        venue = fill.venue
        
        if venue not in self.venue_statistics:
            self.venue_statistics[venue] = VenueFillStatistics(venue=venue)
        
        stats = self.venue_statistics[venue]
        
        # Update counts and totals
        stats.total_fills += 1
        stats.total_quantity += fill.fill_quantity
        stats.total_notional += fill.gross_proceeds
        
        # Update averages
        stats.average_fill_size = stats.total_quantity / stats.total_fills
        
        # Update liquidity metrics
        if fill.liquidity_type == LiquidityType.MAKER:
            maker_count = stats.maker_percentage * stats.total_fills + 1
            stats.maker_percentage = maker_count / stats.total_fills
            stats.taker_percentage = 1.0 - stats.maker_percentage
        elif fill.liquidity_type == LiquidityType.TAKER:
            taker_count = stats.taker_percentage * stats.total_fills + 1
            stats.taker_percentage = taker_count / stats.total_fills
            stats.maker_percentage = 1.0 - stats.taker_percentage
        
        stats.last_update = datetime.now()
    
    def _update_daily_statistics(self, fill: Fill) -> None:
        """Update daily statistics"""
        
        today = datetime.now().date()
        key_prefix = f"{today}_"
        
        self.daily_statistics[f"{key_prefix}total_fills"] += 1
        self.daily_statistics[f"{key_prefix}total_quantity"] += fill.fill_quantity
        self.daily_statistics[f"{key_prefix}total_notional"] += fill.gross_proceeds
        self.daily_statistics[f"{key_prefix}total_fees"] += fill.total_fees
        
        # Symbol-specific stats
        symbol_key = f"{key_prefix}{fill.symbol}_"
        self.daily_statistics[f"{symbol_key}fills"] += 1
        self.daily_statistics[f"{symbol_key}quantity"] += fill.fill_quantity
        self.daily_statistics[f"{symbol_key}notional"] += fill.gross_proceeds
        
        # Venue-specific stats
        venue_key = f"{key_prefix}{fill.venue}_"
        self.daily_statistics[f"{venue_key}fills"] += 1
        self.daily_statistics[f"{venue_key}quantity"] += fill.fill_quantity
        self.daily_statistics[f"{venue_key}notional"] += fill.gross_proceeds
    
    def _emit_fill_event(self, event_type: str, fill: Fill) -> None:
        """Emit fill event"""
        
        event = {
            'type': event_type,
            'fill': fill,
            'timestamp': datetime.now()
        }
        
        for handler in self.fill_event_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in fill event handler: {e}")
    
    def _emit_order_complete_event(self, order_id: str, summary: OrderFillSummary) -> None:
        """Emit order complete event"""
        
        event = {
            'type': 'ORDER_COMPLETE',
            'order_id': order_id,
            'summary': summary,
            'timestamp': datetime.now()
        }
        
        for handler in self.order_complete_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in order complete handler: {e}")


if __name__ == "__main__":
    import time
    
    # Example usage and testing
    print("Testing Fill Manager...")
    
    # Create fill manager
    fm = FillManager("TestFM")
    
    # Add event handlers
    def fill_event_handler(event):
        fill = event['fill']
        print(f"Fill Event: {event['type']} - {fill.symbol} "
              f"{fill.fill_quantity} @ {fill.fill_price:.4f}")
    
    def order_complete_handler(event):
        summary = event['summary']
        print(f"Order Complete: {event['order_id']} - "
              f"Filled: {summary.total_filled_quantity} @ {summary.average_fill_price:.4f}")
    
    fm.add_fill_event_handler(fill_event_handler)
    fm.add_order_complete_handler(order_complete_handler)
    
    # Start fill manager
    fm.start()
    
    try:
        # Add test orders
        print("\nAdding test orders...")
        
        fm.add_order(
            order_id="ORDER_001",
            client_order_id="CLIENT_001",
            symbol="AAPL",
            side="BUY",
            quantity=1000,
            price=150.00
        )
        
        fm.add_order(
            order_id="ORDER_002",
            client_order_id="CLIENT_002",
            symbol="MSFT",
            side="SELL",
            quantity=500,
            price=300.00
        )
        
        # Simulate fills
        print("\nSimulating fills...")
        
        # Partial fill for first order
        fill1 = fm.create_fill_from_execution(
            order_id="ORDER_001",
            client_order_id="CLIENT_001",
            symbol="AAPL",
            side="BUY",
            quantity=400,
            price=149.95,
            venue="NASDAQ",
            execution_id="EXEC_001",
            leaves_quantity=600,
            commission=2.00,
            liquidity_type=LiquidityType.TAKER
        )
        
        fm.process_fill(fill1)
        
        time.sleep(1)
        
        # Complete fill for first order
        fill2 = fm.create_fill_from_execution(
            order_id="ORDER_001",
            client_order_id="CLIENT_001",
            symbol="AAPL",
            side="BUY",
            quantity=600,
            price=150.02,
            venue="ARCA",
            execution_id="EXEC_002",
            leaves_quantity=0,
            commission=3.00,
            liquidity_type=LiquidityType.MAKER
        )
        
        fm.process_fill(fill2)
        
        # Full fill for second order
        fill3 = fm.create_fill_from_execution(
            order_id="ORDER_002",
            client_order_id="CLIENT_002",
            symbol="MSFT",
            side="SELL",
            quantity=500,
            price=300.25,
            venue="NYSE",
            execution_id="EXEC_003",
            leaves_quantity=0,
            commission=2.50,
            liquidity_type=LiquidityType.TAKER
        )
        
        fm.process_fill(fill3)
        
        time.sleep(2)
        
        # Check fill summaries
        print(f"\nFill Summaries:")
        
        for order_id in ["ORDER_001", "ORDER_002"]:
            summary = fm.get_order_fill_summary(order_id)
            
            if summary:
                print(f"  {order_id}:")
                print(f"    Filled Quantity: {summary.total_filled_quantity}")
                print(f"    Average Price: {summary.average_fill_price:.4f}")
                print(f"    Total Fees: ${summary.total_fees:.2f}")
                print(f"    Fill Count: {summary.fill_count}")
                print(f"    Price Improvement: ${summary.price_improvement:.4f}")
        
        # Venue statistics
        print(f"\nVenue Statistics:")
        
        for venue in ["NASDAQ", "ARCA", "NYSE"]:
            stats = fm.get_venue_statistics(venue)
            
            if stats:
                print(f"  {venue}:")
                print(f"    Total Fills: {stats.total_fills}")
                print(f"    Total Quantity: {stats.total_quantity}")
                print(f"    Average Fill Size: {stats.average_fill_size:.0f}")
                print(f"    Maker %: {stats.maker_percentage:.1%}")
                print(f"    Taker %: {stats.taker_percentage:.1%}")
        
        # Daily statistics
        print(f"\nDaily Statistics:")
        daily_stats = fm.get_daily_statistics()
        
        today = datetime.now().date()
        total_fills = daily_stats.get(f"{today}_total_fills", 0)
        total_notional = daily_stats.get(f"{today}_total_notional", 0)
        total_fees = daily_stats.get(f"{today}_total_fees", 0)
        
        print(f"  Total Fills: {total_fills}")
        print(f"  Total Notional: ${total_notional:.2f}")
        print(f"  Total Fees: ${total_fees:.2f}")
        
        # Performance metrics
        print(f"\nPerformance Metrics:")
        metrics = fm.get_performance_metrics()
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        # Test error handling
        print(f"\nTesting error handling...")
        
        # Try to process fill for non-existent order
        bad_fill = fm.create_fill_from_execution(
            order_id="NONEXISTENT",
            client_order_id="BAD_CLIENT",
            symbol="XYZ",
            side="BUY",
            quantity=100,
            price=50.00,
            venue="TEST",
            execution_id="BAD_EXEC"
        )
        
        fm.process_fill(bad_fill)
        
        time.sleep(1)
        
        # Check error metrics
        final_metrics = fm.get_performance_metrics()
        print(f"Processing errors: {final_metrics['processing_errors']}")
        print(f"Match statistics: {final_metrics['match_statistics']}")
        
    finally:
        # Stop fill manager
        fm.stop()
    
    print("\nFill Manager testing completed!")