"""
High Frequency Execution Engine
Ultra-fast execution system for high-frequency trading strategies
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import struct
import mmap
import ctypes
import os
import tempfile
from collections import deque, defaultdict, namedtuple
import warnings


class ExecutionSpeed(Enum):
    """Execution speed categories"""
    NANOSECOND = "NANOSECOND"     # < 100ns target
    MICROSECOND = "MICROSECOND"   # < 10μs target  
    MILLISECOND = "MILLISECOND"   # < 1ms target
    STANDARD = "STANDARD"         # < 100ms target


class MarketDataType(Enum):
    """Market data types for HFT"""
    LEVEL1 = "LEVEL1"             # Best bid/offer
    LEVEL2 = "LEVEL2"             # Order book depth
    TRADES = "TRADES"             # Trade ticks
    IMBALANCE = "IMBALANCE"       # Order imbalances
    AUCTION = "AUCTION"           # Auction data


@dataclass
class MarketTick:
    """High-resolution market data tick"""
    symbol: str
    timestamp_ns: int  # Nanosecond precision
    
    # Level 1 data
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    
    # Trade data
    last_price: Optional[float] = None
    last_size: Optional[float] = None
    
    # Derived fields
    spread: float = field(init=False)
    mid_price: float = field(init=False)
    
    def __post_init__(self):
        self.spread = self.ask_price - self.bid_price
        self.mid_price = (self.bid_price + self.ask_price) / 2.0
    
    @property
    def timestamp_us(self) -> float:
        """Timestamp in microseconds"""
        return self.timestamp_ns / 1000.0
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points"""
        if self.mid_price > 0:
            return (self.spread / self.mid_price) * 10000
        return 0.0


@dataclass
class HFTSignal:
    """High-frequency trading signal"""
    signal_id: str
    symbol: str
    timestamp_ns: int
    
    # Signal properties
    signal_type: str  # 'MOMENTUM', 'MEAN_REVERSION', 'ARBITRAGE', etc.
    direction: str    # 'BUY', 'SELL', 'NEUTRAL'
    strength: float   # Signal strength [0.0, 1.0]
    confidence: float # Signal confidence [0.0, 1.0]
    
    # Execution parameters
    urgency: ExecutionSpeed = ExecutionSpeed.MICROSECOND
    max_quantity: float = 1000.0
    price_limit: Optional[float] = None
    time_horizon_ms: float = 100.0  # Signal validity period
    
    # Risk parameters
    max_adverse_selection_bps: float = 5.0
    max_market_impact_bps: float = 2.0
    
    @property
    def is_expired(self) -> bool:
        """Check if signal has expired"""
        current_time_ns = time.time_ns()
        age_ms = (current_time_ns - self.timestamp_ns) / 1_000_000
        return age_ms > self.time_horizon_ms
    
    @property
    def age_us(self) -> float:
        """Signal age in microseconds"""
        current_time_ns = time.time_ns()
        return (current_time_ns - self.timestamp_ns) / 1000.0


@dataclass
class ExecutionMetrics:
    """Execution performance metrics"""
    order_id: str
    
    # Timestamps (nanoseconds)
    signal_timestamp_ns: int
    decision_timestamp_ns: int
    order_sent_timestamp_ns: int
    ack_timestamp_ns: Optional[int] = None
    fill_timestamp_ns: Optional[int] = None
    
    # Latencies (microseconds)
    decision_latency_us: Optional[float] = None
    order_latency_us: Optional[float] = None
    ack_latency_us: Optional[float] = None
    fill_latency_us: Optional[float] = None
    total_latency_us: Optional[float] = None
    
    # Performance metrics
    slippage_bps: Optional[float] = None
    market_impact_bps: Optional[float] = None
    implementation_shortfall_bps: Optional[float] = None
    
    def calculate_latencies(self) -> None:
        """Calculate all latency metrics"""
        
        if self.decision_timestamp_ns:
            self.decision_latency_us = (
                self.decision_timestamp_ns - self.signal_timestamp_ns
            ) / 1000.0
        
        if self.order_sent_timestamp_ns:
            self.order_latency_us = (
                self.order_sent_timestamp_ns - self.decision_timestamp_ns
            ) / 1000.0
        
        if self.ack_timestamp_ns and self.order_sent_timestamp_ns:
            self.ack_latency_us = (
                self.ack_timestamp_ns - self.order_sent_timestamp_ns
            ) / 1000.0
        
        if self.fill_timestamp_ns and self.order_sent_timestamp_ns:
            self.fill_latency_us = (
                self.fill_timestamp_ns - self.order_sent_timestamp_ns
            ) / 1000.0
        
        if self.fill_timestamp_ns:
            self.total_latency_us = (
                self.fill_timestamp_ns - self.signal_timestamp_ns
            ) / 1000.0


class LowLatencyBuffer:
    """
    Lock-free circular buffer for low-latency data
    Uses memory-mapped files and atomic operations
    """
    
    def __init__(self, buffer_size: int = 1024):
        self.buffer_size = buffer_size
        self.element_size = 1024  # Fixed size per element
        
        # Create memory-mapped buffer
        self.buffer_file = os.path.join(tempfile.gettempdir(), f"hft_buffer_{os.getpid()}_{id(self)}")
        self.total_size = self.buffer_size * self.element_size
        
        # Initialize memory map
        with open(self.buffer_file, 'wb') as f:
            f.write(b'\x00' * self.total_size)
        
        self.mmap_file = open(self.buffer_file, 'r+b')
        self.buffer = mmap.mmap(self.mmap_file.fileno(), 0)
        
        # Atomic counters (simplified - real implementation would use platform-specific atomics)
        self.write_index = 0
        self.read_index = 0
        self.lock = threading.RLock()  # Fallback lock
    
    def write(self, data: bytes) -> bool:
        """Write data to buffer (lock-free)"""
        
        if len(data) > self.element_size:
            return False
        
        with self.lock:  # Simplified locking
            next_write = (self.write_index + 1) % self.buffer_size
            
            if next_write == self.read_index:
                return False  # Buffer full
            
            # Write data
            offset = self.write_index * self.element_size
            self.buffer[offset:offset + len(data)] = data
            
            # Pad remaining bytes
            remaining = self.element_size - len(data)
            if remaining > 0:
                self.buffer[offset + len(data):offset + self.element_size] = b'\x00' * remaining
            
            # Update write index
            self.write_index = next_write
            
            return True
    
    def read(self) -> Optional[bytes]:
        """Read data from buffer (lock-free)"""
        
        with self.lock:  # Simplified locking
            if self.read_index == self.write_index:
                return None  # Buffer empty
            
            # Read data
            offset = self.read_index * self.element_size
            data = bytes(self.buffer[offset:offset + self.element_size])
            
            # Find actual data length (strip null bytes)
            actual_length = len(data.rstrip(b'\x00'))
            data = data[:actual_length]
            
            # Update read index
            self.read_index = (self.read_index + 1) % self.buffer_size
            
            return data
    
    def close(self) -> None:
        """Close and cleanup buffer"""
        
        if hasattr(self, 'buffer'):
            self.buffer.close()
        
        if hasattr(self, 'mmap_file'):
            self.mmap_file.close()
        
        try:
            os.unlink(self.buffer_file)
        except OSError:
            pass


class SignalProcessor:
    """
    Real-time signal processing for HFT
    """
    
    def __init__(self):
        self.signal_handlers: Dict[str, Callable] = {}
        self.active_signals: Dict[str, HFTSignal] = {}
        
        # Performance tracking
        self.processing_times: deque = deque(maxlen=10000)
        self.signal_counts: defaultdict = defaultdict(int)
        
        # Threading
        self.lock = threading.RLock()
    
    def register_signal_handler(self, signal_type: str, handler: Callable) -> None:
        """Register handler for signal type"""
        self.signal_handlers[signal_type] = handler
    
    def process_signal(self, signal: HFTSignal) -> Tuple[bool, str]:
        """
        Process incoming HFT signal
        
        Args:
            signal: HFT signal to process
        
        Returns:
            (success, message)
        """
        
        start_time_ns = time.time_ns()
        
        try:
            # Check if signal is expired
            if signal.is_expired:
                return False, "Signal expired"
            
            # Check signal strength and confidence thresholds
            if signal.strength < 0.3 or signal.confidence < 0.5:
                return False, "Signal below threshold"
            
            # Store active signal
            with self.lock:
                self.active_signals[signal.signal_id] = signal
                self.signal_counts[signal.signal_type] += 1
            
            # Route to appropriate handler
            handler = self.signal_handlers.get(signal.signal_type)
            
            if handler:
                try:
                    result = handler(signal)
                    
                    # Record processing time
                    processing_time_ns = time.time_ns() - start_time_ns
                    self.processing_times.append(processing_time_ns / 1000.0)  # Convert to μs
                    
                    return result
                    
                except Exception as e:
                    return False, f"Handler error: {e}"
            else:
                return False, f"No handler for signal type: {signal.signal_type}"
        
        except Exception as e:
            return False, f"Signal processing error: {e}"
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[HFTSignal]:
        """Get active signals, optionally filtered by symbol"""
        
        with self.lock:
            signals = list(self.active_signals.values())
            
            if symbol:
                signals = [s for s in signals if s.symbol == symbol]
            
            # Remove expired signals
            current_signals = []
            for signal in signals:
                if not signal.is_expired:
                    current_signals.append(signal)
                else:
                    # Remove expired signal
                    self.active_signals.pop(signal.signal_id, None)
            
            return current_signals
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get signal processing statistics"""
        
        with self.lock:
            stats = {
                'total_signals_processed': sum(self.signal_counts.values()),
                'signals_by_type': dict(self.signal_counts),
                'active_signals_count': len(self.active_signals)
            }
            
            if self.processing_times:
                times_array = np.array(self.processing_times)
                stats.update({
                    'average_processing_time_us': np.mean(times_array),
                    'p50_processing_time_us': np.percentile(times_array, 50),
                    'p95_processing_time_us': np.percentile(times_array, 95),
                    'p99_processing_time_us': np.percentile(times_array, 99),
                    'max_processing_time_us': np.max(times_array)
                })
            
            return stats


class HighFrequencyExecutor:
    """
    Ultra-fast order execution engine
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Market data
        self.current_market_tick: Optional[MarketTick] = None
        self.tick_history: deque = deque(maxlen=1000)
        
        # Execution state
        self.pending_orders: Dict[str, Dict] = {}
        self.execution_metrics: Dict[str, ExecutionMetrics] = {}
        
        # Performance optimization
        self.low_latency_buffer = LowLatencyBuffer()
        
        # Risk controls
        self.max_position = 10000.0
        self.current_position = 0.0
        self.max_order_size = 1000.0
        self.min_spread_bps = 1.0
        
        # Execution statistics
        self.execution_stats = {
            'total_orders': 0,
            'successful_executions': 0,
            'rejections': 0,
            'average_execution_time_us': 0.0,
            'total_pnl': 0.0
        }
        
        # Threading
        self.lock = threading.RLock()
    
    def update_market_data(self, tick: MarketTick) -> None:
        """Update market data (ultra-low latency)"""
        
        # Minimal locking for maximum speed
        self.current_market_tick = tick
        self.tick_history.append(tick)
        
        # Trigger any pending market-dependent executions
        self._check_pending_executions()
    
    def execute_signal(self, signal: HFTSignal) -> Tuple[bool, str, Optional[str]]:
        """
        Execute HFT signal with ultra-low latency
        
        Args:
            signal: HFT signal to execute
        
        Returns:
            (success, message, order_id)
        """
        
        execution_start_ns = time.time_ns()
        
        try:
            # Pre-execution validation
            validation_result = self._validate_signal_execution(signal)
            if not validation_result[0]:
                return validation_result + (None,)
            
            # Generate order ID
            order_id = f"HFT_{signal.symbol}_{int(execution_start_ns)}"
            
            # Create execution metrics
            metrics = ExecutionMetrics(
                order_id=order_id,
                signal_timestamp_ns=signal.timestamp_ns,
                decision_timestamp_ns=execution_start_ns,
                order_sent_timestamp_ns=0  # Will be set when order is sent
            )
            
            self.execution_metrics[order_id] = metrics
            
            # Determine execution strategy based on urgency
            if signal.urgency == ExecutionSpeed.NANOSECOND:
                success, message = self._execute_nanosecond(signal, order_id, metrics)
            elif signal.urgency == ExecutionSpeed.MICROSECOND:
                success, message = self._execute_microsecond(signal, order_id, metrics)
            elif signal.urgency == ExecutionSpeed.MILLISECOND:
                success, message = self._execute_millisecond(signal, order_id, metrics)
            else:
                success, message = self._execute_standard(signal, order_id, metrics)
            
            # Update statistics
            with self.lock:
                self.execution_stats['total_orders'] += 1
                
                if success:
                    self.execution_stats['successful_executions'] += 1
                else:
                    self.execution_stats['rejections'] += 1
                
                # Update average execution time
                total_orders = self.execution_stats['total_orders']
                current_avg = self.execution_stats['average_execution_time_us']
                execution_time_us = (time.time_ns() - execution_start_ns) / 1000.0
                
                self.execution_stats['average_execution_time_us'] = (
                    (current_avg * (total_orders - 1) + execution_time_us) / total_orders
                )
            
            return success, message, order_id if success else None
            
        except Exception as e:
            return False, f"Execution error: {e}", None
    
    def _validate_signal_execution(self, signal: HFTSignal) -> Tuple[bool, str]:
        """Validate signal before execution"""
        
        # Check if we have current market data
        if not self.current_market_tick:
            return False, "No market data available"
        
        # Check signal age
        if signal.is_expired:
            return False, "Signal expired"
        
        # Check spread requirements
        if self.current_market_tick.spread_bps < self.min_spread_bps:
            return False, "Spread too tight"
        
        # Check position limits
        order_quantity = min(signal.max_quantity, self.max_order_size)
        
        if signal.direction == "BUY":
            new_position = self.current_position + order_quantity
        else:
            new_position = self.current_position - order_quantity
        
        if abs(new_position) > self.max_position:
            return False, "Position limit exceeded"
        
        # Check market conditions
        if not self._is_market_suitable_for_execution():
            return False, "Market conditions unsuitable"
        
        return True, "Validation passed"
    
    def _is_market_suitable_for_execution(self) -> bool:
        """Check if market conditions are suitable for execution"""
        
        if not self.current_market_tick:
            return False
        
        # Check for reasonable bid/ask sizes
        if (self.current_market_tick.bid_size < 100 or 
            self.current_market_tick.ask_size < 100):
            return False
        
        # Check for reasonable prices (no crossed market)
        if self.current_market_tick.bid_price >= self.current_market_tick.ask_price:
            return False
        
        # Check for excessive volatility (if we have history)
        if len(self.tick_history) >= 10:
            recent_prices = [tick.mid_price for tick in list(self.tick_history)[-10:]]
            price_std = np.std(recent_prices)
            price_mean = np.mean(recent_prices)
            
            if price_mean > 0:
                volatility = price_std / price_mean
                if volatility > 0.01:  # 1% volatility threshold
                    return False
        
        return True
    
    def _execute_nanosecond(
        self,
        signal: HFTSignal,
        order_id: str,
        metrics: ExecutionMetrics
    ) -> Tuple[bool, str]:
        """Execute with nanosecond-level latency requirements"""
        
        # For nanosecond execution, we need:
        # 1. Direct memory access to market data
        # 2. Pre-compiled execution paths
        # 3. Hardware-level optimizations
        
        try:
            # Simulate ultra-fast execution
            order_sent_ns = time.time_ns()
            metrics.order_sent_timestamp_ns = order_sent_ns
            
            # Determine execution price and quantity
            tick = self.current_market_tick
            
            if signal.direction == "BUY":
                # For nanosecond execution, we take liquidity aggressively
                execution_price = tick.ask_price
                execution_quantity = min(signal.max_quantity, tick.ask_size, self.max_order_size)
            else:
                execution_price = tick.bid_price
                execution_quantity = min(signal.max_quantity, tick.bid_size, self.max_order_size)
            
            # Simulate market response (deterministic midpoint values used for reproducibility)
            # Nanosecond execution typical ack: midpoint of 100-500ns -> 300ns
            ack_delay_ns = 300
            # Fill delay: midpoint of 50-200ns -> 125ns
            fill_delay_ns = ack_delay_ns + 125
            
            # Update metrics
            metrics.ack_timestamp_ns = order_sent_ns + ack_delay_ns
            metrics.fill_timestamp_ns = order_sent_ns + fill_delay_ns
            metrics.calculate_latencies()
            
            # Update position
            with self.lock:
                if signal.direction == "BUY":
                    self.current_position += execution_quantity
                else:
                    self.current_position -= execution_quantity
                
                # Calculate P&L (simplified)
                pnl = self._calculate_execution_pnl(signal, execution_price, execution_quantity)
                self.execution_stats['total_pnl'] += pnl
            
            return True, f"Nanosecond execution: {execution_quantity}@{execution_price:.4f}"
            
        except Exception as e:
            return False, f"Nanosecond execution failed: {e}"
    
    def _execute_microsecond(
        self,
        signal: HFTSignal,
        order_id: str,
        metrics: ExecutionMetrics
    ) -> Tuple[bool, str]:
        """Execute with microsecond-level latency requirements"""
        
        try:
            order_sent_ns = time.time_ns()
            metrics.order_sent_timestamp_ns = order_sent_ns
            
            tick = self.current_market_tick
            
            # For microsecond execution, we can be slightly more sophisticated
            # Consider signal strength for pricing
            if signal.direction == "BUY":
                if signal.strength > 0.8:
                    # High strength - take offer
                    execution_price = tick.ask_price
                else:
                    # Lower strength - try to improve by joining bid
                    execution_price = tick.bid_price
            else:
                if signal.strength > 0.8:
                    # High strength - hit bid
                    execution_price = tick.bid_price
                else:
                    # Lower strength - try to improve by joining offer
                    execution_price = tick.ask_price
            
            execution_quantity = min(signal.max_quantity, self.max_order_size)
            
            # Simulate microsecond execution timing (deterministic midpoints)
            # Ack midpoint 1-5μs -> 3000ns
            ack_delay_ns = 3000
            # Fill additional midpoint 0.5-2μs -> 1250ns
            fill_delay_ns = ack_delay_ns + 1250
            
            metrics.ack_timestamp_ns = order_sent_ns + ack_delay_ns
            metrics.fill_timestamp_ns = order_sent_ns + fill_delay_ns
            metrics.calculate_latencies()
            
            # Update position and P&L
            with self.lock:
                if signal.direction == "BUY":
                    self.current_position += execution_quantity
                else:
                    self.current_position -= execution_quantity
                
                pnl = self._calculate_execution_pnl(signal, execution_price, execution_quantity)
                self.execution_stats['total_pnl'] += pnl
            
            return True, f"Microsecond execution: {execution_quantity}@{execution_price:.4f}"
            
        except Exception as e:
            return False, f"Microsecond execution failed: {e}"
    
    def _execute_millisecond(
        self,
        signal: HFTSignal,
        order_id: str,
        metrics: ExecutionMetrics
    ) -> Tuple[bool, str]:
        """Execute with millisecond-level latency requirements"""
        
        try:
            order_sent_ns = time.time_ns()
            metrics.order_sent_timestamp_ns = order_sent_ns
            
            # For millisecond execution, we can use more complex logic
            execution_result = self._optimize_execution_price(signal)
            
            if not execution_result[0]:
                return execution_result
            
            execution_price, execution_quantity = execution_result[1], execution_result[2]
            
            # Simulate millisecond execution timing (deterministic midpoints)
            # Ack midpoint 0.1-1ms -> 550000ns
            ack_delay_ns = 550_000
            # Fill additional midpoint 50k-500k -> 275000ns
            fill_delay_ns = ack_delay_ns + 275_000
            
            metrics.ack_timestamp_ns = order_sent_ns + ack_delay_ns
            metrics.fill_timestamp_ns = order_sent_ns + fill_delay_ns
            metrics.calculate_latencies()
            
            # Update position and P&L
            with self.lock:
                if signal.direction == "BUY":
                    self.current_position += execution_quantity
                else:
                    self.current_position -= execution_quantity
                
                pnl = self._calculate_execution_pnl(signal, execution_price, execution_quantity)
                self.execution_stats['total_pnl'] += pnl
            
            return True, f"Millisecond execution: {execution_quantity}@{execution_price:.4f}"
            
        except Exception as e:
            return False, f"Millisecond execution failed: {e}"
    
    def _execute_standard(
        self,
        signal: HFTSignal,
        order_id: str,
        metrics: ExecutionMetrics
    ) -> Tuple[bool, str]:
        """Execute with standard latency requirements"""
        
        try:
            order_sent_ns = time.time_ns()
            metrics.order_sent_timestamp_ns = order_sent_ns
            
            # For standard execution, we can use full optimization
            execution_result = self._optimize_execution_with_slicing(signal)
            
            if not execution_result[0]:
                return execution_result
            
            total_quantity, avg_price = execution_result[1], execution_result[2]
            
            # Simulate standard execution timing (deterministic midpoints)
            # Ack midpoint 1-10ms -> 5_500_000ns
            ack_delay_ns = 5_500_000
            # Fill additional midpoint 1-5ms -> 3_000_000ns
            fill_delay_ns = ack_delay_ns + 3_000_000
            
            metrics.ack_timestamp_ns = order_sent_ns + ack_delay_ns
            metrics.fill_timestamp_ns = order_sent_ns + fill_delay_ns
            metrics.calculate_latencies()
            
            # Update position and P&L
            with self.lock:
                if signal.direction == "BUY":
                    self.current_position += total_quantity
                else:
                    self.current_position -= total_quantity
                
                pnl = self._calculate_execution_pnl(signal, avg_price, total_quantity)
                self.execution_stats['total_pnl'] += pnl
            
            return True, f"Standard execution: {total_quantity}@{avg_price:.4f}"
            
        except Exception as e:
            return False, f"Standard execution failed: {e}"
    
    def _optimize_execution_price(self, signal: HFTSignal) -> Tuple[bool, float, float]:
        """Optimize execution price based on signal characteristics"""
        
        if not self.current_market_tick:
            return False, 0.0, 0.0
        
        tick = self.current_market_tick
        
        # Base execution parameters
        if signal.direction == "BUY":
            base_price = tick.ask_price
            best_size = tick.ask_size
        else:
            base_price = tick.bid_price
            best_size = tick.bid_size
        
        # Adjust based on signal strength and market conditions
        price_adjustment = 0.0
        
        if signal.strength > 0.9:
            # Very strong signal - pay up/sell down slightly
            price_adjustment = tick.spread * 0.1 * (1 if signal.direction == "BUY" else -1)
        elif signal.strength < 0.5:
            # Weak signal - try to get better price
            price_adjustment = -tick.spread * 0.2 * (1 if signal.direction == "BUY" else -1)
        
        execution_price = base_price + price_adjustment
        execution_quantity = min(signal.max_quantity, best_size, self.max_order_size)
        
        return True, execution_price, execution_quantity
    
    def _optimize_execution_with_slicing(self, signal: HFTSignal) -> Tuple[bool, float, float]:
        """Optimize execution using order slicing"""
        
        if not self.current_market_tick:
            return False, 0.0, 0.0
        
        tick = self.current_market_tick
        target_quantity = min(signal.max_quantity, self.max_order_size)
        
        # Slice large orders to minimize market impact
        slice_size = min(target_quantity, 500)  # Maximum 500 per slice
        remaining_quantity = target_quantity
        total_executed = 0.0
        total_notional = 0.0
        
        while remaining_quantity > 0:
            current_slice = min(remaining_quantity, slice_size)
            
            if signal.direction == "BUY":
                execution_price = tick.ask_price
                available_size = tick.ask_size
            else:
                execution_price = tick.bid_price
                available_size = tick.bid_size
            
            executed_quantity = min(current_slice, available_size)
            total_executed += executed_quantity
            total_notional += executed_quantity * execution_price
            
            remaining_quantity -= executed_quantity
            
            # Break if we can't get full execution
            if executed_quantity < current_slice:
                break
        
        if total_executed > 0:
            avg_price = total_notional / total_executed
            return True, total_executed, avg_price
        else:
            return False, 0.0, 0.0
    
    def _calculate_execution_pnl(
        self,
        signal: HFTSignal,
        execution_price: float,
        execution_quantity: float
    ) -> float:
        """Calculate execution P&L (simplified)"""
        
        if not self.current_market_tick:
            return 0.0
        
        # Use mid price as fair value reference
        fair_value = self.current_market_tick.mid_price
        
        if signal.direction == "BUY":
            # P&L is negative of price paid above fair value
            pnl = (fair_value - execution_price) * execution_quantity
        else:
            # P&L is positive of price received above fair value
            pnl = (execution_price - fair_value) * execution_quantity
        
        return pnl
    
    def _check_pending_executions(self) -> None:
        """Check and execute any pending market-dependent orders"""
        
        # This would be used for conditional orders that wait for specific market conditions
        # For now, it's a placeholder for future functionality
        pass
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        
        with self.lock:
            stats = self.execution_stats.copy()
            
            # Add current state
            stats.update({
                'current_position': self.current_position,
                'position_utilization': abs(self.current_position) / self.max_position,
                'pending_orders': len(self.pending_orders),
                'tick_history_length': len(self.tick_history)
            })
            
            # Add latency statistics from metrics
            if self.execution_metrics:
                all_metrics = list(self.execution_metrics.values())
                
                decision_latencies = [m.decision_latency_us for m in all_metrics if m.decision_latency_us]
                if decision_latencies:
                    stats['average_decision_latency_us'] = np.mean(decision_latencies)
                    stats['p95_decision_latency_us'] = np.percentile(decision_latencies, 95)
                
                fill_latencies = [m.fill_latency_us for m in all_metrics if m.fill_latency_us]
                if fill_latencies:
                    stats['average_fill_latency_us'] = np.mean(fill_latencies)
                    stats['p95_fill_latency_us'] = np.percentile(fill_latencies, 95)
            
            return stats
    
    def get_order_metrics(self, order_id: str) -> Optional[ExecutionMetrics]:
        """Get metrics for specific order"""
        return self.execution_metrics.get(order_id)
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.low_latency_buffer.close()


class HighFrequencyExecutionEngine:
    """
    Main high-frequency execution engine
    Coordinates multiple executors and signal processing
    """
    
    def __init__(self, name: str = "HFTEngine"):
        self.name = name
        
        # Components
        self.signal_processor = SignalProcessor()
        self.executors: Dict[str, HighFrequencyExecutor] = {}
        
        # Market data management
        self.market_data_feed = queue.Queue(maxsize=10000)
        
        # Performance tracking
        self.engine_stats = {
            'signals_processed': 0,
            'orders_executed': 0,
            'total_pnl': 0.0,
            'uptime_seconds': 0.0,
            'start_time': None
        }
        
        # Threading
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []
        self.lock = threading.RLock()
        
        # Register signal handlers
        self._setup_signal_handlers()
    
    def add_symbol(self, symbol: str) -> None:
        """Add symbol for HFT execution"""
        
        if symbol not in self.executors:
            self.executors[symbol] = HighFrequencyExecutor(symbol)
    
    def start(self) -> None:
        """Start HFT execution engine"""
        
        self.is_running = True
        self.engine_stats['start_time'] = time.time()
        
        # Start market data processing thread
        market_data_thread = threading.Thread(
            target=self._market_data_loop,
            name="HFT-MarketData"
        )
        market_data_thread.start()
        self.worker_threads.append(market_data_thread)
        
        print(f"High Frequency Execution Engine {self.name} started")
    
    def stop(self) -> None:
        """Stop HFT execution engine"""
        
        self.is_running = False
        
        # Stop worker threads
        for thread in self.worker_threads:
            thread.join(timeout=2.0)
        
        # Cleanup executors
        for executor in self.executors.values():
            executor.cleanup()
        
        # Update uptime
        if self.engine_stats['start_time']:
            self.engine_stats['uptime_seconds'] = time.time() - self.engine_stats['start_time']
        
        print(f"High Frequency Execution Engine {self.name} stopped")
    
    def process_market_data(self, tick: MarketTick) -> None:
        """Process incoming market data"""
        
        try:
            self.market_data_feed.put(tick, timeout=0.001)  # 1ms timeout
        except queue.Full:
            # Drop old data if queue is full
            try:
                self.market_data_feed.get_nowait()
                self.market_data_feed.put(tick, timeout=0.001)
            except (queue.Empty, queue.Full):
                pass  # Continue if still can't queue
    
    def process_signal(self, signal: HFTSignal) -> Tuple[bool, str, Optional[str]]:
        """
        Process HFT signal for execution
        
        Args:
            signal: HFT signal to process
        
        Returns:
            (success, message, order_id)
        """
        
        try:
            # Process through signal processor
            process_result = self.signal_processor.process_signal(signal)
            
            if not process_result[0]:
                return process_result + (None,)
            
            # Update engine statistics
            with self.lock:
                self.engine_stats['signals_processed'] += 1
            
            return process_result + (None,)  # Signal processing doesn't generate order_id
            
        except Exception as e:
            return False, f"Signal processing error: {e}", None
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics"""
        
        with self.lock:
            stats = self.engine_stats.copy()
            
            # Update uptime
            if stats['start_time']:
                stats['uptime_seconds'] = time.time() - stats['start_time']
            
            # Add signal processor statistics
            stats['signal_processor'] = self.signal_processor.get_processing_statistics()
            
            # Add executor statistics
            stats['executors'] = {}
            total_pnl = 0.0
            total_orders = 0
            
            for symbol, executor in self.executors.items():
                executor_stats = executor.get_execution_statistics()
                stats['executors'][symbol] = executor_stats
                total_pnl += executor_stats.get('total_pnl', 0.0)
                total_orders += executor_stats.get('total_orders', 0)
            
            stats['total_pnl'] = total_pnl
            stats['orders_executed'] = total_orders
            
            return stats
    
    def get_symbol_statistics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get statistics for specific symbol"""
        
        executor = self.executors.get(symbol)
        if executor:
            return executor.get_execution_statistics()
        return None
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for different signal types"""
        
        def momentum_handler(signal: HFTSignal) -> Tuple[bool, str]:
            """Handle momentum signals"""
            
            executor = self.executors.get(signal.symbol)
            if not executor:
                return False, f"No executor for symbol {signal.symbol}"
            
            return executor.execute_signal(signal)[:2]  # Only return success and message
        
        def mean_reversion_handler(signal: HFTSignal) -> Tuple[bool, str]:
            """Handle mean reversion signals"""
            
            executor = self.executors.get(signal.symbol)
            if not executor:
                return False, f"No executor for symbol {signal.symbol}"
            
            # Mean reversion signals might use different execution logic
            return executor.execute_signal(signal)[:2]
        
        def arbitrage_handler(signal: HFTSignal) -> Tuple[bool, str]:
            """Handle arbitrage signals"""
            
            executor = self.executors.get(signal.symbol)
            if not executor:
                return False, f"No executor for symbol {signal.symbol}"
            
            # Arbitrage signals typically require fastest execution
            signal.urgency = ExecutionSpeed.NANOSECOND
            return executor.execute_signal(signal)[:2]
        
        # Register handlers
        self.signal_processor.register_signal_handler('MOMENTUM', momentum_handler)
        self.signal_processor.register_signal_handler('MEAN_REVERSION', mean_reversion_handler)
        self.signal_processor.register_signal_handler('ARBITRAGE', arbitrage_handler)
    
    def _market_data_loop(self) -> None:
        """Market data processing loop"""
        
        while self.is_running:
            try:
                # Get market data tick
                try:
                    tick = self.market_data_feed.get(timeout=0.01)
                except queue.Empty:
                    continue
                
                # Update appropriate executor
                executor = self.executors.get(tick.symbol)
                if executor:
                    executor.update_market_data(tick)
                
            except Exception as e:
                print(f"Error in market data loop: {e}")


if __name__ == "__main__":
    import time
    import random
    
    # Example usage and testing
    print("Testing High Frequency Execution Engine...")
    
    # Create HFT engine
    engine = HighFrequencyExecutionEngine("TestHFTEngine")
    
    # Add symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        engine.add_symbol(symbol)
    
    # Start engine
    engine.start()
    
    try:
        print("\nSimulating market data and signals...")
        
        # Simulate market data
        for i in range(100):
            for symbol in symbols:
                # Create random market tick
                base_price = 100.0 + random.uniform(-5, 5)
                spread = random.uniform(0.01, 0.05)
                
                tick = MarketTick(
                    symbol=symbol,
                    timestamp_ns=time.time_ns(),
                    bid_price=base_price,
                    bid_size=random.uniform(100, 1000),
                    ask_price=base_price + spread,
                    ask_size=random.uniform(100, 1000),
                    last_price=base_price + spread/2,
                    last_size=random.uniform(10, 100)
                )
                
                engine.process_market_data(tick)
            
            # Generate some HFT signals
            if i % 10 == 0:  # Every 10th iteration
                for symbol in random.sample(symbols, 2):  # Random 2 symbols
                    signal = HFTSignal(
                        signal_id=f"SIG_{symbol}_{i}",
                        symbol=symbol,
                        timestamp_ns=time.time_ns(),
                        signal_type=random.choice(['MOMENTUM', 'MEAN_REVERSION', 'ARBITRAGE']),
                        direction=random.choice(['BUY', 'SELL']),
                        strength=random.uniform(0.3, 1.0),
                        confidence=random.uniform(0.5, 1.0),
                        urgency=random.choice([
                            ExecutionSpeed.NANOSECOND,
                            ExecutionSpeed.MICROSECOND,
                            ExecutionSpeed.MILLISECOND
                        ]),
                        max_quantity=random.uniform(100, 1000),
                        time_horizon_ms=random.uniform(50, 500)
                    )
                    
                    success, message, order_id = engine.process_signal(signal)
                    if success:
                        print(f"Signal processed: {signal.signal_type} {signal.direction} "
                              f"{signal.symbol} (strength: {signal.strength:.2f})")
            
            time.sleep(0.001)  # 1ms delay between iterations
        
        time.sleep(2)  # Let processing complete
        
        # Check engine statistics
        print(f"\nEngine Statistics:")
        stats = engine.get_engine_statistics()
        
        print(f"  Uptime: {stats['uptime_seconds']:.2f} seconds")
        print(f"  Signals Processed: {stats['signals_processed']}")
        print(f"  Orders Executed: {stats['orders_executed']}")
        print(f"  Total P&L: ${stats['total_pnl']:.2f}")
        
        # Signal processor statistics
        sp_stats = stats['signal_processor']
        print(f"\n  Signal Processor:")
        print(f"    Total Signals: {sp_stats['total_signals_processed']}")
        print(f"    Active Signals: {sp_stats['active_signals_count']}")
        
        if 'average_processing_time_us' in sp_stats:
            print(f"    Avg Processing Time: {sp_stats['average_processing_time_us']:.2f} μs")
            print(f"    P95 Processing Time: {sp_stats['p95_processing_time_us']:.2f} μs")
            print(f"    P99 Processing Time: {sp_stats['p99_processing_time_us']:.2f} μs")
        
        print(f"    Signals by Type:")
        for signal_type, count in sp_stats['signals_by_type'].items():
            print(f"      {signal_type}: {count}")
        
        # Executor statistics
        print(f"\n  Executor Statistics:")
        for symbol, executor_stats in stats['executors'].items():
            print(f"    {symbol}:")
            print(f"      Total Orders: {executor_stats['total_orders']}")
            print(f"      Successful: {executor_stats['successful_executions']}")
            print(f"      Rejections: {executor_stats['rejections']}")
            print(f"      Current Position: {executor_stats['current_position']}")
            print(f"      P&L: ${executor_stats['total_pnl']:.2f}")
            
            if 'average_execution_time_us' in executor_stats:
                print(f"      Avg Execution Time: {executor_stats['average_execution_time_us']:.2f} μs")
            
            if 'average_decision_latency_us' in executor_stats:
                print(f"      Avg Decision Latency: {executor_stats['average_decision_latency_us']:.2f} μs")
                print(f"      P95 Decision Latency: {executor_stats['p95_decision_latency_us']:.2f} μs")
            
            if 'average_fill_latency_us' in executor_stats:
                print(f"      Avg Fill Latency: {executor_stats['average_fill_latency_us']:.2f} μs")
                print(f"      P95 Fill Latency: {executor_stats['p95_fill_latency_us']:.2f} μs")
        
        # Test individual symbol statistics
        print(f"\nDetailed AAPL Statistics:")
        aapl_stats = engine.get_symbol_statistics("AAPL")
        if aapl_stats:
            print(f"  Position Utilization: {aapl_stats['position_utilization']:.1%}")
            print(f"  Success Rate: {(aapl_stats['successful_executions'] / max(1, aapl_stats['total_orders'])):.1%}")
        
    finally:
        # Stop engine
        engine.stop()
    
    print("\nHigh Frequency Execution Engine testing completed!")