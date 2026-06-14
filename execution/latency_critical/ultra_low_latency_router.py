"""
Ultra Low Latency Order Router
High-performance order routing for latency-critical trading
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import socket
import struct
import mmap
import ctypes
import os
from collections import deque, namedtuple
import warnings


class LatencyTier(Enum):
    """Latency requirement tiers"""
    ULTRA_LOW = "ULTRA_LOW"     # < 1ms
    LOW = "LOW"                 # < 5ms
    NORMAL = "NORMAL"           # < 50ms
    BULK = "BULK"               # No requirement


class VenueLatencyProfile(Enum):
    """Venue latency characteristics"""
    DIRECT_CONNECT = "DIRECT_CONNECT"     # Direct market data feed
    COLOCATED = "COLOCATED"               # Colocated access
    PREMIUM = "PREMIUM"                    # Premium connectivity
    STANDARD = "STANDARD"                  # Standard connectivity


@dataclass
class LatencyMeasurement:
    """Single latency measurement"""
    timestamp: float
    venue: str
    latency_us: float  # Microseconds
    measurement_type: str  # 'ORDER_ACK', 'FILL', 'MARKET_DATA'
    
    @property
    def latency_ms(self) -> float:
        """Latency in milliseconds"""
        return self.latency_us / 1000.0


@dataclass
class VenueConnectivity:
    """Venue connectivity configuration"""
    venue_id: str
    
    # Network configuration
    primary_host: str
    primary_port: int
    backup_host: Optional[str] = None
    backup_port: Optional[int] = None
    
    # Performance characteristics
    latency_profile: VenueLatencyProfile = VenueLatencyProfile.STANDARD
    expected_latency_us: float = 5000.0  # 5ms default
    max_orders_per_second: int = 1000
    
    # Connection settings
    tcp_nodelay: bool = True
    socket_buffer_size: int = 65536
    connection_timeout_ms: int = 1000
    heartbeat_interval_ms: int = 1000
    
    # Circuit breaker settings
    max_consecutive_failures: int = 3
    failure_recovery_time_ms: int = 5000
    
    # Latency thresholds
    latency_warning_threshold_us: float = 10000.0  # 10ms
    latency_error_threshold_us: float = 50000.0    # 50ms


@dataclass
class OrderMetrics:
    """Order-level latency metrics"""
    order_id: str
    
    # Timestamps (microseconds since epoch)
    order_creation_time: float
    routing_start_time: float
    venue_send_time: float
    venue_ack_time: Optional[float] = None
    first_fill_time: Optional[float] = None
    
    # Calculated latencies
    routing_latency_us: Optional[float] = None
    venue_ack_latency_us: Optional[float] = None
    first_fill_latency_us: Optional[float] = None
    
    def update_ack_time(self, ack_time: float) -> None:
        """Update acknowledgment time and calculate latency"""
        self.venue_ack_time = ack_time
        
        if self.venue_send_time:
            self.venue_ack_latency_us = ack_time - self.venue_send_time
    
    def update_fill_time(self, fill_time: float) -> None:
        """Update first fill time and calculate latency"""
        if self.first_fill_time is None:
            self.first_fill_time = fill_time
            
            if self.venue_send_time:
                self.first_fill_latency_us = fill_time - self.venue_send_time


class HighResolutionTimer:
    """High-resolution timer for microsecond precision"""
    
    @staticmethod
    def get_time_us() -> float:
        """Get current time in microseconds since epoch"""
        return time.time_ns() / 1000.0
    
    @staticmethod
    def get_time_ns() -> int:
        """Get current time in nanoseconds since epoch"""
        return time.time_ns()
    
    @staticmethod
    def sleep_us(microseconds: float) -> None:
        """Sleep for specified microseconds (busy wait for precision)"""
        end_time = HighResolutionTimer.get_time_us() + microseconds
        
        while HighResolutionTimer.get_time_us() < end_time:
            pass  # Busy wait


class LatencyMonitor:
    """
    Real-time latency monitoring and alerting
    """
    
    def __init__(self, alert_threshold_us: float = 10000.0):
        self.alert_threshold_us = alert_threshold_us
        
        # Measurement storage
        self.measurements: deque = deque(maxlen=10000)  # Last 10k measurements
        self.venue_stats: Dict[str, Dict] = {}
        
        # Rolling statistics (last 1000 measurements per venue)
        self.rolling_stats: Dict[str, deque] = {}
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
        
        # Threading
        self.lock = threading.RLock()
    
    def record_measurement(self, measurement: LatencyMeasurement) -> None:
        """Record latency measurement"""
        
        with self.lock:
            # Store measurement
            self.measurements.append(measurement)
            
            # Update venue statistics
            venue = measurement.venue
            
            if venue not in self.venue_stats:
                self.venue_stats[venue] = {
                    'count': 0,
                    'total_latency_us': 0.0,
                    'min_latency_us': float('inf'),
                    'max_latency_us': 0.0,
                    'last_measurement_time': 0.0
                }
                self.rolling_stats[venue] = deque(maxlen=1000)
            
            stats = self.venue_stats[venue]
            rolling = self.rolling_stats[venue]
            
            # Update statistics
            stats['count'] += 1
            stats['total_latency_us'] += measurement.latency_us
            stats['min_latency_us'] = min(stats['min_latency_us'], measurement.latency_us)
            stats['max_latency_us'] = max(stats['max_latency_us'], measurement.latency_us)
            stats['last_measurement_time'] = measurement.timestamp
            
            # Update rolling statistics
            rolling.append(measurement.latency_us)
            
            # Check for alerts
            if measurement.latency_us > self.alert_threshold_us:
                self._trigger_alert(measurement)
    
    def get_venue_statistics(self, venue: str) -> Optional[Dict]:
        """Get statistics for a venue"""
        
        with self.lock:
            if venue not in self.venue_stats:
                return None
            
            stats = self.venue_stats[venue]
            rolling = self.rolling_stats[venue]
            
            result = stats.copy()
            
            # Calculate derived statistics
            if stats['count'] > 0:
                result['average_latency_us'] = stats['total_latency_us'] / stats['count']
            else:
                result['average_latency_us'] = 0.0
            
            # Rolling statistics
            if rolling:
                rolling_array = np.array(rolling)
                result['rolling_average_us'] = np.mean(rolling_array)
                result['rolling_std_us'] = np.std(rolling_array)
                result['rolling_p50_us'] = np.percentile(rolling_array, 50)
                result['rolling_p95_us'] = np.percentile(rolling_array, 95)
                result['rolling_p99_us'] = np.percentile(rolling_array, 99)
            
            return result
    
    def get_all_venue_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all venues"""
        
        with self.lock:
            result = {}
            
            for venue in self.venue_stats:
                result[venue] = self.get_venue_statistics(venue)
            
            return result
    
    def add_alert_handler(self, handler: Callable) -> None:
        """Add latency alert handler"""
        self.alert_handlers.append(handler)
    
    def _trigger_alert(self, measurement: LatencyMeasurement) -> None:
        """Trigger latency alert"""
        
        alert = {
            'type': 'HIGH_LATENCY',
            'measurement': measurement,
            'threshold_us': self.alert_threshold_us,
            'timestamp': HighResolutionTimer.get_time_us()
        }
        
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error in alert handler: {e}")


class CircuitBreaker:
    """
    Circuit breaker for venue connections
    """
    
    def __init__(
        self,
        venue_id: str,
        failure_threshold: int = 5,
        recovery_timeout_ms: int = 10000,
        success_threshold: int = 3
    ):
        self.venue_id = venue_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout_ms = recovery_timeout_ms
        self.success_threshold = success_threshold
        
        # State tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.is_open = False
        
        # Threading
        self.lock = threading.RLock()
    
    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        
        with self.lock:
            if not self.is_open:
                return True
            
            # Check if recovery timeout has elapsed
            current_time = HighResolutionTimer.get_time_us()
            time_since_failure = current_time - self.last_failure_time
            
            if time_since_failure > (self.recovery_timeout_ms * 1000):
                # Try to close circuit breaker
                self.is_open = False
                self.failure_count = 0
                self.success_count = 0
                return True
            
            return False
    
    def record_success(self) -> None:
        """Record successful execution"""
        
        with self.lock:
            if self.is_open:
                self.success_count += 1
                
                if self.success_count >= self.success_threshold:
                    self.is_open = False
                    self.failure_count = 0
                    self.success_count = 0
                    print(f"Circuit breaker for {self.venue_id} closed after recovery")
            else:
                # Reset failure count on success
                self.failure_count = 0
    
    def record_failure(self) -> None:
        """Record failed execution"""
        
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = HighResolutionTimer.get_time_us()
            
            if self.failure_count >= self.failure_threshold:
                self.is_open = True
                print(f"Circuit breaker for {self.venue_id} opened after {self.failure_count} failures")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        
        with self.lock:
            return {
                'venue_id': self.venue_id,
                'is_open': self.is_open,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'time_until_retry_ms': max(0, 
                    self.recovery_timeout_ms - 
                    (HighResolutionTimer.get_time_us() - self.last_failure_time) / 1000
                ) if self.is_open else 0
            }


class UltraLowLatencyRouter:
    """
    Ultra-low latency order router
    Optimized for high-frequency trading requirements
    """
    
    def __init__(self, name: str = "ULLRouter"):
        self.name = name
        
        # Venue configurations
        self.venues: Dict[str, VenueConnectivity] = {}
        self.venue_rankings: List[str] = []  # Ordered by preference
        
        # Latency monitoring
        self.latency_monitor = LatencyMonitor()
        self.order_metrics: Dict[str, OrderMetrics] = {}
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Connection management
        self.connections: Dict[str, Any] = {}  # venue_id -> connection
        self.connection_status: Dict[str, bool] = {}
        
        # Routing queues (lock-free where possible)
        self.high_priority_queue = queue.Queue()
        self.normal_priority_queue = queue.Queue()
        
        # Performance optimization
        self.enable_busy_wait = True
        self.cpu_affinity_core = None  # CPU core for routing thread
        
        # Statistics
        self.routing_stats = {
            'total_orders_routed': 0,
            'total_routing_time_us': 0.0,
            'venue_distribution': {},
            'latency_violations': 0,
            'circuit_breaker_blocks': 0
        }
        
        # Threading
        self.is_running = False
        self.routing_threads: List[threading.Thread] = []
        self.lock = threading.RLock()
        
        # Setup alert handler
        self.latency_monitor.add_alert_handler(self._handle_latency_alert)
    
    def add_venue(self, venue_config: VenueConnectivity) -> None:
        """Add venue configuration"""
        
        with self.lock:
            self.venues[venue_config.venue_id] = venue_config
            self.connection_status[venue_config.venue_id] = False
            
            # Create circuit breaker
            self.circuit_breakers[venue_config.venue_id] = CircuitBreaker(
                venue_id=venue_config.venue_id,
                failure_threshold=venue_config.max_consecutive_failures,
                recovery_timeout_ms=venue_config.failure_recovery_time_ms
            )
            
            # Initialize venue statistics
            self.routing_stats['venue_distribution'][venue_config.venue_id] = 0
    
    def set_venue_ranking(self, venue_ids: List[str]) -> None:
        """Set venue preference ranking"""
        
        with self.lock:
            # Validate all venues exist
            for venue_id in venue_ids:
                if venue_id not in self.venues:
                    raise ValueError(f"Unknown venue: {venue_id}")
            
            self.venue_rankings = venue_ids.copy()
    
    def start(self) -> None:
        """Start ultra-low latency router"""
        
        self.is_running = True
        
        # Start routing threads
        for i in range(2):  # High priority and normal priority
            thread = threading.Thread(
                target=self._routing_loop,
                args=(i == 0,),  # is_high_priority
                name=f"ULLRouting-{'High' if i == 0 else 'Normal'}"
            )
            
            # Set CPU affinity if specified
            if self.cpu_affinity_core is not None:
                # Note: This is a simplified example - actual CPU affinity setting 
                # would require platform-specific code
                pass
            
            thread.start()
            self.routing_threads.append(thread)
        
        print(f"Ultra-Low Latency Router {self.name} started")
    
    def stop(self) -> None:
        """Stop ultra-low latency router"""
        
        self.is_running = False
        
        # Stop routing threads
        for thread in self.routing_threads:
            thread.join(timeout=2.0)
        
        # Close connections
        self._close_all_connections()
        
        print(f"Ultra-Low Latency Router {self.name} stopped")
    
    def route_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        price: Optional[float] = None,
        latency_tier: LatencyTier = LatencyTier.NORMAL,
        preferred_venue: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Route order to optimal venue
        
        Args:
            order_id: Order identifier
            symbol: Trading symbol  
            side: Order side
            quantity: Order quantity
            order_type: Order type
            price: Order price (for limit orders)
            latency_tier: Latency requirement
            preferred_venue: Preferred venue (if any)
            **kwargs: Additional order parameters
        
        Returns:
            (success, message, selected_venue)
        """
        
        start_time = HighResolutionTimer.get_time_us()
        
        try:
            # Create order metrics
            metrics = OrderMetrics(
                order_id=order_id,
                order_creation_time=start_time,
                routing_start_time=start_time
            )
            
            self.order_metrics[order_id] = metrics
            
            # Select optimal venue
            selected_venue = self._select_venue(
                symbol=symbol,
                side=side,
                quantity=quantity,
                latency_tier=latency_tier,
                preferred_venue=preferred_venue
            )
            
            if not selected_venue:
                return False, "No suitable venue available", None
            
            # Create order packet
            order_packet = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_type': order_type,
                'price': price,
                'venue': selected_venue,
                'metrics': metrics,
                'timestamp': start_time,
                **kwargs
            }
            
            # Queue order based on latency tier
            if latency_tier in [LatencyTier.ULTRA_LOW, LatencyTier.LOW]:
                try:
                    self.high_priority_queue.put(order_packet, timeout=0.001)  # 1ms timeout
                except queue.Full:
                    return False, "High priority queue full", None
            else:
                try:
                    self.normal_priority_queue.put(order_packet, timeout=0.01)  # 10ms timeout
                except queue.Full:
                    return False, "Normal priority queue full", None
            
            # Update routing latency
            routing_time = HighResolutionTimer.get_time_us() - start_time
            metrics.routing_latency_us = routing_time
            
            # Update statistics
            with self.lock:
                self.routing_stats['total_orders_routed'] += 1
                self.routing_stats['total_routing_time_us'] += routing_time
                self.routing_stats['venue_distribution'][selected_venue] += 1
            
            return True, "Order queued for routing", selected_venue
            
        except Exception as e:
            return False, f"Routing error: {e}", None
    
    def record_venue_ack(self, order_id: str, venue: str, ack_time: Optional[float] = None) -> None:
        """Record venue acknowledgment"""
        
        if ack_time is None:
            ack_time = HighResolutionTimer.get_time_us()
        
        if order_id in self.order_metrics:
            metrics = self.order_metrics[order_id]
            metrics.update_ack_time(ack_time)
            
            # Record latency measurement
            if metrics.venue_ack_latency_us is not None:
                measurement = LatencyMeasurement(
                    timestamp=ack_time,
                    venue=venue,
                    latency_us=metrics.venue_ack_latency_us,
                    measurement_type='ORDER_ACK'
                )
                
                self.latency_monitor.record_measurement(measurement)
            
            # Update circuit breaker
            if venue in self.circuit_breakers:
                self.circuit_breakers[venue].record_success()
    
    def record_venue_fill(self, order_id: str, venue: str, fill_time: Optional[float] = None) -> None:
        """Record venue fill"""
        
        if fill_time is None:
            fill_time = HighResolutionTimer.get_time_us()
        
        if order_id in self.order_metrics:
            metrics = self.order_metrics[order_id]
            metrics.update_fill_time(fill_time)
            
            # Record latency measurement
            if metrics.first_fill_latency_us is not None:
                measurement = LatencyMeasurement(
                    timestamp=fill_time,
                    venue=venue,
                    latency_us=metrics.first_fill_latency_us,
                    measurement_type='FILL'
                )
                
                self.latency_monitor.record_measurement(measurement)
    
    def record_venue_failure(self, venue: str, error_message: str) -> None:
        """Record venue failure"""
        
        if venue in self.circuit_breakers:
            self.circuit_breakers[venue].record_failure()
            
            with self.lock:
                self.routing_stats['circuit_breaker_blocks'] += 1
    
    def get_order_metrics(self, order_id: str) -> Optional[OrderMetrics]:
        """Get order metrics"""
        return self.order_metrics.get(order_id)
    
    def get_venue_latency_stats(self, venue: str) -> Optional[Dict]:
        """Get venue latency statistics"""
        return self.latency_monitor.get_venue_statistics(venue)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics"""
        
        with self.lock:
            stats = self.routing_stats.copy()
            
            # Calculate derived statistics
            if stats['total_orders_routed'] > 0:
                stats['average_routing_time_us'] = (
                    stats['total_routing_time_us'] / stats['total_orders_routed']
                )
            else:
                stats['average_routing_time_us'] = 0.0
            
            # Add circuit breaker status
            stats['circuit_breaker_status'] = {}
            
            for venue_id, cb in self.circuit_breakers.items():
                stats['circuit_breaker_status'][venue_id] = cb.get_status()
            
            # Add venue latency statistics
            stats['venue_latency_stats'] = self.latency_monitor.get_all_venue_statistics()
            
            return stats
    
    def _select_venue(
        self,
        symbol: str,
        side: str,
        quantity: float,
        latency_tier: LatencyTier,
        preferred_venue: Optional[str] = None
    ) -> Optional[str]:
        """Select optimal venue for order"""
        
        # If preferred venue is specified and available, use it
        if preferred_venue and preferred_venue in self.venues:
            cb = self.circuit_breakers.get(preferred_venue)
            
            if cb and cb.can_execute():
                return preferred_venue
        
        # Score all available venues
        venue_scores = []
        
        for venue_id in self.venue_rankings:
            if venue_id not in self.venues:
                continue
            
            venue_config = self.venues[venue_id]
            cb = self.circuit_breakers[venue_id]
            
            # Skip if circuit breaker is open
            if not cb.can_execute():
                continue
            
            # Calculate venue score
            score = self._calculate_venue_score(venue_config, latency_tier)
            venue_scores.append((venue_id, score))
        
        # Sort by score (higher is better)
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return best venue
        if venue_scores:
            return venue_scores[0][0]
        
        return None
    
    def _calculate_venue_score(
        self,
        venue_config: VenueConnectivity,
        latency_tier: LatencyTier
    ) -> float:
        """Calculate venue score for routing decision"""
        
        score = 0.0
        
        # Latency score (higher is better)
        latency_score = 1.0 / (1.0 + venue_config.expected_latency_us / 1000.0)
        score += latency_score * 50.0
        
        # Capacity score
        capacity_score = min(1.0, venue_config.max_orders_per_second / 10000.0)
        score += capacity_score * 20.0
        
        # Profile score
        profile_scores = {
            VenueLatencyProfile.DIRECT_CONNECT: 30.0,
            VenueLatencyProfile.COLOCATED: 25.0,
            VenueLatencyProfile.PREMIUM: 20.0,
            VenueLatencyProfile.STANDARD: 10.0
        }
        
        score += profile_scores.get(venue_config.latency_profile, 0.0)
        
        # Historical performance score
        venue_stats = self.latency_monitor.get_venue_statistics(venue_config.venue_id)
        
        if venue_stats:
            # Penalize high latency
            avg_latency = venue_stats.get('rolling_average_us', venue_config.expected_latency_us)
            latency_penalty = max(0, (avg_latency - venue_config.expected_latency_us) / 1000.0)
            score -= latency_penalty
            
            # Reward consistency (low standard deviation)
            std_latency = venue_stats.get('rolling_std_us', 0)
            consistency_bonus = max(0, 10.0 - std_latency / 1000.0)
            score += consistency_bonus
        
        return score
    
    def _routing_loop(self, is_high_priority: bool) -> None:
        """Main routing loop"""
        
        queue_to_process = self.high_priority_queue if is_high_priority else self.normal_priority_queue
        
        while self.is_running:
            try:
                # Get next order
                try:
                    timeout = 0.001 if is_high_priority else 0.01
                    order_packet = queue_to_process.get(timeout=timeout)
                except queue.Empty:
                    continue
                
                # Process order
                self._process_order_packet(order_packet)
                
            except Exception as e:
                print(f"Error in routing loop ({'high' if is_high_priority else 'normal'}): {e}")
    
    def _process_order_packet(self, order_packet: Dict) -> None:
        """Process individual order packet"""
        
        venue = order_packet['venue']
        metrics = order_packet['metrics']
        
        try:
            # Record venue send time
            send_time = HighResolutionTimer.get_time_us()
            metrics.venue_send_time = send_time
            
            # Simulate sending order to venue
            # In real implementation, this would use actual venue API
            success = self._send_to_venue(venue, order_packet)
            
            if not success:
                # Record failure
                self.record_venue_failure(venue, "Send failed")
                
                # Try to reroute if possible
                self._attempt_reroute(order_packet)
            
        except Exception as e:
            print(f"Error processing order {order_packet['order_id']}: {e}")
            self.record_venue_failure(venue, str(e))
    
    def _send_to_venue(self, venue: str, order_packet: Dict) -> bool:
        """
        Send order to venue (simulated)
        In real implementation, this would use actual venue protocols
        """
        
        # Simulate network delay
        venue_config = self.venues.get(venue)
        
        if not venue_config:
            return False
        
        # Simulate latency based on venue profile
        simulated_latency_us = venue_config.expected_latency_us
        
        if self.enable_busy_wait and simulated_latency_us < 1000:  # < 1ms
            HighResolutionTimer.sleep_us(simulated_latency_us)
        else:
            time.sleep(simulated_latency_us / 1_000_000)  # Convert to seconds
        
        # Simulate 99% success rate
        return np.random.random() > 0.01
    
    def _attempt_reroute(self, order_packet: Dict) -> None:
        """Attempt to reroute failed order"""
        
        # For simplicity, just log the reroute attempt
        print(f"Attempting to reroute order {order_packet['order_id']}")
        
        # In real implementation, would select alternative venue and retry
    
    def _handle_latency_alert(self, alert: Dict) -> None:
        """Handle latency alert"""
        
        measurement = alert['measurement']
        
        print(f"LATENCY ALERT: {measurement.venue} latency "
              f"{measurement.latency_ms:.2f}ms exceeds threshold "
              f"{alert['threshold_us']/1000:.2f}ms")
        
        with self.lock:
            self.routing_stats['latency_violations'] += 1
    
    def _close_all_connections(self) -> None:
        """Close all venue connections"""
        
        for venue_id, connection in self.connections.items():
            try:
                if hasattr(connection, 'close'):
                    connection.close()
            except Exception as e:
                print(f"Error closing connection to {venue_id}: {e}")


if __name__ == "__main__":
    import time
    
    # Example usage and testing
    print("Testing Ultra Low Latency Router...")
    
    # Create router
    router = UltraLowLatencyRouter("TestULLRouter")
    
    # Add venue configurations
    venues = [
        VenueConnectivity(
            venue_id="NASDAQ_DIRECT",
            primary_host="nasdaq.direct.com",
            primary_port=12345,
            latency_profile=VenueLatencyProfile.DIRECT_CONNECT,
            expected_latency_us=500.0,  # 0.5ms
            max_orders_per_second=10000
        ),
        VenueConnectivity(
            venue_id="NYSE_COLOCATION",
            primary_host="nyse.colo.com", 
            primary_port=23456,
            latency_profile=VenueLatencyProfile.COLOCATED,
            expected_latency_us=800.0,  # 0.8ms
            max_orders_per_second=8000
        ),
        VenueConnectivity(
            venue_id="ARCA_PREMIUM",
            primary_host="arca.premium.com",
            primary_port=34567,
            latency_profile=VenueLatencyProfile.PREMIUM,
            expected_latency_us=1500.0,  # 1.5ms
            max_orders_per_second=5000
        )
    ]
    
    for venue in venues:
        router.add_venue(venue)
    
    # Set venue ranking
    router.set_venue_ranking(["NASDAQ_DIRECT", "NYSE_COLOCATION", "ARCA_PREMIUM"])
    
    # Start router
    router.start()
    
    try:
        print("\nRouting test orders...")
        
        # Route high-priority orders
        for i in range(10):
            success, message, venue = router.route_order(
                order_id=f"ULL_ORDER_{i:03d}",
                symbol="AAPL",
                side="BUY",
                quantity=100 * (i + 1),
                order_type="LIMIT",
                price=150.00 + i * 0.01,
                latency_tier=LatencyTier.ULTRA_LOW
            )
            
            print(f"Order {i}: {message} -> {venue}")
            
            if success:
                # Simulate venue acknowledgment
                time.sleep(0.001)  # 1ms delay
                router.record_venue_ack(f"ULL_ORDER_{i:03d}", venue)
                
                # Simulate some fills
                if i % 3 == 0:
                    time.sleep(0.002)  # Additional 2ms for fill
                    router.record_venue_fill(f"ULL_ORDER_{i:03d}", venue)
        
        # Route normal priority orders
        for i in range(5):
            success, message, venue = router.route_order(
                order_id=f"NORMAL_ORDER_{i:03d}",
                symbol="MSFT",
                side="SELL",
                quantity=200 * (i + 1),
                order_type="MARKET",
                latency_tier=LatencyTier.NORMAL
            )
            
            if success:
                router.record_venue_ack(f"NORMAL_ORDER_{i:03d}", venue)
        
        time.sleep(2)  # Let processing complete
        
        # Check routing statistics
        print(f"\nRouting Statistics:")
        stats = router.get_routing_statistics()
        
        print(f"  Total Orders Routed: {stats['total_orders_routed']}")
        print(f"  Average Routing Time: {stats['average_routing_time_us']:.2f} μs")
        print(f"  Latency Violations: {stats['latency_violations']}")
        print(f"  Circuit Breaker Blocks: {stats['circuit_breaker_blocks']}")
        
        print(f"\n  Venue Distribution:")
        for venue, count in stats['venue_distribution'].items():
            print(f"    {venue}: {count}")
        
        print(f"\n  Circuit Breaker Status:")
        for venue, status in stats['circuit_breaker_status'].items():
            print(f"    {venue}: {'OPEN' if status['is_open'] else 'CLOSED'} "
                  f"(failures: {status['failure_count']})")
        
        # Check venue latency statistics
        print(f"\n  Venue Latency Statistics:")
        for venue, latency_stats in stats['venue_latency_stats'].items():
            if latency_stats and latency_stats['count'] > 0:
                print(f"    {venue}:")
                print(f"      Count: {latency_stats['count']}")
                print(f"      Average: {latency_stats['average_latency_us']:.2f} μs")
                print(f"      Rolling P95: {latency_stats.get('rolling_p95_us', 0):.2f} μs") 
                print(f"      Rolling P99: {latency_stats.get('rolling_p99_us', 0):.2f} μs")
        
        # Check individual order metrics
        print(f"\nOrder Metrics Sample:")
        for i in range(3):
            order_id = f"ULL_ORDER_{i:03d}"
            metrics = router.get_order_metrics(order_id)
            
            if metrics:
                print(f"  {order_id}:")
                print(f"    Routing Latency: {metrics.routing_latency_us:.2f} μs")
                
                if metrics.venue_ack_latency_us:
                    print(f"    Venue Ack Latency: {metrics.venue_ack_latency_us:.2f} μs")
                
                if metrics.first_fill_latency_us:
                    print(f"    First Fill Latency: {metrics.first_fill_latency_us:.2f} μs")
        
        # Test circuit breaker by simulating failures
        print(f"\nTesting circuit breaker...")
        
        for i in range(6):  # Exceed failure threshold
            router.record_venue_failure("NASDAQ_DIRECT", f"Test failure {i}")
        
        # Check circuit breaker status
        final_stats = router.get_routing_statistics()
        cb_status = final_stats['circuit_breaker_status']['NASDAQ_DIRECT']
        
        print(f"NASDAQ_DIRECT circuit breaker: {'OPEN' if cb_status['is_open'] else 'CLOSED'}")
        print(f"Failure count: {cb_status['failure_count']}")
        
        # Try routing with circuit breaker open
        success, message, venue = router.route_order(
            order_id="CB_TEST_ORDER",
            symbol="AAPL",
            side="BUY", 
            quantity=100,
            order_type="LIMIT",
            price=150.00,
            latency_tier=LatencyTier.ULTRA_LOW,
            preferred_venue="NASDAQ_DIRECT"
        )
        
        print(f"Circuit breaker test: {message} -> {venue}")
        
    finally:
        # Stop router
        router.stop()
    
    print("\nUltra Low Latency Router testing completed!")