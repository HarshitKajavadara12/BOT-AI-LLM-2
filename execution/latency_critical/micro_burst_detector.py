"""
Micro Burst Detector
Advanced market microstructure analysis for detecting trading opportunities and risks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import time
from collections import deque, defaultdict, namedtuple
import warnings


class BurstType(Enum):
    """Types of micro bursts in market data"""
    VOLUME_BURST = "VOLUME_BURST"         # Sudden volume spike
    PRICE_BURST = "PRICE_BURST"           # Rapid price movement
    SPREAD_BURST = "SPREAD_BURST"         # Spread expansion/contraction
    IMBALANCE_BURST = "IMBALANCE_BURST"   # Order book imbalance
    MOMENTUM_BURST = "MOMENTUM_BURST"     # Directional momentum
    VOLATILITY_BURST = "VOLATILITY_BURST" # Volatility spike


class BurstSeverity(Enum):
    """Severity levels for micro bursts"""
    LOW = "LOW"           # 1-2 standard deviations
    MEDIUM = "MEDIUM"     # 2-3 standard deviations
    HIGH = "HIGH"         # 3-5 standard deviations
    EXTREME = "EXTREME"   # >5 standard deviations


@dataclass
class MarketMicroTick:
    """High-resolution market microstructure data"""
    symbol: str
    timestamp_ns: int
    
    # Level 1 data
    bid_price: float
    bid_size: float
    ask_price: float
    ask_size: float
    
    # Trade data
    last_price: Optional[float] = None
    last_size: Optional[float] = None
    trade_side: Optional[str] = None  # 'BUY', 'SELL', 'UNKNOWN'
    
    # Order book metrics
    total_bid_volume: Optional[float] = None
    total_ask_volume: Optional[float] = None
    bid_levels: Optional[int] = None
    ask_levels: Optional[int] = None
    
    # Market quality metrics
    effective_spread_bps: Optional[float] = None
    realized_spread_bps: Optional[float] = None
    price_impact_bps: Optional[float] = None
    
    @property
    def spread(self) -> float:
        """Bid-ask spread"""
        return self.ask_price - self.bid_price
    
    @property
    def mid_price(self) -> float:
        """Mid price"""
        return (self.bid_price + self.ask_price) / 2.0
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points"""
        return (self.spread / self.mid_price) * 10000 if self.mid_price > 0 else 0.0
    
    @property
    def book_imbalance(self) -> float:
        """Order book imbalance ratio"""
        if self.total_bid_volume is not None and self.total_ask_volume is not None:
            total_volume = self.total_bid_volume + self.total_ask_volume
            if total_volume > 0:
                return (self.total_bid_volume - self.total_ask_volume) / total_volume
        
        # Fallback to top-of-book imbalance
        total_size = self.bid_size + self.ask_size
        if total_size > 0:
            return (self.bid_size - self.ask_size) / total_size
        
        return 0.0


@dataclass
class MicroBurst:
    """Detected micro burst event"""
    burst_id: str
    symbol: str
    burst_type: BurstType
    severity: BurstSeverity
    
    # Timing
    start_time_ns: int
    peak_time_ns: int
    end_time_ns: Optional[int] = None
    
    # Measurements
    baseline_value: float = 0.0
    peak_value: float = 0.0
    magnitude: float = 0.0  # Standard deviations from baseline
    duration_us: Optional[float] = None
    
    # Market impact
    price_impact_bps: Optional[float] = None
    volume_impact: Optional[float] = None
    spread_impact_bps: Optional[float] = None
    
    # Context
    market_conditions: Optional[Dict[str, Any]] = None
    triggering_events: List[str] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Duration in milliseconds"""
        if self.duration_us is not None:
            return self.duration_us / 1000.0
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if burst is still active"""
        return self.end_time_ns is None


class RollingStatistics:
    """
    Efficient rolling statistics calculator
    """
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        # Cached statistics
        self._mean = 0.0
        self._variance = 0.0
        self._std = 0.0
        self._min = float('inf')
        self._max = float('-inf')
        self._dirty = True
    
    def add_value(self, value: float, timestamp: int) -> None:
        """Add new value and timestamp"""
        self.values.append(value)
        self.timestamps.append(timestamp)
        self._dirty = True
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics"""
        if self._dirty:
            self._recalculate_statistics()
        
        return {
            'mean': self._mean,
            'std': self._std,
            'variance': self._variance,
            'min': self._min,
            'max': self._max,
            'count': len(self.values)
        }
    
    def get_z_score(self, value: float) -> float:
        """Get z-score for a value"""
        if self._dirty:
            self._recalculate_statistics()
        
        if self._std > 0:
            return (value - self._mean) / self._std
        return 0.0
    
    def is_outlier(self, value: float, threshold: float = 2.0) -> bool:
        """Check if value is an outlier"""
        return abs(self.get_z_score(value)) > threshold
    
    def _recalculate_statistics(self) -> None:
        """Recalculate cached statistics"""
        if not self.values:
            self._mean = 0.0
            self._variance = 0.0
            self._std = 0.0
            self._min = float('inf')
            self._max = float('-inf')
        else:
            values_array = np.array(self.values)
            self._mean = np.mean(values_array)
            self._variance = np.var(values_array, ddof=1) if len(values_array) > 1 else 0.0
            self._std = np.sqrt(self._variance)
            self._min = np.min(values_array)
            self._max = np.max(values_array)
        
        self._dirty = False


class BurstDetector:
    """
    Base class for burst detection algorithms
    """
    
    def __init__(self, burst_type: BurstType, detection_window: int = 100):
        self.burst_type = burst_type
        self.detection_window = detection_window
        self.rolling_stats = RollingStatistics(detection_window)
        
        # Detection parameters
        self.low_threshold = 2.0      # 2 sigma
        self.medium_threshold = 3.0   # 3 sigma  
        self.high_threshold = 5.0     # 5 sigma
        self.extreme_threshold = 7.0  # 7 sigma
        
        # State tracking
        self.active_bursts: Dict[str, MicroBurst] = {}
        self.completed_bursts: List[MicroBurst] = []
        
        # Performance metrics
        self.detection_count = 0
        self.false_positive_count = 0
    
    @abstractmethod
    def extract_value(self, tick: MarketMicroTick) -> float:
        """Extract relevant value from market tick"""
        pass
    
    def process_tick(self, tick: MarketMicroTick) -> List[MicroBurst]:
        """
        Process market tick and detect bursts
        
        Args:
            tick: Market microstructure tick
        
        Returns:
            List of newly detected or updated bursts
        """
        
        # Extract relevant value
        value = self.extract_value(tick)
        
        if value is None or not np.isfinite(value):
            return []
        
        # Update rolling statistics
        self.rolling_stats.add_value(value, tick.timestamp_ns)
        
        # Get current statistics
        stats = self.rolling_stats.get_statistics()
        
        if stats['count'] < self.detection_window // 2:
            return []  # Not enough data for reliable detection
        
        # Calculate z-score
        z_score = self.rolling_stats.get_z_score(value)
        abs_z_score = abs(z_score)
        
        # Determine severity
        severity = self._classify_severity(abs_z_score)
        
        if severity is None:
            # No burst detected, check for burst completion
            return self._check_burst_completion(tick, value)
        
        # Burst detected
        burst_id = f"{self.burst_type.value}_{tick.symbol}_{tick.timestamp_ns}"
        
        # Check if this extends an existing burst
        existing_burst = self._find_existing_burst(tick.symbol, tick.timestamp_ns)
        
        if existing_burst:
            # Update existing burst
            return self._update_existing_burst(existing_burst, tick, value, z_score, severity)
        else:
            # Create new burst
            return self._create_new_burst(burst_id, tick, value, z_score, severity, stats)
    
    def _classify_severity(self, abs_z_score: float) -> Optional[BurstSeverity]:
        """Classify burst severity based on z-score"""
        
        if abs_z_score >= self.extreme_threshold:
            return BurstSeverity.EXTREME
        elif abs_z_score >= self.high_threshold:
            return BurstSeverity.HIGH
        elif abs_z_score >= self.medium_threshold:
            return BurstSeverity.MEDIUM
        elif abs_z_score >= self.low_threshold:
            return BurstSeverity.LOW
        else:
            return None
    
    def _find_existing_burst(self, symbol: str, timestamp_ns: int) -> Optional[MicroBurst]:
        """Find existing active burst for symbol"""
        
        # Look for active bursts within reasonable time window (e.g., 10ms)
        time_window_ns = 10_000_000  # 10ms
        
        for burst in self.active_bursts.values():
            if (burst.symbol == symbol and 
                burst.is_active and
                (timestamp_ns - burst.peak_time_ns) <= time_window_ns):
                return burst
        
        return None
    
    def _create_new_burst(
        self,
        burst_id: str,
        tick: MarketMicroTick,
        value: float,
        z_score: float,
        severity: BurstSeverity,
        stats: Dict[str, float]
    ) -> List[MicroBurst]:
        """Create new burst detection"""
        
        burst = MicroBurst(
            burst_id=burst_id,
            symbol=tick.symbol,
            burst_type=self.burst_type,
            severity=severity,
            start_time_ns=tick.timestamp_ns,
            peak_time_ns=tick.timestamp_ns,
            baseline_value=stats['mean'],
            peak_value=value,
            magnitude=abs(z_score),
            market_conditions=self._capture_market_conditions(tick)
        )
        
        # Calculate market impact
        burst.price_impact_bps = self._estimate_price_impact(tick, burst)
        burst.spread_impact_bps = self._estimate_spread_impact(tick, burst)
        burst.volume_impact = self._estimate_volume_impact(tick, burst)
        
        self.active_bursts[burst_id] = burst
        self.detection_count += 1
        
        return [burst]
    
    def _update_existing_burst(
        self,
        burst: MicroBurst,
        tick: MarketMicroTick,
        value: float,
        z_score: float,
        severity: BurstSeverity
    ) -> List[MicroBurst]:
        """Update existing burst with new data"""
        
        # Update burst properties if this is a new peak
        if abs(z_score) > burst.magnitude:
            burst.peak_time_ns = tick.timestamp_ns
            burst.peak_value = value
            burst.magnitude = abs(z_score)
            burst.severity = severity
            
            # Update market impact estimates
            burst.price_impact_bps = self._estimate_price_impact(tick, burst)
            burst.spread_impact_bps = self._estimate_spread_impact(tick, burst)
            burst.volume_impact = self._estimate_volume_impact(tick, burst)
        
        return [burst]
    
    def _check_burst_completion(self, tick: MarketMicroTick, value: float) -> List[MicroBurst]:
        """Check if any active bursts should be completed"""
        
        completed = []
        to_remove = []
        
        current_time_ns = tick.timestamp_ns
        completion_threshold_ns = 5_000_000  # 5ms without burst activity
        
        for burst_id, burst in self.active_bursts.items():
            if (burst.symbol == tick.symbol and
                (current_time_ns - burst.peak_time_ns) > completion_threshold_ns):
                
                # Complete the burst
                burst.end_time_ns = current_time_ns
                burst.duration_us = (burst.end_time_ns - burst.start_time_ns) / 1000.0
                
                self.completed_bursts.append(burst)
                completed.append(burst)
                to_remove.append(burst_id)
        
        # Remove completed bursts from active list
        for burst_id in to_remove:
            del self.active_bursts[burst_id]
        
        return completed
    
    def _capture_market_conditions(self, tick: MarketMicroTick) -> Dict[str, Any]:
        """Capture relevant market conditions"""
        
        return {
            'mid_price': tick.mid_price,
            'spread_bps': tick.spread_bps,
            'book_imbalance': tick.book_imbalance,
            'bid_size': tick.bid_size,
            'ask_size': tick.ask_size,
            'timestamp_ns': tick.timestamp_ns
        }
    
    def _estimate_price_impact(self, tick: MarketMicroTick, burst: MicroBurst) -> Optional[float]:
        """Estimate price impact of burst (to be overridden by subclasses)"""
        return None
    
    def _estimate_spread_impact(self, tick: MarketMicroTick, burst: MicroBurst) -> Optional[float]:
        """Estimate spread impact of burst (to be overridden by subclasses)"""
        return None
    
    def _estimate_volume_impact(self, tick: MarketMicroTick, burst: MicroBurst) -> Optional[float]:
        """Estimate volume impact of burst (to be overridden by subclasses)"""
        return None


class VolumeBurstDetector(BurstDetector):
    """Detects sudden volume spikes"""
    
    def __init__(self, detection_window: int = 100):
        super().__init__(BurstType.VOLUME_BURST, detection_window)
    
    def extract_value(self, tick: MarketMicroTick) -> float:
        """Extract volume-related value"""
        if tick.last_size is not None:
            return tick.last_size
        else:
            # Use average of bid/ask sizes as proxy
            return (tick.bid_size + tick.ask_size) / 2.0
    
    def _estimate_volume_impact(self, tick: MarketMicroTick, burst: MicroBurst) -> Optional[float]:
        """Estimate volume impact"""
        if burst.baseline_value > 0:
            return (burst.peak_value - burst.baseline_value) / burst.baseline_value
        return None


class PriceBurstDetector(BurstDetector):
    """Detects rapid price movements"""
    
    def __init__(self, detection_window: int = 50):
        super().__init__(BurstType.PRICE_BURST, detection_window)
        self.previous_price = None
    
    def extract_value(self, tick: MarketMicroTick) -> float:
        """Extract price movement value"""
        current_price = tick.last_price if tick.last_price is not None else tick.mid_price
        
        if self.previous_price is None:
            self.previous_price = current_price
            return 0.0
        
        # Calculate price change in basis points
        if self.previous_price > 0:
            price_change_bps = ((current_price - self.previous_price) / self.previous_price) * 10000
        else:
            price_change_bps = 0.0
        
        self.previous_price = current_price
        return price_change_bps
    
    def _estimate_price_impact(self, tick: MarketMicroTick, burst: MicroBurst) -> Optional[float]:
        """Estimate price impact"""
        return abs(burst.peak_value)  # Peak value is already in basis points


class SpreadBurstDetector(BurstDetector):
    """Detects sudden spread changes"""
    
    def __init__(self, detection_window: int = 100):
        super().__init__(BurstType.SPREAD_BURST, detection_window)
    
    def extract_value(self, tick: MarketMicroTick) -> float:
        """Extract spread value"""
        return tick.spread_bps
    
    def _estimate_spread_impact(self, tick: MarketMicroTick, burst: MicroBurst) -> Optional[float]:
        """Estimate spread impact"""
        if burst.baseline_value > 0:
            return (burst.peak_value - burst.baseline_value) / burst.baseline_value
        return None


class ImbalanceBurstDetector(BurstDetector):
    """Detects order book imbalance changes"""
    
    def __init__(self, detection_window: int = 100):
        super().__init__(BurstType.IMBALANCE_BURST, detection_window)
    
    def extract_value(self, tick: MarketMicroTick) -> float:
        """Extract imbalance value"""
        return tick.book_imbalance


class MomentumBurstDetector(BurstDetector):
    """Detects directional momentum bursts"""
    
    def __init__(self, detection_window: int = 50):
        super().__init__(BurstType.MOMENTUM_BURST, detection_window)
        self.price_history = deque(maxlen=10)
    
    def extract_value(self, tick: MarketMicroTick) -> float:
        """Extract momentum value"""
        current_price = tick.last_price if tick.last_price is not None else tick.mid_price
        self.price_history.append(current_price)
        
        if len(self.price_history) < 5:
            return 0.0
        
        # Calculate momentum as slope of recent price changes
        prices = np.array(self.price_history)
        x = np.arange(len(prices))
        
        # Linear regression slope
        if len(prices) > 1:
            slope = np.polyfit(x, prices, 1)[0]
            # Normalize by average price
            avg_price = np.mean(prices)
            if avg_price > 0:
                return (slope / avg_price) * 10000  # Convert to basis points per tick
        
        return 0.0


class VolatilityBurstDetector(BurstDetector):
    """Detects volatility spikes"""
    
    def __init__(self, detection_window: int = 100):
        super().__init__(BurstType.VOLATILITY_BURST, detection_window)
        self.price_changes = deque(maxlen=20)
    
    def extract_value(self, tick: MarketMicroTick) -> float:
        """Extract volatility value"""
        current_price = tick.last_price if tick.last_price is not None else tick.mid_price
        
        if len(self.price_changes) > 0:
            last_price = self.price_changes[-1] if self.price_changes else current_price
            
            if last_price > 0:
                price_change = ((current_price - last_price) / last_price) * 10000
                self.price_changes.append(current_price)
                
                # Calculate rolling volatility (standard deviation of price changes)
                if len(self.price_changes) >= 10:
                    changes = []
                    for i in range(1, len(self.price_changes)):
                        prev_price = self.price_changes[i-1]
                        curr_price = self.price_changes[i]
                        if prev_price > 0:
                            change = ((curr_price - prev_price) / prev_price) * 10000
                            changes.append(change)
                    
                    if changes:
                        return np.std(changes)
        
        self.price_changes.append(current_price)
        return 0.0


class MicroBurstDetector:
    """
    Main micro burst detection system
    Coordinates multiple detection algorithms
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Initialize detectors
        self.detectors = {
            BurstType.VOLUME_BURST: VolumeBurstDetector(),
            BurstType.PRICE_BURST: PriceBurstDetector(),
            BurstType.SPREAD_BURST: SpreadBurstDetector(),
            BurstType.IMBALANCE_BURST: ImbalanceBurstDetector(),
            BurstType.MOMENTUM_BURST: MomentumBurstDetector(),
            BurstType.VOLATILITY_BURST: VolatilityBurstDetector()
        }
        
        # Event handling
        self.burst_handlers: List[Callable] = []
        
        # Statistics and monitoring
        self.detection_stats = {
            'total_ticks_processed': 0,
            'total_bursts_detected': 0,
            'bursts_by_type': defaultdict(int),
            'bursts_by_severity': defaultdict(int),
            'average_processing_time_us': 0.0
        }
        
        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        
        # Threading
        self.lock = threading.RLock()
    
    def process_tick(self, tick: MarketMicroTick) -> List[MicroBurst]:
        """
        Process market tick through all detectors
        
        Args:
            tick: Market microstructure tick
        
        Returns:
            List of all detected/updated bursts
        """
        
        start_time_ns = time.time_ns()
        all_bursts = []
        
        try:
            # Process through each detector
            for detector in self.detectors.values():
                try:
                    bursts = detector.process_tick(tick)
                    all_bursts.extend(bursts)
                except Exception as e:
                    print(f"Error in {detector.burst_type.value} detector: {e}")
            
            # Update statistics
            with self.lock:
                self.detection_stats['total_ticks_processed'] += 1
                
                for burst in all_bursts:
                    if burst.start_time_ns == tick.timestamp_ns:  # New burst
                        self.detection_stats['total_bursts_detected'] += 1
                        self.detection_stats['bursts_by_type'][burst.burst_type.value] += 1
                        self.detection_stats['bursts_by_severity'][burst.severity.value] += 1
                
                # Update processing time
                processing_time_us = (time.time_ns() - start_time_ns) / 1000.0
                self.processing_times.append(processing_time_us)
                
                if self.processing_times:
                    self.detection_stats['average_processing_time_us'] = np.mean(self.processing_times)
            
            # Emit burst events
            for burst in all_bursts:
                if burst.start_time_ns == tick.timestamp_ns:  # New burst
                    self._emit_burst_event('BURST_DETECTED', burst)
                elif burst.end_time_ns == tick.timestamp_ns:  # Completed burst
                    self._emit_burst_event('BURST_COMPLETED', burst)
                else:  # Updated burst
                    self._emit_burst_event('BURST_UPDATED', burst)
            
            return all_bursts
            
        except Exception as e:
            print(f"Error processing tick: {e}")
            return []
    
    def get_active_bursts(self, burst_type: Optional[BurstType] = None) -> List[MicroBurst]:
        """Get currently active bursts"""
        
        active_bursts = []
        
        for detector in self.detectors.values():
            if burst_type is None or detector.burst_type == burst_type:
                active_bursts.extend(detector.active_bursts.values())
        
        return active_bursts
    
    def get_completed_bursts(
        self,
        burst_type: Optional[BurstType] = None,
        limit: Optional[int] = None
    ) -> List[MicroBurst]:
        """Get completed bursts"""
        
        completed_bursts = []
        
        for detector in self.detectors.values():
            if burst_type is None or detector.burst_type == burst_type:
                completed_bursts.extend(detector.completed_bursts)
        
        # Sort by start time (most recent first)
        completed_bursts.sort(key=lambda b: b.start_time_ns, reverse=True)
        
        if limit:
            completed_bursts = completed_bursts[:limit]
        
        return completed_bursts
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection statistics"""
        
        with self.lock:
            stats = self.detection_stats.copy()
            
            # Add detector-specific statistics
            stats['detector_stats'] = {}
            for burst_type, detector in self.detectors.items():
                stats['detector_stats'][burst_type.value] = {
                    'detection_count': detector.detection_count,
                    'false_positive_count': detector.false_positive_count,
                    'active_bursts': len(detector.active_bursts),
                    'completed_bursts': len(detector.completed_bursts)
                }
            
            # Add performance metrics
            if self.processing_times:
                times_array = np.array(self.processing_times)
                stats['performance'] = {
                    'average_processing_time_us': np.mean(times_array),
                    'p95_processing_time_us': np.percentile(times_array, 95),
                    'p99_processing_time_us': np.percentile(times_array, 99),
                    'max_processing_time_us': np.max(times_array)
                }
            
            return stats
    
    def add_burst_handler(self, handler: Callable) -> None:
        """Add burst event handler"""
        self.burst_handlers.append(handler)
    
    def configure_detector(
        self,
        burst_type: BurstType,
        **kwargs
    ) -> None:
        """Configure specific detector parameters"""
        
        detector = self.detectors.get(burst_type)
        if detector:
            for key, value in kwargs.items():
                if hasattr(detector, key):
                    setattr(detector, key, value)
    
    def reset_statistics(self) -> None:
        """Reset detection statistics"""
        
        with self.lock:
            self.detection_stats = {
                'total_ticks_processed': 0,
                'total_bursts_detected': 0,
                'bursts_by_type': defaultdict(int),
                'bursts_by_severity': defaultdict(int),
                'average_processing_time_us': 0.0
            }
            
            self.processing_times.clear()
            
            # Reset detector statistics
            for detector in self.detectors.values():
                detector.detection_count = 0
                detector.false_positive_count = 0
                detector.completed_bursts.clear()
    
    def _emit_burst_event(self, event_type: str, burst: MicroBurst) -> None:
        """Emit burst event"""
        
        event = {
            'type': event_type,
            'burst': burst,
            'timestamp_ns': time.time_ns()
        }
        
        for handler in self.burst_handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in burst event handler: {e}")


if __name__ == "__main__":
    import random
    import time
    
    # Example usage and testing
    print("Testing Micro Burst Detector...")
    
    # Create detector
    detector = MicroBurstDetector("AAPL")
    
    # Add event handler
    def burst_event_handler(event):
        burst = event['burst']
        print(f"Burst Event: {event['type']} - {burst.burst_type.value} "
              f"{burst.severity.value} (magnitude: {burst.magnitude:.2f}σ)")
    
    detector.add_burst_handler(burst_event_handler)
    
    print("\nSimulating market data with micro bursts...")
    
    # Generate realistic market data with embedded bursts
    base_price = 150.0
    base_volume = 500.0
    
    for i in range(1000):
        # Create base market conditions
        spread_bps = random.uniform(1, 5)
        spread_dollars = (spread_bps / 10000) * base_price
        
        bid_price = base_price - spread_dollars / 2
        ask_price = base_price + spread_dollars / 2
        
        bid_size = base_volume + random.uniform(-100, 100)
        ask_size = base_volume + random.uniform(-100, 100)
        
        # Inject periodic bursts
        if i % 100 == 50:  # Volume burst
            last_size = base_volume * random.uniform(5, 10)  # 5-10x normal volume
        elif i % 150 == 75:  # Price burst
            price_shock = base_price * random.uniform(0.01, 0.03) * random.choice([-1, 1])
            base_price += price_shock
            bid_price += price_shock
            ask_price += price_shock
            last_size = base_volume * 2
        elif i % 200 == 100:  # Spread burst
            spread_multiplier = random.uniform(3, 8)
            spread_dollars *= spread_multiplier
            bid_price = base_price - spread_dollars / 2
            ask_price = base_price + spread_dollars / 2
            last_size = base_volume * 0.5
        else:
            last_size = base_volume + random.uniform(-50, 50)
        
        # Add some price drift
        base_price += random.uniform(-0.01, 0.01)
        
        # Create market tick
        tick = MarketMicroTick(
            symbol="AAPL",
            timestamp_ns=time.time_ns() - (1000 - i) * 1_000_000,  # 1ms intervals
            bid_price=bid_price,
            bid_size=max(100, bid_size),
            ask_price=ask_price,
            ask_size=max(100, ask_size),
            last_price=base_price + random.uniform(-spread_dollars/4, spread_dollars/4),
            last_size=max(10, last_size),
            trade_side=random.choice(['BUY', 'SELL', 'UNKNOWN']),
            total_bid_volume=bid_size * random.uniform(5, 15),
            total_ask_volume=ask_size * random.uniform(5, 15),
            bid_levels=random.randint(5, 20),
            ask_levels=random.randint(5, 20)
        )
        
        # Process tick
        bursts = detector.process_tick(tick)
        
        # Small delay to simulate realistic processing
        time.sleep(0.0001)  # 0.1ms
    
    print("\nAnalyzing detection results...")
    
    # Get detection statistics
    stats = detector.get_detection_statistics()
    
    print(f"Detection Statistics:")
    print(f"  Total Ticks Processed: {stats['total_ticks_processed']}")
    print(f"  Total Bursts Detected: {stats['total_bursts_detected']}")
    print(f"  Average Processing Time: {stats['average_processing_time_us']:.2f} μs")
    
    print(f"\n  Bursts by Type:")
    for burst_type, count in stats['bursts_by_type'].items():
        print(f"    {burst_type}: {count}")
    
    print(f"\n  Bursts by Severity:")
    for severity, count in stats['bursts_by_severity'].items():
        print(f"    {severity}: {count}")
    
    print(f"\n  Detector Performance:")
    for detector_type, detector_stats in stats['detector_stats'].items():
        print(f"    {detector_type}:")
        print(f"      Detections: {detector_stats['detection_count']}")
        print(f"      Active: {detector_stats['active_bursts']}")
        print(f"      Completed: {detector_stats['completed_bursts']}")
    
    if 'performance' in stats:
        perf = stats['performance']
        print(f"\n  Performance Metrics:")
        print(f"    Average: {perf['average_processing_time_us']:.2f} μs")
        print(f"    P95: {perf['p95_processing_time_us']:.2f} μs")
        print(f"    P99: {perf['p99_processing_time_us']:.2f} μs")
        print(f"    Max: {perf['max_processing_time_us']:.2f} μs")
    
    # Get active bursts
    active_bursts = detector.get_active_bursts()
    print(f"\n  Currently Active Bursts: {len(active_bursts)}")
    
    for burst in active_bursts[:5]:  # Show first 5
        print(f"    {burst.burst_type.value} {burst.severity.value} "
              f"(age: {(time.time_ns() - burst.start_time_ns) / 1_000_000:.1f}ms)")
    
    # Get completed bursts
    completed_bursts = detector.get_completed_bursts(limit=10)
    print(f"\n  Recent Completed Bursts: {len(completed_bursts)}")
    
    for burst in completed_bursts[:5]:  # Show first 5
        print(f"    {burst.burst_type.value} {burst.severity.value} "
              f"(duration: {burst.duration_ms:.1f}ms, magnitude: {burst.magnitude:.2f}σ)")
    
    # Test configuration
    print(f"\nTesting detector configuration...")
    
    # Make volume detector more sensitive
    detector.configure_detector(
        BurstType.VOLUME_BURST,
        low_threshold=1.5,
        medium_threshold=2.0
    )
    
    print("Volume detector sensitivity increased")
    
    # Test with more volume bursts
    for i in range(50):
        tick = MarketMicroTick(
            symbol="AAPL",
            timestamp_ns=time.time_ns(),
            bid_price=150.0,
            bid_size=500,
            ask_price=150.05,
            ask_size=500,
            last_size=base_volume * random.uniform(2, 4) if i % 10 == 0 else base_volume
        )
        
        detector.process_tick(tick)
        time.sleep(0.0001)
    
    # Check updated statistics
    final_stats = detector.get_detection_statistics()
    volume_detections = final_stats['bursts_by_type'].get('VOLUME_BURST', 0)
    
    print(f"Volume burst detections after sensitivity increase: {volume_detections}")
    
    print("\nMicro Burst Detector testing completed!")