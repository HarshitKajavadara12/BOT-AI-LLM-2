"""
Nanosecond Timer
High-precision timing utilities for ultra-low latency trading systems
"""

import time
import os
import platform
import ctypes
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod  
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import struct
import mmap
import warnings


class TimerPrecision(Enum):
    """Timer precision levels"""
    NANOSECOND = "NANOSECOND"     # Best available nanosecond precision
    MICROSECOND = "MICROSECOND"   # Microsecond precision
    MILLISECOND = "MILLISECOND"   # Millisecond precision
    SYSTEM = "SYSTEM"             # System default precision


class TimeSource(Enum):
    """Time source types"""
    SYSTEM_MONOTONIC = "SYSTEM_MONOTONIC"     # System monotonic clock
    SYSTEM_REALTIME = "SYSTEM_REALTIME"      # System realtime clock
    TSC = "TSC"                               # CPU Time Stamp Counter
    HPET = "HPET"                            # High Precision Event Timer
    RDTSC = "RDTSC"                          # Read Time Stamp Counter instruction


@dataclass
class TimingMeasurement:
    """Single timing measurement"""
    measurement_id: str
    start_time_ns: int
    end_time_ns: Optional[int] = None
    duration_ns: Optional[int] = None
    precision: TimerPrecision = TimerPrecision.NANOSECOND
    time_source: TimeSource = TimeSource.SYSTEM_MONOTONIC
    
    def complete(self, end_time_ns: Optional[int] = None) -> None:
        """Complete the timing measurement"""
        if end_time_ns is None:
            end_time_ns = NanosecondTimer.get_time_ns()
        
        self.end_time_ns = end_time_ns
        self.duration_ns = end_time_ns - self.start_time_ns
    
    @property
    def duration_us(self) -> Optional[float]:
        """Duration in microseconds"""
        return self.duration_ns / 1000.0 if self.duration_ns is not None else None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """Duration in milliseconds"""
        return self.duration_ns / 1_000_000.0 if self.duration_ns is not None else None
    
    @property
    def is_complete(self) -> bool:
        """Check if measurement is complete"""
        return self.end_time_ns is not None


class HighPrecisionClock:
    """
    High-precision clock abstraction
    """
    
    def __init__(self, time_source: TimeSource = TimeSource.SYSTEM_MONOTONIC):
        self.time_source = time_source
        self.calibration_offset_ns = 0
        self.frequency_hz = self._detect_frequency()
        
        # Platform-specific optimizations
        self.platform = platform.system().lower()
        self._setup_platform_optimizations()
    
    def get_time_ns(self) -> int:
        """Get current time in nanoseconds"""
        
        if self.time_source == TimeSource.SYSTEM_MONOTONIC:
            return time.time_ns()
        elif self.time_source == TimeSource.SYSTEM_REALTIME:
            return int(time.time() * 1_000_000_000)
        elif self.time_source == TimeSource.TSC:
            return self._get_tsc_time_ns()
        elif self.time_source == TimeSource.RDTSC:
            return self._get_rdtsc_time_ns()
        else:
            # Fallback to system monotonic
            return time.time_ns()
    
    def get_resolution_ns(self) -> int:
        """Get timer resolution in nanoseconds"""
        
        if self.time_source in [TimeSource.TSC, TimeSource.RDTSC]:
            # TSC resolution depends on CPU frequency
            if self.frequency_hz > 0:
                return int(1_000_000_000 / self.frequency_hz)
            else:
                return 1  # Assume 1ns resolution if frequency unknown
        else:
            # System clocks typically have 1ns resolution on modern systems
            return 1
    
    def calibrate(self, samples: int = 1000) -> Dict[str, float]:
        """
        Calibrate timer accuracy and precision
        
        Args:
            samples: Number of calibration samples
        
        Returns:
            Calibration results
        """
        
        # Measure timer overhead
        overhead_measurements = []
        for _ in range(samples):
            start = self.get_time_ns()
            end = self.get_time_ns()
            overhead_measurements.append(end - start)
        
        # Measure timer precision (consecutive calls)
        precision_measurements = []
        for _ in range(samples):
            times = [self.get_time_ns() for _ in range(10)]
            deltas = [times[i+1] - times[i] for i in range(len(times)-1)]
            precision_measurements.extend([d for d in deltas if d > 0])
        
        # Calculate statistics
        import numpy as np
        
        results = {
            'overhead_mean_ns': np.mean(overhead_measurements),
            'overhead_std_ns': np.std(overhead_measurements),
            'overhead_min_ns': np.min(overhead_measurements),
            'overhead_max_ns': np.max(overhead_measurements),
            'resolution_ns': self.get_resolution_ns(),
            'frequency_hz': self.frequency_hz
        }
        
        if precision_measurements:
            results.update({
                'precision_mean_ns': np.mean(precision_measurements),
                'precision_std_ns': np.std(precision_measurements),
                'precision_min_ns': np.min(precision_measurements)
            })
        
        return results
    
    def _detect_frequency(self) -> float:
        """Detect timer frequency"""
        
        if self.time_source in [TimeSource.TSC, TimeSource.RDTSC]:
            # Try to detect CPU frequency
            try:
                if self.platform == 'linux':
                    return self._detect_cpu_frequency_linux()
                elif self.platform == 'windows':
                    return self._detect_cpu_frequency_windows()
                else:
                    return 0.0
            except Exception:
                return 0.0
        else:
            return 1_000_000_000.0  # 1 GHz for nanosecond precision
    
    def _setup_platform_optimizations(self) -> None:
        """Setup platform-specific optimizations"""
        
        if self.platform == 'linux':
            self._setup_linux_optimizations()
        elif self.platform == 'windows':
            self._setup_windows_optimizations()
    
    def _setup_linux_optimizations(self) -> None:
        """Setup Linux-specific optimizations"""
        
        try:
            # Set high priority for timing thread
            import os
            os.nice(-19)  # Highest priority
        except (ImportError, PermissionError):
            pass
        
        try:
            # Try to use CLOCK_MONOTONIC_RAW if available
            import ctypes
            import ctypes.util
            
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            CLOCK_MONOTONIC_RAW = 4
            
            class TimeSpec(ctypes.Structure):
                _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]
            
            self._linux_timespec = TimeSpec()
            self._linux_clock_gettime = libc.clock_gettime
            self._linux_clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(TimeSpec)]
            self._linux_clock_id = CLOCK_MONOTONIC_RAW
            
        except Exception:
            pass
    
    def _setup_windows_optimizations(self) -> None:
        """Setup Windows-specific optimizations"""
        
        try:
            # Use QueryPerformanceCounter for high precision
            import ctypes
            from ctypes import wintypes
            
            kernel32 = ctypes.windll.kernel32
            
            # QueryPerformanceCounter
            self._qpc = kernel32.QueryPerformanceCounter
            self._qpc.argtypes = [ctypes.POINTER(wintypes.LARGE_INTEGER)]
            self._qpc.restype = wintypes.BOOL
            
            # QueryPerformanceFrequency
            self._qpf = kernel32.QueryPerformanceFrequency
            self._qpf.argtypes = [ctypes.POINTER(wintypes.LARGE_INTEGER)]
            self._qpf.restype = wintypes.BOOL
            
            # Get frequency
            freq = wintypes.LARGE_INTEGER()
            self._qpf(ctypes.byref(freq))
            self.frequency_hz = freq.value
            
        except Exception:
            pass
    
    def _detect_cpu_frequency_linux(self) -> float:
        """Detect CPU frequency on Linux"""
        
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'cpu MHz' in line:
                        mhz = float(line.split(':')[1].strip())
                        return mhz * 1_000_000  # Convert to Hz
        except Exception:
            pass
        
        # Try alternative method
        try:
            with open('/sys/devices/system/cpu/cpu0/cpufreq/base_frequency', 'r') as f:
                khz = int(f.read().strip())
                return khz * 1000  # Convert to Hz
        except Exception:
            pass
        
        return 0.0
    
    def _detect_cpu_frequency_windows(self) -> float:
        """Detect CPU frequency on Windows"""
        
        try:
            import winreg
            
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            
            mhz, _ = winreg.QueryValueEx(key, "~MHz")
            winreg.CloseKey(key)
            
            return mhz * 1_000_000  # Convert to Hz
            
        except Exception:
            pass
        
        return 0.0
    
    def _get_tsc_time_ns(self) -> int:
        """Get TSC-based time (simplified implementation)"""
        
        # This is a simplified version - real implementation would require
        # platform-specific assembly or system calls
        return time.time_ns()
    
    def _get_rdtsc_time_ns(self) -> int:
        """Get RDTSC-based time (simplified implementation)"""
        
        # This is a simplified version - real implementation would use
        # the RDTSC instruction directly
        return time.time_ns()


class NanosecondTimer:
    """
    High-precision nanosecond timer
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, time_source: TimeSource = TimeSource.SYSTEM_MONOTONIC):
        self.clock = HighPrecisionClock(time_source)
        self.active_measurements: Dict[str, TimingMeasurement] = {}
        self.completed_measurements: List[TimingMeasurement] = []
        
        # Statistics
        self.measurement_count = 0
        self.total_measurement_time_ns = 0
        
        # Threading
        self.lock = threading.RLock()
    
    @classmethod
    def get_instance(cls, time_source: TimeSource = TimeSource.SYSTEM_MONOTONIC) -> 'NanosecondTimer':
        """Get singleton instance"""
        
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(time_source)
        
        return cls._instance
    
    @staticmethod
    def get_time_ns() -> int:
        """Get current time in nanoseconds (static method)"""
        return time.time_ns()
    
    @staticmethod
    def get_time_us() -> float:
        """Get current time in microseconds (static method)"""
        return time.time_ns() / 1000.0
    
    @staticmethod
    def get_time_ms() -> float:
        """Get current time in milliseconds (static method)"""
        return time.time_ns() / 1_000_000.0
    
    @staticmethod
    def sleep_ns(nanoseconds: int) -> None:
        """
        Sleep for specified nanoseconds (busy wait for precision)
        
        Args:
            nanoseconds: Time to sleep in nanoseconds
        """
        
        if nanoseconds <= 0:
            return
        
        end_time = NanosecondTimer.get_time_ns() + nanoseconds
        
        # Use busy wait for sub-microsecond precision
        if nanoseconds < 1000:  # < 1μs
            while NanosecondTimer.get_time_ns() < end_time:
                pass
        else:
            # Use system sleep for longer durations, then busy wait for precision
            sleep_time_s = (nanoseconds - 1000) / 1_000_000_000.0
            time.sleep(sleep_time_s)
            
            # Busy wait for remaining time
            while NanosecondTimer.get_time_ns() < end_time:
                pass
    
    @staticmethod
    def sleep_us(microseconds: float) -> None:
        """Sleep for specified microseconds"""
        NanosecondTimer.sleep_ns(int(microseconds * 1000))
    
    @staticmethod
    def sleep_ms(milliseconds: float) -> None:
        """Sleep for specified milliseconds"""  
        NanosecondTimer.sleep_ns(int(milliseconds * 1_000_000))
    
    def start_measurement(self, measurement_id: str) -> TimingMeasurement:
        """
        Start a timing measurement
        
        Args:
            measurement_id: Unique identifier for measurement
        
        Returns:
            TimingMeasurement object
        """
        
        start_time_ns = self.clock.get_time_ns()
        
        measurement = TimingMeasurement(
            measurement_id=measurement_id,
            start_time_ns=start_time_ns,
            time_source=self.clock.time_source
        )
        
        with self.lock:
            self.active_measurements[measurement_id] = measurement
        
        return measurement
    
    def end_measurement(self, measurement_id: str) -> Optional[TimingMeasurement]:
        """
        End a timing measurement
        
        Args:
            measurement_id: Measurement identifier
        
        Returns:
            Completed TimingMeasurement or None if not found
        """
        
        end_time_ns = self.clock.get_time_ns()
        
        with self.lock:
            measurement = self.active_measurements.pop(measurement_id, None)
            
            if measurement:
                measurement.complete(end_time_ns)
                self.completed_measurements.append(measurement)
                
                # Update statistics
                self.measurement_count += 1
                if measurement.duration_ns:
                    self.total_measurement_time_ns += measurement.duration_ns
                
                return measurement
        
        return None
    
    def get_measurement(self, measurement_id: str) -> Optional[TimingMeasurement]:
        """Get active or completed measurement"""
        
        with self.lock:
            # Check active measurements first
            if measurement_id in self.active_measurements:
                return self.active_measurements[measurement_id]
            
            # Search completed measurements
            for measurement in reversed(self.completed_measurements):
                if measurement.measurement_id == measurement_id:
                    return measurement
        
        return None
    
    def measure_latency(self, func: Callable, *args, **kwargs) -> Tuple[Any, TimingMeasurement]:
        """
        Measure function execution latency
        
        Args:
            func: Function to measure
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            (function_result, timing_measurement)
        """
        
        measurement_id = f"func_{func.__name__}_{id(func)}_{time.time_ns()}"
        
        measurement = self.start_measurement(measurement_id)
        
        try:
            result = func(*args, **kwargs)
        finally:
            self.end_measurement(measurement_id)
        
        return result, measurement
    
    def get_timer_statistics(self) -> Dict[str, Any]:
        """Get timer statistics"""
        
        with self.lock:
            stats = {
                'measurement_count': self.measurement_count,
                'active_measurements': len(self.active_measurements),
                'completed_measurements': len(self.completed_measurements),
                'total_measurement_time_ns': self.total_measurement_time_ns,
                'clock_statistics': {}
            }
            
            # Add average measurement time
            if self.measurement_count > 0:
                stats['average_measurement_time_ns'] = (
                    self.total_measurement_time_ns / self.measurement_count
                )
                stats['average_measurement_time_us'] = stats['average_measurement_time_ns'] / 1000.0
            
            # Add clock calibration data
            try:
                calibration = self.clock.calibrate(samples=100)
                stats['clock_statistics'] = calibration
            except Exception as e:
                stats['clock_statistics'] = {'calibration_error': str(e)}
            
            return stats
    
    def benchmark_timer_overhead(self, iterations: int = 10000) -> Dict[str, float]:
        """
        Benchmark timer overhead
        
        Args:
            iterations: Number of benchmark iterations
        
        Returns:
            Benchmark results
        """
        
        import numpy as np
        
        # Measure get_time_ns overhead
        get_time_measurements = []
        for _ in range(iterations):
            start = time.time_ns()
            _ = self.clock.get_time_ns()
            end = time.time_ns()
            get_time_measurements.append(end - start)
        
        # Measure start/end measurement overhead
        measurement_overheads = []
        for i in range(iterations):
            measurement_id = f"benchmark_{i}"
            
            start = time.time_ns()
            self.start_measurement(measurement_id)
            self.end_measurement(measurement_id)
            end = time.time_ns()
            
            measurement_overheads.append(end - start)
        
        # Calculate statistics
        return {
            'get_time_mean_ns': np.mean(get_time_measurements),
            'get_time_std_ns': np.std(get_time_measurements),
            'get_time_min_ns': np.min(get_time_measurements),
            'get_time_max_ns': np.max(get_time_measurements),
            'measurement_mean_ns': np.mean(measurement_overheads),
            'measurement_std_ns': np.std(measurement_overheads),
            'measurement_min_ns': np.min(measurement_overheads),
            'measurement_max_ns': np.max(measurement_overheads),
            'iterations': iterations
        }
    
    def clear_completed_measurements(self) -> None:
        """Clear completed measurements to free memory"""
        
        with self.lock:
            self.completed_measurements.clear()


class LatencyProfiler:
    """
    Advanced latency profiling system
    """
    
    def __init__(self, name: str = "LatencyProfiler"):
        self.name = name
        self.timer = NanosecondTimer.get_instance()
        
        # Profiling data
        self.profiles: Dict[str, List[TimingMeasurement]] = {}
        self.hierarchical_profiles: Dict[str, Dict] = {}
        
        # Context stack for nested profiling
        self.context_stack: List[str] = []
        
        # Statistics
        self.profile_stats: Dict[str, Dict[str, float]] = {}
        
        # Threading
        self.lock = threading.RLock()
    
    def profile(self, name: str):
        """
        Context manager for profiling code blocks
        
        Args:
            name: Profile name
        
        Usage:
            with profiler.profile("my_function"):
                # code to profile
                pass
        """
        
        return ProfileContext(self, name)
    
    def start_profile(self, name: str) -> str:
        """Start profiling a named section"""
        
        measurement_id = f"{name}_{time.time_ns()}"
        self.timer.start_measurement(measurement_id)
        
        with self.lock:
            if name not in self.profiles:
                self.profiles[name] = []
            
            # Track context for hierarchical profiling
            self.context_stack.append(measurement_id)
        
        return measurement_id
    
    def end_profile(self, measurement_id: str) -> Optional[TimingMeasurement]:
        """End profiling section"""
        
        measurement = self.timer.end_measurement(measurement_id)
        
        if measurement:
            name = measurement.measurement_id.rsplit('_', 1)[0]  # Extract name from ID
            
            with self.lock:
                if name in self.profiles:
                    self.profiles[name].append(measurement)
                
                # Update hierarchical profiles
                if len(self.context_stack) > 1:
                    parent_id = self.context_stack[-2]
                    parent_name = parent_id.rsplit('_', 1)[0]
                    
                    if parent_name not in self.hierarchical_profiles:
                        self.hierarchical_profiles[parent_name] = {}
                    
                    if name not in self.hierarchical_profiles[parent_name]:
                        self.hierarchical_profiles[parent_name][name] = []
                    
                    self.hierarchical_profiles[parent_name][name].append(measurement)
                
                # Remove from context stack
                if measurement_id in self.context_stack:
                    self.context_stack.remove(measurement_id)
        
        return measurement
    
    def get_profile_statistics(self, name: str) -> Optional[Dict[str, float]]:
        """Get statistics for a named profile"""
        
        with self.lock:
            if name not in self.profiles or not self.profiles[name]:
                return None
            
            measurements = self.profiles[name]
            durations = [m.duration_ns for m in measurements if m.duration_ns is not None]
            
            if not durations:
                return None
            
            import numpy as np
            
            stats = {
                'count': len(durations),
                'mean_ns': np.mean(durations),
                'std_ns': np.std(durations),
                'min_ns': np.min(durations),
                'max_ns': np.max(durations),
                'p50_ns': np.percentile(durations, 50),
                'p95_ns': np.percentile(durations, 95),
                'p99_ns': np.percentile(durations, 99),
                'total_ns': np.sum(durations)
            }
            
            # Add microsecond and millisecond versions
            for key in ['mean', 'std', 'min', 'max', 'p50', 'p95', 'p99', 'total']:
                stats[f'{key}_us'] = stats[f'{key}_ns'] / 1000.0
                stats[f'{key}_ms'] = stats[f'{key}_ns'] / 1_000_000.0
            
            return stats
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all profiles"""
        
        with self.lock:
            all_stats = {}
            
            for name in self.profiles:
                stats = self.get_profile_statistics(name)
                if stats:
                    all_stats[name] = stats
            
            return all_stats
    
    def get_hierarchical_statistics(self) -> Dict[str, Dict]:
        """Get hierarchical profiling statistics"""
        
        with self.lock:
            hierarchical_stats = {}
            
            for parent_name, children in self.hierarchical_profiles.items():
                hierarchical_stats[parent_name] = {}
                
                for child_name, measurements in children.items():
                    durations = [m.duration_ns for m in measurements if m.duration_ns is not None]
                    
                    if durations:
                        import numpy as np
                        
                        hierarchical_stats[parent_name][child_name] = {
                            'count': len(durations),
                            'mean_ns': np.mean(durations),
                            'total_ns': np.sum(durations),
                            'percentage': 0.0  # Will be calculated below
                        }
                
                # Calculate percentages
                parent_stats = self.get_profile_statistics(parent_name)
                if parent_stats:
                    parent_total = parent_stats['total_ns']
                    
                    for child_name in hierarchical_stats[parent_name]:
                        child_total = hierarchical_stats[parent_name][child_name]['total_ns']
                        hierarchical_stats[parent_name][child_name]['percentage'] = (
                            (child_total / parent_total) * 100.0 if parent_total > 0 else 0.0
                        )
            
            return hierarchical_stats
    
    def clear_profiles(self) -> None:
        """Clear all profiling data"""
        
        with self.lock:
            self.profiles.clear()
            self.hierarchical_profiles.clear()
            self.context_stack.clear()
            self.profile_stats.clear()
    
    def print_report(self, sort_by: str = 'mean_ns') -> None:
        """Print profiling report"""
        
        print(f"\nLatency Profiling Report - {self.name}")
        print("=" * 60)
        
        all_stats = self.get_all_statistics()
        
        if not all_stats:
            print("No profiling data available")
            return
        
        # Sort profiles by specified metric
        sorted_profiles = sorted(
            all_stats.items(),
            key=lambda x: x[1].get(sort_by, 0),
            reverse=True
        )
        
        # Print header  
        print(f"{'Profile Name':<30} {'Count':<8} {'Mean (μs)':<12} {'P95 (μs)':<12} {'P99 (μs)':<12} {'Total (ms)':<12}")
        print("-" * 100)
        
        # Print profile statistics
        for name, stats in sorted_profiles:
            print(f"{name:<30} {stats['count']:<8} "
                  f"{stats['mean_us']:<12.2f} {stats['p95_us']:<12.2f} "
                  f"{stats['p99_us']:<12.2f} {stats['total_ms']:<12.2f}")
        
        # Print hierarchical information
        hierarchical_stats = self.get_hierarchical_statistics()
        
        if hierarchical_stats:
            print(f"\nHierarchical Breakdown:")
            print("-" * 60)
            
            for parent_name, children in hierarchical_stats.items():
                print(f"\n{parent_name}:")
                
                for child_name, child_stats in children.items():
                    print(f"  {child_name:<25} {child_stats['percentage']:<6.1f}% "
                          f"({child_stats['mean_ns']/1000:.2f} μs avg)")


class ProfileContext:
    """Context manager for profiling"""
    
    def __init__(self, profiler: LatencyProfiler, name: str):
        self.profiler = profiler
        self.name = name
        self.measurement_id = None
    
    def __enter__(self):
        self.measurement_id = self.profiler.start_profile(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.measurement_id:
            self.profiler.end_profile(self.measurement_id)


if __name__ == "__main__":
    import random
    import numpy as np
    
    # Example usage and testing
    print("Testing Nanosecond Timer...")
    
    # Test basic timer functionality
    print("\nBasic Timer Tests:")
    
    timer = NanosecondTimer.get_instance()
    
    # Test static methods
    start_time = NanosecondTimer.get_time_ns()
    NanosecondTimer.sleep_us(100)  # Sleep 100 microseconds
    end_time = NanosecondTimer.get_time_ns()
    
    sleep_duration_us = (end_time - start_time) / 1000.0
    print(f"Sleep test: Requested 100μs, actual {sleep_duration_us:.2f}μs")
    
    # Test measurement functionality
    measurement = timer.start_measurement("test_measurement")
    
    # Simulate some work
    for i in range(1000):
        _ = random.random() * random.random()
    
    completed_measurement = timer.end_measurement("test_measurement")
    
    if completed_measurement:
        print(f"Work simulation took {completed_measurement.duration_us:.2f}μs")
    
    # Test function latency measurement
    def test_function(n: int) -> int:
        """Test function for latency measurement"""
        total = 0
        for i in range(n):
            total += i * i
        return total
    
    result, measurement = timer.measure_latency(test_function, 10000)
    print(f"Function execution took {measurement.duration_us:.2f}μs (result: {result})")
    
    # Test timer statistics
    print(f"\nTimer Statistics:")
    stats = timer.get_timer_statistics()
    
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    # Benchmark timer overhead
    print(f"\nBenchmarking Timer Overhead:")
    benchmark_results = timer.benchmark_timer_overhead(iterations=5000)
    
    for key, value in benchmark_results.items():
        if 'ns' in key:
            print(f"  {key}: {value:.2f} ns")
        else:
            print(f"  {key}: {value}")
    
    # Test profiler
    print(f"\nTesting Latency Profiler:")
    
    profiler = LatencyProfiler("TestProfiler")
    
    # Profile some operations
    for iteration in range(100):
        with profiler.profile("main_loop"):
            
            # Simulate data processing
            with profiler.profile("data_processing"):
                data = np.random.random(1000)
                processed_data = np.fft.fft(data)
            
            # Simulate algorithm execution
            with profiler.profile("algorithm"):
                # Simulate different algorithm complexities
                if iteration % 10 == 0:
                    # Expensive operation occasionally
                    with profiler.profile("expensive_operation"):
                        time.sleep(0.001)  # 1ms
                else:
                    # Quick operation most of the time
                    with profiler.profile("quick_operation"):
                        result = sum(range(100))
            
            # Simulate I/O operation
            with profiler.profile("io_simulation"):
                # Simulate variable I/O latency
                delay_us = random.uniform(10, 100)
                NanosecondTimer.sleep_us(delay_us)
    
    # Print profiling report
    profiler.print_report()
    
    # Test high-precision clock calibration
    print(f"\nHigh-Precision Clock Calibration:")
    
    clock = HighPrecisionClock()
    calibration = clock.calibrate(samples=1000)
    
    print(f"Clock Calibration Results:")
    for key, value in calibration.items():
        if isinstance(value, float):
            if 'ns' in key:
                print(f"  {key}: {value:.2f} ns")
            elif 'hz' in key:
                print(f"  {key}: {value:.0f} Hz")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test precision of timer
    print(f"\nTimer Precision Test:")
    
    precision_measurements = []
    for _ in range(1000):
        start = NanosecondTimer.get_time_ns()
        end = NanosecondTimer.get_time_ns()
        precision_measurements.append(end - start)
    
    # Filter out zero measurements  
    non_zero_measurements = [m for m in precision_measurements if m > 0]
    
    if non_zero_measurements:
        min_precision = min(non_zero_measurements)
        avg_precision = sum(non_zero_measurements) / len(non_zero_measurements)
        print(f"  Minimum measurable time: {min_precision} ns")
        print(f"  Average overhead: {avg_precision:.2f} ns")
        print(f"  Zero measurements: {len(precision_measurements) - len(non_zero_measurements)}")
    
    # Test sleep precision
    print(f"\nSleep Precision Test:")
    
    sleep_tests = [
        (100, "100ns"),
        (1000, "1μs"),
        (10000, "10μs"),
        (100000, "100μs")
    ]
    
    for sleep_ns, description in sleep_tests:
        measurements = []
        
        for _ in range(100):
            start = NanosecondTimer.get_time_ns()
            NanosecondTimer.sleep_ns(sleep_ns)
            end = NanosecondTimer.get_time_ns()
            measurements.append(end - start)
        
        avg_actual = sum(measurements) / len(measurements)
        error_ns = avg_actual - sleep_ns
        error_percent = (error_ns / sleep_ns) * 100 if sleep_ns > 0 else 0
        
        print(f"  {description}: Requested {sleep_ns}ns, "
              f"Average {avg_actual:.0f}ns "
              f"(error: {error_percent:+.1f}%)")
    
    print("\nNanosecond Timer testing completed!")