"""
Change Point Detection Framework
Advanced statistical methods for detecting structural breaks and regime changes in financial time series
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
import math
from collections import defaultdict, deque
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, argrelextrema
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class ChangePointMethod(Enum):
    """Change point detection methods"""
    CUSUM = "CUSUM"                         # Cumulative Sum
    PELT = "PELT"                          # Pruned Exact Linear Time
    BINARY_SEGMENTATION = "BINARY_SEGMENTATION"  # Binary Segmentation
    BAYESIAN = "BAYESIAN"                  # Bayesian Change Point Detection
    KERNEL = "KERNEL"                      # Kernel Change Point Detection
    EWMA = "EWMA"                         # Exponentially Weighted Moving Average
    PAGE_HINKLEY = "PAGE_HINKLEY"         # Page-Hinkley Test


class ChangePointType(Enum):
    """Types of change points"""
    MEAN_SHIFT = "MEAN_SHIFT"             # Change in mean
    VARIANCE_SHIFT = "VARIANCE_SHIFT"     # Change in variance
    TREND_CHANGE = "TREND_CHANGE"         # Change in trend
    CORRELATION_CHANGE = "CORRELATION_CHANGE"  # Change in correlation structure
    DISTRIBUTION_CHANGE = "DISTRIBUTION_CHANGE"  # Change in distribution


@dataclass
class ChangePoint:
    """Individual change point information"""
    
    # Location information
    index: int                            # Index in time series
    date: Optional[datetime] = None       # Date if available
    confidence: float = 0.0               # Confidence level (0-1)
    
    # Change characteristics
    change_type: str = "UNKNOWN"          # Type of change
    magnitude: float = 0.0                # Magnitude of change
    direction: int = 0                    # Direction: -1 (decrease), 0 (neutral), 1 (increase)
    
    # Statistical properties
    test_statistic: float = 0.0           # Test statistic value
    p_value: float = 1.0                  # P-value if available
    
    # Before/after statistics
    before_mean: float = 0.0
    after_mean: float = 0.0
    before_var: float = 0.0
    after_var: float = 0.0
    
    # Segment information
    before_segment_length: int = 0
    after_segment_length: int = 0
    
    # Additional metadata
    method_used: str = ""
    threshold_used: float = 0.0


@dataclass
class ChangePointResults:
    """Results from change point detection"""
    
    # Method information
    method: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Detected change points
    change_points: List[ChangePoint] = field(default_factory=list)
    n_change_points: int = 0
    
    # Time series information
    original_series: Optional[pd.Series] = None
    dates: Optional[pd.DatetimeIndex] = None
    
    # Segmentation results
    segments: List[Dict[str, Any]] = field(default_factory=list)
    segment_statistics: List[Dict[str, float]] = field(default_factory=list)
    
    # Model diagnostics
    total_cost: float = 0.0               # Total segmentation cost
    penalty: float = 0.0                  # Penalty for number of change points
    bic: float = 0.0                      # Bayesian Information Criterion
    aic: float = 0.0                      # Akaike Information Criterion
    
    # Detection statistics
    detection_threshold: float = 0.0
    false_alarm_rate: float = 0.0


class ChangePointDetector(ABC):
    """Abstract base class for change point detection"""
    
    @abstractmethod
    def detect(self, series: pd.Series, **kwargs) -> ChangePointResults:
        """Detect change points in time series"""
        pass


class CUSUMDetector(ChangePointDetector):
    """Cumulative Sum (CUSUM) change point detection"""
    
    def __init__(self, 
                 threshold: float = 5.0,
                 drift: float = 0.5,
                 reset_threshold: float = -5.0):
        
        self.threshold = threshold
        self.drift = drift
        self.reset_threshold = reset_threshold
    
    def detect(self, series: pd.Series, **kwargs) -> ChangePointResults:
        """Detect change points using CUSUM method"""
        
        if len(series) < 3:
            return ChangePointResults(method="CUSUM")
        
        # Standardize series
        standardized = (series - series.mean()) / series.std()
        
        # CUSUM for positive and negative changes
        cusum_pos = np.zeros(len(standardized))
        cusum_neg = np.zeros(len(standardized))
        
        change_points = []
        
        for i in range(1, len(standardized)):
            # Positive CUSUM (upward changes)
            cusum_pos[i] = max(0, cusum_pos[i-1] + standardized.iloc[i] - self.drift)
            
            # Negative CUSUM (downward changes)  
            cusum_neg[i] = min(0, cusum_neg[i-1] + standardized.iloc[i] + self.drift)
            
            # Check for change points
            if cusum_pos[i] > self.threshold:
                # Upward change detected
                change_point = ChangePoint(
                    index=i,
                    date=series.index[i] if hasattr(series.index, 'to_pydatetime') else None,
                    change_type="MEAN_SHIFT",
                    direction=1,
                    test_statistic=cusum_pos[i],
                    method_used="CUSUM_POSITIVE",
                    threshold_used=self.threshold
                )
                
                # Calculate before/after statistics
                before_data = series.iloc[max(0, i-20):i]
                after_data = series.iloc[i:min(len(series), i+20)]
                
                if len(before_data) > 0 and len(after_data) > 0:
                    change_point.before_mean = before_data.mean()
                    change_point.after_mean = after_data.mean()
                    change_point.magnitude = change_point.after_mean - change_point.before_mean
                    change_point.before_var = before_data.var()
                    change_point.after_var = after_data.var()
                
                change_points.append(change_point)
                cusum_pos[i] = 0  # Reset
            
            elif cusum_neg[i] < self.reset_threshold:
                # Downward change detected
                change_point = ChangePoint(
                    index=i,
                    date=series.index[i] if hasattr(series.index, 'to_pydatetime') else None,
                    change_type="MEAN_SHIFT",
                    direction=-1,
                    test_statistic=cusum_neg[i],
                    method_used="CUSUM_NEGATIVE",
                    threshold_used=self.reset_threshold
                )
                
                # Calculate before/after statistics
                before_data = series.iloc[max(0, i-20):i]
                after_data = series.iloc[i:min(len(series), i+20)]
                
                if len(before_data) > 0 and len(after_data) > 0:
                    change_point.before_mean = before_data.mean()
                    change_point.after_mean = after_data.mean()
                    change_point.magnitude = change_point.after_mean - change_point.before_mean
                    change_point.before_var = before_data.var()
                    change_point.after_var = after_data.var()
                
                change_points.append(change_point)
                cusum_neg[i] = 0  # Reset
        
        # Create results
        results = ChangePointResults(
            method="CUSUM",
            parameters={
                'threshold': self.threshold,
                'drift': self.drift,
                'reset_threshold': self.reset_threshold
            },
            change_points=change_points,
            n_change_points=len(change_points),
            original_series=series,
            dates=series.index,
            detection_threshold=self.threshold
        )
        
        # Create segments
        results.segments = self._create_segments(series, change_points)
        
        return results
    
    def _create_segments(self, series: pd.Series, 
                        change_points: List[ChangePoint]) -> List[Dict[str, Any]]:
        """Create segments between change points"""
        
        if not change_points:
            return [{
                'start_idx': 0,
                'end_idx': len(series) - 1,
                'start_date': series.index[0],
                'end_date': series.index[-1],
                'length': len(series),
                'mean': series.mean(),
                'std': series.std(),
                'data': series
            }]
        
        segments = []
        change_indices = sorted([cp.index for cp in change_points])
        
        # First segment
        start_idx = 0
        end_idx = change_indices[0]
        segment_data = series.iloc[start_idx:end_idx]
        
        segments.append({
            'start_idx': start_idx,
            'end_idx': end_idx - 1,
            'start_date': series.index[start_idx],
            'end_date': series.index[end_idx - 1],
            'length': len(segment_data),
            'mean': segment_data.mean() if len(segment_data) > 0 else 0,
            'std': segment_data.std() if len(segment_data) > 0 else 0,
            'data': segment_data
        })
        
        # Middle segments
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            segment_data = series.iloc[start_idx:end_idx]
            
            segments.append({
                'start_idx': start_idx,
                'end_idx': end_idx - 1,
                'start_date': series.index[start_idx],
                'end_date': series.index[end_idx - 1],
                'length': len(segment_data),
                'mean': segment_data.mean() if len(segment_data) > 0 else 0,
                'std': segment_data.std() if len(segment_data) > 0 else 0,
                'data': segment_data
            })
        
        # Last segment
        start_idx = change_indices[-1]
        segment_data = series.iloc[start_idx:]
        
        segments.append({
            'start_idx': start_idx,
            'end_idx': len(series) - 1,
            'start_date': series.index[start_idx],
            'end_date': series.index[-1],
            'length': len(segment_data),
            'mean': segment_data.mean() if len(segment_data) > 0 else 0,
            'std': segment_data.std() if len(segment_data) > 0 else 0,
            'data': segment_data
        })
        
        return segments


class BinarySegmentationDetector(ChangePointDetector):
    """Binary Segmentation change point detection"""
    
    def __init__(self, 
                 min_segment_length: int = 30,
                 max_change_points: int = 10,
                 significance_level: float = 0.05):
        
        self.min_segment_length = min_segment_length
        self.max_change_points = max_change_points
        self.significance_level = significance_level
    
    def detect(self, series: pd.Series, **kwargs) -> ChangePointResults:
        """Detect change points using binary segmentation"""
        
        if len(series) < 2 * self.min_segment_length:
            return ChangePointResults(method="BINARY_SEGMENTATION")
        
        standardized = (series - series.mean()) / series.std()
        
        # Find change points recursively
        change_points = []
        segments_to_process = [(0, len(standardized) - 1)]
        
        while segments_to_process and len(change_points) < self.max_change_points:
            start, end = segments_to_process.pop(0)
            
            if end - start < 2 * self.min_segment_length:
                continue
            
            # Find best change point in this segment
            best_change_point = self._find_best_change_point(
                standardized.iloc[start:end+1], start
            )
            
            if best_change_point is not None:
                change_points.append(best_change_point)
                
                # Add new segments to process
                cp_idx = best_change_point.index
                
                if cp_idx - start >= 2 * self.min_segment_length:
                    segments_to_process.append((start, cp_idx - 1))
                
                if end - cp_idx >= 2 * self.min_segment_length:
                    segments_to_process.append((cp_idx, end))
        
        # Sort change points by index
        change_points.sort(key=lambda cp: cp.index)
        
        # Create results
        results = ChangePointResults(
            method="BINARY_SEGMENTATION",
            parameters={
                'min_segment_length': self.min_segment_length,
                'max_change_points': self.max_change_points,
                'significance_level': self.significance_level
            },
            change_points=change_points,
            n_change_points=len(change_points),
            original_series=series,
            dates=series.index
        )
        
        # Create segments
        results.segments = self._create_segments(series, change_points)
        
        return results
    
    def _find_best_change_point(self, segment: pd.Series, 
                               offset: int) -> Optional[ChangePoint]:
        """Find best change point within a segment"""
        
        n = len(segment)
        best_statistic = 0
        best_idx = -1
        
        # Try each possible change point location
        for i in range(self.min_segment_length, n - self.min_segment_length):
            # Split segment
            left = segment.iloc[:i]
            right = segment.iloc[i:]
            
            # Calculate test statistic (difference in means scaled by pooled variance)
            if len(left) > 1 and len(right) > 1:
                mean_diff = abs(right.mean() - left.mean())
                pooled_var = ((len(left) - 1) * left.var() + (len(right) - 1) * right.var()) / (n - 2)
                
                if pooled_var > 0:
                    test_statistic = mean_diff / np.sqrt(pooled_var * (1/len(left) + 1/len(right)))
                    
                    if test_statistic > best_statistic:
                        best_statistic = test_statistic
                        best_idx = i
        
        # Check significance
        if best_idx >= 0:
            # Approximate p-value using t-distribution
            df = n - 2
            p_value = 2 * (1 - stats.t.cdf(best_statistic, df))
            
            if p_value < self.significance_level:
                actual_idx = offset + best_idx
                
                change_point = ChangePoint(
                    index=actual_idx,
                    date=segment.index[best_idx] if hasattr(segment.index, 'to_pydatetime') else None,
                    change_type="MEAN_SHIFT",
                    test_statistic=best_statistic,
                    p_value=p_value,
                    confidence=1 - p_value,
                    method_used="BINARY_SEGMENTATION"
                )
                
                # Calculate before/after statistics
                left_data = segment.iloc[:best_idx]
                right_data = segment.iloc[best_idx:]
                
                change_point.before_mean = left_data.mean()
                change_point.after_mean = right_data.mean()
                change_point.magnitude = change_point.after_mean - change_point.before_mean
                change_point.direction = 1 if change_point.magnitude > 0 else -1
                change_point.before_var = left_data.var()
                change_point.after_var = right_data.var()
                
                return change_point
        
        return None
    
    def _create_segments(self, series: pd.Series, 
                        change_points: List[ChangePoint]) -> List[Dict[str, Any]]:
        """Create segments between change points"""
        
        if not change_points:
            return [{
                'start_idx': 0,
                'end_idx': len(series) - 1,
                'start_date': series.index[0],
                'end_date': series.index[-1],
                'length': len(series),
                'mean': series.mean(),
                'std': series.std(),
                'data': series
            }]
        
        segments = []
        change_indices = sorted([cp.index for cp in change_points])
        
        # Create segments
        prev_idx = 0
        for cp_idx in change_indices:
            segment_data = series.iloc[prev_idx:cp_idx]
            
            if len(segment_data) > 0:
                segments.append({
                    'start_idx': prev_idx,
                    'end_idx': cp_idx - 1,
                    'start_date': series.index[prev_idx],
                    'end_date': series.index[cp_idx - 1],
                    'length': len(segment_data),
                    'mean': segment_data.mean(),
                    'std': segment_data.std(),
                    'data': segment_data
                })
            
            prev_idx = cp_idx
        
        # Last segment
        segment_data = series.iloc[prev_idx:]
        if len(segment_data) > 0:
            segments.append({
                'start_idx': prev_idx,
                'end_idx': len(series) - 1,
                'start_date': series.index[prev_idx],
                'end_date': series.index[-1],
                'length': len(segment_data),
                'mean': segment_data.mean(),
                'std': segment_data.std(),
                'data': segment_data
            })
        
        return segments


class PageHinkleyDetector(ChangePointDetector):
    """Page-Hinkley test for change point detection"""
    
    def __init__(self, 
                 threshold: float = 150.0,
                 alpha: float = 0.9999,
                 detect_both_directions: bool = True):
        
        self.threshold = threshold
        self.alpha = alpha
        self.detect_both_directions = detect_both_directions
    
    def detect(self, series: pd.Series, **kwargs) -> ChangePointResults:
        """Detect change points using Page-Hinkley test"""
        
        if len(series) < 5:
            return ChangePointResults(method="PAGE_HINKLEY")
        
        change_points = []
        
        # Initialize statistics
        sum_pos = 0
        sum_neg = 0
        min_pos = 0
        max_neg = 0
        
        # Estimate initial parameters
        reference_mean = series.iloc[:min(50, len(series)//2)].mean()
        
        for i in range(1, len(series)):
            current_value = series.iloc[i]
            
            # Update positive sum (for upward changes)
            sum_pos += current_value - reference_mean
            min_pos = min(min_pos, sum_pos)
            
            # Page-Hinkley statistic for upward change
            ph_pos = sum_pos - min_pos
            
            if ph_pos > self.threshold:
                # Upward change detected
                change_point = ChangePoint(
                    index=i,
                    date=series.index[i] if hasattr(series.index, 'to_pydatetime') else None,
                    change_type="MEAN_SHIFT",
                    direction=1,
                    test_statistic=ph_pos,
                    method_used="PAGE_HINKLEY_POSITIVE",
                    threshold_used=self.threshold
                )
                
                # Calculate statistics
                before_data = series.iloc[max(0, i-30):i]
                after_data = series.iloc[i:min(len(series), i+30)]
                
                if len(before_data) > 0 and len(after_data) > 0:
                    change_point.before_mean = before_data.mean()
                    change_point.after_mean = after_data.mean()
                    change_point.magnitude = change_point.after_mean - change_point.before_mean
                
                change_points.append(change_point)
                
                # Reset
                sum_pos = 0
                min_pos = 0
                reference_mean = series.iloc[max(0, i-10):i+1].mean()
            
            if self.detect_both_directions:
                # Update negative sum (for downward changes)
                sum_neg += reference_mean - current_value
                max_neg = max(max_neg, sum_neg)
                
                # Page-Hinkley statistic for downward change  
                ph_neg = sum_neg - max_neg
                
                if ph_neg > self.threshold:
                    # Downward change detected
                    change_point = ChangePoint(
                        index=i,
                        date=series.index[i] if hasattr(series.index, 'to_pydatetime') else None,
                        change_type="MEAN_SHIFT",
                        direction=-1,
                        test_statistic=ph_neg,
                        method_used="PAGE_HINKLEY_NEGATIVE",
                        threshold_used=self.threshold
                    )
                    
                    # Calculate statistics
                    before_data = series.iloc[max(0, i-30):i]
                    after_data = series.iloc[i:min(len(series), i+30)]
                    
                    if len(before_data) > 0 and len(after_data) > 0:
                        change_point.before_mean = before_data.mean()
                        change_point.after_mean = after_data.mean()
                        change_point.magnitude = change_point.after_mean - change_point.before_mean
                    
                    change_points.append(change_point)
                    
                    # Reset
                    sum_neg = 0
                    max_neg = 0
                    reference_mean = series.iloc[max(0, i-10):i+1].mean()
        
        # Create results
        results = ChangePointResults(
            method="PAGE_HINKLEY",
            parameters={
                'threshold': self.threshold,
                'alpha': self.alpha,
                'detect_both_directions': self.detect_both_directions
            },
            change_points=change_points,
            n_change_points=len(change_points),
            original_series=series,
            dates=series.index,
            detection_threshold=self.threshold
        )
        
        # Create segments
        results.segments = self._create_segments(series, change_points)
        
        return results
    
    def _create_segments(self, series: pd.Series, 
                        change_points: List[ChangePoint]) -> List[Dict[str, Any]]:
        """Create segments between change points"""
        
        if not change_points:
            return [{
                'start_idx': 0,
                'end_idx': len(series) - 1,
                'start_date': series.index[0],
                'end_date': series.index[-1],
                'length': len(series),
                'mean': series.mean(),
                'std': series.std(),
                'data': series
            }]
        
        segments = []
        change_indices = sorted([cp.index for cp in change_points])
        
        # Create segments
        prev_idx = 0
        for cp_idx in change_indices:
            segment_data = series.iloc[prev_idx:cp_idx]
            
            if len(segment_data) > 0:
                segments.append({
                    'start_idx': prev_idx,
                    'end_idx': cp_idx - 1,
                    'start_date': series.index[prev_idx],
                    'end_date': series.index[cp_idx - 1],
                    'length': len(segment_data),
                    'mean': segment_data.mean(),
                    'std': segment_data.std(),
                    'data': segment_data
                })
            
            prev_idx = cp_idx
        
        # Last segment
        segment_data = series.iloc[prev_idx:]
        if len(segment_data) > 0:
            segments.append({
                'start_idx': prev_idx,
                'end_idx': len(series) - 1,
                'start_date': series.index[prev_idx],
                'end_date': series.index[-1],
                'length': len(segment_data),
                'mean': segment_data.mean(),
                'std': segment_data.std(),
                'data': segment_data
            })
        
        return segments


class EWMADetector(ChangePointDetector):
    """Exponentially Weighted Moving Average change detection"""
    
    def __init__(self, 
                 lambda_param: float = 0.1,
                 threshold_std: float = 3.0,
                 min_detection_delay: int = 5):
        
        self.lambda_param = lambda_param
        self.threshold_std = threshold_std
        self.min_detection_delay = min_detection_delay
    
    def detect(self, series: pd.Series, **kwargs) -> ChangePointResults:
        """Detect change points using EWMA"""
        
        if len(series) < 10:
            return ChangePointResults(method="EWMA")
        
        # Calculate EWMA
        ewma = series.ewm(alpha=self.lambda_param).mean()
        
        # Calculate control limits
        ewma_std = series.ewm(alpha=self.lambda_param).std()
        upper_limit = ewma + self.threshold_std * ewma_std
        lower_limit = ewma - self.threshold_std * ewma_std
        
        change_points = []
        last_detection = -self.min_detection_delay
        
        for i in range(1, len(series)):
            current_value = series.iloc[i]
            
            # Check if current value exceeds control limits
            if (i - last_detection) >= self.min_detection_delay:
                if current_value > upper_limit.iloc[i]:
                    # Upward change
                    change_point = ChangePoint(
                        index=i,
                        date=series.index[i] if hasattr(series.index, 'to_pydatetime') else None,
                        change_type="MEAN_SHIFT",
                        direction=1,
                        test_statistic=(current_value - ewma.iloc[i]) / ewma_std.iloc[i],
                        method_used="EWMA_UPPER",
                        threshold_used=self.threshold_std
                    )
                    
                    change_points.append(change_point)
                    last_detection = i
                    
                elif current_value < lower_limit.iloc[i]:
                    # Downward change
                    change_point = ChangePoint(
                        index=i,
                        date=series.index[i] if hasattr(series.index, 'to_pydatetime') else None,
                        change_type="MEAN_SHIFT",
                        direction=-1,
                        test_statistic=(current_value - ewma.iloc[i]) / ewma_std.iloc[i],
                        method_used="EWMA_LOWER",
                        threshold_used=self.threshold_std
                    )
                    
                    change_points.append(change_point)
                    last_detection = i
        
        # Calculate before/after statistics for each change point
        for cp in change_points:
            before_data = series.iloc[max(0, cp.index-20):cp.index]
            after_data = series.iloc[cp.index:min(len(series), cp.index+20)]
            
            if len(before_data) > 0 and len(after_data) > 0:
                cp.before_mean = before_data.mean()
                cp.after_mean = after_data.mean()
                cp.magnitude = cp.after_mean - cp.before_mean
                cp.before_var = before_data.var()
                cp.after_var = after_data.var()
        
        # Create results
        results = ChangePointResults(
            method="EWMA",
            parameters={
                'lambda_param': self.lambda_param,
                'threshold_std': self.threshold_std,
                'min_detection_delay': self.min_detection_delay
            },
            change_points=change_points,
            n_change_points=len(change_points),
            original_series=series,
            dates=series.index,
            detection_threshold=self.threshold_std
        )
        
        # Create segments
        results.segments = self._create_segments(series, change_points)
        
        return results
    
    def _create_segments(self, series: pd.Series, 
                        change_points: List[ChangePoint]) -> List[Dict[str, Any]]:
        """Create segments between change points"""
        
        if not change_points:
            return [{
                'start_idx': 0,
                'end_idx': len(series) - 1,
                'start_date': series.index[0],
                'end_date': series.index[-1],
                'length': len(series),
                'mean': series.mean(),
                'std': series.std(),
                'data': series
            }]
        
        segments = []
        change_indices = sorted([cp.index for cp in change_points])
        
        # Create segments
        prev_idx = 0
        for cp_idx in change_indices:
            segment_data = series.iloc[prev_idx:cp_idx]
            
            if len(segment_data) > 0:
                segments.append({
                    'start_idx': prev_idx,
                    'end_idx': cp_idx - 1,
                    'start_date': series.index[prev_idx],
                    'end_date': series.index[cp_idx - 1],
                    'length': len(segment_data),
                    'mean': segment_data.mean(),
                    'std': segment_data.std(),
                    'data': segment_data
                })
            
            prev_idx = cp_idx
        
        # Last segment
        segment_data = series.iloc[prev_idx:]
        if len(segment_data) > 0:
            segments.append({
                'start_idx': prev_idx,
                'end_idx': len(series) - 1,
                'start_date': series.index[prev_idx],
                'end_date': series.index[-1],
                'length': len(segment_data),
                'mean': segment_data.mean(),
                'std': segment_data.std(),
                'data': segment_data
            })
        
        return segments


class ChangePointAnalyzer:
    """
    Comprehensive change point analysis framework
    """
    
    def __init__(self):
        self.detectors = {
            'cusum': CUSUMDetector,
            'binary_segmentation': BinarySegmentationDetector,
            'page_hinkley': PageHinkleyDetector,
            'ewma': EWMADetector
        }
    
    def detect_change_points(self, 
                           series: pd.Series,
                           method: str = 'cusum',
                           **kwargs) -> ChangePointResults:
        """
        Detect change points using specified method
        
        Args:
            series: Time series data
            method: Detection method 
            **kwargs: Method-specific parameters
            
        Returns:
            ChangePointResults object
        """
        
        if method not in self.detectors:
            raise ValueError(f"Unknown method: {method}")
        
        detector_class = self.detectors[method]
        detector = detector_class(**kwargs)
        
        results = detector.detect(series)
        
        return results
    
    def compare_methods(self, 
                       series: pd.Series,
                       methods: List[str] = ['cusum', 'binary_segmentation', 'page_hinkley'],
                       **kwargs) -> Dict[str, ChangePointResults]:
        """Compare different change point detection methods"""
        
        comparison_results = {}
        
        for method in methods:
            if method in self.detectors:
                try:
                    results = self.detect_change_points(series, method, **kwargs)
                    comparison_results[method] = results
                except Exception as e:
                    print(f"Method {method} failed: {e}")
                    continue
        
        return comparison_results
    
    def consensus_detection(self, 
                          series: pd.Series,
                          methods: List[str] = ['cusum', 'binary_segmentation'],
                          consensus_threshold: int = 2,
                          tolerance: int = 5) -> ChangePointResults:
        """Find consensus change points across multiple methods"""
        
        # Run multiple methods
        method_results = self.compare_methods(series, methods)
        
        if not method_results:
            return ChangePointResults(method="CONSENSUS")
        
        # Collect all change points
        all_change_points = []
        for method_name, results in method_results.items():
            for cp in results.change_points:
                cp.method_used = method_name
                all_change_points.append(cp)
        
        # Find consensus points
        consensus_points = []
        
        for cp in all_change_points:
            # Count how many methods detected a change point near this location
            nearby_count = 0
            nearby_points = []
            
            for other_cp in all_change_points:
                if abs(other_cp.index - cp.index) <= tolerance:
                    nearby_count += 1
                    nearby_points.append(other_cp)
            
            # Check if consensus threshold is met
            if nearby_count >= consensus_threshold:
                # Create consensus change point (average location)
                avg_index = int(np.mean([p.index for p in nearby_points]))
                avg_confidence = np.mean([p.confidence for p in nearby_points])
                
                # Check if we already have a consensus point nearby
                already_exists = any(abs(existing.index - avg_index) <= tolerance 
                                   for existing in consensus_points)
                
                if not already_exists:
                    consensus_cp = ChangePoint(
                        index=avg_index,
                        date=series.index[avg_index] if hasattr(series.index, 'to_pydatetime') else None,
                        confidence=avg_confidence,
                        change_type="CONSENSUS",
                        method_used=f"CONSENSUS_{nearby_count}_methods"
                    )
                    
                    # Calculate statistics
                    before_data = series.iloc[max(0, avg_index-20):avg_index]
                    after_data = series.iloc[avg_index:min(len(series), avg_index+20)]
                    
                    if len(before_data) > 0 and len(after_data) > 0:
                        consensus_cp.before_mean = before_data.mean()
                        consensus_cp.after_mean = after_data.mean()
                        consensus_cp.magnitude = consensus_cp.after_mean - consensus_cp.before_mean
                        consensus_cp.direction = 1 if consensus_cp.magnitude > 0 else -1
                    
                    consensus_points.append(consensus_cp)
        
        # Sort by index
        consensus_points.sort(key=lambda cp: cp.index)
        
        # Create results
        results = ChangePointResults(
            method="CONSENSUS",
            parameters={
                'methods_used': methods,
                'consensus_threshold': consensus_threshold,
                'tolerance': tolerance
            },
            change_points=consensus_points,
            n_change_points=len(consensus_points),
            original_series=series,
            dates=series.index
        )
        
        # Create segments
        results.segments = self._create_segments(series, consensus_points)
        
        return results
    
    def _create_segments(self, series: pd.Series, 
                        change_points: List[ChangePoint]) -> List[Dict[str, Any]]:
        """Create segments between change points"""
        
        if not change_points:
            return [{
                'start_idx': 0,
                'end_idx': len(series) - 1,
                'start_date': series.index[0],
                'end_date': series.index[-1],
                'length': len(series),
                'mean': series.mean(),
                'std': series.std(),
                'data': series
            }]
        
        segments = []
        change_indices = sorted([cp.index for cp in change_points])
        
        # Create segments
        prev_idx = 0
        for cp_idx in change_indices:
            segment_data = series.iloc[prev_idx:cp_idx]
            
            if len(segment_data) > 0:
                segments.append({
                    'start_idx': prev_idx,
                    'end_idx': cp_idx - 1,
                    'start_date': series.index[prev_idx],
                    'end_date': series.index[cp_idx - 1],
                    'length': len(segment_data),
                    'mean': segment_data.mean(),
                    'std': segment_data.std(),
                    'data': segment_data
                })
            
            prev_idx = cp_idx
        
        # Last segment
        segment_data = series.iloc[prev_idx:]
        if len(segment_data) > 0:
            segments.append({
                'start_idx': prev_idx,
                'end_idx': len(series) - 1,
                'start_date': series.index[prev_idx],
                'end_date': series.index[-1],
                'length': len(segment_data),
                'mean': segment_data.mean(),
                'std': segment_data.std(),
                'data': segment_data
            })
        
        return segments
    
    def plot_change_point_analysis(self, 
                                 results: ChangePointResults,
                                 save_path: Optional[str] = None):
        """Create change point analysis visualization"""
        
        if results.original_series is None:
            print("No data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        series = results.original_series
        
        # 1. Time series with change points
        ax1 = axes[0]
        ax1.plot(series.index, series.values, 'b-', linewidth=1, alpha=0.7)
        
        # Mark change points
        for cp in results.change_points:
            ax1.axvline(x=series.index[cp.index], color='red', linestyle='--', alpha=0.8)
            ax1.annotate(f'CP{cp.index}', 
                        xy=(series.index[cp.index], series.iloc[cp.index]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        ax1.set_title(f'Change Point Detection - {results.method}')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # 2. Segmented means
        ax2 = axes[1]
        ax2.plot(series.index, series.values, 'b-', linewidth=1, alpha=0.3, label='Original')
        
        # Plot segment means
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, segment in enumerate(results.segments):
            if 'data' in segment:
                segment_data = segment['data']
                ax2.plot(segment_data.index, 
                        [segment['mean']] * len(segment_data),
                        color=colors[i % len(colors)], 
                        linewidth=3, alpha=0.8,
                        label=f"Segment {i+1} (μ={segment['mean']:.4f})")
        
        ax2.set_title('Segmented Means')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Change point analysis plot saved to {save_path}")
        
        plt.show()
    
    def generate_change_point_report(self, results: ChangePointResults) -> str:
        """Generate comprehensive change point analysis report"""
        
        report = []
        report.append("="*70)
        report.append("CHANGE POINT DETECTION ANALYSIS REPORT")
        report.append("="*70)
        
        # Method information
        report.append(f"\nMETHOD INFORMATION:")
        report.append(f"  Detection Method: {results.method}")
        report.append(f"  Parameters: {results.parameters}")
        report.append(f"  Detection Threshold: {results.detection_threshold}")
        
        # Data summary
        if results.original_series is not None:
            series = results.original_series
            report.append(f"\nDATA SUMMARY:")
            report.append(f"  Observations: {len(series)}")
            if results.dates is not None:
                report.append(f"  Period: {results.dates[0]} to {results.dates[-1]}")
            report.append(f"  Mean: {series.mean():.6f}")
            report.append(f"  Std Dev: {series.std():.6f}")
            report.append(f"  Min: {series.min():.6f}")
            report.append(f"  Max: {series.max():.6f}")
        
        # Change point summary
        report.append(f"\nCHANGE POINT SUMMARY:")
        report.append(f"  Total Change Points Detected: {results.n_change_points}")
        
        if results.change_points:
            report.append(f"  Change Point Details:")
            report.append(f"    {'Index':<8} {'Date':<12} {'Type':<15} {'Direction':<10} {'Magnitude':<12} {'Confidence':<10}")
            report.append("    " + "-"*75)
            
            for cp in results.change_points:
                date_str = cp.date.strftime('%Y-%m-%d') if cp.date else 'N/A'
                direction_str = '↑' if cp.direction > 0 else '↓' if cp.direction < 0 else '='
                
                report.append(f"    {cp.index:<8} {date_str:<12} {cp.change_type:<15} "
                             f"{direction_str:<10} {cp.magnitude:<12.6f} {cp.confidence:<10.3f}")
        
        # Segment analysis
        if results.segments:
            report.append(f"\nSEGMENT ANALYSIS:")
            report.append(f"  Total Segments: {len(results.segments)}")
            report.append(f"  Segment Statistics:")
            report.append(f"    {'Segment':<8} {'Start':<8} {'End':<8} {'Length':<8} {'Mean':<12} {'Std Dev':<12}")
            report.append("    " + "-"*70)
            
            for i, segment in enumerate(results.segments):
                report.append(f"    {i+1:<8} {segment['start_idx']:<8} {segment['end_idx']:<8} "
                             f"{segment['length']:<8} {segment['mean']:<12.6f} {segment['std']:<12.6f}")
            
            # Segment comparison
            if len(results.segments) > 1:
                report.append(f"\n  Segment Comparison:")
                means = [seg['mean'] for seg in results.segments]
                stds = [seg['std'] for seg in results.segments]
                
                report.append(f"    Mean Range: {min(means):.6f} to {max(means):.6f}")
                report.append(f"    Std Dev Range: {min(stds):.6f} to {max(stds):.6f}")
                report.append(f"    Largest Mean Change: {max(means) - min(means):.6f}")
        
        # Method-specific information
        if results.method == "CUSUM":
            report.append(f"\nCUSUM-SPECIFIC INFORMATION:")
            report.append(f"  Threshold: {results.parameters.get('threshold', 'N/A')}")
            report.append(f"  Drift: {results.parameters.get('drift', 'N/A')}")
            
        elif results.method == "BINARY_SEGMENTATION":
            report.append(f"\nBINARY SEGMENTATION-SPECIFIC INFORMATION:")
            report.append(f"  Min Segment Length: {results.parameters.get('min_segment_length', 'N/A')}")
            report.append(f"  Max Change Points: {results.parameters.get('max_change_points', 'N/A')}")
            report.append(f"  Significance Level: {results.parameters.get('significance_level', 'N/A')}")
        
        # Statistical tests
        if results.change_points:
            significant_changes = [cp for cp in results.change_points if cp.p_value < 0.05]
            report.append(f"\nSTATISTICAL SIGNIFICANCE:")
            report.append(f"  Significant Changes (p < 0.05): {len(significant_changes)}")
            
            if significant_changes:
                report.append(f"  Most Significant Change:")
                most_sig = min(significant_changes, key=lambda cp: cp.p_value)
                report.append(f"    Index: {most_sig.index}")
                report.append(f"    P-value: {most_sig.p_value:.6f}")
                report.append(f"    Test Statistic: {most_sig.test_statistic:.3f}")
        
        return "\n".join(report)
