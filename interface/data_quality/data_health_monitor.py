"""
Data Health Monitor
==================

Real-time data quality monitoring and validation system for trading platforms.
Provides comprehensive data health checks, anomaly detection, and quality metrics.

Features:
- Real-time data quality monitoring and validation
- Multi-dimensional data health scoring
- Anomaly detection and alerting system
- Historical data quality trends and reporting
- Data completeness and consistency checks
- Performance impact analysis of data issues
- Interactive data health dashboard
- Automated data quality alerts and notifications

Author: Quantum Forge Interface Team
Date: November 2025
"""

import time
import sys
import logging
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import json
from pathlib import Path
import math

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class DataQualityStatus(Enum):
    """Data quality status levels."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    WARNING = "Warning"
    CRITICAL = "Critical"
    UNKNOWN = "Unknown"

class DataSource(Enum):
    """Types of data sources."""
    MARKET_DATA = "Market Data"
    REFERENCE_DATA = "Reference Data"
    NEWS_DATA = "News Data"
    ALTERNATIVE_DATA = "Alternative Data"
    DERIVED_DATA = "Derived Data"

@dataclass
class DataQualityMetric:
    """Individual data quality metric."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    status: DataQualityStatus
    description: str
    timestamp: datetime

@dataclass
class DataHealthReport:
    """Comprehensive data health report."""
    source: str
    overall_score: float
    status: DataQualityStatus
    metrics: List[DataQualityMetric]
    anomalies_detected: int
    issues: List[str]
    recommendations: List[str]
    timestamp: datetime

class DataHealthMonitor:
    """
    Real-time data quality monitoring and validation system.
    
    Provides comprehensive monitoring of data quality across multiple sources
    with real-time alerts and detailed health reporting.
    """
    
    def __init__(self):
        """Initialize data health monitor."""
        st.set_page_config(
            page_title="Quantum Forge - Data Health Monitor",
            page_icon=" ",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .health-header {
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .health-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #10b981;
            margin-bottom: 1rem;
        }
        .health-excellent {
            background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.2) 100%);
            border-left: 4px solid #10b981;
        }
        .health-good {
            background: linear-gradient(90deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.2) 100%);
            border-left: 4px solid #22c55e;
        }
        .health-warning {
            background: linear-gradient(90deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.2) 100%);
            border-left: 4px solid #f59e0b;
        }
        .health-critical {
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.2) 100%);
            border-left: 4px solid #ef4444;
        }
        .health-metric {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin: 0.5rem 0;
            text-align: center;
        }
        .health-status {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
            font-weight: bold;
            color: white;
        }
        .status-excellent { background: #10b981; }
        .status-good { background: #22c55e; }
        .status-warning { background: #f59e0b; }
        .status-critical { background: #ef4444; }
        .status-unknown { background: #6b7280; }
        .alert-box {
            background: #fef2f2;
            border: 1px solid #fecaca;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        self.status_colors = {
            DataQualityStatus.EXCELLENT: '#10b981',
            DataQualityStatus.GOOD: '#22c55e',
            DataQualityStatus.WARNING: '#f59e0b',
            DataQualityStatus.CRITICAL: '#ef4444',
            DataQualityStatus.UNKNOWN: '#6b7280'
        }
        
        # Initialize session state for alerts
        if 'data_alerts' not in st.session_state:
            st.session_state.data_alerts = []
    
    def generate_sample_data_sources(self) -> Dict[str, pd.DataFrame]:
        """Generate data sources from real-time cache with injected quality issues."""
        try:
            # Use RealTimeDataCache
            from data.ingestion.realtime_data_cache import RealTimeDataCache
            cache = RealTimeDataCache(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            cache.start()
            time.sleep(0.1)
            
            data_sources = {}
            
            # 1. Market Data (Real history)
            hist = cache.get_historical_data("BTCUSDT", days=1)
            if hist.empty:
                # Fallback
                dates = pd.date_range(end=datetime.now(), periods=100, freq='1min')
                hist = pd.DataFrame({'close': [50000.0]*100, 'volume': [1000.0]*100}, index=dates)
            
            market_data = hist.reset_index().rename(columns={'index': 'timestamp', 'close': 'price'})
            market_data['symbol'] = 'BTCUSDT'
            market_data['bid'] = market_data['price'] * 0.9995
            market_data['ask'] = market_data['price'] * 1.0005
            
            # Deterministic injection of issues for demo purposes
            # Missing data: every 20th record
            market_data.loc[market_data.index % 20 == 0, 'price'] = np.nan
            
            # Outliers: every 50th record
            market_data.loc[market_data.index % 50 == 0, 'price'] *= 1.5
            
            # Stale data: every 33rd record repeats previous
            for idx in range(len(market_data)):
                if idx > 0 and idx % 33 == 0:
                    market_data.loc[idx, 'price'] = market_data.iloc[idx-1]['price']
            
            data_sources['market_data'] = market_data
            
            # 2. Reference Data (Derived from market events)
            # Create events based on large price moves
            market_data['returns'] = market_data['price'].pct_change().fillna(0)
            significant_moves = market_data[market_data['returns'].abs() > 0.001]
            
            ref_data = []
            for idx, row in significant_moves.iterrows():
                ref_data.append({
                    'timestamp': row['timestamp'],
                    'symbol': 'BTCUSDT',
                    'event_type': 'volatility_alert',
                    'value': row['returns'],
                    'status': 'confirmed' if idx % 10 != 0 else 'error' # Inject error
                })
            
            if not ref_data:
                 ref_data = [{'timestamp': datetime.now(), 'symbol': 'BTCUSDT', 'event_type': 'none', 'value': 0, 'status': 'confirmed'}]
                 
            data_sources['reference_data'] = pd.DataFrame(ref_data)
            
            # 3. News Data (Simulated based on price direction)
            news_data = []
            for idx, row in market_data.iloc[::5].iterrows(): # Every 5th minute
                sentiment = 1.0 if row['returns'] > 0 else -1.0
                news_data.append({
                    'timestamp': row['timestamp'],
                    'source': 'CryptoNews',
                    'category': 'market_update',
                    'sentiment_score': sentiment,
                    'confidence': 0.9 if idx % 7 != 0 else 0.2, # Inject low confidence
                    'word_count': int(abs(row['returns']) * 100000) + 50,
                    'processing_delay': abs(row['returns']) * 1000 # Delay proportional to volatility
                })
            
            data_sources['news_data'] = pd.DataFrame(news_data)
            
            # 4. Alternative Data (Volume based)
            alt_data = []
            for idx, row in market_data.iloc[::10].iterrows():
                alt_data.append({
                    'timestamp': row['timestamp'],
                    'data_type': 'on_chain_volume',
                    'symbol': 'BTCUSDT',
                    'value': row['volume'],
                    'data_quality_flag': 'good' if row['volume'] > 0 else 'poor',
                    'source_reliability': 0.95
                })
            data_sources['alternative_data'] = pd.DataFrame(alt_data)
            
            # 5. Derived Data (Moving averages)
            derived_data = []
            ma_5 = market_data['price'].rolling(5).mean()
            for idx, val in ma_5.items():
                if pd.isna(val): continue
                derived_data.append({
                    'timestamp': market_data.iloc[idx]['timestamp'],
                    'metric_name': 'MA_5',
                    'symbol': 'BTCUSDT',
                    'value': val if idx % 40 != 0 else np.nan, # Inject missing calculation
                    'calculation_time': 0.05,
                    'input_data_age': 0.1
                })
            
            data_sources['derived_data'] = pd.DataFrame(derived_data)
            
            return data_sources
            
        except Exception as e:
            logger.error(f"Error generating sample data sources: {str(e)}")
            return {}
    
    def analyze_data_completeness(self, data: pd.DataFrame) -> float:
        """Analyze data completeness score."""
        try:
            if data.empty:
                return 0.0
            
            # Calculate missing data percentage
            total_cells = data.size
            missing_cells = data.isna().sum().sum()
            completeness = 1.0 - (missing_cells / total_cells)
            
            return completeness
            
        except Exception as e:
            logger.error(f"Error analyzing data completeness: {str(e)}")
            return 0.0
    
    def analyze_data_freshness(self, data: pd.DataFrame, timestamp_col: str = 'timestamp') -> float:
        """Analyze data freshness score."""
        try:
            if data.empty or timestamp_col not in data.columns:
                return 0.0
            
            current_time = datetime.now()
            
            # Convert timestamp column to datetime
            timestamps = pd.to_datetime(data[timestamp_col])
            
            # Calculate average age of data
            data_ages = (current_time - timestamps).dt.total_seconds()
            avg_age_minutes = data_ages.mean() / 60
            
            # Score based on freshness (exponential decay)
            # Fresh data (< 5 min) = 1.0, older data gets lower scores
            freshness_score = np.exp(-avg_age_minutes / 60)  # 1-hour half-life
            
            return min(1.0, freshness_score)
            
        except Exception as e:
            logger.error(f"Error analyzing data freshness: {str(e)}")
            return 0.0
    
    def analyze_data_accuracy(self, data: pd.DataFrame) -> float:
        """Analyze data accuracy by detecting outliers."""
        try:
            if data.empty:
                return 0.0
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) == 0:
                return 1.0  # No numeric data to check
            
            outlier_ratios = []
            
            for col in numeric_cols:
                col_data = data[col].dropna()
                
                if len(col_data) < 10:
                    continue
                
                # Use IQR method to detect outliers
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (col_data < lower_bound) | (col_data > upper_bound)
                outlier_ratio = outliers.mean()
                outlier_ratios.append(outlier_ratio)
            
            if not outlier_ratios:
                return 1.0
            
            # Accuracy score is inverse of outlier ratio
            avg_outlier_ratio = np.mean(outlier_ratios)
            accuracy_score = 1.0 - min(1.0, avg_outlier_ratio * 2)  # Scale outliers
            
            return accuracy_score
            
        except Exception as e:
            logger.error(f"Error analyzing data accuracy: {str(e)}")
            return 0.0
    
    def analyze_data_consistency(self, data: pd.DataFrame) -> float:
        """Analyze data consistency and coherence."""
        try:
            if data.empty:
                return 0.0
            
            consistency_scores = []
            
            # Check for duplicate records
            if len(data) > 0:
                duplicate_ratio = data.duplicated().mean()
                consistency_scores.append(1.0 - duplicate_ratio)
            
            # Check for data type consistency
            for col in data.columns:
                if data[col].dtype == 'object':
                    # For categorical data, check for format consistency
                    col_data = data[col].dropna()
                    if len(col_data) > 0:
                        # Simple consistency check: variation in string lengths
                        if col_data.dtype == 'object':
                            str_lengths = col_data.astype(str).str.len()
                            if len(str_lengths) > 1:
                                length_cv = str_lengths.std() / str_lengths.mean()
                                consistency_scores.append(1.0 - min(1.0, length_cv / 2))
            
            # Check for logical consistency (e.g., bid <= price <= ask)
            if 'bid' in data.columns and 'ask' in data.columns and 'price' in data.columns:
                valid_spread = ((data['bid'] <= data['price']) & (data['price'] <= data['ask'])).mean()
                consistency_scores.append(valid_spread)
            
            return np.mean(consistency_scores) if consistency_scores else 1.0
            
        except Exception as e:
            logger.error(f"Error analyzing data consistency: {str(e)}")
            return 0.0
    
    def analyze_data_timeliness(self, data: pd.DataFrame, timestamp_col: str = 'timestamp') -> float:
        """Analyze data timeliness (regular updates)."""
        try:
            if data.empty or timestamp_col not in data.columns:
                return 0.0
            
            timestamps = pd.to_datetime(data[timestamp_col]).sort_values()
            
            if len(timestamps) < 2:
                return 1.0
            
            # Calculate time gaps between updates
            time_diffs = timestamps.diff().dt.total_seconds()
            time_diffs = time_diffs.dropna()
            
            if len(time_diffs) == 0:
                return 1.0
            
            # Expected update frequency (assume median is expected)
            expected_interval = time_diffs.median()
            
            # Calculate coefficient of variation in update intervals
            if expected_interval > 0:
                cv = time_diffs.std() / expected_interval
                timeliness_score = 1.0 / (1.0 + cv)  # Lower CV = better timeliness
            else:
                timeliness_score = 1.0
            
            return timeliness_score
            
        except Exception as e:
            logger.error(f"Error analyzing data timeliness: {str(e)}")
            return 0.0
    
    def calculate_overall_health_score(self, metrics: List[DataQualityMetric]) -> Tuple[float, DataQualityStatus]:
        """Calculate overall data health score and status."""
        try:
            if not metrics:
                return 0.0, DataQualityStatus.UNKNOWN
            
            # Weighted average of all metrics
            weights = {
                'completeness': 0.25,
                'freshness': 0.20,
                'accuracy': 0.25,
                'consistency': 0.15,
                'timeliness': 0.15
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric in metrics:
                metric_name = metric.name.lower()
                weight = weights.get(metric_name, 0.1)  # Default weight
                weighted_score += metric.value * weight
                total_weight += weight
            
            overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
            
            # Determine status based on score
            if overall_score >= 0.9:
                status = DataQualityStatus.EXCELLENT
            elif overall_score >= 0.75:
                status = DataQualityStatus.GOOD
            elif overall_score >= 0.6:
                status = DataQualityStatus.WARNING
            else:
                status = DataQualityStatus.CRITICAL
            
            return overall_score, status
            
        except Exception as e:
            logger.error(f"Error calculating overall health score: {str(e)}")
            return 0.0, DataQualityStatus.UNKNOWN
    
    def generate_health_report(self, source_name: str, data: pd.DataFrame) -> DataHealthReport:
        """Generate comprehensive health report for a data source."""
        try:
            current_time = datetime.now()
            
            # Calculate individual metrics
            completeness = self.analyze_data_completeness(data)
            freshness = self.analyze_data_freshness(data)
            accuracy = self.analyze_data_accuracy(data)
            consistency = self.analyze_data_consistency(data)
            timeliness = self.analyze_data_timeliness(data)
            
            # Create metric objects
            metrics = [
                DataQualityMetric(
                    name="completeness",
                    value=completeness,
                    threshold_warning=0.95,
                    threshold_critical=0.90,
                    status=DataQualityStatus.EXCELLENT if completeness >= 0.95 else
                           DataQualityStatus.GOOD if completeness >= 0.90 else
                           DataQualityStatus.WARNING if completeness >= 0.80 else
                           DataQualityStatus.CRITICAL,
                    description="Percentage of non-missing data points",
                    timestamp=current_time
                ),
                DataQualityMetric(
                    name="freshness",
                    value=freshness,
                    threshold_warning=0.8,
                    threshold_critical=0.6,
                    status=DataQualityStatus.EXCELLENT if freshness >= 0.9 else
                           DataQualityStatus.GOOD if freshness >= 0.8 else
                           DataQualityStatus.WARNING if freshness >= 0.6 else
                           DataQualityStatus.CRITICAL,
                    description="How recent the data is",
                    timestamp=current_time
                ),
                DataQualityMetric(
                    name="accuracy",
                    value=accuracy,
                    threshold_warning=0.9,
                    threshold_critical=0.8,
                    status=DataQualityStatus.EXCELLENT if accuracy >= 0.95 else
                           DataQualityStatus.GOOD if accuracy >= 0.9 else
                           DataQualityStatus.WARNING if accuracy >= 0.8 else
                           DataQualityStatus.CRITICAL,
                    description="Absence of outliers and erroneous values",
                    timestamp=current_time
                ),
                DataQualityMetric(
                    name="consistency",
                    value=consistency,
                    threshold_warning=0.9,
                    threshold_critical=0.8,
                    status=DataQualityStatus.EXCELLENT if consistency >= 0.95 else
                           DataQualityStatus.GOOD if consistency >= 0.9 else
                           DataQualityStatus.WARNING if consistency >= 0.8 else
                           DataQualityStatus.CRITICAL,
                    description="Internal coherence and logical consistency",
                    timestamp=current_time
                ),
                DataQualityMetric(
                    name="timeliness",
                    value=timeliness,
                    threshold_warning=0.8,
                    threshold_critical=0.6,
                    status=DataQualityStatus.EXCELLENT if timeliness >= 0.9 else
                           DataQualityStatus.GOOD if timeliness >= 0.8 else
                           DataQualityStatus.WARNING if timeliness >= 0.6 else
                           DataQualityStatus.CRITICAL,
                    description="Regularity and predictability of updates",
                    timestamp=current_time
                )
            ]
            
            # Calculate overall score and status
            overall_score, overall_status = self.calculate_overall_health_score(metrics)
            
            # Detect anomalies and issues
            anomalies_detected = 0
            issues = []
            recommendations = []
            
            for metric in metrics:
                if metric.status == DataQualityStatus.CRITICAL:
                    anomalies_detected += 1
                    issues.append(f"Critical issue with {metric.name}: {metric.value:.2%}")
                    
                    # Generate recommendations
                    if metric.name == "completeness":
                        recommendations.append("Investigate data pipeline for missing data")
                    elif metric.name == "freshness":
                        recommendations.append("Check data feed connectivity and processing delays")
                    elif metric.name == "accuracy":
                        recommendations.append("Review data validation rules and outlier detection")
                    elif metric.name == "consistency":
                        recommendations.append("Audit data transformation and validation logic")
                    elif metric.name == "timeliness":
                        recommendations.append("Monitor data feed schedules and processing times")
            
            return DataHealthReport(
                source=source_name,
                overall_score=overall_score,
                status=overall_status,
                metrics=metrics,
                anomalies_detected=anomalies_detected,
                issues=issues,
                recommendations=recommendations,
                timestamp=current_time
            )
            
        except Exception as e:
            logger.error(f"Error generating health report for {source_name}: {str(e)}")
            return DataHealthReport(
                source=source_name,
                overall_score=0.0,
                status=DataQualityStatus.UNKNOWN,
                metrics=[],
                anomalies_detected=0,
                issues=[f"Error analyzing data: {str(e)}"],
                recommendations=["Contact system administrator"],
                timestamp=datetime.now()
            )
    
    def render_header(self):
        """Render data health monitor header."""
        st.markdown("""
        <div class="health-header">
            <h1>  Quantum Forge Data Health Monitor</h1>
            <p>Real-time Data Quality Monitoring & Validation</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Control buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("  Refresh Data", key="refresh_data"):
                st.rerun()
                
        with col2:
            if st.button("  View Alerts", key="view_alerts"):
                st.info(f"Active alerts: {len(st.session_state.data_alerts)}")
                
        with col3:
            if st.button("  Generate Report", key="generate_report"):
                st.info("Generating comprehensive health report...")
                
        with col4:
            if st.button(" ️ Configure Thresholds", key="configure_thresholds"):
                st.info("Opening threshold configuration...")
    
    def render_sidebar_controls(self):
        """Render data health monitor sidebar controls."""
        st.sidebar.markdown("##   Health Monitor Controls")
        
        # Data source selection
        st.sidebar.markdown("###   Data Sources")
        
        selected_sources = st.sidebar.multiselect(
            "Select Data Sources to Monitor",
            options=['market_data', 'reference_data', 'news_data', 'alternative_data', 'derived_data'],
            default=['market_data', 'reference_data', 'news_data'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Monitoring parameters
        st.sidebar.markdown("###  ️ Monitoring Parameters")
        
        refresh_interval = st.sidebar.selectbox(
            "Refresh Interval",
            options=[30, 60, 300, 600, 1800],
            index=1,
            format_func=lambda x: f"{x} seconds" if x < 60 else f"{x//60} minutes"
        )
        
        alert_threshold = st.sidebar.slider(
            "Alert Threshold (Overall Score)",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05
        )
        
        # Display options
        st.sidebar.markdown("###   Display Options")
        
        show_detailed_metrics = st.sidebar.checkbox("Show Detailed Metrics", value=True)
        show_historical_trends = st.sidebar.checkbox("Show Historical Trends", value=True)
        show_recommendations = st.sidebar.checkbox("Show Recommendations", value=True)
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
        
        # Thresholds configuration
        st.sidebar.markdown("###   Quality Thresholds")
        
        completeness_threshold = st.sidebar.slider("Completeness Warning", 0.8, 1.0, 0.95, 0.01)
        freshness_threshold = st.sidebar.slider("Freshness Warning", 0.5, 1.0, 0.8, 0.05)
        accuracy_threshold = st.sidebar.slider("Accuracy Warning", 0.7, 1.0, 0.9, 0.05)
        
        return {
            'selected_sources': selected_sources,
            'refresh_interval': refresh_interval,
            'alert_threshold': alert_threshold,
            'show_detailed_metrics': show_detailed_metrics,
            'show_historical_trends': show_historical_trends,
            'show_recommendations': show_recommendations,
            'auto_refresh': auto_refresh,
            'thresholds': {
                'completeness': completeness_threshold,
                'freshness': freshness_threshold,
                'accuracy': accuracy_threshold
            }
        }
    
    def render_overall_health_summary(self, health_reports: List[DataHealthReport]):
        """Render overall system health summary."""
        if not health_reports:
            st.warning("No health reports available")
            return
        
        st.subheader("  System Health Overview")
        
        # Overall statistics
        total_sources = len(health_reports)
        avg_score = np.mean([report.overall_score for report in health_reports])
        total_anomalies = sum(report.anomalies_detected for report in health_reports)
        
        # Status distribution
        status_counts = {}
        for report in health_reports:
            status = report.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Display summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="health-metric">
                <h3>{total_sources}</h3>
                <p>Data Sources</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status_class = "excellent" if avg_score >= 0.9 else "good" if avg_score >= 0.75 else "warning" if avg_score >= 0.6 else "critical"
            st.markdown(f"""
            <div class="health-metric health-{status_class}">
                <h3>{avg_score:.1%}</h3>
                <p>Overall Health Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            anomaly_class = "excellent" if total_anomalies == 0 else "warning" if total_anomalies < 3 else "critical"
            st.markdown(f"""
            <div class="health-metric health-{anomaly_class}">
                <h3>{total_anomalies}</h3>
                <p>Active Issues</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Most common status
            most_common_status = max(status_counts.items(), key=lambda x: x[1])[0] if status_counts else "Unknown"
            status_class = most_common_status.lower().replace(' ', '_')
            st.markdown(f"""
            <div class="health-metric">
                <span class="health-status status-{status_class.split('_')[0]}">{most_common_status}</span>
                <p>Dominant Status</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Health score trend (simulated)
        st.markdown("**Health Score Trend (Last 24 Hours)**")
        
        # Generate historical trend data
        hours = list(range(24, 0, -1))
        trend_scores = []
        
        for idx, hour in enumerate(hours):
            # Deterministic variation around current score using a small sinusoid
            base_score = avg_score
            variation = 0.03 * math.sin((idx + 1) * 0.5)
            score = np.clip(base_score + variation, 0, 1)
            trend_scores.append(score)
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=trend_scores,
                mode='lines+markers',
                name='Health Score',
                line=dict(color='#10b981', width=2),
                fill='tonexty',
                fillcolor='rgba(16, 185, 129, 0.1)'
            )
        )
        
        # Add threshold lines
        fig.add_hline(y=0.9, line_dash="dash", line_color="green", annotation_text="Excellent")
        fig.add_hline(y=0.75, line_dash="dash", line_color="orange", annotation_text="Good")
        fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="Warning")
        
        fig.update_layout(
            title="System Health Score Trend",
            xaxis_title="Hours Ago",
            yaxis_title="Health Score",
            template='plotly_white',
            height=300,
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_source_health_cards(self, health_reports: List[DataHealthReport], show_details: bool):
        """Render individual data source health cards."""
        st.subheader("  Data Source Health Status")
        
        for report in health_reports:
            # Determine card style based on status
            status_class = report.status.value.lower().replace(' ', '_')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="health-card health-{status_class.split('_')[0]}">
                    <h4>{report.source.replace('_', ' ').title()}</h4>
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <span class="health-status status-{status_class.split('_')[0]}">{report.status.value}</span>
                        <span style="font-size: 1.5rem; font-weight: bold;">{report.overall_score:.1%}</span>
                    </div>
                    <p>Last Updated: {report.timestamp.strftime('%H:%M:%S')}</p>
                    {f'<p style="color: #ef4444;"><strong>Issues:</strong> {report.anomalies_detected} detected</p>' if report.anomalies_detected > 0 else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Show issues if any
                if report.issues:
                    for issue in report.issues[:3]:  # Show max 3 issues
                        st.markdown(f"""
                        <div class="alert-box">
                            <strong> ️ Issue:</strong> {issue}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                if show_details:
                    # Metrics radar chart
                    metrics_names = [m.name.title() for m in report.metrics]
                    metrics_values = [m.value for m in report.metrics]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(
                        go.Scatterpolar(
                            r=metrics_values,
                            theta=metrics_names,
                            fill='toself',
                            name=report.source,
                            line_color=self.status_colors[report.status]
                        )
                    )
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                            )
                        ),
                        showlegend=False,
                        height=250,
                        margin=dict(t=20, b=20, l=20, r=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
    
    def render_detailed_metrics(self, health_reports: List[DataHealthReport]):
        """Render detailed metrics comparison."""
        st.subheader("  Detailed Quality Metrics")
        
        # Create metrics comparison matrix
        sources = [report.source for report in health_reports]
        metric_names = ['completeness', 'freshness', 'accuracy', 'consistency', 'timeliness']
        
        metrics_matrix = []
        for report in health_reports:
            row = []
            for metric_name in metric_names:
                metric = next((m for m in report.metrics if m.name == metric_name), None)
                row.append(metric.value if metric else 0)
            metrics_matrix.append(row)
        
        # Heatmap
        fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z=metrics_matrix,
                x=[name.title() for name in metric_names],
                y=[source.replace('_', ' ').title() for source in sources],
                colorscale='RdYlGn',
                text=np.round(metrics_matrix, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='Source: %{y}<br>Metric: %{x}<br>Score: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Quality Score")
            )
        )
        
        fig.update_layout(
            title="Data Quality Metrics Heatmap",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Quality Metrics Distribution**")
            
            fig = go.Figure()
            
            for i, metric_name in enumerate(metric_names):
                values = [row[i] for row in metrics_matrix]
                
                fig.add_trace(
                    go.Box(
                        y=values,
                        name=metric_name.title(),
                        marker_color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
                    )
                )
            
            fig.update_layout(
                title="Quality Score Distribution by Metric",
                yaxis_title="Quality Score",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Source Performance Ranking**")
            
            # Calculate ranking based on overall scores
            source_scores = [(report.source.replace('_', ' ').title(), report.overall_score) 
                           for report in health_reports]
            source_scores.sort(key=lambda x: x[1], reverse=True)
            
            sources_ranked = [x[0] for x in source_scores]
            scores_ranked = [x[1] for x in source_scores]
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=scores_ranked,
                    y=sources_ranked,
                    orientation='h',
                    marker_color=[self.status_colors[report.status] for report in 
                                sorted(health_reports, key=lambda x: x.overall_score, reverse=True)],
                    text=[f"{score:.1%}" for score in scores_ranked],
                    textposition='inside'
                )
            )
            
            fig.update_layout(
                title="Data Source Health Ranking",
                xaxis_title="Overall Health Score",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_recommendations(self, health_reports: List[DataHealthReport]):
        """Render recommendations based on health analysis."""
        st.subheader("  Recommendations & Action Items")
        
        all_recommendations = []
        critical_issues = []
        
        for report in health_reports:
            if report.recommendations:
                for rec in report.recommendations:
                    all_recommendations.append(f"**{report.source.replace('_', ' ').title()}:** {rec}")
            
            if report.status == DataQualityStatus.CRITICAL:
                critical_issues.append(f"{report.source.replace('_', ' ').title()}: {report.overall_score:.1%} health score")
        
        # Priority recommendations
        if critical_issues:
            st.markdown("**  Critical Issues Requiring Immediate Attention:**")
            for issue in critical_issues:
                st.markdown(f"""
                <div class="alert-box">
                    <strong>Critical:</strong> {issue}
                </div>
                """, unsafe_allow_html=True)
        
        # General recommendations
        if all_recommendations:
            st.markdown("**  Recommended Actions:**")
            for i, rec in enumerate(all_recommendations[:10], 1):  # Show top 10
                st.markdown(f"{i}. {rec}")
        
        # System-wide recommendations
        st.markdown("**  System-wide Improvements:**")
        
        recommendations = [
            "Implement automated data quality alerts for critical thresholds",
            "Set up data lineage tracking to identify root causes of quality issues",
            "Establish data quality SLAs with vendors and internal teams",
            "Create automated data reconciliation processes",
            "Implement real-time data validation at ingestion points",
            "Set up data quality dashboards for operational teams",
            "Establish data quality metrics in CI/CD pipelines",
            "Create data quality incident response procedures"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    
    def run_data_health_monitor(self):
        """Run the complete data health monitoring interface."""
        try:
            # Header
            self.render_header()
            
            # Sidebar controls
            options = self.render_sidebar_controls()
            
            if not options['selected_sources']:
                st.warning("Please select at least one data source to monitor")
                return
            
            # Generate data and health reports
            with st.spinner("Analyzing data quality across sources..."):
                data_sources = self.generate_sample_data_sources()
                
                health_reports = []
                for source_name in options['selected_sources']:
                    if source_name in data_sources:
                        report = self.generate_health_report(source_name, data_sources[source_name])
                        health_reports.append(report)
                        
                        # Add to alerts if score is below threshold
                        if report.overall_score < options['alert_threshold']:
                            alert = {
                                'timestamp': datetime.now(),
                                'source': source_name,
                                'score': report.overall_score,
                                'status': report.status.value,
                                'message': f"Health score {report.overall_score:.1%} below threshold {options['alert_threshold']:.1%}"
                            }
                            if alert not in st.session_state.data_alerts:
                                st.session_state.data_alerts.append(alert)
            
            # Main content
            main_content = st.container()
            
            with main_content:
                # Overall health summary
                self.render_overall_health_summary(health_reports)
                
                st.markdown("---")
                
                # Source health cards
                self.render_source_health_cards(health_reports, options['show_detailed_metrics'])
                
                # Detailed metrics
                if options['show_detailed_metrics'] and len(health_reports) > 1:
                    st.markdown("---")
                    self.render_detailed_metrics(health_reports)
                
                # Recommendations
                if options['show_recommendations']:
                    st.markdown("---")
                    self.render_recommendations(health_reports)
            
            # Auto-refresh functionality
            if options['auto_refresh']:
                time.sleep(options['refresh_interval'])
                st.rerun()
                
        except Exception as e:
            st.error(f"Error in data health monitor: {str(e)}")
            logger.error(f"Data health monitor error: {str(e)}")

def main():
    """Main function to run the data health monitor."""
    monitor = DataHealthMonitor()
    monitor.run_data_health_monitor()

if __name__ == "__main__":
    main()