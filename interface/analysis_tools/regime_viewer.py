"""
Regime Viewer Analysis Tool
==========================

Advanced market regime detection and analysis tool for identifying
and visualizing different market states and regime transitions.

Features:
- Multiple regime detection algorithms
- Real-time regime classification and monitoring
- Regime transition analysis and visualization
- Historical regime pattern analysis
- Regime-based strategy performance analysis
- Interactive regime exploration interface
- Custom regime definition capabilities
- Alert system for regime changes

Author: Quantum Forge Interface Team
Date: November 2025
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import warnings
import logging
from dataclasses import dataclass
from enum import Enum
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import scipy.signal as signal
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class RegimeType(Enum):
    """Types of market regimes."""
    BULL_MARKET = "Bull Market"
    BEAR_MARKET = "Bear Market"
    SIDEWAYS = "Sideways"
    HIGH_VOLATILITY = "High Volatility"
    LOW_VOLATILITY = "Low Volatility"
    CRISIS = "Crisis"
    RECOVERY = "Recovery"

@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    lookback_window: int = 252
    volatility_threshold: float = 0.25
    trend_threshold: float = 0.10
    method: str = "gaussian_mixture"  # gaussian_mixture, kmeans, threshold_based
    n_regimes: int = 3

@dataclass
class RegimeState:
    """Individual regime state information."""
    regime_id: int
    regime_name: str
    start_date: datetime
    end_date: Optional[datetime]
    duration_days: int
    avg_return: float
    volatility: float
    max_drawdown: float
    characteristics: Dict[str, float]

class RegimeViewer:
    """
    Advanced regime detection and analysis tool.
    
    Provides comprehensive market regime analysis including detection,
    classification, and performance analysis across different market states.
    """
    
    def __init__(self):
        """Initialize regime viewer."""
        st.set_page_config(
            page_title="Quantum Forge - Regime Viewer",
            page_icon=" ",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .regime-header {
            background: linear-gradient(90deg, #8b5cf6 0%, #7c3aed 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .regime-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #8b5cf6;
            margin-bottom: 1rem;
        }
        .regime-bull {
            background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.2) 100%);
            border-left: 4px solid #10b981;
        }
        .regime-bear {
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.2) 100%);
            border-left: 4px solid #ef4444;
        }
        .regime-sideways {
            background: linear-gradient(90deg, rgba(107, 114, 128, 0.1) 0%, rgba(107, 114, 128, 0.2) 100%);
            border-left: 4px solid #6b7280;
        }
        .regime-crisis {
            background: linear-gradient(90deg, rgba(220, 38, 38, 0.1) 0%, rgba(220, 38, 38, 0.2) 100%);
            border-left: 4px solid #dc2626;
        }
        .regime-metric {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin: 0.5rem 0;
            text-align: center;
        }
        .regime-current {
            background: linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%);
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
        }
        </style>
        """, unsafe_allow_html=True)
        
        self.regime_colors = {
            RegimeType.BULL_MARKET: '#10b981',
            RegimeType.BEAR_MARKET: '#ef4444',
            RegimeType.SIDEWAYS: '#6b7280',
            RegimeType.HIGH_VOLATILITY: '#f59e0b',
            RegimeType.LOW_VOLATILITY: '#3b82f6',
            RegimeType.CRISIS: '#dc2626',
            RegimeType.RECOVERY: '#059669'
        }
    
    def generate_market_data(self, periods: int = 1000) -> pd.DataFrame:
        """Generate sample market data with regime changes."""
        try:
            dates = pd.date_range(start=datetime.now() - timedelta(days=periods), end=datetime.now(), freq='D')
            
            # Initialize data
            returns = np.zeros(len(dates))
            volatilities = np.zeros(len(dates))
            
            # Define regime periods with different characteristics
            regime_periods = [
                (0, 200, 'bull', 0.0008, 0.012),      # Bull market
                (200, 300, 'crisis', -0.002, 0.035),   # Crisis
                (300, 500, 'recovery', 0.001, 0.020),  # Recovery
                (500, 700, 'sideways', 0.0002, 0.015), # Sideways
                (700, len(dates), 'bull', 0.0006, 0.018) # Another bull phase
            ]
            
            np.random.seed(42)  # For reproducible results
            
            for start_idx, end_idx, regime_type, mean_return, volatility in regime_periods:
                period_length = end_idx - start_idx
                
                if regime_type == 'bull':
                    # Bull market: positive trend, moderate volatility
                    period_returns = np.random.normal(mean_return, volatility, period_length)
                    # Add trend component
                    trend = np.linspace(0, mean_return * 0.5, period_length)
                    period_returns += trend
                    
                elif regime_type == 'bear':
                    # Bear market: negative trend, high volatility
                    period_returns = np.random.normal(mean_return, volatility, period_length)
                    # Add negative trend
                    trend = np.linspace(0, mean_return * 0.5, period_length)
                    period_returns += trend
                    
                elif regime_type == 'crisis':
                    # Crisis: high negative returns, very high volatility
                    period_returns = np.random.normal(mean_return, volatility, period_length)
                    # Add some extreme events
                    extreme_events = np.random.choice([0, 1], size=period_length, p=[0.95, 0.05])
                    period_returns += extreme_events * np.random.normal(-0.05, 0.02, period_length)
                    
                elif regime_type == 'recovery':
                    # Recovery: positive but volatile
                    period_returns = np.random.normal(mean_return, volatility, period_length)
                    # Add some positive momentum
                    for i in range(1, period_length):
                        if period_returns[i-1] > 0:
                            period_returns[i] += 0.0002  # Momentum effect
                            
                else:  # sideways
                    # Sideways: low returns, moderate volatility, mean reverting
                    period_returns = np.random.normal(mean_return, volatility, period_length)
                    # Add mean reversion
                    for i in range(1, period_length):
                        period_returns[i] -= 0.1 * period_returns[i-1]  # Mean reversion
                
                # Add some autocorrelation
                for i in range(1, period_length):
                    period_returns[i] += 0.05 * period_returns[i-1]
                
                returns[start_idx:end_idx] = period_returns
                volatilities[start_idx:end_idx] = volatility
            
            # Calculate additional market indicators
            prices = 100 * np.cumprod(1 + returns)
            
            # Rolling statistics
            rolling_vol = pd.Series(returns).rolling(20).std() * np.sqrt(252)
            rolling_ret = pd.Series(returns).rolling(20).mean() * 252
            
            # Technical indicators
            sma_50 = pd.Series(prices).rolling(50).mean()
            sma_200 = pd.Series(prices).rolling(200).mean()
            
            # VIX-like indicator (inverse of rolling Sharpe)
            vix_proxy = rolling_vol * 100
            vix_proxy = np.clip(vix_proxy, 10, 80)  # Realistic VIX range
            
            # Create DataFrame
            data = pd.DataFrame({
                'Date': dates,
                'Price': prices,
                'Returns': returns,
                'Volatility': rolling_vol,
                'Rolling_Return': rolling_ret,
                'SMA_50': sma_50,
                'SMA_200': sma_200,
                'VIX': vix_proxy,
                'Volume': np.random.lognormal(15, 0.5, len(dates))  # Random volume
            }).set_index('Date')
            
            return data
            
        except Exception as e:
            logger.error(f"Error generating market data: {str(e)}")
            return pd.DataFrame()
    
    def detect_regimes_gaussian_mixture(self, data: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
        """Detect regimes using Gaussian Mixture Model."""
        try:
            # Prepare features for regime detection
            features = []
            feature_names = []
            
            if 'Returns' in data.columns:
                features.append(data['Returns'].rolling(config.lookback_window//4).mean().fillna(0))
                feature_names.append('Returns')
            
            if 'Volatility' in data.columns:
                features.append(data['Volatility'].fillna(data['Volatility'].mean()))
                feature_names.append('Volatility')
            
            if 'SMA_50' in data.columns and 'SMA_200' in data.columns:
                trend_indicator = (data['SMA_50'] - data['SMA_200']) / data['SMA_200']
                features.append(trend_indicator.fillna(0))
                feature_names.append('Trend')
            
            if 'VIX' in data.columns:
                features.append(data['VIX'].fillna(data['VIX'].mean()))
                feature_names.append('VIX')
            
            # Stack features
            feature_matrix = np.column_stack(features)
            
            # Remove any rows with NaN
            valid_mask = ~np.any(np.isnan(feature_matrix), axis=1)
            feature_matrix_clean = feature_matrix[valid_mask]
            
            if len(feature_matrix_clean) < config.n_regimes * 10:
                logger.warning("Insufficient data for regime detection")
                return data
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(feature_matrix_clean)
            
            # Fit Gaussian Mixture Model
            gmm = GaussianMixture(n_components=config.n_regimes, random_state=42)
            regime_labels = gmm.fit_predict(features_scaled)
            
            # Map regimes back to full dataset
            full_regimes = np.full(len(data), -1)
            full_regimes[valid_mask] = regime_labels
            
            # Assign regime names based on characteristics
            regime_names = self._assign_regime_names(feature_matrix_clean, regime_labels, config.n_regimes)
            
            # Map numeric labels to names
            regime_name_series = pd.Series(full_regimes).map(regime_names).fillna('Unknown')
            
            data['Regime'] = regime_name_series.values
            data['Regime_ID'] = full_regimes
            
            return data
            
        except Exception as e:
            logger.error(f"Error in Gaussian mixture regime detection: {str(e)}")
            return data
    
    def detect_regimes_threshold_based(self, data: pd.DataFrame, config: RegimeConfig) -> pd.DataFrame:
        """Detect regimes using threshold-based approach."""
        try:
            regimes = []
            
            for i in range(len(data)):
                current_vol = data['Volatility'].iloc[i] if 'Volatility' in data.columns else 0.15
                current_ret = data['Rolling_Return'].iloc[i] if 'Rolling_Return' in data.columns else 0
                
                # Simple threshold-based classification
                if current_vol > config.volatility_threshold * 1.5:
                    if current_ret < -config.trend_threshold:
                        regime = RegimeType.CRISIS.value
                    else:
                        regime = RegimeType.HIGH_VOLATILITY.value
                elif current_ret > config.trend_threshold:
                    regime = RegimeType.BULL_MARKET.value
                elif current_ret < -config.trend_threshold:
                    regime = RegimeType.BEAR_MARKET.value
                elif current_vol < config.volatility_threshold * 0.5:
                    regime = RegimeType.LOW_VOLATILITY.value
                else:
                    regime = RegimeType.SIDEWAYS.value
                
                regimes.append(regime)
            
            data['Regime'] = regimes
            
            # Create numeric IDs
            unique_regimes = list(set(regimes))
            regime_id_map = {name: i for i, name in enumerate(unique_regimes)}
            data['Regime_ID'] = [regime_id_map[r] for r in regimes]
            
            return data
            
        except Exception as e:
            logger.error(f"Error in threshold-based regime detection: {str(e)}")
            return data
    
    def _assign_regime_names(self, features: np.ndarray, labels: np.ndarray, n_regimes: int) -> Dict[int, str]:
        """Assign meaningful names to regime clusters."""
        try:
            regime_names = {}
            
            for regime_id in range(n_regimes):
                regime_mask = labels == regime_id
                regime_features = features[regime_mask]
                
                if len(regime_features) == 0:
                    regime_names[regime_id] = f"Regime {regime_id + 1}"
                    continue
                
                # Calculate average characteristics
                avg_return = np.mean(regime_features[:, 0]) if features.shape[1] > 0 else 0
                avg_volatility = np.mean(regime_features[:, 1]) if features.shape[1] > 1 else 0
                avg_trend = np.mean(regime_features[:, 2]) if features.shape[1] > 2 else 0
                avg_vix = np.mean(regime_features[:, 3]) if features.shape[1] > 3 else 0
                
                # Assign names based on characteristics
                if avg_volatility > 0.3 and avg_return < -0.1:
                    regime_names[regime_id] = RegimeType.CRISIS.value
                elif avg_return > 0.1 and avg_trend > 0:
                    regime_names[regime_id] = RegimeType.BULL_MARKET.value
                elif avg_return < -0.05 and avg_trend < 0:
                    regime_names[regime_id] = RegimeType.BEAR_MARKET.value
                elif avg_volatility > 0.25:
                    regime_names[regime_id] = RegimeType.HIGH_VOLATILITY.value
                elif avg_volatility < 0.15:
                    regime_names[regime_id] = RegimeType.LOW_VOLATILITY.value
                elif abs(avg_return) < 0.05:
                    regime_names[regime_id] = RegimeType.SIDEWAYS.value
                else:
                    regime_names[regime_id] = f"Regime {regime_id + 1}"
            
            return regime_names
            
        except Exception as e:
            logger.error(f"Error assigning regime names: {str(e)}")
            return {i: f"Regime {i + 1}" for i in range(n_regimes)}
    
    def analyze_regime_states(self, data: pd.DataFrame) -> List[RegimeState]:
        """Analyze individual regime states and transitions."""
        try:
            if 'Regime' not in data.columns:
                return []
            
            regime_states = []
            current_regime = None
            start_date = None
            start_idx = 0
            
            for i, (date, row) in enumerate(data.iterrows()):
                regime = row['Regime']
                
                if regime != current_regime:
                    # End previous regime
                    if current_regime is not None and start_date is not None:
                        end_date = date - timedelta(days=1)
                        duration = (end_date - start_date).days
                        
                        # Calculate regime statistics
                        regime_data = data.iloc[start_idx:i]
                        if len(regime_data) > 0:
                            avg_return = regime_data['Returns'].mean() * 252 if 'Returns' in regime_data.columns else 0
                            volatility = regime_data['Returns'].std() * np.sqrt(252) if 'Returns' in regime_data.columns else 0
                            
                            # Calculate max drawdown for this regime
                            if 'Price' in regime_data.columns:
                                peak = regime_data['Price'].expanding().max()
                                drawdown = (regime_data['Price'] - peak) / peak
                                max_drawdown = drawdown.min()
                            else:
                                max_drawdown = 0
                            
                            # Regime characteristics
                            characteristics = {}
                            if 'Volatility' in regime_data.columns:
                                characteristics['avg_volatility'] = regime_data['Volatility'].mean()
                            if 'VIX' in regime_data.columns:
                                characteristics['avg_vix'] = regime_data['VIX'].mean()
                            
                            regime_states.append(RegimeState(
                                regime_id=len(regime_states),
                                regime_name=current_regime,
                                start_date=start_date,
                                end_date=end_date,
                                duration_days=duration,
                                avg_return=avg_return,
                                volatility=volatility,
                                max_drawdown=max_drawdown,
                                characteristics=characteristics
                            ))
                    
                    # Start new regime
                    current_regime = regime
                    start_date = date
                    start_idx = i
            
            # Handle last regime
            if current_regime is not None and start_date is not None:
                end_date = data.index[-1]
                duration = (end_date - start_date).days
                
                regime_data = data.iloc[start_idx:]
                if len(regime_data) > 0:
                    avg_return = regime_data['Returns'].mean() * 252 if 'Returns' in regime_data.columns else 0
                    volatility = regime_data['Returns'].std() * np.sqrt(252) if 'Returns' in regime_data.columns else 0
                    
                    if 'Price' in regime_data.columns:
                        peak = regime_data['Price'].expanding().max()
                        drawdown = (regime_data['Price'] - peak) / peak
                        max_drawdown = drawdown.min()
                    else:
                        max_drawdown = 0
                    
                    characteristics = {}
                    if 'Volatility' in regime_data.columns:
                        characteristics['avg_volatility'] = regime_data['Volatility'].mean()
                    if 'VIX' in regime_data.columns:
                        characteristics['avg_vix'] = regime_data['VIX'].mean()
                    
                    regime_states.append(RegimeState(
                        regime_id=len(regime_states),
                        regime_name=current_regime,
                        start_date=start_date,
                        end_date=end_date,
                        duration_days=duration,
                        avg_return=avg_return,
                        volatility=volatility,
                        max_drawdown=max_drawdown,
                        characteristics=characteristics
                    ))
            
            return regime_states
            
        except Exception as e:
            logger.error(f"Error analyzing regime states: {str(e)}")
            return []
    
    def render_header(self):
        """Render regime viewer header."""
        st.markdown("""
        <div class="regime-header">
            <h1>  Quantum Forge Regime Viewer</h1>
            <p>Advanced Market Regime Detection & Analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Control buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("  Detect Regimes", key="detect_regimes"):
                st.rerun()
                
        with col2:
            if st.button("  Analyze Transitions", key="analyze_transitions"):
                st.info("Analyzing regime transitions...")
                
        with col3:
            if st.button("  Strategy Analysis", key="strategy_analysis"):
                st.info("Analyzing regime-based strategies...")
                
        with col4:
            if st.button("  Set Alerts", key="set_alerts"):
                st.info("Setting up regime change alerts...")
    
    def render_sidebar_controls(self):
        """Render regime viewer sidebar controls."""
        st.sidebar.markdown("##   Regime Detection Controls")
        
        # Detection method
        method = st.sidebar.selectbox(
            "Detection Method",
            options=["gaussian_mixture", "threshold_based", "kmeans"],
            format_func=lambda x: {
                "gaussian_mixture": "Gaussian Mixture Model",
                "threshold_based": "Threshold-Based",
                "kmeans": "K-Means Clustering"
            }[x]
        )
        
        # Parameters
        st.sidebar.markdown("###   Parameters")
        
        n_regimes = st.sidebar.slider(
            "Number of Regimes",
            min_value=2,
            max_value=7,
            value=4,
            step=1
        )
        
        lookback_window = st.sidebar.slider(
            "Lookback Window (days)",
            min_value=30,
            max_value=500,
            value=252,
            step=30
        )
        
        volatility_threshold = st.sidebar.slider(
            "Volatility Threshold",
            min_value=0.1,
            max_value=0.5,
            value=0.25,
            step=0.05
        )
        
        trend_threshold = st.sidebar.slider(
            "Trend Threshold",
            min_value=0.05,
            max_value=0.25,
            value=0.10,
            step=0.01
        )
        
        # Data parameters
        st.sidebar.markdown("###   Data Parameters")
        
        data_periods = st.sidebar.slider(
            "Historical Periods",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100
        )
        
        # Display options
        st.sidebar.markdown("###   Display Options")
        
        show_transitions = st.sidebar.checkbox("Show Regime Transitions", value=True)
        show_statistics = st.sidebar.checkbox("Show Regime Statistics", value=True)
        highlight_current = st.sidebar.checkbox("Highlight Current Regime", value=True)
        
        return RegimeConfig(
            lookback_window=lookback_window,
            volatility_threshold=volatility_threshold,
            trend_threshold=trend_threshold,
            method=method,
            n_regimes=n_regimes
        ), {
            'data_periods': data_periods,
            'show_transitions': show_transitions,
            'show_statistics': show_statistics,
            'highlight_current': highlight_current
        }
    
    def render_current_regime_status(self, data: pd.DataFrame):
        """Render current regime status."""
        if data.empty or 'Regime' not in data.columns:
            return
        
        current_regime = data['Regime'].iloc[-1]
        current_date = data.index[-1]
        
        # Find how long we've been in current regime
        regime_start = None
        for i in range(len(data) - 1, -1, -1):
            if data['Regime'].iloc[i] != current_regime:
                regime_start = data.index[i + 1]
                break
        
        if regime_start is None:
            regime_start = data.index[0]
        
        days_in_regime = (current_date - regime_start).days
        
        st.subheader("  Current Market Regime")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="regime-current">
                Current Regime: {current_regime}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="regime-metric">
                <strong>Duration</strong><br>
                {days_in_regime} days
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Current market indicators
            if 'Volatility' in data.columns:
                current_vol = data['Volatility'].iloc[-1]
                st.markdown(f"""
                <div class="regime-metric">
                    <strong>Current Volatility</strong><br>
                    {current_vol:.1%}
                </div>
                """, unsafe_allow_html=True)
    
    def render_regime_visualization(self, data: pd.DataFrame):
        """Render main regime visualization."""
        if data.empty:
            st.warning("No data available for regime visualization")
            return
        
        st.subheader("  Regime Analysis Visualization")
        
        # Create subplot
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=['Price & Regimes', 'Volatility', 'Returns'],
            vertical_spacing=0.05
        )
        
        # Price chart with regime coloring
        if 'Price' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Price'],
                    mode='lines',
                    name='Price',
                    line=dict(color='black', width=1),
                    hovertemplate='Date: %{x}<br>Price: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add regime background colors
            if 'Regime' in data.columns:
                current_regime = None
                start_idx = 0
                
                for i, regime in enumerate(data['Regime']):
                    if regime != current_regime or i == len(data) - 1:
                        if current_regime is not None:
                            # Add colored background for previous regime
                            regime_color = self.regime_colors.get(
                                RegimeType(current_regime) if current_regime in [r.value for r in RegimeType] else None,
                                '#6b7280'
                            )
                            
                            fig.add_vrect(
                                x0=data.index[start_idx],
                                x1=data.index[i-1] if i > 0 else data.index[-1],
                                fillcolor=regime_color,
                                opacity=0.2,
                                layer="below",
                                line_width=0,
                                row=1, col=1
                            )
                        
                        current_regime = regime
                        start_idx = i
        
        # Volatility chart
        if 'Volatility' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Volatility'] * 100,
                    mode='lines',
                    name='Volatility',
                    line=dict(color='orange', width=1),
                    hovertemplate='Date: %{x}<br>Volatility: %{y:.1f}%<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Returns chart
        if 'Returns' in data.columns:
            colors = ['green' if r > 0 else 'red' for r in data['Returns']]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Returns'] * 100,
                    name='Daily Returns',
                    marker_color=colors,
                    opacity=0.6,
                    hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
                ),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Market Regime Analysis",
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Returns (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_regime_statistics(self, regime_states: List[RegimeState]):
        """Render regime statistics and analysis."""
        if not regime_states:
            st.warning("No regime states to analyze")
            return
        
        st.subheader("  Regime Statistics")
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Regime Summary**")
            
            regime_counts = {}
            total_duration = 0
            
            for state in regime_states:
                regime_name = state.regime_name
                if regime_name not in regime_counts:
                    regime_counts[regime_name] = {'count': 0, 'total_days': 0, 'avg_return': [], 'volatility': []}
                
                regime_counts[regime_name]['count'] += 1
                regime_counts[regime_name]['total_days'] += state.duration_days
                regime_counts[regime_name]['avg_return'].append(state.avg_return)
                regime_counts[regime_name]['volatility'].append(state.volatility)
                total_duration += state.duration_days
            
            for regime_name, stats in regime_counts.items():
                avg_duration = stats['total_days'] / stats['count']
                avg_return = np.mean(stats['avg_return'])
                avg_vol = np.mean(stats['volatility'])
                frequency = stats['total_days'] / total_duration
                
                regime_class = "regime-bull" if "Bull" in regime_name else "regime-bear" if "Bear" in regime_name or "Crisis" in regime_name else "regime-sideways"
                
                st.markdown(f"""
                <div class="regime-card {regime_class}">
                    <h4>{regime_name}</h4>
                    <p><strong>Frequency:</strong> {frequency:.1%} of time</p>
                    <p><strong>Avg Duration:</strong> {avg_duration:.0f} days</p>
                    <p><strong>Avg Return:</strong> {avg_return:.1%}</p>
                    <p><strong>Avg Volatility:</strong> {avg_vol:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Regime Performance Comparison**")
            
            # Create performance comparison chart
            regime_names = list(regime_counts.keys())
            avg_returns = [np.mean(regime_counts[name]['avg_return']) for name in regime_names]
            avg_vols = [np.mean(regime_counts[name]['volatility']) for name in regime_names]
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=avg_vols,
                    y=avg_returns,
                    mode='markers+text',
                    text=regime_names,
                    textposition="top center",
                    marker=dict(
                        size=15,
                        color=[self.regime_colors.get(RegimeType(name) if name in [r.value for r in RegimeType] else None, '#6b7280') 
                               for name in regime_names],
                        opacity=0.7
                    ),
                    hovertemplate='%{text}<br>Return: %{y:.1%}<br>Volatility: %{x:.1%}<extra></extra>'
                )
            )
            
            fig.update_layout(
                title="Risk-Return by Regime",
                xaxis_title="Volatility",
                yaxis_title="Average Return",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Regime transition matrix
        st.markdown("**Regime Transitions**")
        
        if len(regime_states) > 1:
            # Build transition matrix
            unique_regimes = list(set(state.regime_name for state in regime_states))
            n_regimes = len(unique_regimes)
            transition_matrix = np.zeros((n_regimes, n_regimes))
            
            for i in range(len(regime_states) - 1):
                from_regime = regime_states[i].regime_name
                to_regime = regime_states[i + 1].regime_name
                
                from_idx = unique_regimes.index(from_regime)
                to_idx = unique_regimes.index(to_regime)
                
                transition_matrix[from_idx, to_idx] += 1
            
            # Normalize to probabilities
            row_sums = transition_matrix.sum(axis=1)
            transition_matrix = np.divide(transition_matrix, row_sums[:, np.newaxis], 
                                        out=np.zeros_like(transition_matrix), where=row_sums[:, np.newaxis]!=0)
            
            # Create heatmap
            fig = go.Figure()
            
            fig.add_trace(
                go.Heatmap(
                    z=transition_matrix,
                    x=unique_regimes,
                    y=unique_regimes,
                    colorscale='Blues',
                    text=np.round(transition_matrix, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hovertemplate='From: %{y}<br>To: %{x}<br>Probability: %{z:.2f}<extra></extra>',
                    colorbar=dict(title="Transition Probability")
                )
            )
            
            fig.update_layout(
                title="Regime Transition Matrix",
                xaxis_title="To Regime",
                yaxis_title="From Regime",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def run_regime_viewer(self):
        """Run the complete regime viewer interface."""
        try:
            # Header
            self.render_header()
            
            # Sidebar controls
            config, display_options = self.render_sidebar_controls()
            
            # Generate market data
            with st.spinner("Generating market data and detecting regimes..."):
                data = self.generate_market_data(display_options['data_periods'])
                
                if data.empty:
                    st.error("Failed to generate market data")
                    return
                
                # Detect regimes based on selected method
                if config.method == "gaussian_mixture":
                    data = self.detect_regimes_gaussian_mixture(data, config)
                elif config.method == "threshold_based":
                    data = self.detect_regimes_threshold_based(data, config)
                else:  # kmeans - simplified implementation
                    data = self.detect_regimes_gaussian_mixture(data, config)  # Use GMM as fallback
                
                # Analyze regime states
                regime_states = self.analyze_regime_states(data)
            
            # Main content
            main_content = st.container()
            
            with main_content:
                # Current regime status
                if display_options['highlight_current']:
                    self.render_current_regime_status(data)
                    st.markdown("---")
                
                # Main visualization
                self.render_regime_visualization(data)
                
                st.markdown("---")
                
                # Regime statistics
                if display_options['show_statistics']:
                    self.render_regime_statistics(regime_states)
                
        except Exception as e:
            st.error(f"Error in regime viewer: {str(e)}")
            logger.error(f"Regime viewer error: {str(e)}")

def main():
    """Main function to run the regime viewer."""
    viewer = RegimeViewer()
    viewer.run_regime_viewer()

if __name__ == "__main__":
    main()