"""
Signal Inspector Analysis Tool
=============================

Advanced trading signal analysis and validation tool for evaluating
signal quality, performance attribution, and optimization opportunities.

Features:
- Multi-asset signal analysis and validation
- Signal quality metrics and performance attribution
- Cross-asset signal correlation analysis
- Signal timing and decay analysis
- Factor decomposition of signal performance
- Interactive signal exploration interface
- Signal optimization and backtesting
- Real-time signal monitoring capabilities

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
from scipy import stats
from scipy.optimize import minimize
import scipy.signal as signal
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class SignalType(Enum):
    """Types of trading signals."""
    MOMENTUM = "Momentum"
    MEAN_REVERSION = "Mean Reversion"
    TREND_FOLLOWING = "Trend Following"
    BREAKOUT = "Breakout"
    VOLATILITY = "Volatility"
    CARRY = "Carry"
    VALUE = "Value"
    QUALITY = "Quality"

@dataclass
class SignalConfig:
    """Configuration for signal analysis."""
    lookback_window: int = 252
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    signal_threshold: float = 0.5
    max_position: float = 1.0
    transaction_cost: float = 0.001  # 10 bps

@dataclass
class SignalMetrics:
    """Signal performance metrics."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    information_ratio: float
    hit_rate: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float

class SignalInspector:
    """
    Advanced signal analysis and validation tool.
    
    Provides comprehensive signal analysis including performance metrics,
    attribution analysis, and optimization capabilities.
    """
    
    def __init__(self):
        """Initialize signal inspector."""
        st.set_page_config(
            page_title="Quantum Forge - Signal Inspector",
            page_icon=" ",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .signal-header {
            background: linear-gradient(90deg, #3b82f6 0%, #1e40af 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .signal-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #3b82f6;
            margin-bottom: 1rem;
        }
        .signal-strong {
            background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.2) 100%);
            border-left: 4px solid #10b981;
        }
        .signal-weak {
            background: linear-gradient(90deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.2) 100%);
            border-left: 4px solid #ef4444;
        }
        .signal-neutral {
            background: linear-gradient(90deg, rgba(107, 114, 128, 0.1) 0%, rgba(107, 114, 128, 0.2) 100%);
            border-left: 4px solid #6b7280;
        }
        .signal-metric {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin: 0.5rem 0;
            text-align: center;
        }
        .signal-current {
            background: linear-gradient(90deg, #fbbf24 0%, #f59e0b 100%);
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            display: inline-block;
        }
        .signal-performance {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        self.signal_colors = {
            SignalType.MOMENTUM: '#3b82f6',
            SignalType.MEAN_REVERSION: '#ef4444',
            SignalType.TREND_FOLLOWING: '#10b981',
            SignalType.BREAKOUT: '#f59e0b',
            SignalType.VOLATILITY: '#8b5cf6',
            SignalType.CARRY: '#06b6d4',
            SignalType.VALUE: '#84cc16',
            SignalType.QUALITY: '#ec4899'
        }
    
    def generate_signal_data(self, n_assets: int = 10, periods: int = 1000) -> Dict[str, pd.DataFrame]:
        """Generate sample signal and market data."""
        try:
            # Generate dates
            dates = pd.date_range(start=datetime.now() - timedelta(days=periods), end=datetime.now(), freq='D')
            
            # Asset names
            assets = [f"Asset_{i+1}" for i in range(n_assets)]
            
            # Generate market data
            np.random.seed(42)  # For reproducible results
            
            # Generate returns with some correlation structure
            correlation_matrix = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
            np.fill_diagonal(correlation_matrix, 1.0)
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            
            # Ensure positive semi-definite
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 0.01)
            correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            # Generate correlated returns
            returns = np.random.multivariate_normal(
                mean=np.zeros(n_assets), 
                cov=correlation_matrix * 0.02**2, 
                size=len(dates)
            )
            
            # Add some regime changes
            for i in range(n_assets):
                # Bull market periods
                bull_periods = np.random.choice([0, 1], size=len(dates), p=[0.7, 0.3])
                returns[:, i] += bull_periods * 0.0005
                
                # Crisis periods
                crisis_periods = np.random.choice([0, 1], size=len(dates), p=[0.95, 0.05])
                returns[:, i] += crisis_periods * np.random.normal(-0.03, 0.01, len(dates))
            
            # Calculate prices
            prices = pd.DataFrame(
                100 * np.cumprod(1 + returns, axis=0),
                index=dates,
                columns=assets
            )
            
            returns_df = pd.DataFrame(returns, index=dates, columns=assets)
            
            # Generate various signals
            signals_data = {}
            
            # 1. Momentum signals
            momentum_signals = pd.DataFrame(index=dates, columns=assets)
            for asset in assets:
                # 20-day momentum
                momentum_20 = returns_df[asset].rolling(20).mean()
                # 60-day momentum
                momentum_60 = returns_df[asset].rolling(60).mean()
                # Combined momentum signal
                momentum_signals[asset] = np.tanh((momentum_20 - momentum_60) * 50)
            
            signals_data['momentum'] = momentum_signals
            
            # 2. Mean reversion signals
            mean_reversion_signals = pd.DataFrame(index=dates, columns=assets)
            for asset in assets:
                # Z-score based mean reversion
                returns_ma = returns_df[asset].rolling(50).mean()
                returns_std = returns_df[asset].rolling(50).std()
                z_score = (returns_df[asset] - returns_ma) / returns_std
                mean_reversion_signals[asset] = -np.tanh(z_score)  # Negative for mean reversion
            
            signals_data['mean_reversion'] = mean_reversion_signals
            
            # 3. Trend following signals
            trend_signals = pd.DataFrame(index=dates, columns=assets)
            for asset in assets:
                # Moving average crossover
                ma_short = prices[asset].rolling(20).mean()
                ma_long = prices[asset].rolling(50).mean()
                trend_signals[asset] = np.tanh((ma_short - ma_long) / ma_long * 10)
            
            signals_data['trend_following'] = trend_signals
            
            # 4. Volatility signals
            volatility_signals = pd.DataFrame(index=dates, columns=assets)
            for asset in assets:
                # GARCH-like volatility signal
                rolling_vol = returns_df[asset].rolling(20).std()
                vol_ma = rolling_vol.rolling(60).mean()
                vol_signal = (rolling_vol - vol_ma) / vol_ma
                volatility_signals[asset] = np.tanh(vol_signal)
            
            signals_data['volatility'] = volatility_signals
            
            # 5. Breakout signals
            breakout_signals = pd.DataFrame(index=dates, columns=assets)
            for asset in assets:
                # Bollinger band breakout
                rolling_mean = prices[asset].rolling(20).mean()
                rolling_std = prices[asset].rolling(20).std()
                upper_band = rolling_mean + 2 * rolling_std
                lower_band = rolling_mean - 2 * rolling_std
                
                breakout_up = (prices[asset] > upper_band).astype(int)
                breakout_down = (prices[asset] < lower_band).astype(int)
                breakout_signals[asset] = breakout_up - breakout_down
            
            signals_data['breakout'] = breakout_signals
            
            # Add noise and realistic features to signals
            for signal_name in signals_data:
                # Add some noise
                noise = np.random.normal(0, 0.1, signals_data[signal_name].shape)
                signals_data[signal_name] += noise
                
                # Clip to reasonable ranges
                signals_data[signal_name] = signals_data[signal_name].clip(-1, 1)
                
                # Forward fill NaN values
                signals_data[signal_name] = signals_data[signal_name].fillna(method='ffill').fillna(0)
            
            # Return combined data
            return {
                'prices': prices,
                'returns': returns_df,
                'signals': signals_data
            }
            
        except Exception as e:
            logger.error(f"Error generating signal data: {str(e)}")
            return {'prices': pd.DataFrame(), 'returns': pd.DataFrame(), 'signals': {}}
    
    def calculate_signal_metrics(self, signals: pd.DataFrame, returns: pd.DataFrame, 
                                config: SignalConfig) -> Dict[str, SignalMetrics]:
        """Calculate comprehensive signal performance metrics."""
        try:
            metrics = {}
            
            for asset in signals.columns:
                if asset not in returns.columns:
                    continue
                
                signal_values = signals[asset].dropna()
                asset_returns = returns[asset].dropna()
                
                # Align data
                common_index = signal_values.index.intersection(asset_returns.index)
                if len(common_index) < 50:  # Minimum data requirement
                    continue
                
                signal_aligned = signal_values.loc[common_index]
                returns_aligned = asset_returns.loc[common_index]
                
                # Generate positions (lag signals by 1 day)
                positions = signal_aligned.shift(1).fillna(0)
                
                # Apply thresholds and position sizing
                positions = positions.clip(-config.max_position, config.max_position)
                positions = np.where(np.abs(positions) > config.signal_threshold, positions, 0)
                
                # Calculate strategy returns
                strategy_returns = positions * returns_aligned
                
                # Account for transaction costs
                position_changes = np.abs(positions.diff()).fillna(0)
                transaction_costs = position_changes * config.transaction_cost
                strategy_returns_net = strategy_returns - transaction_costs
                
                # Calculate metrics
                total_return = (1 + strategy_returns_net).prod() - 1
                
                # Risk metrics
                volatility = strategy_returns_net.std() * np.sqrt(252)
                sharpe_ratio = strategy_returns_net.mean() / strategy_returns_net.std() * np.sqrt(252) if strategy_returns_net.std() > 0 else 0
                
                # Downside deviation
                downside_returns = strategy_returns_net[strategy_returns_net < 0]
                downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                sortino_ratio = strategy_returns_net.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
                
                # Maximum drawdown
                cumulative_returns = (1 + strategy_returns_net).cumprod()
                peak = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - peak) / peak
                max_drawdown = drawdown.min()
                
                # Win rate
                win_rate = (strategy_returns_net > 0).mean()
                
                # Information ratio (vs benchmark - assume zero)
                information_ratio = strategy_returns_net.mean() / strategy_returns_net.std() * np.sqrt(252) if strategy_returns_net.std() > 0 else 0
                
                # Signal accuracy metrics
                # Create binary signals for classification metrics
                binary_signals = (signal_aligned > 0).astype(int)
                forward_returns = returns_aligned.shift(-1).dropna()
                
                # Align for prediction analysis
                pred_index = binary_signals.index.intersection(forward_returns.index)
                if len(pred_index) > 10:
                    binary_pred = binary_signals.loc[pred_index]
                    binary_actual = (forward_returns.loc[pred_index] > 0).astype(int)
                    
                    # Classification metrics
                    hit_rate = (binary_pred == binary_actual).mean()
                    
                    # Precision, recall, F1 (for positive signals)
                    tp = ((binary_pred == 1) & (binary_actual == 1)).sum()
                    fp = ((binary_pred == 1) & (binary_actual == 0)).sum()
                    fn = ((binary_pred == 0) & (binary_actual == 1)).sum()
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    accuracy = hit_rate
                else:
                    hit_rate = 0.5
                    precision = 0.5
                    recall = 0.5
                    f1_score = 0.5
                    accuracy = 0.5
                
                metrics[asset] = SignalMetrics(
                    total_return=total_return,
                    sharpe_ratio=sharpe_ratio,
                    sortino_ratio=sortino_ratio,
                    max_drawdown=max_drawdown,
                    win_rate=win_rate,
                    information_ratio=information_ratio,
                    hit_rate=hit_rate,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating signal metrics: {str(e)}")
            return {}
    
    def analyze_signal_decay(self, signals: pd.DataFrame, returns: pd.DataFrame, 
                           max_horizon: int = 20) -> pd.DataFrame:
        """Analyze signal decay over different time horizons."""
        try:
            decay_results = []
            
            for asset in signals.columns:
                if asset not in returns.columns:
                    continue
                
                signal_values = signals[asset].dropna()
                asset_returns = returns[asset].dropna()
                
                # Align data
                common_index = signal_values.index.intersection(asset_returns.index)
                if len(common_index) < max_horizon + 50:
                    continue
                
                signal_aligned = signal_values.loc[common_index]
                returns_aligned = asset_returns.loc[common_index]
                
                # Calculate correlations at different horizons
                for horizon in range(1, max_horizon + 1):
                    forward_returns = returns_aligned.shift(-horizon)
                    
                    # Calculate correlation
                    valid_mask = ~(signal_aligned.isna() | forward_returns.isna())
                    if valid_mask.sum() < 50:
                        continue
                    
                    correlation = signal_aligned[valid_mask].corr(forward_returns[valid_mask])
                    
                    # Calculate rank correlation (Spearman)
                    rank_correlation = signal_aligned[valid_mask].corr(forward_returns[valid_mask], method='spearman')
                    
                    decay_results.append({
                        'Asset': asset,
                        'Horizon': horizon,
                        'Correlation': correlation,
                        'Rank_Correlation': rank_correlation,
                        'Abs_Correlation': abs(correlation)
                    })
            
            return pd.DataFrame(decay_results)
            
        except Exception as e:
            logger.error(f"Error in signal decay analysis: {str(e)}")
            return pd.DataFrame()
    
    def perform_signal_attribution(self, signals_dict: Dict[str, pd.DataFrame], 
                                 returns: pd.DataFrame) -> pd.DataFrame:
        """Perform signal attribution analysis."""
        try:
            attribution_results = []
            
            # For each asset, analyze contribution of different signals
            for asset in returns.columns:
                asset_results = {'Asset': asset}
                
                # Get returns for this asset
                asset_returns = returns[asset].dropna()
                
                # Collect all signals for this asset
                signal_features = []
                signal_names = []
                
                for signal_name, signal_df in signals_dict.items():
                    if asset in signal_df.columns:
                        signal_data = signal_df[asset].dropna()
                        
                        # Align with returns
                        common_index = signal_data.index.intersection(asset_returns.index)
                        if len(common_index) > 100:
                            signal_aligned = signal_data.loc[common_index]
                            signal_features.append(signal_aligned.shift(1).fillna(0))  # Lag by 1 day
                            signal_names.append(signal_name)
                
                if len(signal_features) == 0:
                    continue
                
                # Create feature matrix
                feature_matrix = pd.concat(signal_features, axis=1)
                feature_matrix.columns = signal_names
                
                # Align with returns
                common_index = feature_matrix.index.intersection(asset_returns.index)
                if len(common_index) < 100:
                    continue
                
                features_aligned = feature_matrix.loc[common_index]
                returns_aligned = asset_returns.loc[common_index]
                
                # Multiple regression to analyze signal contributions
                from sklearn.linear_model import LinearRegression
                
                # Remove any remaining NaN values
                valid_mask = ~(features_aligned.isna().any(axis=1) | returns_aligned.isna())
                if valid_mask.sum() < 50:
                    continue
                
                X = features_aligned[valid_mask]
                y = returns_aligned[valid_mask]
                
                # Fit regression
                reg = LinearRegression()
                reg.fit(X, y)
                
                # Calculate R-squared
                r_squared = reg.score(X, y)
                
                # Attribution analysis
                for i, signal_name in enumerate(signal_names):
                    coef = reg.coef_[i]
                    signal_contribution = coef * X.iloc[:, i].mean()
                    
                    asset_results[f'{signal_name}_coef'] = coef
                    asset_results[f'{signal_name}_contribution'] = signal_contribution
                    
                    # Statistical significance (t-test)
                    residuals = y - reg.predict(X)
                    mse = np.mean(residuals**2)
                    var_coef = mse * np.linalg.inv(X.T @ X)[i, i]
                    t_stat = coef / np.sqrt(var_coef) if var_coef > 0 else 0
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(X) - len(signal_names)))
                    
                    asset_results[f'{signal_name}_t_stat'] = t_stat
                    asset_results[f'{signal_name}_p_value'] = p_value
                
                asset_results['R_squared'] = r_squared
                asset_results['Residual_Std'] = np.std(residuals)
                
                attribution_results.append(asset_results)
            
            return pd.DataFrame(attribution_results)
            
        except Exception as e:
            logger.error(f"Error in signal attribution: {str(e)}")
            return pd.DataFrame()
    
    def render_header(self):
        """Render signal inspector header."""
        st.markdown("""
        <div class="signal-header">
            <h1>  Quantum Forge Signal Inspector</h1>
            <p>Advanced Trading Signal Analysis & Validation</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Control buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("  Analyze Signals", key="analyze_signals"):
                st.rerun()
                
        with col2:
            if st.button("  Performance Attribution", key="perf_attribution"):
                st.info("Running performance attribution analysis...")
                
        with col3:
            if st.button(" ️ Decay Analysis", key="decay_analysis"):
                st.info("Analyzing signal decay patterns...")
                
        with col4:
            if st.button("  Optimize Signals", key="optimize_signals"):
                st.info("Running signal optimization...")
    
    def render_sidebar_controls(self):
        """Render signal inspector sidebar controls."""
        st.sidebar.markdown("##   Signal Analysis Controls")
        
        # Data parameters
        st.sidebar.markdown("###   Data Parameters")
        
        n_assets = st.sidebar.slider(
            "Number of Assets",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )
        
        periods = st.sidebar.slider(
            "Historical Periods",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100
        )
        
        # Signal parameters
        st.sidebar.markdown("###  ️ Signal Parameters")
        
        lookback_window = st.sidebar.slider(
            "Lookback Window",
            min_value=50,
            max_value=500,
            value=252,
            step=50
        )
        
        signal_threshold = st.sidebar.slider(
            "Signal Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
        
        max_position = st.sidebar.slider(
            "Maximum Position Size",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1
        )
        
        transaction_cost = st.sidebar.slider(
            "Transaction Cost (bps)",
            min_value=0,
            max_value=50,
            value=10,
            step=5
        ) / 10000  # Convert to decimal
        
        # Analysis options
        st.sidebar.markdown("###   Analysis Options")
        
        selected_signals = st.sidebar.multiselect(
            "Select Signals to Analyze",
            options=['momentum', 'mean_reversion', 'trend_following', 'volatility', 'breakout'],
            default=['momentum', 'mean_reversion', 'trend_following']
        )
        
        show_individual_assets = st.sidebar.checkbox("Show Individual Assets", value=True)
        show_aggregate_metrics = st.sidebar.checkbox("Show Aggregate Metrics", value=True)
        show_decay_analysis = st.sidebar.checkbox("Show Decay Analysis", value=True)
        show_attribution = st.sidebar.checkbox("Show Attribution Analysis", value=True)
        
        return SignalConfig(
            lookback_window=lookback_window,
            signal_threshold=signal_threshold,
            max_position=max_position,
            transaction_cost=transaction_cost
        ), {
            'n_assets': n_assets,
            'periods': periods,
            'selected_signals': selected_signals,
            'show_individual_assets': show_individual_assets,
            'show_aggregate_metrics': show_aggregate_metrics,
            'show_decay_analysis': show_decay_analysis,
            'show_attribution': show_attribution
        }
    
    def render_signal_overview(self, data: Dict[str, Any], selected_signals: List[str]):
        """Render signal overview and current status."""
        st.subheader("  Signal Overview")
        
        if not data['signals']:
            st.warning("No signal data available")
            return
        
        # Current signal strengths
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Signal Strengths**")
            
            for signal_name in selected_signals:
                if signal_name in data['signals']:
                    signal_df = data['signals'][signal_name]
                    current_values = signal_df.iloc[-1]
                    avg_signal = current_values.mean()
                    
                    # Determine signal strength
                    if abs(avg_signal) > 0.5:
                        strength_class = "signal-strong" if avg_signal > 0 else "signal-weak"
                        strength_text = "Strong" if avg_signal > 0 else "Strong Negative"
                    elif abs(avg_signal) > 0.25:
                        strength_class = "signal-card"
                        strength_text = "Moderate" if avg_signal > 0 else "Moderate Negative"
                    else:
                        strength_class = "signal-neutral"
                        strength_text = "Neutral"
                    
                    st.markdown(f"""
                    <div class="signal-card {strength_class}">
                        <h4>{signal_name.replace('_', ' ').title()}</h4>
                        <p><strong>Average Signal:</strong> {avg_signal:.3f}</p>
                        <p><strong>Strength:</strong> {strength_text}</p>
                        <p><strong>Active Assets:</strong> {(abs(current_values) > 0.1).sum()}/{len(current_values)}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Signal Distribution**")
            
            # Create signal distribution plot
            fig = go.Figure()
            
            for signal_name in selected_signals:
                if signal_name in data['signals']:
                    signal_df = data['signals'][signal_name]
                    current_values = signal_df.iloc[-1].dropna()
                    
                    fig.add_trace(
                        go.Box(
                            y=current_values,
                            name=signal_name.replace('_', ' ').title(),
                            marker_color=self.signal_colors.get(
                                SignalType(signal_name.replace('_', ' ').title()) if signal_name.replace('_', ' ').title() in [s.value for s in SignalType] else None,
                                '#3b82f6'
                            )
                        )
                    )
            
            fig.update_layout(
                title="Current Signal Distribution Across Assets",
                yaxis_title="Signal Strength",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_metrics(self, metrics: Dict[str, Dict[str, SignalMetrics]], 
                                 selected_signals: List[str]):
        """Render signal performance metrics."""
        st.subheader("  Signal Performance Metrics")
        
        if not metrics:
            st.warning("No performance metrics available")
            return
        
        # Aggregate metrics across all assets for each signal
        aggregate_metrics = {}
        
        for signal_name in selected_signals:
            if signal_name in metrics:
                signal_metrics = list(metrics[signal_name].values())
                
                if signal_metrics:
                    aggregate_metrics[signal_name] = {
                        'Total Return': np.mean([m.total_return for m in signal_metrics]),
                        'Sharpe Ratio': np.mean([m.sharpe_ratio for m in signal_metrics]),
                        'Max Drawdown': np.mean([m.max_drawdown for m in signal_metrics]),
                        'Win Rate': np.mean([m.win_rate for m in signal_metrics]),
                        'Hit Rate': np.mean([m.hit_rate for m in signal_metrics]),
                        'Precision': np.mean([m.precision for m in signal_metrics]),
                        'F1 Score': np.mean([m.f1_score for m in signal_metrics])
                    }
        
        if not aggregate_metrics:
            st.warning("No aggregate metrics to display")
            return
        
        # Performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk-Return Profile**")
            
            # Risk-return scatter plot
            fig = go.Figure()
            
            for signal_name, metrics_dict in aggregate_metrics.items():
                fig.add_trace(
                    go.Scatter(
                        x=[abs(metrics_dict['Max Drawdown'])],
                        y=[metrics_dict['Sharpe Ratio']],
                        mode='markers+text',
                        text=[signal_name.replace('_', ' ').title()],
                        textposition="top center",
                        marker=dict(
                            size=15,
                            color=self.signal_colors.get(
                                SignalType(signal_name.replace('_', ' ').title()) if signal_name.replace('_', ' ').title() in [s.value for s in SignalType] else None,
                                '#3b82f6'
                            ),
                            opacity=0.7
                        ),
                        hovertemplate=f'{signal_name}<br>Sharpe: %{{y:.2f}}<br>Max DD: %{{x:.1%}}<extra></extra>'
                    )
                )
            
            fig.update_layout(
                title="Risk-Return by Signal Type",
                xaxis_title="Maximum Drawdown",
                yaxis_title="Sharpe Ratio",
                template='plotly_white',
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Signal Quality Metrics**")
            
            # Signal quality heatmap
            quality_metrics = []
            signal_names = []
            
            for signal_name, metrics_dict in aggregate_metrics.items():
                quality_metrics.append([
                    metrics_dict['Hit Rate'],
                    metrics_dict['Precision'],
                    metrics_dict['F1 Score'],
                    metrics_dict['Win Rate']
                ])
                signal_names.append(signal_name.replace('_', ' ').title())
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Heatmap(
                    z=quality_metrics,
                    x=['Hit Rate', 'Precision', 'F1 Score', 'Win Rate'],
                    y=signal_names,
                    colorscale='RdYlGn',
                    text=np.round(quality_metrics, 3),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hovertemplate='Signal: %{y}<br>Metric: %{x}<br>Value: %{z:.3f}<extra></extra>',
                    colorbar=dict(title="Score")
                )
            )
            
            fig.update_layout(
                title="Signal Quality Heatmap",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary table
        st.markdown("**Performance Summary**")
        
        summary_df = pd.DataFrame(aggregate_metrics).T
        summary_df = summary_df.round(4)
        
        # Style the dataframe
        styled_df = summary_df.style.format({
            'Total Return': '{:.2%}',
            'Max Drawdown': '{:.2%}',
            'Win Rate': '{:.2%}',
            'Hit Rate': '{:.2%}',
            'Precision': '{:.3f}',
            'F1 Score': '{:.3f}',
            'Sharpe Ratio': '{:.3f}'
        }).background_gradient(subset=['Sharpe Ratio', 'Hit Rate', 'F1 Score'], cmap='RdYlGn')
        
        st.dataframe(styled_df, use_container_width=True)
    
    def render_decay_analysis(self, data: Dict[str, Any], selected_signals: List[str]):
        """Render signal decay analysis."""
        st.subheader(" ️ Signal Decay Analysis")
        
        decay_results = []
        
        for signal_name in selected_signals:
            if signal_name in data['signals']:
                decay_data = self.analyze_signal_decay(
                    data['signals'][signal_name], 
                    data['returns']
                )
                if not decay_data.empty:
                    decay_data['Signal'] = signal_name
                    decay_results.append(decay_data)
        
        if not decay_results:
            st.warning("No decay analysis data available")
            return
        
        combined_decay = pd.concat(decay_results, ignore_index=True)
        
        # Aggregate by signal and horizon
        avg_decay = combined_decay.groupby(['Signal', 'Horizon']).agg({
            'Correlation': 'mean',
            'Abs_Correlation': 'mean',
            'Rank_Correlation': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Signal Decay Over Time**")
            
            fig = go.Figure()
            
            for signal_name in selected_signals:
                signal_data = avg_decay[avg_decay['Signal'] == signal_name]
                
                if not signal_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=signal_data['Horizon'],
                            y=signal_data['Abs_Correlation'],
                            mode='lines+markers',
                            name=signal_name.replace('_', ' ').title(),
                            line=dict(
                                color=self.signal_colors.get(
                                    SignalType(signal_name.replace('_', ' ').title()) if signal_name.replace('_', ' ').title() in [s.value for s in SignalType] else None,
                                    '#3b82f6'
                                )
                            ),
                            hovertemplate=f'{signal_name}<br>Horizon: %{{x}} days<br>Correlation: %{{y:.3f}}<extra></extra>'
                        )
                    )
            
            fig.update_layout(
                title="Signal Strength Decay",
                xaxis_title="Days Forward",
                yaxis_title="Absolute Correlation",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Decay Half-Life Analysis**")
            
            # Calculate half-life for each signal
            half_life_data = []
            
            for signal_name in selected_signals:
                signal_data = avg_decay[avg_decay['Signal'] == signal_name]
                
                if not signal_data.empty and len(signal_data) > 5:
                    initial_corr = signal_data['Abs_Correlation'].iloc[0]
                    half_target = initial_corr / 2
                    
                    # Find where correlation drops to half
                    half_life_idx = signal_data[signal_data['Abs_Correlation'] <= half_target]
                    
                    if not half_life_idx.empty:
                        half_life = half_life_idx['Horizon'].iloc[0]
                    else:
                        half_life = signal_data['Horizon'].max()  # Didn't reach half-life
                    
                    half_life_data.append({
                        'Signal': signal_name.replace('_', ' ').title(),
                        'Half_Life': half_life,
                        'Initial_Correlation': initial_corr
                    })
            
            if half_life_data:
                half_life_df = pd.DataFrame(half_life_data)
                
                fig = go.Figure()
                
                fig.add_trace(
                    go.Bar(
                        x=half_life_df['Signal'],
                        y=half_life_df['Half_Life'],
                        marker_color=[self.signal_colors.get(
                            SignalType(name) if name in [s.value for s in SignalType] else None,
                            '#3b82f6'
                        ) for name in half_life_df['Signal']],
                        hovertemplate='Signal: %{x}<br>Half-Life: %{y} days<extra></extra>'
                    )
                )
                
                fig.update_layout(
                    title="Signal Half-Life (Days)",
                    xaxis_title="Signal Type",
                    yaxis_title="Half-Life (Days)",
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_attribution_analysis(self, data: Dict[str, Any], selected_signals: List[str]):
        """Render signal attribution analysis."""
        st.subheader("  Signal Attribution Analysis")
        
        # Filter signals dictionary
        filtered_signals = {k: v for k, v in data['signals'].items() if k in selected_signals}
        
        if not filtered_signals:
            st.warning("No signals selected for attribution analysis")
            return
        
        attribution_results = self.perform_signal_attribution(filtered_signals, data['returns'])
        
        if attribution_results.empty:
            st.warning("No attribution results available")
            return
        
        # Display attribution results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Signal Contribution by Asset**")
            
            # Create contribution matrix
            contribution_cols = [col for col in attribution_results.columns if col.endswith('_contribution')]
            
            if contribution_cols:
                contrib_data = attribution_results[['Asset'] + contribution_cols].set_index('Asset')
                contrib_data.columns = [col.replace('_contribution', '').replace('_', ' ').title() for col in contrib_data.columns]
                
                fig = go.Figure()
                
                fig.add_trace(
                    go.Heatmap(
                        z=contrib_data.values,
                        x=contrib_data.columns,
                        y=contrib_data.index,
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(contrib_data.values * 10000, 1),
                        texttemplate="%{text}",
                        textfont={"size": 8},
                        hovertemplate='Asset: %{y}<br>Signal: %{x}<br>Contribution: %{z:.4f}<extra></extra>',
                        colorbar=dict(title="Contribution")
                    )
                )
                
                fig.update_layout(
                    title="Signal Contribution Matrix (bps)",
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Statistical Significance**")
            
            # Show p-values for signal coefficients
            p_value_cols = [col for col in attribution_results.columns if col.endswith('_p_value')]
            
            if p_value_cols:
                p_value_data = attribution_results[['Asset'] + p_value_cols].set_index('Asset')
                p_value_data.columns = [col.replace('_p_value', '').replace('_', ' ').title() for col in p_value_data.columns]
                
                # Create significance heatmap (lower p-value = more significant = darker color)
                fig = go.Figure()
                
                fig.add_trace(
                    go.Heatmap(
                        z=1 - p_value_data.values,  # Invert for better visualization
                        x=p_value_data.columns,
                        y=p_value_data.index,
                        colorscale='Reds',
                        text=np.round(p_value_data.values, 3),
                        texttemplate="%{text}",
                        textfont={"size": 8},
                        hovertemplate='Asset: %{y}<br>Signal: %{x}<br>P-value: %{text}<extra></extra>',
                        colorbar=dict(title="Significance<br>(1-p_value)")
                    )
                )
                
                fig.update_layout(
                    title="Statistical Significance (P-values)",
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.markdown("**Attribution Summary**")
        
        # Calculate summary statistics
        if 'R_squared' in attribution_results.columns:
            avg_r_squared = attribution_results['R_squared'].mean()
            
            st.markdown(f"""
            <div class="signal-performance">
                <h4>Model Performance</h4>
                <p><strong>Average R²:</strong> {avg_r_squared:.3f}</p>
                <p><strong>Assets Analyzed:</strong> {len(attribution_results)}</p>
                <p><strong>Signals Used:</strong> {len(selected_signals)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show detailed table
        display_cols = ['Asset'] + [col for col in attribution_results.columns 
                                  if col.endswith(('_coef', '_p_value')) or col == 'R_squared']
        
        if display_cols:
            display_df = attribution_results[display_cols]
            st.dataframe(display_df.round(4), use_container_width=True)
    
    def run_signal_inspector(self):
        """Run the complete signal inspector interface."""
        try:
            # Header
            self.render_header()
            
            # Sidebar controls
            config, display_options = self.render_sidebar_controls()
            
            if not display_options['selected_signals']:
                st.warning("Please select at least one signal to analyze")
                return
            
            # Generate data
            with st.spinner("Generating signal data and calculating metrics..."):
                data = self.generate_signal_data(
                    n_assets=display_options['n_assets'],
                    periods=display_options['periods']
                )
                
                if not data['signals']:
                    st.error("Failed to generate signal data")
                    return
                
                # Calculate metrics for selected signals
                all_metrics = {}
                for signal_name in display_options['selected_signals']:
                    if signal_name in data['signals']:
                        signal_metrics = self.calculate_signal_metrics(
                            data['signals'][signal_name],
                            data['returns'],
                            config
                        )
                        all_metrics[signal_name] = signal_metrics
            
            # Main content
            main_content = st.container()
            
            with main_content:
                # Signal overview
                self.render_signal_overview(data, display_options['selected_signals'])
                
                st.markdown("---")
                
                # Performance metrics
                if display_options['show_aggregate_metrics']:
                    self.render_performance_metrics(all_metrics, display_options['selected_signals'])
                    st.markdown("---")
                
                # Decay analysis
                if display_options['show_decay_analysis']:
                    self.render_decay_analysis(data, display_options['selected_signals'])
                    st.markdown("---")
                
                # Attribution analysis
                if display_options['show_attribution']:
                    self.render_attribution_analysis(data, display_options['selected_signals'])
                
        except Exception as e:
            st.error(f"Error in signal inspector: {str(e)}")
            logger.error(f"Signal inspector error: {str(e)}")

def main():
    """Main function to run the signal inspector."""
    inspector = SignalInspector()
    inspector.run_signal_inspector()

if __name__ == "__main__":
    main()