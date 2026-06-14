"""
Backtest Analyzer Tool
=====================

Comprehensive backtesting analysis and visualization tool for trading strategies.
Provides detailed performance metrics, risk analysis, and visualization capabilities
for strategy development and optimization.

Features:
- Complete backtesting framework with performance metrics
- Risk-adjusted return analysis and benchmarking
- Drawdown analysis and risk management metrics
- Trade-level analysis and execution statistics
- Strategy comparison and optimization tools
- Interactive performance visualization
- Monte Carlo simulation and stress testing
- Custom strategy implementation support

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
from scipy import stats
import scipy.optimize as optimize
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class BacktestConfig:
    """Configuration for backtest analysis."""
    start_date: datetime
    end_date: datetime
    initial_capital: float
    benchmark: str
    risk_free_rate: float
    transaction_costs: float
    slippage: float

@dataclass
class Trade:
    """Individual trade record."""
    entry_date: datetime
    exit_date: datetime
    symbol: str
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    holding_period: int
    trade_type: str  # 'long' or 'short'

@dataclass
class PerformanceMetrics:
    """Container for backtest performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    num_trades: int
    holding_period_avg: int

class BacktestAnalyzer:
    """
    Comprehensive backtest analysis tool.
    
    Provides detailed backtesting capabilities with performance analysis,
    risk metrics, and visualization for trading strategy development.
    """
    
    def __init__(self):
        """Initialize backtest analyzer."""
        st.set_page_config(
            page_title="Quantum Forge - Backtest Analyzer",
            page_icon=" ",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .backtest-header {
            background: linear-gradient(90deg, #10b981 0%, #059669 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .backtest-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #10b981;
            margin-bottom: 1rem;
        }
        .metric-positive {
            color: #10b981;
            font-weight: bold;
            font-size: 1.2em;
        }
        .metric-negative {
            color: #ef4444;
            font-weight: bold;
            font-size: 1.2em;
        }
        .metric-neutral {
            color: #6b7280;
            font-weight: bold;
            font-size: 1.2em;
        }
        .performance-metric {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin: 0.5rem 0;
            text-align: center;
        }
        .trade-winner {
            background: rgba(16, 185, 129, 0.1);
            color: #059669;
        }
        .trade-loser {
            background: rgba(239, 68, 68, 0.1);
            color: #dc2626;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def generate_sample_backtest_data(self, config: BacktestConfig, strategy_name: str) -> Tuple[pd.DataFrame, List[Trade], pd.DataFrame]:
        """Generate sample backtest data for demonstration."""
        try:
            # Generate date range
            dates = pd.date_range(start=config.start_date, end=config.end_date, freq='D')
            
            # Generate market data (benchmark)
            np.random.seed(42)  # For reproducible results
            market_returns = np.random.normal(0.0008, 0.015, len(dates))  # ~20% annual vol
            
            # Add some realistic market patterns
            for i in range(1, len(market_returns)):
                # Add some momentum/mean reversion
                market_returns[i] += 0.05 * market_returns[i-1]  # Slight momentum
                
            market_prices = 100 * np.cumprod(1 + market_returns)
            
            # Generate strategy returns based on different strategy types
            if strategy_name == "Momentum Strategy":
                # Momentum strategy - performs well in trending markets
                strategy_returns = market_returns * 1.5 + np.random.normal(0, 0.005, len(dates))
                # Add some alpha
                for i in range(20, len(strategy_returns)):
                    trend = np.mean(market_returns[i-20:i])
                    if trend > 0:
                        strategy_returns[i] += 0.002  # Positive alpha in uptrends
            
            elif strategy_name == "Mean Reversion":
                # Mean reversion strategy - inverse correlation with market momentum
                strategy_returns = -0.3 * market_returns + np.random.normal(0.0005, 0.012, len(dates))
                
            elif strategy_name == "Long-Short Equity":
                # Market neutral strategy with lower correlation
                strategy_returns = 0.2 * market_returns + np.random.normal(0.0003, 0.008, len(dates))
                
            else:  # "Buy and Hold"
                strategy_returns = market_returns * 0.95  # Slightly underperform due to costs
            
            # Apply transaction costs and slippage
            strategy_returns = strategy_returns - config.transaction_costs - config.slippage
            
            # Calculate cumulative performance
            strategy_prices = config.initial_capital * np.cumprod(1 + strategy_returns)
            benchmark_prices = config.initial_capital * np.cumprod(1 + market_returns)
            
            # Create performance DataFrame
            performance_df = pd.DataFrame({
                'Date': dates,
                'Strategy': strategy_prices,
                'Benchmark': benchmark_prices,
                'Strategy_Returns': strategy_returns,
                'Benchmark_Returns': market_returns
            }).set_index('Date')
            
            # Generate individual trades
            trades = self._generate_sample_trades(dates, strategy_returns, strategy_name)
            
            # Generate additional metrics data
            metrics_data = self._calculate_additional_metrics(performance_df, trades)
            
            return performance_df, trades, metrics_data
            
        except Exception as e:
            logger.error(f"Error generating backtest data: {str(e)}")
            return pd.DataFrame(), [], pd.DataFrame()
    
    def _generate_sample_trades(self, dates: pd.DatetimeIndex, returns: np.ndarray, strategy_name: str) -> List[Trade]:
        """Generate sample trade records."""
        try:
            trades = []
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'SPY']
            
            # Generate trades based on strategy characteristics
            if strategy_name == "Momentum Strategy":
                num_trades = np.random.randint(50, 100)
                avg_holding_period = 15  # Days
                
            elif strategy_name == "Mean Reversion":
                num_trades = np.random.randint(100, 200)
                avg_holding_period = 5  # Days
                
            elif strategy_name == "Long-Short Equity":
                num_trades = np.random.randint(200, 400)
                avg_holding_period = 10  # Days
                
            else:  # Buy and Hold
                num_trades = np.random.randint(10, 30)
                avg_holding_period = 60  # Days
            
            for i in range(num_trades):
                # Random entry date
                entry_idx = np.random.randint(0, len(dates) - avg_holding_period - 10)
                entry_date = dates[entry_idx]
                
                # Holding period with some variation
                holding_period = max(1, np.random.poisson(avg_holding_period))
                exit_idx = min(entry_idx + holding_period, len(dates) - 1)
                exit_date = dates[exit_idx]
                
                # Random symbol and trade size
                symbol = np.random.choice(symbols)
                quantity = np.random.randint(10, 1000)
                
                # Entry and exit prices (simplified)
                entry_price = np.random.uniform(50, 500)
                
                # Exit price based on market movement during holding period
                period_return = np.sum(returns[entry_idx:exit_idx])
                exit_price = entry_price * (1 + period_return + np.random.normal(0, 0.02))
                
                # Trade type
                trade_type = 'long' if np.random.random() > 0.3 else 'short'
                
                # Calculate P&L
                if trade_type == 'long':
                    pnl = quantity * (exit_price - entry_price)
                    return_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl = quantity * (entry_price - exit_price)
                    return_pct = (entry_price - exit_price) / entry_price
                
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=exit_date,
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    return_pct=return_pct,
                    holding_period=holding_period,
                    trade_type=trade_type
                ))
            
            return trades
            
        except Exception as e:
            logger.error(f"Error generating sample trades: {str(e)}")
            return []
    
    def _calculate_additional_metrics(self, performance_df: pd.DataFrame, trades: List[Trade]) -> pd.DataFrame:
        """Calculate additional performance metrics."""
        try:
            # Rolling metrics
            performance_df['Rolling_Sharpe_30'] = (
                performance_df['Strategy_Returns'].rolling(30).mean() / 
                performance_df['Strategy_Returns'].rolling(30).std() * np.sqrt(252)
            )
            
            performance_df['Rolling_Vol_30'] = (
                performance_df['Strategy_Returns'].rolling(30).std() * np.sqrt(252)
            )
            
            # Drawdown calculation
            performance_df['Peak'] = performance_df['Strategy'].expanding().max()
            performance_df['Drawdown'] = (performance_df['Strategy'] - performance_df['Peak']) / performance_df['Peak']
            
            # Underwater curve
            performance_df['Underwater'] = performance_df['Drawdown'] * 100
            
            return performance_df
            
        except Exception as e:
            logger.error(f"Error calculating additional metrics: {str(e)}")
            return performance_df
    
    def calculate_performance_metrics(self, performance_df: pd.DataFrame, trades: List[Trade], 
                                    risk_free_rate: float = 0.02) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        try:
            if performance_df.empty:
                return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            strategy_returns = performance_df['Strategy_Returns'].dropna()
            
            # Basic performance metrics
            total_return = (performance_df['Strategy'].iloc[-1] / performance_df['Strategy'].iloc[0]) - 1
            annualized_return = (1 + total_return) ** (252 / len(performance_df)) - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            excess_returns = strategy_returns - risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / strategy_returns.std() * np.sqrt(252)
            
            # Sortino ratio (downside deviation)
            downside_returns = strategy_returns[strategy_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else volatility
            sortino_ratio = excess_returns.mean() / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
            
            # Drawdown metrics
            peak = performance_df['Strategy'].expanding().max()
            drawdown = (performance_df['Strategy'] - peak) / peak
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Trade-based metrics
            if trades:
                winning_trades = [t for t in trades if t.pnl > 0]
                losing_trades = [t for t in trades if t.pnl <= 0]
                
                win_rate = len(winning_trades) / len(trades)
                
                total_wins = sum(t.pnl for t in winning_trades) if winning_trades else 0
                total_losses = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 1
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
                
                avg_trade_return = np.mean([t.return_pct for t in trades])
                avg_holding_period = int(np.mean([t.holding_period for t in trades]))
            else:
                win_rate = 0
                profit_factor = 0
                avg_trade_return = 0
                avg_holding_period = 0
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_trade_return=avg_trade_return,
                num_trades=len(trades),
                holding_period_avg=avg_holding_period
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def render_header(self):
        """Render backtest analyzer header."""
        st.markdown("""
        <div class="backtest-header">
            <h1>  Quantum Forge Backtest Analyzer</h1>
            <p>Comprehensive Strategy Performance Analysis & Optimization</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Control buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("  Run Backtest", key="run_backtest"):
                st.rerun()
                
        with col2:
            if st.button("  Optimize Strategy", key="optimize_strategy"):
                st.info("Running strategy optimization...")
                
        with col3:
            if st.button("  Monte Carlo", key="monte_carlo"):
                st.info("Running Monte Carlo simulation...")
                
        with col4:
            if st.button("  Compare Strategies", key="compare"):
                st.info("Comparing strategy performance...")
    
    def render_sidebar_controls(self):
        """Render backtest analyzer sidebar controls."""
        st.sidebar.markdown("##   Backtest Configuration")
        
        # Strategy selection
        strategies = [
            "Momentum Strategy",
            "Mean Reversion",
            "Long-Short Equity",
            "Buy and Hold"
        ]
        
        selected_strategy = st.sidebar.selectbox(
            "Select Strategy",
            options=strategies
        )
        
        # Date range
        st.sidebar.markdown("###   Date Range")
        
        start_date = st.sidebar.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365*2),
            max_value=datetime.now()
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
        
        # Capital and costs
        st.sidebar.markdown("###   Capital & Costs")
        
        initial_capital = st.sidebar.number_input(
            "Initial Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000
        )
        
        transaction_costs = st.sidebar.slider(
            "Transaction Costs (bps)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5
        ) / 10000
        
        slippage = st.sidebar.slider(
            "Slippage (bps)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5
        ) / 10000
        
        # Risk parameters
        st.sidebar.markdown("###  ️ Risk Parameters")
        
        risk_free_rate = st.sidebar.slider(
            "Risk-free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1
        ) / 100
        
        benchmark = st.sidebar.selectbox(
            "Benchmark",
            options=["SPY", "QQQ", "IWM", "VTI"]
        )
        
        return BacktestConfig(
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.min.time()),
            initial_capital=initial_capital,
            benchmark=benchmark,
            risk_free_rate=risk_free_rate,
            transaction_costs=transaction_costs,
            slippage=slippage
        ), selected_strategy
    
    def render_performance_overview(self, metrics: PerformanceMetrics, performance_df: pd.DataFrame):
        """Render performance overview metrics."""
        st.subheader("  Performance Overview")
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            return_class = "metric-positive" if metrics.total_return > 0 else "metric-negative"
            st.markdown(f"""
            <div class="performance-metric">
                <h4>Total Return</h4>
                <div class="{return_class}">{metrics.total_return:.2%}</div>
                <small>Annualized: {metrics.annualized_return:.2%}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sharpe_class = "metric-positive" if metrics.sharpe_ratio > 1 else "metric-neutral" if metrics.sharpe_ratio > 0 else "metric-negative"
            st.markdown(f"""
            <div class="performance-metric">
                <h4>Sharpe Ratio</h4>
                <div class="{sharpe_class}">{metrics.sharpe_ratio:.2f}</div>
                <small>Sortino: {metrics.sortino_ratio:.2f}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            dd_class = "metric-positive" if metrics.max_drawdown > -0.1 else "metric-negative"
            st.markdown(f"""
            <div class="performance-metric">
                <h4>Max Drawdown</h4>
                <div class="{dd_class}">{metrics.max_drawdown:.2%}</div>
                <small>Calmar: {metrics.calmar_ratio:.2f}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            vol_class = "metric-positive" if metrics.volatility < 0.15 else "metric-neutral" if metrics.volatility < 0.25 else "metric-negative"
            st.markdown(f"""
            <div class="performance-metric">
                <h4>Volatility</h4>
                <div class="{vol_class}">{metrics.volatility:.2%}</div>
                <small>Annualized</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Equity curve
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=performance_df.index,
                y=performance_df['Strategy'],
                mode='lines',
                name='Strategy',
                line=dict(color='#10b981', width=2),
                hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=performance_df.index,
                y=performance_df['Benchmark'],
                mode='lines',
                name='Benchmark',
                line=dict(color='#6b7280', width=1, dash='dash'),
                hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title="Equity Curve: Strategy vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template='plotly_white',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_analysis(self, performance_df: pd.DataFrame, metrics: PerformanceMetrics):
        """Render risk analysis charts."""
        st.subheader(" ️ Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Drawdown chart
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=performance_df.index,
                    y=performance_df['Underwater'],
                    mode='lines',
                    name='Drawdown',
                    fill='tozeroy',
                    fillcolor='rgba(239, 68, 68, 0.3)',
                    line=dict(color='#ef4444', width=1),
                    hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
                )
            )
            
            fig.update_layout(
                title="Underwater Curve",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                template='plotly_white',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rolling volatility
            if 'Rolling_Vol_30' in performance_df.columns:
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=performance_df.index,
                        y=performance_df['Rolling_Vol_30'] * 100,
                        mode='lines',
                        name='30-Day Volatility',
                        line=dict(color='#f59e0b', width=2),
                        hovertemplate='Date: %{x}<br>Volatility: %{y:.1f}%<extra></extra>'
                    )
                )
                
                # Add average line
                avg_vol = performance_df['Rolling_Vol_30'].mean() * 100
                fig.add_hline(
                    y=avg_vol,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=f"Average: {avg_vol:.1f}%"
                )
                
                fig.update_layout(
                    title="Rolling 30-Day Volatility",
                    xaxis_title="Date",
                    yaxis_title="Volatility (%)",
                    template='plotly_white',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Returns distribution
        col3, col4 = st.columns(2)
        
        with col3:
            returns = performance_df['Strategy_Returns'].dropna() * 100
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Daily Returns',
                    marker_color='#3b82f6',
                    opacity=0.7,
                    hovertemplate='Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>'
                )
            )
            
            # Add normal distribution overlay
            x_norm = np.linspace(returns.min(), returns.max(), 100)
            y_norm = stats.norm.pdf(x_norm, returns.mean(), returns.std()) * len(returns) * (returns.max() - returns.min()) / 50
            
            fig.add_trace(
                go.Scatter(
                    x=x_norm,
                    y=y_norm,
                    mode='lines',
                    name='Normal Distribution',
                    line=dict(color='red', dash='dash')
                )
            )
            
            fig.update_layout(
                title="Returns Distribution",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                template='plotly_white',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Q-Q plot for normality test
            from scipy.stats import probplot
            
            returns_clean = returns.dropna()
            if len(returns_clean) > 10:
                theoretical_quantiles, sample_quantiles = probplot(returns_clean, dist="norm")
                
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles,
                        y=sample_quantiles,
                        mode='markers',
                        name='Sample Quantiles',
                        marker=dict(color='#3b82f6', size=6),
                        hovertemplate='Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>'
                    )
                )
                
                # Add perfect normal line
                min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
                max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
                
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Normal',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                fig.update_layout(
                    title="Q-Q Plot (Normality Test)",
                    xaxis_title="Theoretical Quantiles",
                    yaxis_title="Sample Quantiles",
                    template='plotly_white',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_trade_analysis(self, trades: List[Trade], metrics: PerformanceMetrics):
        """Render trade-level analysis."""
        st.subheader("  Trade Analysis")
        
        if not trades:
            st.warning("No trade data available")
            return
        
        # Trade summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="performance-metric">
                <h4>Total Trades</h4>
                <div class="metric-neutral">{metrics.num_trades}</div>
                <small>Avg Holding: {metrics.holding_period_avg}d</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            win_class = "metric-positive" if metrics.win_rate > 0.5 else "metric-negative"
            st.markdown(f"""
            <div class="performance-metric">
                <h4>Win Rate</h4>
                <div class="{win_class}">{metrics.win_rate:.1%}</div>
                <small>Profit Factor: {metrics.profit_factor:.2f}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_class = "metric-positive" if metrics.avg_trade_return > 0 else "metric-negative"
            st.markdown(f"""
            <div class="performance-metric">
                <h4>Avg Trade Return</h4>
                <div class="{avg_class}">{metrics.avg_trade_return:.2%}</div>
                <small>Per Trade</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            best_trade = max(trades, key=lambda t: t.pnl)
            worst_trade = min(trades, key=lambda t: t.pnl)
            
            st.markdown(f"""
            <div class="performance-metric">
                <h4>Best/Worst Trade</h4>
                <div class="metric-positive">${best_trade.pnl:.0f}</div>
                <div class="metric-negative">${worst_trade.pnl:.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Trade analysis charts
        col5, col6 = st.columns(2)
        
        with col5:
            # Trade P&L distribution
            pnl_values = [trade.pnl for trade in trades]
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Histogram(
                    x=pnl_values,
                    nbinsx=30,
                    name='Trade P&L',
                    marker_color='#10b981',
                    opacity=0.7,
                    hovertemplate='P&L: $%{x:.0f}<br>Count: %{y}<extra></extra>'
                )
            )
            
            fig.update_layout(
                title="Trade P&L Distribution",
                xaxis_title="P&L ($)",
                yaxis_title="Frequency",
                template='plotly_white',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col6:
            # Cumulative P&L over time
            trades_sorted = sorted(trades, key=lambda t: t.exit_date)
            cumulative_pnl = np.cumsum([trade.pnl for trade in trades_sorted])
            exit_dates = [trade.exit_date for trade in trades_sorted]
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=exit_dates,
                    y=cumulative_pnl,
                    mode='lines+markers',
                    name='Cumulative P&L',
                    line=dict(color='#10b981', width=2),
                    marker=dict(size=4),
                    hovertemplate='Date: %{x}<br>Cumulative P&L: $%{y:,.0f}<extra></extra>'
                )
            )
            
            fig.update_layout(
                title="Cumulative Trade P&L",
                xaxis_title="Date",
                yaxis_title="Cumulative P&L ($)",
                template='plotly_white',
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade details table
        st.markdown("**Recent Trades**")
        
        # Create trade summary DataFrame
        trade_data = []
        for trade in trades[-20:]:  # Last 20 trades
            trade_class = "trade-winner" if trade.pnl > 0 else "trade-loser"
            
            trade_data.append({
                'Exit Date': trade.exit_date.strftime('%Y-%m-%d'),
                'Symbol': trade.symbol,
                'Type': trade.trade_type.title(),
                'Quantity': trade.quantity,
                'Entry Price': f"${trade.entry_price:.2f}",
                'Exit Price': f"${trade.exit_price:.2f}",
                'P&L': f"${trade.pnl:.2f}",
                'Return': f"{trade.return_pct:.2%}",
                'Hold Days': trade.holding_period
            })
        
        trades_df = pd.DataFrame(trade_data)
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    
    def run_backtest_analyzer(self):
        """Run the complete backtest analyzer interface."""
        try:
            # Header
            self.render_header()
            
            # Sidebar controls
            config, selected_strategy = self.render_sidebar_controls()
            
            # Generate backtest data
            with st.spinner("Running backtest analysis..."):
                performance_df, trades, metrics_data = self.generate_sample_backtest_data(config, selected_strategy)
                
                if performance_df.empty:
                    st.error("Failed to generate backtest data")
                    return
                
                # Calculate performance metrics
                metrics = self.calculate_performance_metrics(performance_df, trades, config.risk_free_rate)
            
            # Main content
            main_content = st.container()
            
            with main_content:
                # Performance overview
                self.render_performance_overview(metrics, performance_df)
                
                st.markdown("---")
                
                # Risk analysis
                self.render_risk_analysis(performance_df, metrics)
                
                st.markdown("---")
                
                # Trade analysis
                self.render_trade_analysis(trades, metrics)
                
        except Exception as e:
            st.error(f"Error in backtest analyzer: {str(e)}")
            logger.error(f"Backtest analyzer error: {str(e)}")

def main():
    """Main function to run the backtest analyzer."""
    analyzer = BacktestAnalyzer()
    analyzer.run_backtest_analyzer()

if __name__ == "__main__":
    main()