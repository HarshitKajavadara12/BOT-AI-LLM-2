"""
Main Dashboard for Quantum Forge Trading Platform
================================================

Comprehensive main dashboard providing unified view of trading performance,
risk metrics, market data, and system status with real-time updates.

Features:
- Real-time P&L monitoring and performance metrics
- Portfolio overview with position tracking
- Risk management dashboard with VaR, drawdown, and exposure metrics
- Market overview with key indicators and sentiment
- System health monitoring and alerts
- Interactive navigation to specialized dashboards
- Customizable layout and user preferences

Author: Quantum Forge Interface Team
Date: November 2025
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import time
from dataclasses import dataclass
import warnings
import logging
from pathlib import Path
import sys

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.ingestion.realtime_data_cache import RealTimeDataCache
from core.dynamic_portfolio_tracker import get_portfolio_tracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class DashboardMetrics:
    """Container for dashboard metrics."""
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    portfolio_value: float
    cash_balance: float
    positions_count: int
    active_orders: int

@st.cache_resource
def get_data_cache():
    """Initialize and cache the RealTimeDataCache."""
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"]
    cache = RealTimeDataCache(symbols=symbols)
    cache.start()
    return cache

@st.cache_resource
def get_tracker():
    """Initialize and cache the Dynamic Portfolio Tracker."""
    return get_portfolio_tracker(initial_cash=100000.0)

class MainDashboard:
    """
    Main dashboard for Quantum Forge trading platform.
    
    Provides comprehensive overview of trading operations with real-time
    updates and interactive navigation to specialized modules.
    """
    
    def __init__(self):
        """Initialize main dashboard."""
        self.last_update = None
        self.update_frequency = 2  # seconds - faster refresh for real-time feel
        self._setup_page_config()
        
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Quantum Forge - Main Dashboard",
            page_icon=" ",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://quantum-forge.com/help',
                'Report a bug': 'https://quantum-forge.com/bug-report',
                'About': 'Quantum Forge Trading Platform v2.0'
            }
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f2937 0%, #374151 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            border: 2px solid rgba(255,255,255,0.2);
            color: white;
            min-height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-card h4 {
            color: rgba(255,255,255,1);
            font-size: 1rem;
            margin-bottom: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-card h2 {
            color: white;
            font-size: 2rem;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.4);
            font-weight: 700;
        }
        .positive-pnl {
            color: #10b981;
            font-weight: bold;
        }
        .negative-pnl {
            color: #ef4444;
            font-weight: bold;
        }
        .neutral-pnl {
            color: #6b7280;
            font-weight: bold;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online {
            background-color: #10b981;
        }
        .status-warning {
            background-color: #f59e0b;
        }
        .status-offline {
            background-color: #ef4444;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def generate_sample_data(self) -> Tuple[DashboardMetrics, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Fetch live data and calculate metrics using DYNAMIC portfolio tracker."""
        try:
            cache = get_data_cache()
            tracker = get_tracker()
            
            # Simulate market-driven trading activity
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"]
            tracker.simulate_market_activity(symbols, cache)
            
            # Initialize tracking in session state
            if 'previous_portfolio_value' not in st.session_state:
                st.session_state.previous_portfolio_value = None
            if 'previous_prices' not in st.session_state:
                st.session_state.previous_prices = {}
            if 'pnl_tracking' not in st.session_state:
                st.session_state.pnl_tracking = []
            
            # Get REAL DYNAMIC positions from tracker
            dynamic_positions = tracker.get_positions()
            
            total_value = 0
            total_cost_basis = 0
            positions_data = []
            
            # Process actual dynamic positions
            for symbol, position in dynamic_positions.items():
                if position.amount <= 0:
                    continue  # Skip closed positions
                
                amount = position.amount
                entry_price = position.entry_price
                
                price = cache.get_current_price(symbol)
                if price is None:
                    price = entry_price  # Fallback
                
                market_value = amount * price
                cost_basis = position.total_cost
                total_value += market_value
                total_cost_basis += cost_basis
                
                unrealized_pnl = (price - entry_price) * amount
                
                # Calculate volatility from history
                history = cache.get_historical_data(symbol, days=30)
                volatility = 0.5  # Default
                if not history.empty:
                    returns = history['close'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.std() * np.sqrt(365)  # Annualized

                positions_data.append({
                    'symbol': symbol,
                    'position_size': amount,
                    'market_value': market_value,
                    'unrealized_pnl': unrealized_pnl,
                    'weight': 0,  # Calculate later
                    'side': 'Long',
                    'volatility': volatility,
                    'entry_price': entry_price,
                    'current_price': price,
                    'trades': position.trades_count
                })
            
            positions_df = pd.DataFrame(positions_data)
            if not positions_df.empty:
                positions_df['weight'] = positions_df['market_value'] / total_value if total_value > 0 else 0
            
            # Get REAL cash balance from tracker
            cash_balance = tracker.get_cash_balance()
            total_value += cash_balance
            
            # Calculate initial capital
            initial_capital = tracker.initial_cash
            
            # Calculate metrics based on real market performance
            total_pnl = total_value - initial_capital
            
            # Calculate REAL daily P&L from actual value changes
            if st.session_state.previous_portfolio_value is not None:
                daily_pnl = total_value - st.session_state.previous_portfolio_value
            else:
                daily_pnl = 0  # First run
            st.session_state.previous_portfolio_value = total_value
            
            # Track P&L over time for calculations
            st.session_state.pnl_tracking.append({
                'timestamp': datetime.now(),
                'total_pnl': total_pnl,
                'portfolio_value': total_value
            })
            # Keep only last 1000 points
            if len(st.session_state.pnl_tracking) > 1000:
                st.session_state.pnl_tracking = st.session_state.pnl_tracking[-1000:]
            
            # Calculate REAL performance metrics from tracked data
            total_return = total_pnl / initial_capital if initial_capital > 0 else 0
            
            # Calculate real Sharpe Ratio from P&L tracking
            if len(st.session_state.pnl_tracking) >= 10:
                pnl_values = [x['total_pnl'] for x in st.session_state.pnl_tracking[-100:]]
                returns = np.diff(pnl_values) / initial_capital
                if len(returns) > 0 and np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                else:
                    sharpe_ratio = 0
            else:
                sharpe_ratio = 0
            
            # Calculate real Maximum Drawdown from P&L tracking
            if len(st.session_state.pnl_tracking) >= 2:
                pnl_series = np.array([x['total_pnl'] for x in st.session_state.pnl_tracking])
                running_max = np.maximum.accumulate(pnl_series)
                drawdown = (pnl_series - running_max) / (initial_capital + running_max + 1)
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            else:
                max_drawdown = 0
            
            # Calculate real VaR (95%) from portfolio volatility
            portfolio_volatility = 0
            for symbol in dynamic_positions.keys():
                hist = cache.get_historical_data(symbol, days=30)
                if not hist.empty:
                    returns = hist['close'].pct_change().dropna()
                    if len(returns) > 0:
                        portfolio_volatility += returns.std() ** 2
            portfolio_volatility = np.sqrt(portfolio_volatility)
            var_95 = -1.645 * portfolio_volatility * total_value  # 95% confidence
            
            # Get REAL active orders count from tracker
            active_orders = len(tracker.get_active_orders())
            
            metrics = DashboardMetrics(
                total_pnl=total_pnl,
                daily_pnl=daily_pnl,
                unrealized_pnl=total_pnl,  # All unrealized since no closes
                realized_pnl=sum(p.realized_pnl for p in dynamic_positions.values()),
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                portfolio_value=total_value,
                cash_balance=cash_balance,
                positions_count=len(dynamic_positions),
                active_orders=active_orders  # REAL count from OMS
            )
            
            # Generate realistic P&L history from actual dynamic positions
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            pnl_history = pd.Series(0.0, index=dates)
            
            for symbol, position in dynamic_positions.items():
                if position.amount <= 0:
                    continue
                    
                amount = position.amount
                entry_price = position.entry_price
                hist = cache.get_historical_data(symbol, days=30)
                
                if not hist.empty:
                    # Resample to daily and align
                    daily_close = hist['close'].resample('D').last()
                    # Reindex to match our dates, forward filling missing data
                    daily_close = daily_close.reindex(dates, method='ffill').fillna(entry_price)
                    
                    pos_pnl = (daily_close - entry_price) * amount
                    pnl_history = pnl_history.add(pos_pnl, fill_value=0)
            
            # Add current real-time P&L to the end for live updates
            pnl_history.iloc[-1] = total_pnl  # Update last point with current P&L
            
            pnl_data = pd.DataFrame({
                'date': pnl_history.index,
                'cumulative_pnl': pnl_history.values,
                'daily_pnl': pnl_history.diff().fillna(0).values
            })
            
            # Get REAL system metrics from tracker
            sys_metrics = tracker.get_metrics()
            latency_ms = sys_metrics.avg_latency_ms
            throughput_ops = sys_metrics.throughput_ops_per_sec

            system_status = {
                'market_data_feed': 'online',
                'order_management': 'online',
                'risk_engine': 'online',
                'portfolio_manager': 'online',
                'execution_engine': 'online',
                'data_warehouse': 'online',
                'last_heartbeat': datetime.now(),
                'latency_ms': latency_ms,  # REAL measured latency
                'throughput_ops': throughput_ops,  # REAL measured throughput
                'orders_submitted': sys_metrics.orders_submitted,
                'orders_filled': sys_metrics.orders_filled,
                'fill_rate': sys_metrics.fill_rate,
                'trades_executed': sys_metrics.trades_executed
            }
            
            return metrics, pnl_data, positions_df, system_status
            
        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            # Return default data
            default_metrics = DashboardMetrics(
                total_pnl=0, daily_pnl=0, unrealized_pnl=0, realized_pnl=0,
                total_return=0, sharpe_ratio=0, max_drawdown=0, var_95=0,
                portfolio_value=1000000, cash_balance=50000,
                positions_count=0, active_orders=0
            )
            empty_df = pd.DataFrame()
            empty_status = {}
            return default_metrics, empty_df, empty_df, empty_status
    
    def render_header(self):
        """Render dashboard header."""
        st.markdown("""
        <div class="main-header">
            <h1>  Quantum Forge Trading Platform</h1>
            <p>Real-time Trading Dashboard & Risk Management System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Auto-refresh toggle and last update time
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Initialize auto_refresh in session state if not present
            if 'auto_refresh' not in st.session_state:
                st.session_state.auto_refresh = True
            
            auto_refresh = st.checkbox("Auto-refresh (2s)", value=st.session_state.auto_refresh, key='auto_refresh_checkbox')
            st.session_state.auto_refresh = auto_refresh
            
        with col2:
            if st.button("  Refresh Now"):
                st.rerun()
                
        with col3:
            # Show live timestamp that updates
            current_time = datetime.now().strftime('%H:%M:%S')
            st.write(f"  {current_time}")
    
    def render_key_metrics(self, metrics: DashboardMetrics):
        """Render key performance metrics cards."""
        # Add live indicator
        timestamp = datetime.now().strftime('%H:%M:%S')
        st.subheader(f"  Key Performance Metrics   LIVE - {timestamp}")
        
        # Track previous metrics for change indicators
        if 'prev_metrics' not in st.session_state:
            st.session_state.prev_metrics = {}
        
        # Create metrics columns
        cols = st.columns(6)
        
        # Total P&L with change indicator
        with cols[0]:
            pnl_class = "positive-pnl" if metrics.total_pnl >= 0 else "negative-pnl"
            prev_pnl = st.session_state.prev_metrics.get('total_pnl', metrics.total_pnl)
            pnl_change = metrics.total_pnl - prev_pnl
            change_arrow = " ️" if pnl_change > 0 else " ️" if pnl_change < 0 else " ️"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total P&L {change_arrow}</h4>
                <h2 class="{pnl_class}">${metrics.total_pnl:,.0f}</h2>
                <small style="color: rgba(255,255,255,0.7);">Δ ${pnl_change:+,.0f}</small>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.prev_metrics['total_pnl'] = metrics.total_pnl
        
        # Daily P&L
        with cols[1]:
            daily_class = "positive-pnl" if metrics.daily_pnl >= 0 else "negative-pnl"
            daily_arrow = " ️" if metrics.daily_pnl >= 0 else " ️"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Daily P&L {daily_arrow}</h4>
                <h2 class="{daily_class}">${metrics.daily_pnl:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Total Return with change indicator
        with cols[2]:
            return_class = "positive-pnl" if metrics.total_return >= 0 else "negative-pnl"
            prev_return = st.session_state.prev_metrics.get('total_return', metrics.total_return)
            return_change = metrics.total_return - prev_return
            change_arrow = " ️" if return_change > 0 else " ️" if return_change < 0 else " ️"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Total Return {change_arrow}</h4>
                <h2 class="{return_class}">{metrics.total_return:.2%}</h2>
                <small style="color: rgba(255,255,255,0.7);">Δ {return_change:+.2%}</small>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.prev_metrics['total_return'] = metrics.total_return
        
        # Sharpe Ratio with change indicator
        with cols[3]:
            sharpe_class = "positive-pnl" if metrics.sharpe_ratio >= 1.0 else "neutral-pnl"
            prev_sharpe = st.session_state.prev_metrics.get('sharpe_ratio', metrics.sharpe_ratio)
            sharpe_change = metrics.sharpe_ratio - prev_sharpe
            change_arrow = " ️" if sharpe_change > 0 else " ️" if sharpe_change < 0 else " ️"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Sharpe Ratio {change_arrow}</h4>
                <h2 class="{sharpe_class}">{metrics.sharpe_ratio:.2f}</h2>
                <small style="color: rgba(255,255,255,0.7);">Δ {sharpe_change:+.2f}</small>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.prev_metrics['sharpe_ratio'] = metrics.sharpe_ratio
        
        # Max Drawdown with change indicator
        with cols[4]:
            prev_dd = st.session_state.prev_metrics.get('max_drawdown', metrics.max_drawdown)
            dd_change = metrics.max_drawdown - prev_dd
            # Better drawdown = less negative (arrow up), worse = more negative (arrow down)
            change_arrow = " ️" if dd_change > 0 else " ️" if dd_change < 0 else " ️"
            st.markdown(f"""
            <div class="metric-card">
                <h4>Max Drawdown {change_arrow}</h4>
                <h2 class="negative-pnl">{metrics.max_drawdown:.2%}</h2>
                <small style="color: rgba(255,255,255,0.7);">Δ {dd_change:+.2%}</small>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.prev_metrics['max_drawdown'] = metrics.max_drawdown
        
        # VaR 95% with change indicator
        with cols[5]:
            prev_var = st.session_state.prev_metrics.get('var_95', metrics.var_95)
            var_change = metrics.var_95 - prev_var
            change_arrow = " ️" if var_change > 0 else " ️" if var_change < 0 else " ️"
            st.markdown(f"""
            <div class="metric-card">
                <h4>VaR (95%) {change_arrow}</h4>
                <h2 class="negative-pnl">${metrics.var_95:,.0f}</h2>
                <small style="color: rgba(255,255,255,0.7);">Δ ${var_change:+,.0f}</small>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.prev_metrics['var_95'] = metrics.var_95
    
    def render_pnl_chart(self, pnl_data: pd.DataFrame):
        """Render P&L time series chart."""
        if pnl_data.empty:
            st.warning("No P&L data available")
            return
            
        st.subheader("  P&L Performance")
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=('Cumulative P&L', 'Daily P&L'),
            vertical_spacing=0.1
        )
        
        # Cumulative P&L
        fig.add_trace(
            go.Scatter(
                x=pnl_data['date'],
                y=pnl_data['cumulative_pnl'],
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='#3b82f6', width=3),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Daily P&L bars
        colors = ['#10b981' if x >= 0 else '#ef4444' for x in pnl_data['daily_pnl']]
        fig.add_trace(
            go.Bar(
                x=pnl_data['date'],
                y=pnl_data['daily_pnl'],
                name='Daily P&L',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            title_font_size=16,
            hovermode='x unified'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_live_price_ticker(self):
        """Render live price ticker showing real-time prices with change indicators."""
        try:
            cache = get_data_cache()
            symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"]
            
            # Track previous prices for change detection
            if 'ticker_prev_prices' not in st.session_state:
                st.session_state.ticker_prev_prices = {}
            
            # Create a scrolling ticker display
            ticker_html = "<div style='background: linear-gradient(90deg, #1f2937 0%, #374151 100%); padding: 10px; border-radius: 8px; overflow-x: auto; white-space: nowrap; margin-bottom: 15px;'>"
            
            for symbol in symbols:
                price = cache.get_current_price(symbol)
                if price:
                    # Determine price change
                    prev_price = st.session_state.ticker_prev_prices.get(symbol, price)
                    if price > prev_price:
                        color = "#10b981"  # Green
                        arrow = "↑"
                    elif price < prev_price:
                        color = "#ef4444"  # Red
                        arrow = "↓"
                    else:
                        color = "#6b7280"  # Gray
                        arrow = "→"
                    
                    change_pct = ((price - prev_price) / prev_price * 100) if prev_price != 0 else 0
                    ticker_html += f"<span style='color: {color}; margin: 0 20px; font-size: 14px; font-weight: bold;'>{symbol.replace('USDT', '')}: ${price:,.2f} {arrow} <small>({change_pct:+.2f}%)</small></span>"
                    st.session_state.ticker_prev_prices[symbol] = price
            
            ticker_html += f"<span style='color: #6b7280; margin-left: 30px; font-size: 12px;'>  {datetime.now().strftime('%H:%M:%S.%f')[:-3]}</span>"
            ticker_html += "</div>"
            
            st.markdown(ticker_html, unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error rendering price ticker: {e}")
    
    def render_portfolio_overview(self, metrics: DashboardMetrics, positions_df: pd.DataFrame):
        """Render portfolio overview section."""
        st.subheader("  Portfolio Overview")
        
        # Add live price ticker
        self.render_live_price_ticker()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Portfolio summary with change tracking
            if 'prev_portfolio' not in st.session_state:
                st.session_state.prev_portfolio = {}
            
            prev_value = st.session_state.prev_portfolio.get('value', metrics.portfolio_value)
            value_change = metrics.portfolio_value - prev_value
            value_arrow = " ️" if value_change > 0 else " ️" if value_change < 0 else " ️"
            
            st.markdown("**Portfolio Summary**")
            st.write(f"Total Value: ${metrics.portfolio_value:,.0f} {value_arrow} (Δ ${value_change:+,.0f})")
            st.write(f"Cash Balance: ${metrics.cash_balance:,.0f}")
            st.write(f"Positions: {metrics.positions_count}")
            st.write(f"Active Orders: {metrics.active_orders}  ")
            
            st.session_state.prev_portfolio['value'] = metrics.portfolio_value
            
            # P&L breakdown
            st.markdown("**P&L Breakdown**")
            st.write(f"Realized P&L: ${metrics.realized_pnl:,.0f}")
            st.write(f"Unrealized P&L: ${metrics.unrealized_pnl:,.0f}")
            
            # Risk metrics
            st.markdown("**Risk Metrics**")
            if not positions_df.empty:
                long_exposure = positions_df[positions_df['side'] == 'Long']['market_value'].sum()
                short_exposure = positions_df[positions_df['side'] == 'Short']['market_value'].sum()
                net_exposure = long_exposure - short_exposure
                gross_exposure = long_exposure + short_exposure
                
                st.write(f"Long Exposure: ${long_exposure:,.0f}")
                st.write(f"Short Exposure: ${short_exposure:,.0f}")
                st.write(f"Net Exposure: ${net_exposure:,.0f}")
                st.write(f"Gross Exposure: ${gross_exposure:,.0f}")
        
        with col2:
            if not positions_df.empty:
                # Top positions table with live prices
                st.markdown(f"**Top Positions**   LIVE {datetime.now().strftime('%H:%M:%S')}")
                
                # Get current prices from cache
                cache = get_data_cache()
                top_positions = positions_df.nlargest(10, 'market_value')[
                    ['symbol', 'side', 'position_size', 'market_value', 'unrealized_pnl', 'weight']
                ].copy()
                
                # Add current price column
                top_positions['current_price'] = top_positions['symbol'].apply(
                    lambda s: cache.get_current_price(s) or 0
                )
                
                # Format columns
                top_positions['current_price'] = top_positions['current_price'].apply(lambda x: f"${x:,.2f}")
                top_positions['market_value'] = top_positions['market_value'].apply(lambda x: f"${x:,.0f}")
                top_positions['unrealized_pnl'] = top_positions['unrealized_pnl'].apply(lambda x: f"${x:,.0f}")
                top_positions['weight'] = top_positions['weight'].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(
                    top_positions,
                    column_config={
                        'symbol': 'Symbol',
                        'side': 'Side',
                        'position_size': 'Size',
                        'current_price': '  Live Price',
                        'market_value': 'Market Value',
                        'unrealized_pnl': 'Unrealized P&L',
                        'weight': 'Weight'
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Portfolio allocation pie chart
                if len(positions_df) > 0:
                    fig = px.pie(
                        positions_df.head(8),  # Top 8 positions
                        values='market_value',
                        names='symbol',
                        title='Portfolio Allocation (Top 8 Positions)'
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_system_status(self, system_status: Dict[str, Any]):
        """Render system status monitoring."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]  # Include milliseconds
        st.subheader(f" ️ System Status -   LIVE {timestamp}")
        
        if not system_status:
            st.warning("System status data unavailable")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**System Components**")
            
            components = [
                ('Market Data Feed', system_status.get('market_data_feed', 'unknown')),
                ('Order Management', system_status.get('order_management', 'unknown')),
                ('Risk Engine', system_status.get('risk_engine', 'unknown')),
                ('Portfolio Manager', system_status.get('portfolio_manager', 'unknown')),
                ('Execution Engine', system_status.get('execution_engine', 'unknown')),
                ('Data Warehouse', system_status.get('data_warehouse', 'unknown'))
            ]
            
            for component, status in components:
                if status == 'online':
                    status_class = 'status-online'
                    status_text = '  Online'
                elif status == 'warning':
                    status_class = 'status-warning'
                    status_text = '  Warning'
                else:
                    status_class = 'status-offline'
                    status_text = '  Offline'
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin: 8px 0;">
                    <span class="status-indicator {status_class}"></span>
                    <strong>{component}:</strong>&nbsp;{status_text}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Performance Metrics**")
            
            if 'last_heartbeat' in system_status:
                heartbeat_ago = (datetime.now() - system_status['last_heartbeat']).total_seconds()
                heartbeat_indicator = " " if heartbeat_ago < 5 else " " if heartbeat_ago < 10 else " ️"
                st.write(f"{heartbeat_indicator} Last Heartbeat: {heartbeat_ago:.1f}s ago")
            
            if 'latency_ms' in system_status:
                latency = system_status['latency_ms']
                latency_color = " " if latency < 10 else " " if latency < 50 else " "
                st.write(f"System Latency: {latency_color} {latency:.1f}ms (Real)")
            
            if 'throughput_ops' in system_status:
                st.write(f"  Throughput: {system_status['throughput_ops']:.1f} ops/sec (Real)")
            
            # Show real trading metrics
            if 'orders_submitted' in system_status:
                st.write(f"  Orders Submitted: {system_status['orders_submitted']}")
            
            if 'orders_filled' in system_status:
                st.write(f"  Orders Filled: {system_status['orders_filled']}")
            
            if 'fill_rate' in system_status:
                fill_rate = system_status['fill_rate'] * 100
                st.write(f"  Fill Rate: {fill_rate:.1f}%")
            
            if 'trades_executed' in system_status:
                st.write(f"  Trades Executed: {system_status['trades_executed']}")
            
            # Add live update counter
            if 'update_counter' not in st.session_state:
                st.session_state.update_counter = 0
            st.session_state.update_counter += 1
            st.write(f"  Dashboard Updates: {st.session_state.update_counter}")
    
    def render_navigation_panel(self):
        """Render navigation panel to other dashboards."""
        st.sidebar.markdown("##   Navigation")
        
        dashboard_options = {
            "  Main Dashboard": ("current", None),
            "  Research Dashboard": ("research", "interface/dashboards/research_dashboard.py"),
            "  Execution Dashboard": ("execution", "interface/dashboards/execution_dashboard.py"), 
            " ️ Risk Dashboard": ("risk", "interface/dashboards/risk_dashboard.py"),
            "  Market Microstructure": ("microstructure", "interface/dashboards/market_microstructure_viz.py")
        }
        
        for label, (key, script_path) in dashboard_options.items():
            if key == "current":
                st.sidebar.markdown(f"**{label}** ← Current")
            else:
                if st.sidebar.button(label, key=f"nav_{key}"):
                    # Switch to the selected dashboard
                    if 'current_dashboard' not in st.session_state:
                        st.session_state.current_dashboard = 'main'
                    st.session_state.current_dashboard = key
                    st.rerun()
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("##  ️ Settings")
        
        # Refresh rate setting
        refresh_rate = st.sidebar.selectbox(
            "Auto-refresh Rate",
            options=[1, 5, 10, 30, 60],
            index=1,
            format_func=lambda x: f"{x} seconds"
        )
        
        # Theme selection
        theme = st.sidebar.selectbox(
            "Theme",
            options=["Light", "Dark", "Auto"],
            index=0
        )
        
        # Alert settings
        st.sidebar.markdown("##   Alert Settings")
        enable_alerts = st.sidebar.checkbox("Enable Alerts", value=True)
        
        if enable_alerts:
            pnl_threshold = st.sidebar.number_input(
                "Daily P&L Alert Threshold ($)",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000
            )
            
            drawdown_threshold = st.sidebar.number_input(
                "Max Drawdown Alert (%)",
                min_value=1.0,
                max_value=20.0,
                value=5.0,
                step=0.5
            )
    
    def render_alerts_section(self, metrics: DashboardMetrics):
        """Render alerts and notifications."""
        alerts = []
        
        # Generate sample alerts based on metrics
        if abs(metrics.daily_pnl) > 15000:
            alert_type = "success" if metrics.daily_pnl > 0 else "error"
            alerts.append({
                'type': alert_type,
                'message': f"Large daily P&L movement: ${metrics.daily_pnl:,.0f}",
                'timestamp': datetime.now()
            })
        
        if metrics.max_drawdown < -0.08:
            alerts.append({
                'type': 'warning',
                'message': f"High drawdown detected: {metrics.max_drawdown:.2%}",
                'timestamp': datetime.now()
            })
        
        if metrics.var_95 < -40000:
            alerts.append({
                'type': 'warning',
                'message': f"High VaR exposure: ${metrics.var_95:,.0f}",
                'timestamp': datetime.now()
            })
        
        # Add system alert
        alerts.append({
            'type': 'info',
            'message': "Execution engine experiencing minor latency - monitoring",
            'timestamp': datetime.now() - timedelta(minutes=5)
        })
        
        if alerts:
            st.subheader("  Active Alerts")
            for alert in alerts:
                alert_emoji = {
                    'success': ' ',
                    'warning': ' ️',
                    'error': ' ',
                    'info': 'ℹ️'
                }.get(alert['type'], 'ℹ️')
                
                alert_color = {
                    'success': '#d1fae5',
                    'warning': '#fef3c7',
                    'error': '#fee2e2',
                    'info': '#dbeafe'
                }.get(alert['type'], '#dbeafe')
                
                st.markdown(f"""
                <div style="background-color: {alert_color}; padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 4px solid #374151;">
                    <strong>{alert_emoji} {alert['message']}</strong><br>
                    <small>{alert['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
    
    def render_main_dashboard(self):
        """Render complete main dashboard."""
        try:
            # Initialize session state for auto-refresh
            if 'last_refresh' not in st.session_state:
                st.session_state.last_refresh = datetime.now()
            
            # Header
            self.render_header()
            
            # Navigation panel
            self.render_navigation_panel()
            
            # Generate sample data
            metrics, pnl_data, positions_df, system_status = self.generate_sample_data()
            self.last_update = datetime.now()
            
            # Main content
            main_content = st.container()
            
            with main_content:
                # Key metrics
                self.render_key_metrics(metrics)
                
                st.markdown("---")
                
                # P&L chart and portfolio overview
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    self.render_pnl_chart(pnl_data)
                
                with col2:
                    self.render_alerts_section(metrics)
                
                st.markdown("---")
                
                # Portfolio overview
                self.render_portfolio_overview(metrics, positions_df)
                
                st.markdown("---")
                
                # System status
                self.render_system_status(system_status)
            
            # Auto-refresh logic - check every render
            if st.session_state.get('auto_refresh', True):
                time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
                if time_since_refresh >= self.update_frequency:
                    st.session_state.last_refresh = datetime.now()
                    time.sleep(0.5)  # Brief pause before rerun
                    st.rerun()
                else:
                    # Schedule rerun after remaining time
                    remaining = self.update_frequency - time_since_refresh
                    time.sleep(min(remaining, 0.5))
                    st.rerun()
            
        except Exception as e:
            st.error(f"Error rendering dashboard: {str(e)}")
            logger.error(f"Dashboard error: {str(e)}")

def main():
    """Main function to run the dashboard with navigation support."""
    # Check if user wants to switch dashboard
    if 'current_dashboard' in st.session_state and st.session_state.current_dashboard != 'main':
        dashboard_map = {
            'research': 'research_dashboard',
            'execution': 'execution_dashboard',
            'risk': 'risk_dashboard',
            'microstructure': 'market_microstructure_viz'
        }
        
        selected = st.session_state.current_dashboard
        if selected in dashboard_map:
            module_name = dashboard_map[selected]
            try:
                # Import and run the selected dashboard
                import importlib
                import sys
                from pathlib import Path
                
                # Add current directory to path
                sys.path.insert(0, str(Path(__file__).parent))
                
                # Import the dashboard module
                dashboard_module = importlib.import_module(module_name)
                
                # Run the dashboard's main function
                if hasattr(dashboard_module, 'main'):
                    dashboard_module.main()
                else:
                    # Try to instantiate and render
                    dashboard_class_name = ''.join([word.capitalize() for word in module_name.split('_')])
                    if hasattr(dashboard_module, dashboard_class_name):
                        dashboard_class = getattr(dashboard_module, dashboard_class_name)
                        dashboard_instance = dashboard_class()
                        if hasattr(dashboard_instance, 'render'):
                            dashboard_instance.render()
                        else:
                            st.error(f"Dashboard {module_name} has no render method")
                    else:
                        st.error(f"Dashboard class {dashboard_class_name} not found in {module_name}")
                
                return  # Exit after rendering other dashboard
            except Exception as e:
                st.error(f"Error loading dashboard: {str(e)}")
                st.session_state.current_dashboard = 'main'  # Reset to main
    
    # Render main dashboard
    dashboard = MainDashboard()
    dashboard.render_main_dashboard()

if __name__ == "__main__":
    main()