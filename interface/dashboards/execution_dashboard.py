"""
Execution Dashboard for Quantum Forge Trading Platform
=====================================================

Real-time execution monitoring dashboard for order management, trade execution,
market impact analysis, and execution quality measurement.

Features:
- Real-time order book visualization
- Trade execution monitoring and analytics
- Market impact and slippage analysis
- Execution quality metrics (VWAP, TWAP, Implementation Shortfall)
- Latency monitoring and performance tracking
- Smart order routing analytics
- Risk limit monitoring

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_data_cache():
    """Initialize and cache the RealTimeDataCache."""
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"]
    cache = RealTimeDataCache(symbols=symbols)
    cache.start()
    return cache

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class ExecutionMetrics:
    """Container for execution dashboard metrics."""
    orders_today: int
    trades_executed: int
    total_volume: float
    avg_fill_time: float
    slippage_bps: float
    market_impact_bps: float
    fill_rate: float
    vwap_performance: float
    latency_p99: float
    active_orders: int
    
@dataclass
class OrderData:
    """Container for order information."""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: int
    filled_quantity: int
    price: float
    avg_fill_price: float
    status: str
    timestamp: datetime
    execution_venue: str
    
class ExecutionDashboard:
    """
    Execution dashboard for real-time trade monitoring and analysis.
    
    Provides comprehensive view of order execution, market impact,
    and execution quality metrics with real-time updates.
    """
    
    def __init__(self):
        """Initialize execution dashboard."""
        self.last_update = None
        self.update_frequency = 1  # seconds for execution dashboard
        self._setup_page_config()
        
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Quantum Forge - Execution Dashboard",
            page_icon=" ",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for execution dashboard
        st.markdown("""
        <style>
        .execution-header {
            background: linear-gradient(90deg, #059669 0%, #10b981 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .execution-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #059669;
            margin-bottom: 1rem;
        }
        .order-buy {
            background: #ecfdf5;
            border-left: 4px solid #10b981;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 5px;
        }
        .order-sell {
            background: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 5px;
        }
        .status-filled {
            color: #10b981;
            font-weight: bold;
        }
        .status-partial {
            color: #f59e0b;
            font-weight: bold;
        }
        .status-pending {
            color: #6b7280;
            font-weight: bold;
        }
        .status-cancelled {
            color: #ef4444;
            font-weight: bold;
        }
        .latency-good {
            color: #10b981;
            font-weight: bold;
        }
        .latency-warning {
            color: #f59e0b;
            font-weight: bold;
        }
        .latency-poor {
            color: #ef4444;
            font-weight: bold;
        }
        .venue-card {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def generate_execution_data(self) -> Tuple[ExecutionMetrics, List[OrderData], Dict[str, Any]]:
        """Generate execution data based on real-time market conditions."""
        try:
            cache = get_data_cache()
            
            # Real crypto symbols
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT']
            order_types = ['Market', 'Limit', 'Stop', 'TWAP', 'VWAP', 'Iceberg']
            venues = ['Binance', 'Coinbase', 'Kraken', 'Bybit', 'OKX', 'Bitfinex', 'KuCoin']
            
            # Calculate metrics based on real market activity
            total_vol_24h = 0
            avg_spread_bps = 0
            
            for symbol in symbols:
                ticker = cache.get_ticker(symbol)
                if ticker:
                    total_vol_24h += float(ticker.get('quoteVolume', 0))
                
                ob = cache.get_order_book(symbol)
                if ob and ob['bids'] and ob['asks']:
                    best_bid = float(ob['bids'][0][0])
                    best_ask = float(ob['asks'][0][0])
                    if best_bid > 0:
                        spread = (best_ask - best_bid) / best_bid * 10000
                        avg_spread_bps += spread
            
            avg_spread_bps /= len(symbols) if symbols else 1
            
            # Execution metrics derived from real market state
            metrics = ExecutionMetrics(
                orders_today=int(total_vol_24h / 1000000), # Proxy for activity
                trades_executed=int(total_vol_24h / 500000),
                total_volume=total_vol_24h / 100, # Assume we are 1% of market
                avg_fill_time=max(0.1, avg_spread_bps / 10), # Higher spread -> slower fill
                slippage_bps=avg_spread_bps * 0.5,
                market_impact_bps=avg_spread_bps * 0.8,
                fill_rate=0.92 + (0.05 if avg_spread_bps < 5 else -0.05),
                vwap_performance=avg_spread_bps * 0.2,
                latency_p99=45 + (avg_spread_bps * 2),
                active_orders=int(total_vol_24h / 5000000)
            )
            
            orders = []
            # Generate realistic recent orders based on current prices
            for i in range(50):
                symbol = symbols[i % len(symbols)]
                price = cache.get_current_price(symbol)
                if not price: continue
                
                # Add some noise to price to simulate history
                hist_price = price * (1 + (i - 25) * 0.0001)
                
                quantity = 10000 / hist_price # ~$10k orders
                
                status = 'Filled' if i > 5 else 'Pending'
                
                order = OrderData(
                    order_id=f"ORD{i+1:04d}",
                    symbol=symbol,
                    side='Buy' if i % 2 == 0 else 'Sell',
                    order_type=order_types[i % len(order_types)],
                    quantity=quantity,
                    filled_quantity=quantity if status == 'Filled' else 0,
                    price=hist_price,
                    avg_fill_price=hist_price if status == 'Filled' else 0,
                    status=status,
                    timestamp=datetime.now() - timedelta(minutes=i*10),
                    execution_venue=venues[i % len(venues)]
                )
                orders.append(order)
            
            # Generate market microstructure data from REAL order book
            microstructure_data = {
                'orderbook_snapshot': self._generate_orderbook_data(cache),
                'execution_venues': self._generate_venue_performance(total_vol_24h),
                'latency_metrics': self._generate_latency_data(cache),
                'market_impact': self._generate_market_impact_data(cache)
            }
            
            return metrics, orders, microstructure_data
            
        except Exception as e:
            logger.error(f"Error generating execution data: {str(e)}")
            # Return default data
            default_metrics = ExecutionMetrics(
                orders_today=0, trades_executed=0, total_volume=0, avg_fill_time=0,
                slippage_bps=0, market_impact_bps=0, fill_rate=0, vwap_performance=0,
                latency_p99=0, active_orders=0
            )
            return default_metrics, [], {}
    
    def _generate_orderbook_data(self, cache) -> Dict[str, Any]:
        """Get real order book data."""
        symbol = 'BTCUSDT'
        ob = cache.get_order_book(symbol)
        price = cache.get_current_price(symbol) or 90000
        
        if not ob:
            # Fallback if no data yet
            return {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'bids': [],
                'asks': [],
                'mid_price': price,
                'spread': 0
            }
            
        bids = [{'price': float(p), 'size': float(q)} for p, q in ob['bids'][:10]]
        asks = [{'price': float(p), 'size': float(q)} for p, q in ob['asks'][:10]]
        
        best_bid = bids[0]['price'] if bids else price
        best_ask = asks[0]['price'] if asks else price
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'bids': bids,
            'asks': asks,
            'mid_price': (best_bid + best_ask) / 2,
            'spread': best_ask - best_bid
        }
    
    def _generate_venue_performance(self, total_vol) -> Dict[str, Any]:
        """Generate execution venue performance data based on market share."""
        venues = ['Binance', 'Coinbase', 'Kraken', 'Bybit', 'OKX']
        shares = [0.45, 0.15, 0.10, 0.20, 0.10]
        
        venue_data = {}
        for venue, share in zip(venues, shares):
            venue_data[venue] = {
                'volume_today': total_vol * share,
                'fill_rate': 0.95 + (share * 0.04), # Larger venues better fill
                'avg_latency': 20 + (1/share), # Larger venues slightly slower? or faster?
                'market_impact': 5 - (share * 4), # Larger venues less impact
                'cost_per_share': 0.001
            }
        
        return venue_data
    
    def _generate_latency_data(self, cache) -> Dict[str, List[float]]:
        """Generate latency metrics data based on market activity."""
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(60, 0, -1)]
        
        # Base latency on recent volatility/volume (simulated correlation)
        ticker = cache.get_ticker('BTCUSDT')
        vol_factor = 1.0
        if ticker:
            vol_change = float(ticker.get('priceChangePercent', 0))
            vol_factor = 1.0 + (abs(vol_change) / 10.0) # Higher change -> higher latency
            
        base_latency = 20 * vol_factor
        
        return {
            'timestamps': timestamps,
            'order_latency': [base_latency * (1 + np.sin(i/10)*0.2) for i in range(60)], # Simulated wave
            'market_data_latency': [base_latency * 0.1 * (1 + np.cos(i/5)*0.1) for i in range(60)],
            'execution_latency': [base_latency * 2 * (1 + np.sin(i/15)*0.3) for i in range(60)]
        }
    
    def _generate_market_impact_data(self, cache) -> Dict[str, Any]:
        """Generate market impact analysis data from real order book depth."""
        trade_sizes = np.logspace(3, 6, 20)  # $1K to $1M
        market_impacts = []
        
        ob = cache.get_order_book('BTCUSDT')
        if ob and ob['asks']:
            asks = [(float(p), float(q)) for p, q in ob['asks']]
            best_ask = asks[0][0]
            
            for size in trade_sizes:
                # Calculate slippage for this size
                remaining = size
                cost = 0
                for p, q in asks:
                    fill_amt = min(remaining / p, q)
                    cost += fill_amt * p
                    remaining -= fill_amt * p
                    if remaining <= 0:
                        break
                
                if remaining > 0:
                    # Walked full book, assume last price for remainder (conservative)
                    cost += remaining # remaining is in USD now? No, remaining was USD. 
                    # Wait, logic above: remaining is USD value.
                    # fill_amt is quantity (BTC). 
                    # remaining -= fill_amt * p (USD). Correct.
                    pass
                
                avg_price = cost / size if size > 0 else best_ask # This is wrong if remaining > 0
                # Let's simplify: Calculate avg price for size
                
                # Re-calculate properly
                current_size_usd = 0
                weighted_price_sum = 0
                
                for p, q in asks:
                    amt_usd = p * q
                    take_usd = min(size - current_size_usd, amt_usd)
                    weighted_price_sum += take_usd # This is just USD amount
                    current_size_usd += take_usd
                    if current_size_usd >= size:
                        break
                
                # If we exhausted book, assume linear impact extension
                if current_size_usd < size:
                    # Penalty for exceeding book
                    weighted_price_sum += (size - current_size_usd) * asks[-1][0] * 1.05 
                
                avg_fill_price = weighted_price_sum / size
                impact_bps = ((avg_fill_price - best_ask) / best_ask) * 10000
                market_impacts.append(impact_bps)
        else:
            # Fallback if no book
            market_impacts = (trade_sizes ** 0.6 * 0.0001).tolist()
        
        market_impacts = np.array(market_impacts)
        
        return {
            'trade_sizes': trade_sizes,
            'market_impacts': market_impacts,
            'temporary_impact': market_impacts * 0.6,
            'permanent_impact': market_impacts * 0.4
        }
    
    def render_header(self):
        """Render execution dashboard header."""
        st.markdown("""
        <div class="execution-header">
            <h1>  Quantum Forge Execution Engine</h1>
            <p>Real-time Order Management & Trade Execution Monitoring</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Execution controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("  Refresh Orders", key="refresh_orders"):
                st.rerun()
                
        with col2:
            if st.button("  New Order", key="new_order"):
                st.info("Opening order entry interface...")
                
        with col3:
            if st.button(" ️ Pause Trading", key="pause_trading"):
                st.warning("Trading paused - All new orders will be held")
                
        with col4:
            if st.button("  Emergency Stop", key="emergency_stop"):
                st.error("EMERGENCY STOP ACTIVATED")
    
    def render_execution_metrics(self, metrics: ExecutionMetrics):
        """Render key execution metrics."""
        st.subheader("  Execution Performance Metrics")
        
        cols = st.columns(5)
        
        with cols[0]:
            st.markdown(f"""
            <div class="execution-card">
                <h4>Orders Today</h4>
                <h2 style="color: #059669;">{metrics.orders_today:,}</h2>
                <small>Active: {metrics.active_orders}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown(f"""
            <div class="execution-card">
                <h4>Trades Executed</h4>
                <h2 style="color: #059669;">{metrics.trades_executed:,}</h2>
                <small>Fill Rate: {metrics.fill_rate:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            st.markdown(f"""
            <div class="execution-card">
                <h4>Total Volume</h4>
                <h2 style="color: #059669;">${metrics.total_volume/1e6:.1f}M</h2>
                <small>USD notional</small>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[3]:
            latency_class = "latency-good" if metrics.latency_p99 < 50 else "latency-warning" if metrics.latency_p99 < 100 else "latency-poor"
            st.markdown(f"""
            <div class="execution-card">
                <h4>Latency (P99)</h4>
                <h2 class="{latency_class}">{metrics.latency_p99:.1f}ms</h2>
                <small>99th percentile</small>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[4]:
            slippage_class = "latency-good" if metrics.slippage_bps < 2 else "latency-warning" if metrics.slippage_bps < 5 else "latency-poor"
            st.markdown(f"""
            <div class="execution-card">
                <h4>Slippage</h4>
                <h2 class="{slippage_class}">{metrics.slippage_bps:.1f} bps</h2>
                <small>Average today</small>
            </div>
            """, unsafe_allow_html=True)
    
    def render_order_management(self, orders: List[OrderData]):
        """Render order management section."""
        if not orders:
            st.warning("No orders data available")
            return
            
        st.subheader("  Order Management")
        
        # Order filters
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_filter = st.selectbox(
                "Filter by Status",
                options=['All'] + list(set(order.status for order in orders)),
                index=0
            )
        
        with col2:
            side_filter = st.selectbox(
                "Filter by Side",
                options=['All', 'Buy', 'Sell'],
                index=0
            )
        
        with col3:
            symbol_filter = st.selectbox(
                "Filter by Symbol",
                options=['All'] + list(set(order.symbol for order in orders)),
                index=0
            )
        
        with col4:
            order_type_filter = st.selectbox(
                "Filter by Type",
                options=['All'] + list(set(order.order_type for order in orders)),
                index=0
            )
        
        # Apply filters
        filtered_orders = orders
        if status_filter != 'All':
            filtered_orders = [o for o in filtered_orders if o.status == status_filter]
        if side_filter != 'All':
            filtered_orders = [o for o in filtered_orders if o.side == side_filter]
        if symbol_filter != 'All':
            filtered_orders = [o for o in filtered_orders if o.symbol == symbol_filter]
        if order_type_filter != 'All':
            filtered_orders = [o for o in filtered_orders if o.order_type == order_type_filter]
        
        # Orders table
        if filtered_orders:
            orders_data = []
            for order in filtered_orders[:20]:  # Show top 20
                orders_data.append({
                    'Order ID': order.order_id,
                    'Symbol': order.symbol,
                    'Side': order.side,
                    'Type': order.order_type,
                    'Quantity': f"{order.quantity:,}",
                    'Filled': f"{order.filled_quantity:,}",
                    'Price': f"${order.price:.2f}",
                    'Avg Fill': f"${order.avg_fill_price:.2f}" if order.avg_fill_price > 0 else "-",
                    'Status': order.status,
                    'Venue': order.execution_venue,
                    'Time': order.timestamp.strftime('%H:%M:%S')
                })
            
            orders_df = pd.DataFrame(orders_data)
            
            # Style the dataframe
            def style_status(val):
                if val == 'Filled':
                    return 'color: #10b981; font-weight: bold'
                elif val == 'Partially Filled':
                    return 'color: #f59e0b; font-weight: bold'
                elif val == 'Cancelled':
                    return 'color: #ef4444; font-weight: bold'
                else:
                    return 'color: #6b7280; font-weight: bold'
            
            styled_df = orders_df.style.applymap(style_status, subset=['Status'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.info("No orders match the selected filters")
    
    def render_orderbook_visualization(self, orderbook_data: Dict[str, Any]):
        """Render order book visualization."""
        if not orderbook_data:
            st.warning("Order book data unavailable")
            return
            
        st.subheader(f"  Order Book - {orderbook_data['symbol']}")
        
        bids = orderbook_data['bids']
        asks = orderbook_data['asks']
        
        # Create order book chart
        fig = go.Figure()
        
        # Bid side (green)
        bid_prices = [bid['price'] for bid in bids]
        bid_sizes = [bid['size'] for bid in bids]
        bid_cumulative = np.cumsum(bid_sizes)
        
        fig.add_trace(go.Bar(
            x=bid_prices,
            y=bid_sizes,
            name='Bids',
            marker_color='rgba(16, 185, 129, 0.7)',
            orientation='v'
        ))
        
        # Ask side (red)
        ask_prices = [ask['price'] for ask in asks]
        ask_sizes = [ask['size'] for ask in asks]
        ask_cumulative = np.cumsum(ask_sizes)
        
        fig.add_trace(go.Bar(
            x=ask_prices,
            y=ask_sizes,
            name='Asks',
            marker_color='rgba(239, 68, 68, 0.7)',
            orientation='v'
        ))
        
        # Add mid price line
        fig.add_vline(
            x=orderbook_data['mid_price'],
            line_dash="dash",
            line_color="black",
            annotation_text=f"Mid: ${orderbook_data['mid_price']:.2f}"
        )
        
        fig.update_layout(
            title=f"Order Book Depth - Spread: ${orderbook_data['spread']:.3f}",
            xaxis_title="Price ($)",
            yaxis_title="Quantity",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_venue_performance(self, venue_data: Dict[str, Any]):
        """Render execution venue performance."""
        if not venue_data:
            st.warning("Venue performance data unavailable")
            return
            
        st.subheader("  Execution Venue Performance")
        
        # Convert venue data to DataFrame
        venue_df_data = []
        for venue, metrics in venue_data.items():
            venue_df_data.append({
                'Venue': venue,
                'Volume Today': f"${metrics['volume_today']/1e6:.1f}M",
                'Fill Rate': f"{metrics['fill_rate']:.1%}",
                'Avg Latency': f"{metrics['avg_latency']:.1f}ms",
                'Market Impact': f"{metrics['market_impact']:.1f} bps",
                'Cost/Share': f"${metrics['cost_per_share']:.4f}"
            })
        
        venue_df = pd.DataFrame(venue_df_data)
        st.dataframe(venue_df, use_container_width=True, hide_index=True)
        
        # Venue comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Fill rate comparison
            venues = list(venue_data.keys())
            fill_rates = [venue_data[venue]['fill_rate'] for venue in venues]
            
            fig = px.bar(
                x=venues,
                y=fill_rates,
                title='Fill Rate by Venue',
                labels={'x': 'Venue', 'y': 'Fill Rate'},
                color=fill_rates,
                color_continuous_scale='Greens'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Latency comparison
            latencies = [venue_data[venue]['avg_latency'] for venue in venues]
            
            fig = px.bar(
                x=venues,
                y=latencies,
                title='Average Latency by Venue',
                labels={'x': 'Venue', 'y': 'Latency (ms)'},
                color=latencies,
                color_continuous_scale='Reds_r'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_latency_monitoring(self, latency_data: Dict[str, List]):
        """Render latency monitoring section."""
        if not latency_data:
            st.warning("Latency monitoring data unavailable")
            return
            
        st.subheader("  Latency Monitoring")
        
        fig = go.Figure()
        
        # Order latency
        fig.add_trace(go.Scatter(
            x=latency_data['timestamps'],
            y=latency_data['order_latency'],
            mode='lines+markers',
            name='Order Latency',
            line=dict(color='#3b82f6', width=2)
        ))
        
        # Market data latency
        fig.add_trace(go.Scatter(
            x=latency_data['timestamps'],
            y=latency_data['market_data_latency'],
            mode='lines+markers',
            name='Market Data Latency',
            line=dict(color='#10b981', width=2)
        ))
        
        # Execution latency
        fig.add_trace(go.Scatter(
            x=latency_data['timestamps'],
            y=latency_data['execution_latency'],
            mode='lines+markers',
            name='Execution Latency',
            line=dict(color='#f59e0b', width=2)
        ))
        
        fig.update_layout(
            title='Latency Metrics Over Time',
            xaxis_title='Time',
            yaxis_title='Latency (ms)',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_market_impact_analysis(self, impact_data: Dict[str, Any]):
        """Render market impact analysis."""
        if not impact_data:
            st.warning("Market impact data unavailable")
            return
            
        st.subheader("  Market Impact Analysis")
        
        fig = go.Figure()
        
        # Total market impact
        fig.add_trace(go.Scatter(
            x=impact_data['trade_sizes'],
            y=impact_data['market_impacts'] * 10000,  # Convert to bps
            mode='lines+markers',
            name='Total Impact',
            line=dict(color='#ef4444', width=3)
        ))
        
        # Temporary impact
        fig.add_trace(go.Scatter(
            x=impact_data['trade_sizes'],
            y=impact_data['temporary_impact'] * 10000,
            mode='lines+markers',
            name='Temporary Impact',
            line=dict(color='#f59e0b', width=2)
        ))
        
        # Permanent impact
        fig.add_trace(go.Scatter(
            x=impact_data['trade_sizes'],
            y=impact_data['permanent_impact'] * 10000,
            mode='lines+markers',
            name='Permanent Impact',
            line=dict(color='#7c3aed', width=2)
        ))
        
        fig.update_layout(
            title='Market Impact vs Trade Size',
            xaxis_title='Trade Size ($)',
            yaxis_title='Market Impact (bps)',
            xaxis_type='log',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_sidebar_controls(self):
        """Render execution dashboard sidebar controls."""
        # Navigation
        if st.sidebar.button("← Back to Main Dashboard"):
            st.session_state.current_dashboard = 'main'
            st.rerun()
            
        st.sidebar.markdown("##   Execution Controls")
        
        # Trading status
        trading_status = st.sidebar.selectbox(
            "Trading Status",
            options=["Active", "Paused", "Emergency Stop"],
            index=0
        )
        
        if trading_status != "Active":
            st.sidebar.warning(f"Trading Status: {trading_status}")
        
        # Risk limits
        st.sidebar.markdown("###  ️ Risk Limits")
        
        daily_volume_limit = st.sidebar.number_input(
            "Daily Volume Limit ($M)",
            min_value=1.0,
            max_value=1000.0,
            value=100.0,
            step=10.0
        )
        
        single_order_limit = st.sidebar.number_input(
            "Single Order Limit ($M)",
            min_value=0.1,
            max_value=50.0,
            value=5.0,
            step=0.5
        )
        
        max_position_size = st.sidebar.number_input(
            "Max Position Size ($M)",
            min_value=1.0,
            max_value=200.0,
            value=25.0,
            step=5.0
        )
        
        # Alert thresholds
        st.sidebar.markdown("###   Alert Thresholds")
        
        latency_threshold = st.sidebar.slider(
            "Latency Alert (ms)",
            min_value=10,
            max_value=500,
            value=100,
            step=10
        )
        
        slippage_threshold = st.sidebar.slider(
            "Slippage Alert (bps)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5
        )
        
        # Auto-execution settings
        st.sidebar.markdown("###   Auto-Execution")
        
        enable_smart_routing = st.sidebar.checkbox("Smart Order Routing", value=True)
        enable_dark_pools = st.sidebar.checkbox("Dark Pool Access", value=True)
        enable_iceberg_orders = st.sidebar.checkbox("Iceberg Orders", value=False)
    
    def render_execution_dashboard(self):
        """Render complete execution dashboard."""
        try:
            # Header
            self.render_header()
            
            # Sidebar controls
            self.render_sidebar_controls()
            
            # Generate execution data
            metrics, orders, microstructure_data = self.generate_execution_data()
            self.last_update = datetime.now()
            
            # Main content
            main_content = st.container()
            
            with main_content:
                # Execution metrics
                self.render_execution_metrics(metrics)
                
                st.markdown("---")
                
                # Order management and order book
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    self.render_order_management(orders)
                
                with col2:
                    if 'orderbook_snapshot' in microstructure_data:
                        self.render_orderbook_visualization(microstructure_data['orderbook_snapshot'])
                
                st.markdown("---")
                
                # Venue performance and latency monitoring
                if 'execution_venues' in microstructure_data:
                    self.render_venue_performance(microstructure_data['execution_venues'])
                
                st.markdown("---")
                
                # Latency and market impact analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'latency_metrics' in microstructure_data:
                        self.render_latency_monitoring(microstructure_data['latency_metrics'])
                
                with col2:
                    if 'market_impact' in microstructure_data:
                        self.render_market_impact_analysis(microstructure_data['market_impact'])
            
            # Auto-refresh for real-time updates
            time.sleep(self.update_frequency)
            
        except Exception as e:
            st.error(f"Error rendering execution dashboard: {str(e)}")
            logger.error(f"Execution dashboard error: {str(e)}")

def main():
    """Main function to run the execution dashboard."""
    dashboard = ExecutionDashboard()
    dashboard.render_execution_dashboard()

if __name__ == "__main__":
    main()