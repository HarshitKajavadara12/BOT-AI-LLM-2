"""
Order Book Visualization Components
==================================

Specialized components for visualizing order book data, market depth,
and order flow dynamics with real-time updates and advanced features.

Features:
- Real-time order book ladder visualization
- Market depth charts with bid/ask visualization
- Order book heatmap with time dimension
- Order flow and imbalance analysis
- Liquidity analysis and aggregation levels
- Price level clustering and visualization
- Interactive order book controls
- Alert system for significant order book events

Author: Quantum Forge Interface Team
Date: November 2025
"""

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class OrderBookLevel:
    """Represents a single order book level."""
    price: float
    size: int
    orders: int
    side: str  # 'bid' or 'ask'
    timestamp: datetime = None

@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot."""
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    mid_price: float
    spread: float

class OrderBookViz:
    """
    Advanced order book visualization components.
    
    Provides comprehensive visualization tools for order book data,
    market depth analysis, and real-time order flow monitoring.
    """
    
    def __init__(self):
        """Initialize order book visualization components."""
        self.default_colors = {
            'bid': '#10b981',          # Green for bids
            'ask': '#ef4444',          # Red for asks
            'bid_dark': '#059669',     # Dark green
            'ask_dark': '#dc2626',     # Dark red
            'mid_price': '#6366f1',    # Blue for mid price
            'spread': '#f59e0b',       # Orange for spread
            'background': '#f8fafc',   # Light background
            'grid': '#e2e8f0'          # Grid color
        }
        
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'family': "Arial, monospace", 'size': 12},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'hovermode': 'closest'
        }
    
    def create_order_book_ladder(
        self,
        snapshot: OrderBookSnapshot,
        levels: int = 10,
        show_orders: bool = True,
        show_size_bars: bool = True,
        **kwargs
    ) -> go.Figure:
        """
        Create traditional order book ladder visualization.
        
        Args:
            snapshot: Order book snapshot data
            levels: Number of levels to display on each side
            show_orders: Whether to show order count
            show_size_bars: Whether to show size as horizontal bars
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            fig = go.Figure()
            
            # Sort and limit levels
            bids = sorted(snapshot.bids, key=lambda x: x.price, reverse=True)[:levels]
            asks = sorted(snapshot.asks, key=lambda x: x.price)[:levels]
            
            if not bids or not asks:
                return self._create_error_chart("No order book data available")
            
            # Prepare data for visualization
            all_levels = asks[::-1] + bids  # Asks on top (reversed), bids on bottom
            prices = [level.price for level in all_levels]
            sizes = [level.size for level in all_levels]
            orders = [level.orders for level in all_levels]
            sides = [level.side for level in all_levels]
            
            # Create y-axis positions
            y_positions = list(range(len(all_levels)))
            
            # Add size bars if requested
            if show_size_bars:
                max_size = max(sizes)
                for i, (size, side) in enumerate(zip(sizes, sides)):
                    color = self.default_colors['ask'] if side == 'ask' else self.default_colors['bid']
                    
                    fig.add_trace(
                        go.Bar(
                            x=[size],
                            y=[y_positions[i]],
                            orientation='h',
                            name=f'{side.title()} Size',
                            marker_color=color,
                            opacity=0.3,
                            width=0.8,
                            showlegend=False,
                            hovertemplate=f'Price: {prices[i]:.2f}<br>Size: {size:,}<br>Orders: {orders[i]}<extra></extra>'
                        )
                    )
            
            # Add price levels as scatter points
            for i, (price, size, order_count, side) in enumerate(zip(prices, sizes, orders, sides)):
                color = self.default_colors['ask_dark'] if side == 'ask' else self.default_colors['bid_dark']
                
                fig.add_trace(
                    go.Scatter(
                        x=[0],
                        y=[y_positions[i]],
                        mode='markers+text',
                        marker=dict(
                            color=color,
                            size=12,
                            symbol='circle'
                        ),
                        text=f"{price:.2f}  {size:,}" + (f"  ({order_count})" if show_orders else ""),
                        textposition="middle right" if side == 'bid' else "middle left",
                        textfont=dict(
                            color=color,
                            size=11,
                            family="monospace"
                        ),
                        name=f'{side.title()} Levels',
                        showlegend=False,
                        hovertemplate=f'Price: {price:.2f}<br>Size: {size:,}<br>Orders: {order_count}<extra></extra>'
                    )
                )
            
            # Add mid price line
            mid_y = len(asks) - 0.5
            fig.add_hline(
                y=mid_y,
                line_dash="dash",
                line_color=self.default_colors['mid_price'],
                annotation_text=f"Mid: ${snapshot.mid_price:.2f} | Spread: ${snapshot.spread:.3f}",
                annotation_position="right"
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title="Order Book Ladder",
                xaxis=dict(
                    title="Size",
                    showgrid=False,
                    zeroline=True,
                    visible=show_size_bars
                ),
                yaxis=dict(
                    title="",
                    showgrid=False,
                    showticklabels=False
                ),
                height=max(400, len(all_levels) * 25),
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating order book ladder: {str(e)}")
            return self._create_error_chart(f"Error creating order book ladder: {str(e)}")
    
    def create_market_depth_chart(
        self,
        snapshot: OrderBookSnapshot,
        cumulative: bool = True,
        levels: int = 20,
        **kwargs
    ) -> go.Figure:
        """
        Create market depth chart showing cumulative liquidity.
        
        Args:
            snapshot: Order book snapshot data
            cumulative: Whether to show cumulative depth
            levels: Number of levels to include
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            fig = go.Figure()
            
            # Sort levels
            bids = sorted(snapshot.bids, key=lambda x: x.price, reverse=True)[:levels]
            asks = sorted(snapshot.asks, key=lambda x: x.price)[:levels]
            
            if not bids or not asks:
                return self._create_error_chart("No market depth data available")
            
            # Calculate cumulative sizes
            if cumulative:
                bid_sizes = np.cumsum([level.size for level in bids])
                ask_sizes = np.cumsum([level.size for level in asks])
            else:
                bid_sizes = [level.size for level in bids]
                ask_sizes = [level.size for level in asks]
            
            bid_prices = [level.price for level in bids]
            ask_prices = [level.price for level in asks]
            
            # Add bid side
            fig.add_trace(
                go.Scatter(
                    x=bid_prices,
                    y=bid_sizes,
                    mode='lines',
                    name='Bids',
                    line=dict(
                        color=self.default_colors['bid'],
                        width=3
                    ),
                    fill='tozeroy',
                    fillcolor=f'rgba(16, 185, 129, 0.3)',
                    hovertemplate='Price: %{x:.2f}<br>Cumulative Size: %{y:,}<extra></extra>'
                )
            )
            
            # Add ask side
            fig.add_trace(
                go.Scatter(
                    x=ask_prices,
                    y=ask_sizes,
                    mode='lines',
                    name='Asks',
                    line=dict(
                        color=self.default_colors['ask'],
                        width=3
                    ),
                    fill='tozeroy',
                    fillcolor=f'rgba(239, 68, 68, 0.3)',
                    hovertemplate='Price: %{x:.2f}<br>Cumulative Size: %{y:,}<extra></extra>'
                )
            )
            
            # Add mid price line
            fig.add_vline(
                x=snapshot.mid_price,
                line_dash="dash",
                line_color=self.default_colors['mid_price'],
                annotation_text=f"Mid: ${snapshot.mid_price:.2f}"
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title="Market Depth" + (" (Cumulative)" if cumulative else ""),
                xaxis_title="Price ($)",
                yaxis_title="Size" + (" (Cumulative)" if cumulative else ""),
                height=400,
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating market depth chart: {str(e)}")
            return self._create_error_chart(f"Error creating market depth chart: {str(e)}")
    
    def create_order_book_heatmap(
        self,
        snapshots: List[OrderBookSnapshot],
        price_range: Optional[Tuple[float, float]] = None,
        time_window: int = 30,
        **kwargs
    ) -> go.Figure:
        """
        Create order book heatmap showing liquidity over time.
        
        Args:
            snapshots: List of order book snapshots over time
            price_range: Optional price range to focus on
            time_window: Time window in minutes
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if not snapshots:
                return self._create_error_chart("No order book snapshots available")
            
            # Determine price range
            if price_range is None:
                all_prices = []
                for snapshot in snapshots:
                    all_prices.extend([level.price for level in snapshot.bids + snapshot.asks])
                min_price = min(all_prices)
                max_price = max(all_prices)
                price_range = (min_price, max_price)
            
            # Create price grid
            price_step = (price_range[1] - price_range[0]) / 100
            price_levels = np.arange(price_range[0], price_range[1], price_step)
            
            # Create time grid
            timestamps = [snapshot.timestamp for snapshot in snapshots]
            
            # Create liquidity matrix
            liquidity_matrix = np.zeros((len(timestamps), len(price_levels)))
            
            for i, snapshot in enumerate(snapshots):
                # Map order book levels to price grid
                for level in snapshot.bids + snapshot.asks:
                    price_idx = np.argmin(np.abs(price_levels - level.price))
                    if 0 <= price_idx < len(price_levels):
                        liquidity_matrix[i, price_idx] += level.size
            
            # Create heatmap
            fig = go.Figure()
            
            fig.add_trace(
                go.Heatmap(
                    x=price_levels,
                    y=timestamps,
                    z=liquidity_matrix,
                    colorscale='Viridis',
                    hovertemplate='Time: %{y}<br>Price: %{x:.2f}<br>Liquidity: %{z:,.0f}<extra></extra>',
                    colorbar=dict(title="Liquidity")
                )
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title="Order Book Liquidity Heatmap",
                xaxis_title="Price ($)",
                yaxis_title="Time",
                height=500,
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating order book heatmap: {str(e)}")
            return self._create_error_chart(f"Error creating order book heatmap: {str(e)}")
    
    def create_order_imbalance_chart(
        self,
        snapshots: List[OrderBookSnapshot],
        levels: int = 5,
        **kwargs
    ) -> go.Figure:
        """
        Create order book imbalance chart over time.
        
        Args:
            snapshots: List of order book snapshots
            levels: Number of levels to consider for imbalance
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if not snapshots:
                return self._create_error_chart("No order book snapshots available")
            
            timestamps = []
            imbalances = []
            bid_volumes = []
            ask_volumes = []
            
            for snapshot in snapshots:
                # Calculate imbalance for top N levels
                top_bids = sorted(snapshot.bids, key=lambda x: x.price, reverse=True)[:levels]
                top_asks = sorted(snapshot.asks, key=lambda x: x.price)[:levels]
                
                bid_volume = sum(level.size for level in top_bids)
                ask_volume = sum(level.size for level in top_asks)
                
                total_volume = bid_volume + ask_volume
                imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
                
                timestamps.append(snapshot.timestamp)
                imbalances.append(imbalance)
                bid_volumes.append(bid_volume)
                ask_volumes.append(ask_volume)
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=['Order Book Imbalance', 'Bid vs Ask Volume'],
                vertical_spacing=0.1
            )
            
            # Add imbalance line
            colors = ['#10b981' if x > 0 else '#ef4444' if x < 0 else '#6b7280' for x in imbalances]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=imbalances,
                    mode='lines+markers',
                    name='Imbalance',
                    line=dict(color=self.default_colors['mid_price'], width=2),
                    marker=dict(color=colors, size=4),
                    hovertemplate='Time: %{x}<br>Imbalance: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add zero line
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                row=1, col=1
            )
            
            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=timestamps,
                    y=bid_volumes,
                    name='Bid Volume',
                    marker_color=self.default_colors['bid'],
                    opacity=0.7,
                    hovertemplate='Time: %{x}<br>Bid Volume: %{y:,}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=timestamps,
                    y=[-v for v in ask_volumes],  # Negative for visual separation
                    name='Ask Volume',
                    marker_color=self.default_colors['ask'],
                    opacity=0.7,
                    hovertemplate='Time: %{x}<br>Ask Volume: %{y:,}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title="Order Book Imbalance Analysis",
                height=600,
                **kwargs
            )
            
            fig.update_yaxes(title_text="Imbalance", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating order imbalance chart: {str(e)}")
            return self._create_error_chart(f"Error creating order imbalance chart: {str(e)}")
    
    def create_spread_analysis_chart(
        self,
        snapshots: List[OrderBookSnapshot],
        **kwargs
    ) -> go.Figure:
        """
        Create bid-ask spread analysis chart.
        
        Args:
            snapshots: List of order book snapshots
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if not snapshots:
                return self._create_error_chart("No order book snapshots available")
            
            timestamps = [snapshot.timestamp for snapshot in snapshots]
            spreads = [snapshot.spread for snapshot in snapshots]
            mid_prices = [snapshot.mid_price for snapshot in snapshots]
            spread_bps = [(spread / mid_price) * 10000 for spread, mid_price in zip(spreads, mid_prices)]
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.6, 0.4],
                subplot_titles=['Bid-Ask Spread ($)', 'Spread (Basis Points)'],
                vertical_spacing=0.1
            )
            
            # Add spread in dollars
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=spreads,
                    mode='lines',
                    name='Spread ($)',
                    line=dict(color=self.default_colors['spread'], width=2),
                    hovertemplate='Time: %{x}<br>Spread: $%{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add spread in basis points
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=spread_bps,
                    mode='lines',
                    name='Spread (bps)',
                    line=dict(color=self.default_colors['ask'], width=2),
                    hovertemplate='Time: %{x}<br>Spread: %{y:.1f} bps<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add average lines
            avg_spread = np.mean(spreads)
            avg_spread_bps = np.mean(spread_bps)
            
            fig.add_hline(
                y=avg_spread,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Avg: ${avg_spread:.3f}",
                row=1, col=1
            )
            
            fig.add_hline(
                y=avg_spread_bps,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Avg: {avg_spread_bps:.1f} bps",
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title="Bid-Ask Spread Analysis",
                height=500,
                **kwargs
            )
            
            fig.update_yaxes(title_text="Spread ($)", row=1, col=1)
            fig.update_yaxes(title_text="Spread (bps)", row=2, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating spread analysis chart: {str(e)}")
            return self._create_error_chart(f"Error creating spread analysis chart: {str(e)}")
    
    def create_liquidity_profile(
        self,
        snapshot: OrderBookSnapshot,
        price_range_pct: float = 0.02,
        **kwargs
    ) -> go.Figure:
        """
        Create liquidity profile around current price.
        
        Args:
            snapshot: Order book snapshot
            price_range_pct: Price range as percentage of mid price
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            # Define price range
            price_range = snapshot.mid_price * price_range_pct
            min_price = snapshot.mid_price - price_range
            max_price = snapshot.mid_price + price_range
            
            # Filter levels within range
            relevant_bids = [level for level in snapshot.bids if level.price >= min_price]
            relevant_asks = [level for level in snapshot.asks if level.price <= max_price]
            
            if not relevant_bids and not relevant_asks:
                return self._create_error_chart("No liquidity data in specified range")
            
            fig = go.Figure()
            
            # Add bid liquidity
            if relevant_bids:
                bid_prices = [level.price for level in relevant_bids]
                bid_sizes = [level.size for level in relevant_bids]
                
                fig.add_trace(
                    go.Bar(
                        x=bid_prices,
                        y=bid_sizes,
                        name='Bid Liquidity',
                        marker_color=self.default_colors['bid'],
                        opacity=0.7,
                        width=0.01,
                        hovertemplate='Price: %{x:.2f}<br>Size: %{y:,}<extra></extra>'
                    )
                )
            
            # Add ask liquidity
            if relevant_asks:
                ask_prices = [level.price for level in relevant_asks]
                ask_sizes = [level.size for level in relevant_asks]
                
                fig.add_trace(
                    go.Bar(
                        x=ask_prices,
                        y=ask_sizes,
                        name='Ask Liquidity',
                        marker_color=self.default_colors['ask'],
                        opacity=0.7,
                        width=0.01,
                        hovertemplate='Price: %{x:.2f}<br>Size: %{y:,}<extra></extra>'
                    )
                )
            
            # Add mid price line
            fig.add_vline(
                x=snapshot.mid_price,
                line_dash="dash",
                line_color=self.default_colors['mid_price'],
                annotation_text=f"Mid: ${snapshot.mid_price:.2f}"
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=f"Liquidity Profile (±{price_range_pct:.1%})",
                xaxis_title="Price ($)",
                yaxis_title="Size",
                height=400,
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating liquidity profile: {str(e)}")
            return self._create_error_chart(f"Error creating liquidity profile: {str(e)}")
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart when data processing fails."""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Order Book Error:<br>{error_message}",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            **self.default_layout,
            height=400,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig

# Convenience functions for quick order book visualization
def quick_order_book_ladder(bids: List[dict], asks: List[dict], **kwargs) -> go.Figure:
    """Quick order book ladder creation."""
    # Convert dictionaries to OrderBookLevel objects
    bid_levels = [OrderBookLevel(b['price'], b['size'], b.get('orders', 1), 'bid') for b in bids]
    ask_levels = [OrderBookLevel(a['price'], a['size'], a.get('orders', 1), 'ask') for a in asks]
    
    # Create snapshot
    mid_price = (bid_levels[0].price + ask_levels[0].price) / 2 if bid_levels and ask_levels else 0
    spread = ask_levels[0].price - bid_levels[0].price if bid_levels and ask_levels else 0
    
    snapshot = OrderBookSnapshot(
        timestamp=datetime.now(),
        bids=bid_levels,
        asks=ask_levels,
        mid_price=mid_price,
        spread=spread
    )
    
    viz = OrderBookViz()
    return viz.create_order_book_ladder(snapshot, **kwargs)

def quick_market_depth(bids: List[dict], asks: List[dict], **kwargs) -> go.Figure:
    """Quick market depth chart creation."""
    # Convert dictionaries to OrderBookLevel objects
    bid_levels = [OrderBookLevel(b['price'], b['size'], b.get('orders', 1), 'bid') for b in bids]
    ask_levels = [OrderBookLevel(a['price'], a['size'], a.get('orders', 1), 'ask') for a in asks]
    
    # Create snapshot
    mid_price = (bid_levels[0].price + ask_levels[0].price) / 2 if bid_levels and ask_levels else 0
    spread = ask_levels[0].price - bid_levels[0].price if bid_levels and ask_levels else 0
    
    snapshot = OrderBookSnapshot(
        timestamp=datetime.now(),
        bids=bid_levels,
        asks=ask_levels,
        mid_price=mid_price,
        spread=spread
    )
    
    viz = OrderBookViz()
    return viz.create_market_depth_chart(snapshot, **kwargs)