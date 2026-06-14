"""
Live Ticker Components
=====================

Real-time price ticker and market data display components for
financial dashboards and trading interfaces.

Features:
- Real-time price tickers with color-coded changes
- Market status indicators and trading session info
- Multi-asset price streaming displays
- Customizable ticker layouts and styling
- Alert integration for price movements
- Volume and volatility indicators
- Market breadth and sector rotation displays
- News ticker integration capabilities

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
from enum import Enum
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class MarketStatus(Enum):
    """Market session status."""
    PRE_MARKET = "pre_market"
    OPEN = "open"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"

@dataclass
class TickerData:
    """Container for ticker data."""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open_price: float
    last_trade_time: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None

@dataclass
class TickerConfig:
    """Configuration for ticker display."""
    update_interval: int = 1000  # milliseconds
    show_change_arrows: bool = True
    show_volume: bool = True
    show_market_cap: bool = False
    color_positive: str = '#10b981'
    color_negative: str = '#ef4444'
    color_neutral: str = '#6b7280'
    background_color: str = '#f8fafc'
    text_color: str = '#1f2937'

class LiveTicker:
    """
    Live ticker components for real-time market data display.
    
    Provides comprehensive real-time ticker displays with customizable
    layouts, styling, and integration capabilities.
    """
    
    def __init__(self, config: Optional[TickerConfig] = None):
        """Initialize live ticker components."""
        self.config = config or TickerConfig()
        self.last_prices = {}  # For calculating price changes
        
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'family': "Arial, monospace", 'size': 11},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': dict(l=10, r=10, t=30, b=10)
        }
    
    def generate_sample_ticker_data(self, symbols: List[str]) -> List[TickerData]:
        """Generate sample ticker data for demonstration."""
        try:
            ticker_data = []
            current_time = datetime.now()
            
            for symbol in symbols:
                # Generate realistic price movement
                base_price = np.random.uniform(50, 500)
                
                # Simulate price with some persistence
                if symbol in self.last_prices:
                    # Add some momentum to price changes
                    change_factor = np.random.normal(0, 0.02)  # 2% volatility
                    new_price = self.last_prices[symbol]['price'] * (1 + change_factor)
                else:
                    new_price = base_price
                
                # Calculate daily change (from theoretical open)
                open_price = new_price * np.random.uniform(0.95, 1.05)
                change = new_price - open_price
                change_percent = (change / open_price) * 100
                
                # Generate other market data
                high = max(new_price, open_price) * np.random.uniform(1.0, 1.03)
                low = min(new_price, open_price) * np.random.uniform(0.97, 1.0)
                volume = np.random.randint(100000, 10000000)
                
                # Bid/ask spread
                spread_pct = np.random.uniform(0.001, 0.01)  # 0.1% to 1%
                spread = new_price * spread_pct
                bid = new_price - spread / 2
                ask = new_price + spread / 2
                
                ticker = TickerData(
                    symbol=symbol,
                    price=new_price,
                    change=change,
                    change_percent=change_percent,
                    volume=volume,
                    high=high,
                    low=low,
                    open_price=open_price,
                    last_trade_time=current_time - timedelta(seconds=np.random.randint(1, 30)),
                    bid=bid,
                    ask=ask,
                    market_cap=new_price * np.random.randint(100000000, 1000000000),  # Shares outstanding
                    pe_ratio=np.random.uniform(10, 30)
                )
                
                ticker_data.append(ticker)
                
                # Store for next update
                self.last_prices[symbol] = {
                    'price': new_price,
                    'timestamp': current_time
                }
            
            return ticker_data
            
        except Exception as e:
            logger.error(f"Error generating ticker data: {str(e)}")
            return []
    
    def create_horizontal_ticker(
        self,
        ticker_data: List[TickerData],
        title: str = "Market Ticker",
        **kwargs
    ) -> go.Figure:
        """
        Create horizontal scrolling ticker display.
        
        Args:
            ticker_data: List of ticker data objects
            title: Ticker title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if not ticker_data:
                return self._create_error_chart("No ticker data available")
            
            fig = go.Figure()
            
            # Prepare data for display
            x_positions = list(range(len(ticker_data)))
            symbols = [ticker.symbol for ticker in ticker_data]
            prices = [ticker.price for ticker in ticker_data]
            changes = [ticker.change_percent for ticker in ticker_data]
            
            # Create color map based on price changes
            colors = []
            for change in changes:
                if change > 0:
                    colors.append(self.config.color_positive)
                elif change < 0:
                    colors.append(self.config.color_negative)
                else:
                    colors.append(self.config.color_neutral)
            
            # Add price bars
            fig.add_trace(
                go.Bar(
                    x=x_positions,
                    y=changes,
                    marker_color=colors,
                    text=[f"{symbol}<br>${price:.2f}<br>{change:+.2f}%" 
                          for symbol, price, change in zip(symbols, prices, changes)],
                    textposition="auto",
                    textfont=dict(color="white", size=10),
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False
                )
            )
            
            # Customize layout for ticker
            fig.update_layout(
                **self.default_layout,
                title=title,
                xaxis=dict(
                    tickmode='array',
                    tickvals=x_positions,
                    ticktext=symbols,
                    showgrid=False
                ),
                yaxis=dict(
                    title="Change %",
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    zeroline=True,
                    zerolinecolor='black',
                    zerolinewidth=1
                ),
                height=200,
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating horizontal ticker: {str(e)}")
            return self._create_error_chart(f"Error creating horizontal ticker: {str(e)}")
    
    def create_price_grid_ticker(
        self,
        ticker_data: List[TickerData],
        columns: int = 4,
        title: str = "Price Grid",
        **kwargs
    ) -> go.Figure:
        """
        Create grid-based ticker display.
        
        Args:
            ticker_data: List of ticker data objects
            columns: Number of columns in grid
            title: Ticker title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if not ticker_data:
                return self._create_error_chart("No ticker data available")
            
            # Calculate grid dimensions
            rows = (len(ticker_data) + columns - 1) // columns
            
            fig = make_subplots(
                rows=rows,
                cols=columns,
                subplot_titles=[ticker.symbol for ticker in ticker_data[:rows*columns]],
                vertical_spacing=0.1,
                horizontal_spacing=0.1
            )
            
            for i, ticker in enumerate(ticker_data):
                row = (i // columns) + 1
                col = (i % columns) + 1
                
                if row > rows:
                    break
                
                # Determine color based on change
                if ticker.change_percent > 0:
                    color = self.config.color_positive
                    arrow = " "
                elif ticker.change_percent < 0:
                    color = self.config.color_negative
                    arrow = " "
                else:
                    color = self.config.color_neutral
                    arrow = "→"
                
                # Add gauge-like indicator
                fig.add_trace(
                    go.Indicator(
                        mode="number+delta",
                        value=ticker.price,
                        delta={
                            'reference': ticker.open_price,
                            'position': "bottom",
                            'valueformat': '.2f'
                        },
                        number={
                            'prefix': "$",
                            'font': {'size': 16, 'color': color}
                        },
                        title={
                            'text': f"{ticker.symbol}<br>{arrow} {ticker.change_percent:+.2f}%",
                            'font': {'size': 12}
                        },
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ),
                    row=row, col=col
                )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=title,
                height=150 * rows,
                showlegend=False,
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price grid ticker: {str(e)}")
            return self._create_error_chart(f"Error creating price grid ticker: {str(e)}")
    
    def create_market_overview_ticker(
        self,
        ticker_data: List[TickerData],
        indices_data: Optional[List[TickerData]] = None,
        title: str = "Market Overview",
        **kwargs
    ) -> go.Figure:
        """
        Create comprehensive market overview ticker.
        
        Args:
            ticker_data: List of individual stock ticker data
            indices_data: Optional list of market indices data
            title: Ticker title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if not ticker_data:
                return self._create_error_chart("No ticker data available")
            
            # Calculate market statistics
            total_volume = sum(ticker.volume for ticker in ticker_data)
            avg_change = np.mean([ticker.change_percent for ticker in ticker_data])
            advancing = sum(1 for ticker in ticker_data if ticker.change_percent > 0)
            declining = sum(1 for ticker in ticker_data if ticker.change_percent < 0)
            unchanged = len(ticker_data) - advancing - declining
            
            # Create subplots
            fig = make_subplots(
                rows=2 if indices_data else 1,
                cols=2,
                subplot_titles=[
                    "Market Breadth",
                    "Top Movers",
                    "Major Indices" if indices_data else "",
                    "Volume Leaders"
                ],
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]] if indices_data else 
                      [[{"type": "pie"}, {"type": "bar"}]]
            )
            
            # Market breadth pie chart
            fig.add_trace(
                go.Pie(
                    labels=["Advancing", "Declining", "Unchanged"],
                    values=[advancing, declining, unchanged],
                    marker_colors=[self.config.color_positive, 
                                 self.config.color_negative, 
                                 self.config.color_neutral],
                    textinfo="label+percent",
                    hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Top movers
            sorted_tickers = sorted(ticker_data, key=lambda x: abs(x.change_percent), reverse=True)[:10]
            mover_symbols = [ticker.symbol for ticker in sorted_tickers]
            mover_changes = [ticker.change_percent for ticker in sorted_tickers]
            mover_colors = [self.config.color_positive if x > 0 else self.config.color_negative 
                           for x in mover_changes]
            
            fig.add_trace(
                go.Bar(
                    x=mover_symbols,
                    y=mover_changes,
                    marker_color=mover_colors,
                    name="Top Movers",
                    hovertemplate='Symbol: %{x}<br>Change: %{y:.2f}%<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Major indices (if provided)
            if indices_data:
                index_symbols = [idx.symbol for idx in indices_data]
                index_changes = [idx.change_percent for idx in indices_data]
                index_colors = [self.config.color_positive if x > 0 else self.config.color_negative 
                               for x in index_changes]
                
                fig.add_trace(
                    go.Bar(
                        x=index_symbols,
                        y=index_changes,
                        marker_color=index_colors,
                        name="Indices",
                        hovertemplate='Index: %{x}<br>Change: %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Volume leaders
            volume_leaders = sorted(ticker_data, key=lambda x: x.volume, reverse=True)[:10]
            volume_symbols = [ticker.symbol for ticker in volume_leaders]
            volume_values = [ticker.volume / 1000000 for ticker in volume_leaders]  # In millions
            
            fig.add_trace(
                go.Bar(
                    x=volume_symbols,
                    y=volume_values,
                    marker_color=self.config.color_neutral,
                    name="Volume Leaders",
                    hovertemplate='Symbol: %{x}<br>Volume: %{y:.1f}M<extra></extra>'
                ),
                row=2 if indices_data else 1, col=2
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=title,
                height=600 if indices_data else 400,
                showlegend=False,
                **kwargs
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Change %", row=1, col=2)
            if indices_data:
                fig.update_yaxes(title_text="Change %", row=2, col=1)
            fig.update_yaxes(title_text="Volume (M)", row=2 if indices_data else 1, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating market overview ticker: {str(e)}")
            return self._create_error_chart(f"Error creating market overview ticker: {str(e)}")
    
    def create_real_time_price_chart(
        self,
        symbol: str,
        price_history: pd.DataFrame,
        current_ticker: TickerData,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create real-time price chart with ticker information.
        
        Args:
            symbol: Stock symbol
            price_history: DataFrame with historical price data
            current_ticker: Current ticker data
            title: Chart title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if price_history.empty:
                return self._create_error_chart("No price history available")
            
            if title is None:
                title = f"{symbol} - Real-time Price"
            
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=[title, 'Volume'],
                vertical_spacing=0.1
            )
            
            # Price line
            fig.add_trace(
                go.Scatter(
                    x=price_history.index,
                    y=price_history['price'] if 'price' in price_history.columns else price_history['close'],
                    mode='lines',
                    name='Price',
                    line=dict(
                        color=self.config.color_positive if current_ticker.change >= 0 else self.config.color_negative,
                        width=2
                    ),
                    hovertemplate='Time: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Current price point
            fig.add_trace(
                go.Scatter(
                    x=[current_ticker.last_trade_time],
                    y=[current_ticker.price],
                    mode='markers',
                    name='Current Price',
                    marker=dict(
                        color=self.config.color_positive if current_ticker.change >= 0 else self.config.color_negative,
                        size=10,
                        symbol='circle'
                    ),
                    hovertemplate=f'Current: ${current_ticker.price:.2f}<br>Change: {current_ticker.change_percent:+.2f}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Volume bars (if available)
            if 'volume' in price_history.columns:
                fig.add_trace(
                    go.Bar(
                        x=price_history.index,
                        y=price_history['volume'],
                        name='Volume',
                        marker_color=self.config.color_neutral,
                        opacity=0.6,
                        hovertemplate='Time: %{x}<br>Volume: %{y:,}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Add annotations for key levels
            fig.add_hline(
                y=current_ticker.high,
                line_dash="dash",
                line_color="green",
                annotation_text=f"High: ${current_ticker.high:.2f}",
                row=1, col=1
            )
            
            fig.add_hline(
                y=current_ticker.low,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Low: ${current_ticker.low:.2f}",
                row=1, col=1
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                height=500,
                **kwargs
            )
            
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_xaxes(title_text="Time", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating real-time price chart: {str(e)}")
            return self._create_error_chart(f"Error creating real-time price chart: {str(e)}")
    
    def create_sector_rotation_ticker(
        self,
        sector_data: Dict[str, float],
        title: str = "Sector Rotation",
        **kwargs
    ) -> go.Figure:
        """
        Create sector rotation ticker display.
        
        Args:
            sector_data: Dictionary of sector names and performance
            title: Ticker title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if not sector_data:
                return self._create_error_chart("No sector data available")
            
            # Sort sectors by performance
            sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1], reverse=True)
            
            sectors = [item[0] for item in sorted_sectors]
            performance = [item[1] for item in sorted_sectors]
            
            # Color code by performance
            colors = [self.config.color_positive if perf > 0 else self.config.color_negative 
                     for perf in performance]
            
            fig = go.Figure()
            
            # Create horizontal bar chart
            fig.add_trace(
                go.Bar(
                    x=performance,
                    y=sectors,
                    orientation='h',
                    marker_color=colors,
                    text=[f"{perf:+.2f}%" for perf in performance],
                    textposition="auto",
                    textfont=dict(color="white"),
                    hovertemplate='Sector: %{y}<br>Performance: %{x:.2f}%<extra></extra>'
                )
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=title,
                xaxis_title="Performance (%)",
                yaxis_title="Sectors",
                height=max(300, len(sectors) * 25),
                **kwargs
            )
            
            # Add zero line
            fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sector rotation ticker: {str(e)}")
            return self._create_error_chart(f"Error creating sector rotation ticker: {str(e)}")
    
    def get_market_status(self) -> MarketStatus:
        """Determine current market status based on time."""
        try:
            current_time = datetime.now()
            hour = current_time.hour
            minute = current_time.minute
            
            # Simple market hours (can be enhanced with timezone handling)
            if hour < 9 or (hour == 9 and minute < 30):
                return MarketStatus.PRE_MARKET
            elif (hour > 16) or (hour == 16 and minute > 0):
                return MarketStatus.AFTER_HOURS
            elif hour < 7:  # Weekend or very early
                return MarketStatus.CLOSED
            else:
                return MarketStatus.OPEN
                
        except Exception as e:
            logger.error(f"Error determining market status: {str(e)}")
            return MarketStatus.CLOSED
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart when data processing fails."""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Ticker Error:<br>{error_message}",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            **self.default_layout,
            height=300,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig

# Convenience functions for quick ticker creation
def quick_horizontal_ticker(symbols: List[str], **kwargs) -> go.Figure:
    """Quick horizontal ticker creation."""
    ticker = LiveTicker()
    ticker_data = ticker.generate_sample_ticker_data(symbols)
    return ticker.create_horizontal_ticker(ticker_data, **kwargs)

def quick_price_grid(symbols: List[str], columns: int = 4, **kwargs) -> go.Figure:
    """Quick price grid ticker creation."""
    ticker = LiveTicker()
    ticker_data = ticker.generate_sample_ticker_data(symbols)
    return ticker.create_price_grid_ticker(ticker_data, columns, **kwargs)

def quick_market_overview(symbols: List[str], **kwargs) -> go.Figure:
    """Quick market overview ticker creation."""
    ticker = LiveTicker()
    ticker_data = ticker.generate_sample_ticker_data(symbols)
    return ticker.create_market_overview_ticker(ticker_data, **kwargs)