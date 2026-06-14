"""
Advanced Plotly Chart Components
==============================

Reusable high-performance Plotly chart components for financial data visualization.
Provides standardized charting components with consistent styling and advanced features.

Features:
- Real-time updating financial charts (candlestick, line, bar)
- Advanced technical indicators and overlays
- Interactive crossfilter and zoom capabilities
- Professional styling and theming
- High-performance rendering for large datasets
- Customizable annotations and alerts
- Multi-timeframe analysis support
- Export capabilities (PNG, SVG, HTML)

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
class ChartConfig:
    """Configuration for chart styling and behavior."""
    theme: str = 'plotly_white'
    color_palette: List[str] = None
    height: int = 400
    show_toolbar: bool = True
    show_legend: bool = True
    auto_refresh: bool = False
    refresh_interval: int = 1000  # milliseconds

    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = [
                '#6366f1', '#8b5cf6', '#ec4899', '#f59e0b',
                '#10b981', '#3b82f6', '#ef4444', '#6b7280'
            ]

class AdvancedPlotlyCharts:
    """
    Advanced Plotly chart components for financial data visualization.
    
    Provides high-performance, reusable chart components with consistent
    styling and advanced financial charting capabilities.
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """Initialize chart components with configuration."""
        self.config = config or ChartConfig()
        self._setup_default_layout()
        
    def _setup_default_layout(self):
        """Setup default layout configuration."""
        self.default_layout = {
            'template': self.config.theme,
            'height': self.config.height,
            'showlegend': self.config.show_legend,
            'hovermode': 'x unified',
            'xaxis': {
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': 'rgba(128, 128, 128, 0.2)',
                'showspikes': True,
                'spikecolor': "orange",
                'spikesnap': "cursor",
                'spikemode': "across"
            },
            'yaxis': {
                'showgrid': True,
                'gridwidth': 1,
                'gridcolor': 'rgba(128, 128, 128, 0.2)',
                'showspikes': True,
                'spikecolor': "orange",
                'spikesnap': "cursor",
                'spikemode': "across"
            },
            'font': {
                'family': "Arial, sans-serif",
                'size': 12,
                'color': "#2d3748"
            },
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': dict(l=50, r=50, t=50, b=50)
        }
    
    def create_candlestick_chart(
        self,
        data: pd.DataFrame,
        title: str = "Price Chart",
        volume: bool = True,
        indicators: Optional[List[str]] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create advanced candlestick chart with volume and technical indicators.
        
        Args:
            data: DataFrame with OHLCV data
            title: Chart title
            volume: Whether to include volume subplot
            indicators: List of technical indicators to add
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            # Validate data
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in data.columns for col in required_cols):
                raise ValueError(f"Data must contain columns: {required_cols}")
            
            # Create subplots
            subplot_specs = [[{"secondary_y": True}]]
            if volume and 'Volume' in data.columns:
                subplot_specs.append([{"secondary_y": False}])
            
            fig = make_subplots(
                rows=len(subplot_specs),
                cols=1,
                subplot_titles=[title] + (['Volume'] if volume and 'Volume' in data.columns else []),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3] if volume and 'Volume' in data.columns else [1.0],
                specs=subplot_specs
            )
            
            # Add candlestick
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color='#10b981',
                    decreasing_line_color='#ef4444',
                    increasing_fillcolor='rgba(16, 185, 129, 0.7)',
                    decreasing_fillcolor='rgba(239, 68, 68, 0.7)'
                ),
                row=1, col=1
            )
            
            # Add technical indicators
            if indicators:
                self._add_technical_indicators(fig, data, indicators, row=1)
            
            # Add volume
            if volume and 'Volume' in data.columns:
                colors = ['#10b981' if close >= open else '#ef4444' 
                         for close, open in zip(data['Close'], data['Open'])]
                
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(**self.default_layout)
            fig.update_layout(
                title=title,
                xaxis_rangeslider_visible=False,
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {str(e)}")
            return self._create_error_chart(f"Error creating candlestick chart: {str(e)}")
    
    def create_line_chart(
        self,
        data: Union[pd.DataFrame, pd.Series],
        title: str = "Line Chart",
        x_column: Optional[str] = None,
        y_columns: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create multi-line chart with advanced styling.
        
        Args:
            data: DataFrame or Series with data
            title: Chart title
            x_column: Column name for x-axis (default: index)
            y_columns: List of column names for y-axis
            colors: Custom colors for lines
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            fig = go.Figure()
            
            if isinstance(data, pd.Series):
                # Single series
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data.values,
                        mode='lines',
                        name=data.name or 'Value',
                        line=dict(
                            color=colors[0] if colors else self.config.color_palette[0],
                            width=2
                        ),
                        hovertemplate='%{x}<br>%{y:.2f}<extra></extra>'
                    )
                )
            else:
                # DataFrame with multiple columns
                x_data = data[x_column] if x_column else data.index
                plot_columns = y_columns or [col for col in data.select_dtypes(include=[np.number]).columns]
                
                for i, col in enumerate(plot_columns):
                    color = colors[i] if colors and i < len(colors) else self.config.color_palette[i % len(self.config.color_palette)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=data[col],
                            mode='lines',
                            name=col,
                            line=dict(color=color, width=2),
                            hovertemplate=f'{col}: %{{y:.2f}}<extra></extra>'
                        )
                    )
            
            # Update layout
            fig.update_layout(**self.default_layout)
            fig.update_layout(title=title, **kwargs)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating line chart: {str(e)}")
            return self._create_error_chart(f"Error creating line chart: {str(e)}")
    
    def create_bar_chart(
        self,
        data: Union[pd.DataFrame, pd.Series],
        title: str = "Bar Chart",
        orientation: str = 'v',
        color_column: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create bar chart with advanced styling and color coding.
        
        Args:
            data: DataFrame or Series with data
            title: Chart title
            orientation: 'v' for vertical, 'h' for horizontal
            color_column: Column name for color coding
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            fig = go.Figure()
            
            if isinstance(data, pd.Series):
                # Single series bar chart
                colors = [self.config.color_palette[0]] * len(data)
                if data.min() < 0:  # Color negative values differently
                    colors = ['#10b981' if x >= 0 else '#ef4444' for x in data.values]
                
                fig.add_trace(
                    go.Bar(
                        x=data.index if orientation == 'v' else data.values,
                        y=data.values if orientation == 'v' else data.index,
                        name=data.name or 'Value',
                        marker_color=colors,
                        orientation=orientation,
                        hovertemplate='%{x}<br>%{y:.2f}<extra></extra>' if orientation == 'v' else '%{y}<br>%{x:.2f}<extra></extra>'
                    )
                )
            else:
                # DataFrame bar chart
                for i, col in enumerate(data.select_dtypes(include=[np.number]).columns):
                    color = self.config.color_palette[i % len(self.config.color_palette)]
                    
                    fig.add_trace(
                        go.Bar(
                            x=data.index if orientation == 'v' else data[col],
                            y=data[col] if orientation == 'v' else data.index,
                            name=col,
                            marker_color=color,
                            orientation=orientation,
                            hovertemplate=f'{col}: %{{y:.2f}}<extra></extra>' if orientation == 'v' else f'{col}: %{{x:.2f}}<extra></extra>'
                        )
                    )
            
            # Update layout
            fig.update_layout(**self.default_layout)
            fig.update_layout(title=title, **kwargs)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            return self._create_error_chart(f"Error creating bar chart: {str(e)}")
    
    def create_scatter_plot(
        self,
        data: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str = "Scatter Plot",
        size_column: Optional[str] = None,
        color_column: Optional[str] = None,
        trendline: bool = False,
        **kwargs
    ) -> go.Figure:
        """
        Create scatter plot with optional size and color coding.
        
        Args:
            data: DataFrame with data
            x_column: Column name for x-axis
            y_column: Column name for y-axis
            title: Chart title
            size_column: Column name for marker size
            color_column: Column name for color coding
            trendline: Whether to add trendline
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            fig = go.Figure()
            
            # Prepare marker properties
            marker_props = {
                'color': self.config.color_palette[0],
                'size': 8,
                'opacity': 0.7
            }
            
            if color_column and color_column in data.columns:
                marker_props['color'] = data[color_column]
                marker_props['colorscale'] = 'Viridis'
                marker_props['showscale'] = True
                marker_props['colorbar'] = dict(title=color_column)
            
            if size_column and size_column in data.columns:
                marker_props['size'] = data[size_column]
                marker_props['sizemode'] = 'diameter'
                marker_props['sizeref'] = 2. * max(data[size_column]) / (40.**2)
                marker_props['sizemin'] = 4
            
            # Add scatter trace
            fig.add_trace(
                go.Scatter(
                    x=data[x_column],
                    y=data[y_column],
                    mode='markers',
                    name='Data Points',
                    marker=marker_props,
                    hovertemplate=f'{x_column}: %{{x}}<br>{y_column}: %{{y}}<extra></extra>'
                )
            )
            
            # Add trendline
            if trendline:
                z = np.polyfit(data[x_column].dropna(), data[y_column].dropna(), 1)
                p = np.poly1d(z)
                
                fig.add_trace(
                    go.Scatter(
                        x=data[x_column],
                        y=p(data[x_column]),
                        mode='lines',
                        name='Trendline',
                        line=dict(color='red', dash='dash'),
                        hovertemplate='Trendline<extra></extra>'
                    )
                )
            
            # Update layout
            fig.update_layout(**self.default_layout)
            fig.update_layout(
                title=title,
                xaxis_title=x_column,
                yaxis_title=y_column,
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating scatter plot: {str(e)}")
            return self._create_error_chart(f"Error creating scatter plot: {str(e)}")
    
    def create_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Heatmap",
        color_scale: str = 'RdBu',
        show_values: bool = True,
        **kwargs
    ) -> go.Figure:
        """
        Create correlation heatmap with advanced styling.
        
        Args:
            data: DataFrame with correlation matrix or similar data
            title: Chart title
            color_scale: Color scale for heatmap
            show_values: Whether to show values in cells
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            fig = go.Figure()
            
            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=data.values,
                    x=data.columns,
                    y=data.index,
                    colorscale=color_scale,
                    text=data.round(3).values if show_values else None,
                    texttemplate="%{text}" if show_values else None,
                    textfont={"size": 10},
                    hovertemplate='%{y} vs %{x}<br>Value: %{z:.3f}<extra></extra>',
                    colorbar=dict(title="Value")
                )
            )
            
            # Update layout
            fig.update_layout(**self.default_layout)
            fig.update_layout(
                title=title,
                xaxis=dict(side="bottom"),
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {str(e)}")
            return self._create_error_chart(f"Error creating heatmap: {str(e)}")
    
    def create_waterfall_chart(
        self,
        data: pd.Series,
        title: str = "Waterfall Chart",
        **kwargs
    ) -> go.Figure:
        """
        Create waterfall chart for P&L or other cumulative analysis.
        
        Args:
            data: Series with values for waterfall
            title: Chart title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            fig = go.Figure()
            
            # Calculate cumulative values
            cumulative = data.cumsum()
            
            # Colors for positive and negative values
            colors = ['#10b981' if x >= 0 else '#ef4444' for x in data.values]
            
            # Add waterfall trace
            fig.add_trace(
                go.Waterfall(
                    x=data.index,
                    y=data.values,
                    name="P&L Components",
                    measure=['relative'] * len(data),
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    increasing={"marker": {"color": "#10b981"}},
                    decreasing={"marker": {"color": "#ef4444"}},
                    totals={"marker": {"color": "#6366f1"}},
                    hovertemplate='%{x}<br>Value: %{y:,.0f}<br>Cumulative: %{cumulativevalue:,.0f}<extra></extra>'
                )
            )
            
            # Update layout
            fig.update_layout(**self.default_layout)
            fig.update_layout(title=title, **kwargs)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating waterfall chart: {str(e)}")
            return self._create_error_chart(f"Error creating waterfall chart: {str(e)}")
    
    def create_gauge_chart(
        self,
        value: float,
        title: str = "Gauge",
        min_value: float = 0,
        max_value: float = 100,
        thresholds: Optional[List[Tuple[float, str, str]]] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create gauge chart for KPI visualization.
        
        Args:
            value: Current value to display
            title: Chart title
            min_value: Minimum value for gauge
            max_value: Maximum value for gauge
            thresholds: List of (threshold, color, label) tuples
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            fig = go.Figure()
            
            # Default thresholds
            if thresholds is None:
                thresholds = [
                    (max_value * 0.3, '#ef4444', 'Low'),
                    (max_value * 0.7, '#f59e0b', 'Medium'),
                    (max_value, '#10b981', 'High')
                ]
            
            # Create gauge
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=value,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': title},
                    delta={'reference': max_value * 0.5},
                    gauge={
                        'axis': {'range': [None, max_value]},
                        'bar': {'color': self.config.color_palette[0]},
                        'steps': [
                            {'range': [min_value, thresholds[0][0]], 'color': 'lightgray'},
                            {'range': [thresholds[0][0], thresholds[1][0]], 'color': 'gray'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': max_value * 0.9
                        }
                    }
                )
            )
            
            # Update layout
            fig.update_layout(
                height=self.config.height,
                font={'color': "#2d3748", 'family': "Arial"},
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating gauge chart: {str(e)}")
            return self._create_error_chart(f"Error creating gauge chart: {str(e)}")
    
    def _add_technical_indicators(self, fig: go.Figure, data: pd.DataFrame, indicators: List[str], row: int = 1):
        """Add technical indicators to existing chart."""
        try:
            for indicator in indicators:
                if indicator.upper() == 'SMA20':
                    sma20 = data['Close'].rolling(window=20).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=sma20,
                            mode='lines',
                            name='SMA(20)',
                            line=dict(color='orange', width=1)
                        ),
                        row=row, col=1
                    )
                
                elif indicator.upper() == 'SMA50':
                    sma50 = data['Close'].rolling(window=50).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=sma50,
                            mode='lines',
                            name='SMA(50)',
                            line=dict(color='purple', width=1)
                        ),
                        row=row, col=1
                    )
                
                elif indicator.upper() == 'BOLLINGER':
                    sma20 = data['Close'].rolling(window=20).mean()
                    std20 = data['Close'].rolling(window=20).std()
                    upper_band = sma20 + (std20 * 2)
                    lower_band = sma20 - (std20 * 2)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=upper_band,
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='lightblue', width=1, dash='dash'),
                            showlegend=False
                        ),
                        row=row, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=lower_band,
                            mode='lines',
                            name='BB Lower',
                            line=dict(color='lightblue', width=1, dash='dash'),
                            fill='tonexty',
                            fillcolor='rgba(173, 216, 230, 0.1)',
                            showlegend=False
                        ),
                        row=row, col=1
                    )
                    
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart when data processing fails."""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Chart Error:<br>{error_message}",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        
        fig.update_layout(
            height=self.config.height,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig
    
    def update_chart_theme(self, theme: str):
        """Update chart theme."""
        self.config.theme = theme
        self.default_layout['template'] = theme
    
    def export_chart(self, fig: go.Figure, filename: str, format: str = 'png', **kwargs):
        """Export chart to file."""
        try:
            if format.lower() == 'png':
                fig.write_image(filename, **kwargs)
            elif format.lower() == 'html':
                fig.write_html(filename, **kwargs)
            elif format.lower() == 'svg':
                fig.write_image(filename, format='svg', **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logger.info(f"Chart exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting chart: {str(e)}")

# Convenience functions for quick chart creation
def quick_candlestick(data: pd.DataFrame, title: str = "Price Chart", **kwargs) -> go.Figure:
    """Quick candlestick chart creation."""
    charts = AdvancedPlotlyCharts()
    return charts.create_candlestick_chart(data, title, **kwargs)

def quick_line_chart(data: Union[pd.DataFrame, pd.Series], title: str = "Line Chart", **kwargs) -> go.Figure:
    """Quick line chart creation."""
    charts = AdvancedPlotlyCharts()
    return charts.create_line_chart(data, title, **kwargs)

def quick_bar_chart(data: Union[pd.DataFrame, pd.Series], title: str = "Bar Chart", **kwargs) -> go.Figure:
    """Quick bar chart creation."""
    charts = AdvancedPlotlyCharts()
    return charts.create_bar_chart(data, title, **kwargs)

def quick_scatter_plot(data: pd.DataFrame, x_column: str, y_column: str, title: str = "Scatter Plot", **kwargs) -> go.Figure:
    """Quick scatter plot creation."""
    charts = AdvancedPlotlyCharts()
    return charts.create_scatter_plot(data, x_column, y_column, title, **kwargs)

def quick_heatmap(data: pd.DataFrame, title: str = "Heatmap", **kwargs) -> go.Figure:
    """Quick heatmap creation."""
    charts = AdvancedPlotlyCharts()
    return charts.create_heatmap(data, title, **kwargs)