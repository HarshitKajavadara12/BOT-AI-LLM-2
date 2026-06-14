"""
P&L Waterfall Chart Components
=============================

Advanced P&L waterfall visualization components for portfolio performance
analysis, trade attribution, and profit/loss decomposition.

Features:
- Multi-level P&L waterfall charts
- Trade-by-trade contribution analysis
- Factor attribution waterfalls
- Portfolio decomposition charts
- Performance bridge analysis
- Interactive drill-down capabilities
- Custom styling and annotations
- Export and reporting features

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class WaterfallType(Enum):
    """Types of waterfall charts."""
    STANDARD = "standard"
    CUMULATIVE = "cumulative"
    BRIDGE = "bridge"
    ATTRIBUTION = "attribution"

@dataclass
class WaterfallItem:
    """Individual item in waterfall chart."""
    name: str
    value: float
    category: str = "component"
    color: Optional[str] = None
    is_total: bool = False
    description: Optional[str] = None

@dataclass
class WaterfallConfig:
    """Configuration for waterfall charts."""
    show_connectors: bool = True
    show_totals: bool = True
    color_positive: str = '#10b981'
    color_negative: str = '#ef4444'
    color_total: str = '#6366f1'
    color_neutral: str = '#6b7280'
    connector_color: str = '#94a3b8'

class PnLWaterfall:
    """
    Advanced P&L waterfall chart components.
    
    Provides comprehensive visualization tools for P&L analysis,
    trade attribution, and performance decomposition.
    """
    
    def __init__(self, config: Optional[WaterfallConfig] = None):
        """Initialize P&L waterfall components."""
        self.config = config or WaterfallConfig()
        
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'family': "Arial, sans-serif", 'size': 12},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'hovermode': 'x unified'
        }
    
    def create_pnl_waterfall(
        self,
        items: List[WaterfallItem],
        title: str = "P&L Waterfall",
        starting_value: float = 0,
        **kwargs
    ) -> go.Figure:
        """
        Create P&L waterfall chart.
        
        Args:
            items: List of waterfall items
            title: Chart title
            starting_value: Starting value for waterfall
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if not items:
                return self._create_error_chart("No P&L data provided")
            
            fig = go.Figure()
            
            # Prepare data
            names = ['Starting Position'] + [item.name for item in items] + ['Final Position']
            values = [starting_value] + [item.value for item in items] + [0]  # Final will be calculated
            
            # Calculate final position
            final_position = starting_value + sum(item.value for item in items)
            values[-1] = final_position
            
            # Prepare measure types
            measures = ['absolute'] + ['relative'] * len(items) + ['total']
            
            # Prepare colors
            colors = [self.config.color_total]  # Starting position
            
            for item in items:
                if item.color:
                    colors.append(item.color)
                elif item.value > 0:
                    colors.append(self.config.color_positive)
                elif item.value < 0:
                    colors.append(self.config.color_negative)
                else:
                    colors.append(self.config.color_neutral)
            
            colors.append(self.config.color_total)  # Final position
            
            # Create waterfall trace
            fig.add_trace(
                go.Waterfall(
                    name="P&L Components",
                    orientation="v",
                    measure=measures,
                    x=names,
                    y=values,
                    connector={
                        "line": {
                            "color": self.config.connector_color,
                            "dash": "dot",
                            "width": 2
                        },
                        "visible": self.config.show_connectors
                    },
                    increasing={
                        "marker": {"color": self.config.color_positive}
                    },
                    decreasing={
                        "marker": {"color": self.config.color_negative}
                    },
                    totals={
                        "marker": {"color": self.config.color_total}
                    },
                    hovertemplate='%{x}<br>Value: %{y:$,.0f}<br>Running Total: %{cumulativevalue:$,.0f}<extra></extra>'
                )
            )
            
            # Add value annotations
            cumulative = starting_value
            y_positions = [starting_value]
            
            for item in items:
                cumulative += item.value
                y_positions.append(cumulative)
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=title,
                xaxis=dict(
                    tickangle=45,
                    title="Components"
                ),
                yaxis=dict(
                    title="P&L ($)",
                    tickformat="$,.0f"
                ),
                height=500,
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating P&L waterfall: {str(e)}")
            return self._create_error_chart(f"Error creating P&L waterfall: {str(e)}")
    
    def create_trade_attribution_waterfall(
        self,
        trades_data: pd.DataFrame,
        title: str = "Trade Attribution Waterfall",
        **kwargs
    ) -> go.Figure:
        """
        Create trade-by-trade attribution waterfall.
        
        Args:
            trades_data: DataFrame with trade data (symbol, pnl, quantity, etc.)
            title: Chart title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if trades_data.empty:
                return self._create_error_chart("No trade data provided")
            
            # Ensure required columns exist
            if 'pnl' not in trades_data.columns:
                return self._create_error_chart("Trade data must contain 'pnl' column")
            
            # Sort trades by P&L impact
            trades_sorted = trades_data.sort_values('pnl', ascending=False)
            
            # Create waterfall items
            items = []
            for _, trade in trades_sorted.iterrows():
                symbol = trade.get('symbol', f'Trade {len(items)+1}')
                pnl = trade['pnl']
                
                # Determine category and color
                if pnl > 0:
                    category = "Winner"
                    color = self.config.color_positive
                elif pnl < 0:
                    category = "Loser"
                    color = self.config.color_negative
                else:
                    category = "Breakeven"
                    color = self.config.color_neutral
                
                items.append(WaterfallItem(
                    name=symbol,
                    value=pnl,
                    category=category,
                    color=color,
                    description=f"P&L: ${pnl:,.2f}"
                ))
            
            # Create the waterfall
            fig = self.create_pnl_waterfall(items, title, **kwargs)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating trade attribution waterfall: {str(e)}")
            return self._create_error_chart(f"Error creating trade attribution waterfall: {str(e)}")
    
    def create_factor_attribution_waterfall(
        self,
        factor_contributions: Dict[str, float],
        title: str = "Factor Attribution Waterfall",
        starting_value: float = 0,
        **kwargs
    ) -> go.Figure:
        """
        Create factor attribution waterfall chart.
        
        Args:
            factor_contributions: Dictionary of factor names and contributions
            title: Chart title
            starting_value: Starting portfolio value
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if not factor_contributions:
                return self._create_error_chart("No factor contribution data provided")
            
            # Sort factors by contribution magnitude
            sorted_factors = sorted(factor_contributions.items(), 
                                  key=lambda x: abs(x[1]), reverse=True)
            
            # Create waterfall items
            items = []
            for factor_name, contribution in sorted_factors:
                items.append(WaterfallItem(
                    name=factor_name,
                    value=contribution,
                    category="Factor",
                    description=f"Contribution: ${contribution:,.2f}"
                ))
            
            # Create the waterfall
            fig = self.create_pnl_waterfall(items, title, starting_value, **kwargs)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating factor attribution waterfall: {str(e)}")
            return self._create_error_chart(f"Error creating factor attribution waterfall: {str(e)}")
    
    def create_portfolio_bridge_chart(
        self,
        period_data: Dict[str, float],
        title: str = "Portfolio Performance Bridge",
        **kwargs
    ) -> go.Figure:
        """
        Create portfolio performance bridge chart.
        
        Args:
            period_data: Dictionary with period performance components
            title: Chart title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if not period_data:
                return self._create_error_chart("No period data provided")
            
            # Expected keys for portfolio bridge
            expected_components = [
                'starting_nav', 'trading_pnl', 'funding_costs', 
                'management_fees', 'performance_fees', 'other_expenses',
                'dividends', 'interest_income'
            ]
            
            # Create items from available data
            items = []
            starting_nav = period_data.get('starting_nav', 0)
            
            component_mapping = {
                'trading_pnl': 'Trading P&L',
                'funding_costs': 'Funding Costs',
                'management_fees': 'Management Fees',
                'performance_fees': 'Performance Fees',
                'other_expenses': 'Other Expenses',
                'dividends': 'Dividends',
                'interest_income': 'Interest Income'
            }
            
            for key, display_name in component_mapping.items():
                if key in period_data and period_data[key] != 0:
                    items.append(WaterfallItem(
                        name=display_name,
                        value=period_data[key],
                        category="Performance Component"
                    ))
            
            if not items:
                return self._create_error_chart("No meaningful performance components found")
            
            # Create the waterfall
            fig = self.create_pnl_waterfall(items, title, starting_nav, **kwargs)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating portfolio bridge chart: {str(e)}")
            return self._create_error_chart(f"Error creating portfolio bridge chart: {str(e)}")
    
    def create_multi_period_waterfall(
        self,
        period_data: Dict[str, Dict[str, float]],
        title: str = "Multi-Period P&L Analysis",
        **kwargs
    ) -> go.Figure:
        """
        Create multi-period waterfall analysis.
        
        Args:
            period_data: Dictionary of periods with component data
            title: Chart title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if not period_data:
                return self._create_error_chart("No multi-period data provided")
            
            periods = list(period_data.keys())
            n_periods = len(periods)
            
            fig = make_subplots(
                rows=n_periods,
                cols=1,
                subplot_titles=[f"Period: {period}" for period in periods],
                vertical_spacing=0.05
            )
            
            for i, (period, data) in enumerate(period_data.items()):
                # Create items for this period
                items = []
                for component, value in data.items():
                    if component != 'starting_value' and value != 0:
                        items.append(WaterfallItem(
                            name=component.replace('_', ' ').title(),
                            value=value
                        ))
                
                starting_value = data.get('starting_value', 0)
                
                if items:
                    # Create waterfall for this period
                    names = ['Start'] + [item.name for item in items] + ['End']
                    values = [starting_value] + [item.value for item in items] + [0]
                    values[-1] = starting_value + sum(item.value for item in items)
                    
                    measures = ['absolute'] + ['relative'] * len(items) + ['total']
                    
                    fig.add_trace(
                        go.Waterfall(
                            name=f"Period {i+1}",
                            orientation="v",
                            measure=measures,
                            x=names,
                            y=values,
                            showlegend=False,
                            connector={"line": {"color": self.config.connector_color}},
                            increasing={"marker": {"color": self.config.color_positive}},
                            decreasing={"marker": {"color": self.config.color_negative}},
                            totals={"marker": {"color": self.config.color_total}}
                        ),
                        row=i+1, col=1
                    )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=title,
                height=300 * n_periods,
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating multi-period waterfall: {str(e)}")
            return self._create_error_chart(f"Error creating multi-period waterfall: {str(e)}")
    
    def create_sector_attribution_waterfall(
        self,
        sector_data: pd.DataFrame,
        title: str = "Sector Attribution Waterfall",
        **kwargs
    ) -> go.Figure:
        """
        Create sector attribution waterfall chart.
        
        Args:
            sector_data: DataFrame with sector attribution data
            title: Chart title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if sector_data.empty:
                return self._create_error_chart("No sector data provided")
            
            required_cols = ['sector', 'contribution']
            if not all(col in sector_data.columns for col in required_cols):
                return self._create_error_chart(f"Data must contain columns: {required_cols}")
            
            # Sort by contribution
            sector_sorted = sector_data.sort_values('contribution', ascending=False)
            
            # Create waterfall items
            items = []
            for _, row in sector_sorted.iterrows():
                items.append(WaterfallItem(
                    name=row['sector'],
                    value=row['contribution'],
                    category="Sector",
                    description=f"Sector: {row['sector']}, Contribution: ${row['contribution']:,.2f}"
                ))
            
            # Create the waterfall
            fig = self.create_pnl_waterfall(items, title, **kwargs)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating sector attribution waterfall: {str(e)}")
            return self._create_error_chart(f"Error creating sector attribution waterfall: {str(e)}")
    
    def create_cumulative_pnl_waterfall(
        self,
        daily_pnl: pd.Series,
        title: str = "Cumulative P&L Waterfall",
        max_bars: int = 20,
        **kwargs
    ) -> go.Figure:
        """
        Create cumulative P&L waterfall showing daily contributions.
        
        Args:
            daily_pnl: Series with daily P&L values
            title: Chart title
            max_bars: Maximum number of bars to show
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if daily_pnl.empty:
                return self._create_error_chart("No daily P&L data provided")
            
            # Take last N days or aggregate if too many
            if len(daily_pnl) > max_bars:
                # Aggregate older periods
                recent_days = daily_pnl.tail(max_bars - 1)
                older_sum = daily_pnl.head(len(daily_pnl) - (max_bars - 1)).sum()
                
                # Create aggregated series
                aggregated_pnl = pd.Series([older_sum], index=['Prior Period'])
                aggregated_pnl = pd.concat([aggregated_pnl, recent_days])
            else:
                aggregated_pnl = daily_pnl
            
            # Create waterfall items
            items = []
            for date, pnl in aggregated_pnl.items():
                if isinstance(date, str):
                    date_str = date
                else:
                    date_str = date.strftime('%m/%d') if hasattr(date, 'strftime') else str(date)
                
                items.append(WaterfallItem(
                    name=date_str,
                    value=pnl,
                    category="Daily P&L",
                    description=f"Date: {date_str}, P&L: ${pnl:,.2f}"
                ))
            
            # Create the waterfall
            fig = self.create_pnl_waterfall(items, title, **kwargs)
            
            # Customize for time series
            fig.update_xaxes(tickangle=45)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating cumulative P&L waterfall: {str(e)}")
            return self._create_error_chart(f"Error creating cumulative P&L waterfall: {str(e)}")
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart when data processing fails."""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Waterfall Error:<br>{error_message}",
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

# Convenience functions for quick waterfall creation
def quick_pnl_waterfall(components: Dict[str, float], starting_value: float = 0, **kwargs) -> go.Figure:
    """Quick P&L waterfall creation from dictionary."""
    items = [WaterfallItem(name=name, value=value) for name, value in components.items()]
    waterfall = PnLWaterfall()
    return waterfall.create_pnl_waterfall(items, starting_value=starting_value, **kwargs)

def quick_trade_waterfall(trades_df: pd.DataFrame, **kwargs) -> go.Figure:
    """Quick trade attribution waterfall creation."""
    waterfall = PnLWaterfall()
    return waterfall.create_trade_attribution_waterfall(trades_df, **kwargs)

def quick_factor_waterfall(factors: Dict[str, float], **kwargs) -> go.Figure:
    """Quick factor attribution waterfall creation."""
    waterfall = PnLWaterfall()
    return waterfall.create_factor_attribution_waterfall(factors, **kwargs)