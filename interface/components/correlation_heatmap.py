"""
Correlation Heatmap Visualization Components
==========================================

Advanced correlation analysis and heatmap visualization components
for financial data, portfolio analysis, and risk management.

Features:
- Interactive correlation heatmaps with clustering
- Time-varying correlation analysis
- Portfolio correlation decomposition
- Hierarchical clustering and dendrogram visualization
- Factor correlation analysis
- Rolling correlation tracking
- Statistical significance testing
- Custom color schemes and styling

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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class CorrelationConfig:
    """Configuration for correlation visualization."""
    method: str = 'pearson'  # pearson, spearman, kendall
    cluster: bool = True
    show_values: bool = True
    show_significance: bool = False
    significance_level: float = 0.05
    color_scale: str = 'RdBu'
    diverging_center: float = 0.0

class CorrelationHeatmap:
    """
    Advanced correlation heatmap visualization components.
    
    Provides comprehensive correlation analysis and visualization tools
    for financial data, portfolio analysis, and risk management.
    """
    
    def __init__(self, config: Optional[CorrelationConfig] = None):
        """Initialize correlation heatmap components."""
        self.config = config or CorrelationConfig()
        
        self.default_layout = {
            'template': 'plotly_white',
            'font': {'family': "Arial, sans-serif", 'size': 11},
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'hovermode': 'closest'
        }
        
        # Color scales for different types of analysis
        self.color_scales = {
            'RdBu': 'RdBu',
            'RdYlBu': 'RdYlBu', 
            'Viridis': 'Viridis',
            'Plasma': 'Plasma',
            'Custom_Financial': [
                [0.0, '#d73027'],    # Strong negative (red)
                [0.25, '#f46d43'],   # Moderate negative (orange)
                [0.5, '#ffffff'],    # No correlation (white)
                [0.75, '#74add1'],   # Moderate positive (light blue)
                [1.0, '#313695']     # Strong positive (dark blue)
            ]
        }
    
    def create_correlation_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Correlation Matrix",
        annotations: bool = True,
        cluster_method: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create correlation heatmap with optional clustering.
        
        Args:
            data: DataFrame with numerical data
            title: Chart title
            annotations: Whether to show correlation values
            cluster_method: Clustering method ('ward', 'complete', 'average')
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            # Calculate correlation matrix
            corr_matrix = data.corr(method=self.config.method)
            
            if corr_matrix.empty:
                return self._create_error_chart("No numerical data for correlation")
            
            # Apply clustering if requested
            if cluster_method or self.config.cluster:
                method = cluster_method or 'ward'
                corr_matrix = self._cluster_correlation_matrix(corr_matrix, method)
            
            # Calculate statistical significance if requested
            p_values = None
            if self.config.show_significance:
                p_values = self._calculate_significance(data)
                if cluster_method or self.config.cluster:
                    p_values = p_values.reindex(corr_matrix.index, axis=0).reindex(corr_matrix.columns, axis=1)
            
            fig = go.Figure()
            
            # Prepare text annotations
            text_values = None
            if annotations or self.config.show_values:
                if self.config.show_significance and p_values is not None:
                    text_values = self._format_text_with_significance(corr_matrix, p_values)
                else:
                    text_values = np.round(corr_matrix.values, 3).astype(str)
            
            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale=self.config.color_scale,
                    zmid=self.config.diverging_center,
                    zmin=-1,
                    zmax=1,
                    text=text_values,
                    texttemplate="%{text}" if text_values is not None else None,
                    textfont={"size": 9},
                    hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
                    colorbar=dict(
                        title="Correlation",
                        titleside="right",
                        tickmode="linear",
                        tick0=-1,
                        dtick=0.5
                    )
                )
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=title,
                xaxis=dict(
                    side="bottom",
                    tickangle=45
                ),
                yaxis=dict(
                    tickmode="linear"
                ),
                height=max(400, len(corr_matrix) * 25),
                width=max(400, len(corr_matrix) * 25),
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return self._create_error_chart(f"Error creating correlation heatmap: {str(e)}")
    
    def create_rolling_correlation_chart(
        self,
        data: pd.DataFrame,
        asset1: str,
        asset2: str,
        window: int = 30,
        title: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        """
        Create rolling correlation chart between two assets.
        
        Args:
            data: DataFrame with price/return data
            asset1: First asset column name
            asset2: Second asset column name
            window: Rolling window size
            title: Chart title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            if asset1 not in data.columns or asset2 not in data.columns:
                return self._create_error_chart(f"Assets {asset1} or {asset2} not found in data")
            
            # Calculate rolling correlation
            rolling_corr = data[asset1].rolling(window=window).corr(data[asset2])
            
            if title is None:
                title = f"Rolling Correlation: {asset1} vs {asset2} ({window}d window)"
            
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=[title, 'Asset Prices (Normalized)'],
                vertical_spacing=0.1
            )
            
            # Add rolling correlation
            fig.add_trace(
                go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr.values,
                    mode='lines',
                    name=f'Rolling Correlation ({window}d)',
                    line=dict(color='#6366f1', width=2),
                    hovertemplate='Date: %{x}<br>Correlation: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add correlation bands
            fig.add_hline(y=0.7, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)
            fig.add_hline(y=0.3, line_dash="dash", line_color="orange", opacity=0.5, row=1, col=1)
            fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5, row=1, col=1)
            fig.add_hline(y=-0.3, line_dash="dash", line_color="orange", opacity=0.5, row=1, col=1)
            fig.add_hline(y=-0.7, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
            
            # Add normalized asset prices
            normalized_data = data[[asset1, asset2]].div(data[[asset1, asset2]].iloc[0])
            
            fig.add_trace(
                go.Scatter(
                    x=normalized_data.index,
                    y=normalized_data[asset1],
                    mode='lines',
                    name=asset1,
                    line=dict(color='#10b981', width=1.5),
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=normalized_data.index,
                    y=normalized_data[asset2],
                    mode='lines',
                    name=asset2,
                    line=dict(color='#ef4444', width=1.5),
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                height=600,
                **kwargs
            )
            
            fig.update_yaxes(title_text="Correlation", range=[-1, 1], row=1, col=1)
            fig.update_yaxes(title_text="Normalized Price", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating rolling correlation chart: {str(e)}")
            return self._create_error_chart(f"Error creating rolling correlation chart: {str(e)}")
    
    def create_correlation_dendrogram(
        self,
        data: pd.DataFrame,
        method: str = 'ward',
        title: str = "Correlation Dendrogram",
        **kwargs
    ) -> go.Figure:
        """
        Create hierarchical clustering dendrogram based on correlations.
        
        Args:
            data: DataFrame with numerical data
            method: Clustering method
            title: Chart title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            # Calculate correlation matrix and distance matrix
            corr_matrix = data.corr(method=self.config.method)
            distance_matrix = 1 - np.abs(corr_matrix)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distance_matrix, method=method)
            
            # Create dendrogram
            dend = dendrogram(linkage_matrix, labels=corr_matrix.columns, no_plot=True)
            
            fig = go.Figure()
            
            # Add dendrogram traces
            for i, d in zip(dend['icoord'], dend['dcoord']):
                fig.add_trace(
                    go.Scatter(
                        x=i,
                        y=d,
                        mode='lines',
                        line=dict(color='#6366f1', width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
            
            # Add labels
            fig.update_layout(
                **self.default_layout,
                title=title,
                xaxis=dict(
                    tickmode='array',
                    tickvals=dend['leaves'],
                    ticktext=[corr_matrix.columns[i] for i in dend['leaves']],
                    tickangle=45
                ),
                yaxis_title="Distance",
                height=500,
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation dendrogram: {str(e)}")
            return self._create_error_chart(f"Error creating correlation dendrogram: {str(e)}")
    
    def create_factor_correlation_analysis(
        self,
        returns_data: pd.DataFrame,
        factor_data: pd.DataFrame,
        title: str = "Factor Correlation Analysis",
        **kwargs
    ) -> go.Figure:
        """
        Create factor correlation analysis visualization.
        
        Args:
            returns_data: DataFrame with asset returns
            factor_data: DataFrame with factor returns
            title: Chart title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            # Calculate correlations between assets and factors
            correlations = []
            for asset in returns_data.columns:
                asset_corrs = {}
                for factor in factor_data.columns:
                    # Align data
                    aligned_data = pd.concat([returns_data[asset], factor_data[factor]], axis=1).dropna()
                    if len(aligned_data) > 10:  # Minimum data points
                        corr, _ = pearsonr(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
                        asset_corrs[factor] = corr
                    else:
                        asset_corrs[factor] = np.nan
                correlations.append(asset_corrs)
            
            # Create correlation matrix
            corr_df = pd.DataFrame(correlations, index=returns_data.columns)
            
            fig = go.Figure()
            
            # Create heatmap
            fig.add_trace(
                go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale=self.config.color_scale,
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    text=np.round(corr_df.values, 3),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hovertemplate='Asset: %{y}<br>Factor: %{x}<br>Correlation: %{z:.3f}<extra></extra>',
                    colorbar=dict(title="Factor Correlation")
                )
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=title,
                xaxis=dict(
                    title="Factors",
                    side="bottom",
                    tickangle=45
                ),
                yaxis=dict(
                    title="Assets"
                ),
                height=max(400, len(corr_df) * 30),
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating factor correlation analysis: {str(e)}")
            return self._create_error_chart(f"Error creating factor correlation analysis: {str(e)}")
    
    def create_correlation_network(
        self,
        data: pd.DataFrame,
        threshold: float = 0.5,
        title: str = "Correlation Network",
        **kwargs
    ) -> go.Figure:
        """
        Create network visualization of asset correlations.
        
        Args:
            data: DataFrame with numerical data
            threshold: Minimum correlation threshold for edges
            title: Chart title
            **kwargs: Additional chart parameters
        
        Returns:
            Plotly Figure object
        """
        try:
            # Calculate correlation matrix
            corr_matrix = data.corr(method=self.config.method)
            
            # Create network data
            nodes = list(corr_matrix.columns)
            edges = []
            
            for i, asset1 in enumerate(nodes):
                for j, asset2 in enumerate(nodes):
                    if i < j:  # Avoid duplicates
                        corr_val = corr_matrix.loc[asset1, asset2]
                        if abs(corr_val) >= threshold:
                            edges.append({
                                'source': i,
                                'target': j,
                                'weight': abs(corr_val),
                                'correlation': corr_val
                            })
            
            if not edges:
                return self._create_error_chart(f"No correlations above threshold {threshold}")
            
            # Create network layout (circular for simplicity)
            n = len(nodes)
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            x_pos = np.cos(angles)
            y_pos = np.sin(angles)
            
            fig = go.Figure()
            
            # Add edges
            for edge in edges:
                x0, y0 = x_pos[edge['source']], y_pos[edge['source']]
                x1, y1 = x_pos[edge['target']], y_pos[edge['target']]
                
                # Color based on correlation
                color = '#10b981' if edge['correlation'] > 0 else '#ef4444'
                
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        mode='lines',
                        line=dict(
                            color=color,
                            width=edge['weight'] * 5  # Scale line width by correlation
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
            
            # Add nodes
            fig.add_trace(
                go.Scatter(
                    x=x_pos,
                    y=y_pos,
                    mode='markers+text',
                    marker=dict(
                        size=20,
                        color='#6366f1',
                        line=dict(width=2, color='white')
                    ),
                    text=nodes,
                    textposition="middle center",
                    textfont=dict(color='white', size=10),
                    hovertemplate='Asset: %{text}<extra></extra>',
                    showlegend=False
                )
            )
            
            # Update layout
            fig.update_layout(
                **self.default_layout,
                title=title,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                height=600,
                showlegend=False,
                **kwargs
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation network: {str(e)}")
            return self._create_error_chart(f"Error creating correlation network: {str(e)}")
    
    def _cluster_correlation_matrix(self, corr_matrix: pd.DataFrame, method: str) -> pd.DataFrame:
        """Apply hierarchical clustering to correlation matrix."""
        try:
            # Convert correlation to distance
            distance_matrix = 1 - np.abs(corr_matrix)
            
            # Perform clustering
            linkage_matrix = linkage(distance_matrix, method=method)
            dend = dendrogram(linkage_matrix, no_plot=True)
            
            # Reorder matrix based on clustering
            clustered_order = dend['leaves']
            clustered_columns = [corr_matrix.columns[i] for i in clustered_order]
            
            return corr_matrix.reindex(clustered_columns, axis=0).reindex(clustered_columns, axis=1)
            
        except Exception as e:
            logger.error(f"Error clustering correlation matrix: {str(e)}")
            return corr_matrix
    
    def _calculate_significance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical significance of correlations."""
        try:
            n_assets = len(data.columns)
            p_values = np.ones((n_assets, n_assets))
            
            for i, asset1 in enumerate(data.columns):
                for j, asset2 in enumerate(data.columns):
                    if i != j:
                        clean_data = data[[asset1, asset2]].dropna()
                        if len(clean_data) > 3:
                            _, p_val = pearsonr(clean_data[asset1], clean_data[asset2])
                            p_values[i, j] = p_val
            
            return pd.DataFrame(p_values, index=data.columns, columns=data.columns)
            
        except Exception as e:
            logger.error(f"Error calculating significance: {str(e)}")
            return pd.DataFrame(np.ones((len(data.columns), len(data.columns))), 
                              index=data.columns, columns=data.columns)
    
    def _format_text_with_significance(self, corr_matrix: pd.DataFrame, p_values: pd.DataFrame) -> np.ndarray:
        """Format text annotations with significance indicators."""
        text_matrix = np.empty(corr_matrix.shape, dtype=object)
        
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                corr_val = corr_matrix.iloc[i, j]
                p_val = p_values.iloc[i, j]
                
                # Add significance indicators
                if p_val < 0.001:
                    sig_indicator = "***"
                elif p_val < 0.01:
                    sig_indicator = "**"
                elif p_val < self.config.significance_level:
                    sig_indicator = "*"
                else:
                    sig_indicator = ""
                
                text_matrix[i, j] = f"{corr_val:.3f}{sig_indicator}"
        
        return text_matrix
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create error chart when data processing fails."""
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Correlation Error:<br>{error_message}",
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

# Convenience functions for quick correlation analysis
def quick_correlation_heatmap(data: pd.DataFrame, **kwargs) -> go.Figure:
    """Quick correlation heatmap creation."""
    heatmap = CorrelationHeatmap()
    return heatmap.create_correlation_heatmap(data, **kwargs)

def quick_rolling_correlation(data: pd.DataFrame, asset1: str, asset2: str, window: int = 30, **kwargs) -> go.Figure:
    """Quick rolling correlation chart creation."""
    heatmap = CorrelationHeatmap()
    return heatmap.create_rolling_correlation_chart(data, asset1, asset2, window, **kwargs)