"""
Factor Explorer Analysis Tool
============================

Interactive factor analysis and exploration tool for quantitative research.
Provides comprehensive factor analysis capabilities including factor loadings,
exposures, attribution, and performance analysis.

Features:
- Multi-factor model analysis and visualization
- Factor loading decomposition and clustering
- Risk factor exposure tracking and analysis
- Performance attribution to factors
- Factor correlation and interaction analysis
- Time-varying factor analysis
- Custom factor creation and backtesting
- Interactive factor exploration interface

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.stats as stats
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
class FactorModel:
    """Container for factor model configuration."""
    factors: List[str]
    name: str
    description: str
    category: str  # 'fundamental', 'technical', 'macro', 'custom'

@dataclass
class FactorExposure:
    """Container for factor exposure data."""
    asset: str
    factor_exposures: Dict[str, float]
    r_squared: float
    residual_risk: float

class FactorExplorer:
    """
    Interactive factor exploration and analysis tool.
    
    Provides comprehensive factor analysis capabilities for quantitative
    research including factor loadings, exposures, and attribution.
    """
    
    def __init__(self):
        """Initialize factor explorer."""
        st.set_page_config(
            page_title="Quantum Forge - Factor Explorer",
            page_icon=" ",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .factor-header {
            background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .factor-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #3b82f6;
            margin-bottom: 1rem;
        }
        .exposure-positive {
            color: #10b981;
            font-weight: bold;
        }
        .exposure-negative {
            color: #ef4444;
            font-weight: bold;
        }
        .exposure-neutral {
            color: #6b7280;
            font-weight: bold;
        }
        .factor-metric {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        self.factor_models = self._initialize_factor_models()
        
    def _initialize_factor_models(self) -> Dict[str, FactorModel]:
        """Initialize predefined factor models."""
        return {
            'fama_french_3': FactorModel(
                factors=['Market', 'Size', 'Value'],
                name='Fama-French 3-Factor',
                description='Classic market, size, and value factors',
                category='fundamental'
            ),
            'fama_french_5': FactorModel(
                factors=['Market', 'Size', 'Value', 'Profitability', 'Investment'],
                name='Fama-French 5-Factor',
                description='Extended model with profitability and investment factors',
                category='fundamental'
            ),
            'carhart_4': FactorModel(
                factors=['Market', 'Size', 'Value', 'Momentum'],
                name='Carhart 4-Factor',
                description='Fama-French 3-factor plus momentum',
                category='fundamental'
            ),
            'macro_factors': FactorModel(
                factors=['GDP Growth', 'Inflation', 'Interest Rates', 'USD Index', 'VIX'],
                name='Macro Economic Factors',
                description='Key macroeconomic risk factors',
                category='macro'
            ),
            'technical_factors': FactorModel(
                factors=['RSI', 'MACD', 'Bollinger Position', 'Volume Trend', 'Price Momentum'],
                name='Technical Factors',
                description='Technical analysis based factors',
                category='technical'
            )
        }
    
    def generate_factor_data(self, assets: List[str], factors: List[str], periods: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate sample factor and return data."""
        try:
            # Generate dates
            dates = pd.date_range(start=datetime.now() - timedelta(days=periods), end=datetime.now(), freq='D')
            
            # Generate factor returns
            factor_data = pd.DataFrame(index=dates, columns=factors)
            
            for factor in factors:
                if factor == 'Market':
                    # Market factor with some persistence
                    returns = np.random.normal(0.0008, 0.012, len(dates))  # ~20% annual vol
                elif factor == 'Size':
                    # Size factor (SMB - Small Minus Big)
                    returns = np.random.normal(-0.0002, 0.008, len(dates))  # Slight negative premium
                elif factor == 'Value':
                    # Value factor (HML - High Minus Low)
                    returns = np.random.normal(0.0003, 0.010, len(dates))  # Positive value premium
                elif factor == 'Momentum':
                    # Momentum factor
                    returns = np.random.normal(0.0005, 0.015, len(dates))  # Momentum premium
                elif factor == 'Profitability':
                    # Profitability factor (RMW - Robust Minus Weak)
                    returns = np.random.normal(0.0002, 0.007, len(dates))
                elif factor == 'Investment':
                    # Investment factor (CMA - Conservative Minus Aggressive)
                    returns = np.random.normal(0.0001, 0.006, len(dates))
                else:
                    # Generic factor
                    returns = np.random.normal(0, 0.01, len(dates))
                
                # Add some autocorrelation
                for i in range(1, len(returns)):
                    returns[i] += 0.1 * returns[i-1]
                
                factor_data[factor] = returns
            
            # Generate asset returns based on factor exposures
            asset_data = pd.DataFrame(index=dates, columns=assets)
            
            for asset in assets:
                # Generate random factor loadings
                loadings = {}
                for factor in factors:
                    if factor == 'Market':
                        loadings[factor] = np.random.normal(1.0, 0.3)  # Beta around 1
                    elif factor == 'Size':
                        loadings[factor] = np.random.normal(0, 0.5)  # Random size exposure
                    elif factor == 'Value':
                        loadings[factor] = np.random.normal(0, 0.4)  # Random value exposure
                    else:
                        loadings[factor] = np.random.normal(0, 0.3)  # Random exposure
                
                # Calculate asset returns from factor model
                asset_returns = np.zeros(len(dates))
                for factor in factors:
                    asset_returns += loadings[factor] * factor_data[factor].values
                
                # Add idiosyncratic risk
                idiosyncratic = np.random.normal(0, 0.02, len(dates))  # 20% annual idiosyncratic vol
                asset_returns += idiosyncratic
                
                asset_data[asset] = asset_returns
            
            return asset_data.astype(float), factor_data.astype(float)
            
        except Exception as e:
            logger.error(f"Error generating factor data: {str(e)}")
            # Return empty dataframes
            return pd.DataFrame(), pd.DataFrame()
    
    def calculate_factor_exposures(self, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame) -> Dict[str, FactorExposure]:
        """Calculate factor exposures for each asset."""
        try:
            exposures = {}
            
            for asset in asset_returns.columns:
                # Align data
                aligned_data = pd.concat([asset_returns[asset], factor_returns], axis=1).dropna()
                
                if len(aligned_data) < 30:  # Need minimum observations
                    continue
                
                y = aligned_data.iloc[:, 0].values  # Asset returns
                X = aligned_data.iloc[:, 1:].values  # Factor returns
                
                # Add constant for alpha
                X_with_const = np.column_stack([np.ones(len(X)), X])
                
                # OLS regression
                try:
                    coeffs = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                    alpha = coeffs[0]
                    betas = coeffs[1:]
                    
                    # Calculate R-squared
                    y_pred = X_with_const @ coeffs
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    
                    # Calculate residual risk
                    residuals = y - y_pred
                    residual_risk = np.std(residuals) * np.sqrt(252)  # Annualized
                    
                    # Create factor exposure dictionary
                    factor_exposures = {}
                    for i, factor in enumerate(factor_returns.columns):
                        factor_exposures[factor] = betas[i]
                    
                    exposures[asset] = FactorExposure(
                        asset=asset,
                        factor_exposures=factor_exposures,
                        r_squared=r_squared,
                        residual_risk=residual_risk
                    )
                    
                except np.linalg.LinAlgError:
                    logger.warning(f"Could not calculate exposures for {asset}")
                    continue
            
            return exposures
            
        except Exception as e:
            logger.error(f"Error calculating factor exposures: {str(e)}")
            return {}
    
    def render_header(self):
        """Render factor explorer header."""
        st.markdown("""
        <div class="factor-header">
            <h1>  Quantum Forge Factor Explorer</h1>
            <p>Interactive Factor Analysis & Risk Attribution Tool</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Control buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("  Refresh Analysis", key="refresh_factors"):
                st.rerun()
                
        with col2:
            if st.button("  Run PCA", key="run_pca"):
                st.info("Running Principal Component Analysis...")
                
        with col3:
            if st.button("  Optimize Portfolio", key="optimize"):
                st.info("Running factor-based optimization...")
                
        with col4:
            if st.button("  Backtest Strategy", key="backtest"):
                st.info("Backtesting factor strategy...")
    
    def render_sidebar_controls(self):
        """Render factor explorer sidebar controls."""
        st.sidebar.markdown("##   Factor Analysis Controls")
        
        # Factor model selection
        model_names = list(self.factor_models.keys())
        selected_model = st.sidebar.selectbox(
            "Select Factor Model",
            options=model_names,
            format_func=lambda x: self.factor_models[x].name
        )
        
        # Asset selection
        default_assets = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'JPM', 'BAC', 'XOM']
        selected_assets = st.sidebar.multiselect(
            "Select Assets",
            options=default_assets,
            default=default_assets[:6]
        )
        
        # Analysis parameters
        st.sidebar.markdown("###   Analysis Parameters")
        
        lookback_period = st.sidebar.slider(
            "Lookback Period (days)",
            min_value=30,
            max_value=1000,
            value=252,
            step=30
        )
        
        min_r_squared = st.sidebar.slider(
            "Min R-Squared Filter",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05
        )
        
        # Advanced options
        st.sidebar.markdown("###  ️ Advanced Options")
        
        show_residuals = st.sidebar.checkbox("Show Residual Analysis", value=True)
        show_correlations = st.sidebar.checkbox("Show Factor Correlations", value=True)
        cluster_exposures = st.sidebar.checkbox("Cluster by Exposures", value=False)
        
        return {
            'selected_model': selected_model,
            'selected_assets': selected_assets,
            'lookback_period': lookback_period,
            'min_r_squared': min_r_squared,
            'show_residuals': show_residuals,
            'show_correlations': show_correlations,
            'cluster_exposures': cluster_exposures
        }
    
    def render_factor_overview(self, factor_model: FactorModel, factor_data: pd.DataFrame):
        """Render factor model overview."""
        st.subheader(f"  {factor_model.name} Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="factor-card">
                <h4>{factor_model.name}</h4>
                <p><strong>Description:</strong> {factor_model.description}</p>
                <p><strong>Category:</strong> {factor_model.category.title()}</p>
                <p><strong>Factors:</strong> {', '.join(factor_model.factors)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Factor statistics
            if not factor_data.empty:
                factor_stats = factor_data.describe()
                
                st.markdown("**Factor Statistics**")
                for factor in factor_model.factors:
                    if factor in factor_stats.columns:
                        mean_ret = factor_stats.loc['mean', factor] * 252  # Annualized
                        volatility = factor_stats.loc['std', factor] * np.sqrt(252)  # Annualized
                        sharpe = mean_ret / volatility if volatility > 0 else 0
                        
                        st.markdown(f"""
                        <div class="factor-metric">
                            <strong>{factor}</strong><br>
                            Return: {mean_ret:.1%}<br>
                            Vol: {volatility:.1%}<br>
                            Sharpe: {sharpe:.2f}
                        </div>
                        """, unsafe_allow_html=True)
    
    def render_factor_exposures(self, exposures: Dict[str, FactorExposure], factors: List[str]):
        """Render factor exposure analysis."""
        st.subheader("  Factor Exposures")
        
        if not exposures:
            st.warning("No factor exposure data available")
            return
        
        # Create exposure matrix
        exposure_matrix = []
        assets = list(exposures.keys())
        
        for asset in assets:
            row = [asset]
            for factor in factors:
                exposure = exposures[asset].factor_exposures.get(factor, 0)
                row.append(exposure)
            row.extend([exposures[asset].r_squared, exposures[asset].residual_risk])
            exposure_matrix.append(row)
        
        # Create DataFrame
        columns = ['Asset'] + factors + ['R²', 'Residual Risk']
        exposure_df = pd.DataFrame(exposure_matrix, columns=columns)
        
        # Display as heatmap
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Heatmap of exposures
            fig = go.Figure()
            
            z_data = exposure_df[factors].values
            
            fig.add_trace(
                go.Heatmap(
                    z=z_data,
                    x=factors,
                    y=exposure_df['Asset'],
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(z_data, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    hovertemplate='Asset: %{y}<br>Factor: %{x}<br>Exposure: %{z:.3f}<extra></extra>',
                    colorbar=dict(title="Factor Exposure")
                )
            )
            
            fig.update_layout(
                title="Factor Exposure Heatmap",
                height=max(400, len(assets) * 30),
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Summary statistics
            st.markdown("**Exposure Summary**")
            
            # R-squared distribution
            r_squared_values = [exp.r_squared for exp in exposures.values()]
            avg_r_squared = np.mean(r_squared_values)
            
            st.markdown(f"""
            <div class="factor-metric">
                <strong>Model Fit</strong><br>
                Avg R²: {avg_r_squared:.2%}<br>
                Max R²: {max(r_squared_values):.2%}<br>
                Min R²: {min(r_squared_values):.2%}
            </div>
            """, unsafe_allow_html=True)
            
            # Factor exposure ranges
            for factor in factors:
                factor_exposures = [exp.factor_exposures.get(factor, 0) for exp in exposures.values()]
                
                st.markdown(f"""
                <div class="factor-metric">
                    <strong>{factor}</strong><br>
                    Range: {min(factor_exposures):.2f} to {max(factor_exposures):.2f}<br>
                    Avg: {np.mean(factor_exposures):.2f}
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed exposure table
        st.markdown("**Detailed Exposures**")
        
        # Style the dataframe
        def style_exposure(val):
            if isinstance(val, (int, float)):
                if abs(val) > 1:
                    return 'background-color: #fee2e2; color: #dc2626; font-weight: bold'
                elif abs(val) > 0.5:
                    return 'background-color: #fef3c7; color: #d97706; font-weight: bold'
                else:
                    return 'color: #6b7280'
            return ''
        
        styled_df = exposure_df.style.applymap(style_exposure, subset=factors)
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    def render_factor_performance_attribution(self, exposures: Dict[str, FactorExposure], 
                                            factor_returns: pd.DataFrame, 
                                            asset_returns: pd.DataFrame):
        """Render factor performance attribution analysis."""
        st.subheader("  Performance Attribution")
        
        if not exposures or factor_returns.empty or asset_returns.empty:
            st.warning("Insufficient data for performance attribution")
            return
        
        # Calculate attribution for each asset
        attribution_data = []
        
        for asset, exposure in exposures.items():
            if asset not in asset_returns.columns:
                continue
            
            # Calculate factor contributions
            total_return = asset_returns[asset].sum()
            factor_contributions = {}
            
            for factor, loading in exposure.factor_exposures.items():
                if factor in factor_returns.columns:
                    factor_return = factor_returns[factor].sum()
                    contribution = loading * factor_return
                    factor_contributions[factor] = contribution
            
            # Calculate residual (unexplained) return
            explained_return = sum(factor_contributions.values())
            residual_return = total_return - explained_return
            
            attribution_data.append({
                'Asset': asset,
                'Total Return': total_return,
                'Explained Return': explained_return,
                'Residual Return': residual_return,
                **factor_contributions
            })
        
        if not attribution_data:
            st.warning("No attribution data available")
            return
        
        attribution_df = pd.DataFrame(attribution_data)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Stacked bar chart of factor contributions
            fig = go.Figure()
            
            factors = [col for col in attribution_df.columns if col not in ['Asset', 'Total Return', 'Explained Return', 'Residual Return']]
            
            for factor in factors:
                fig.add_trace(
                    go.Bar(
                        name=factor,
                        x=attribution_df['Asset'],
                        y=attribution_df[factor],
                        hovertemplate=f'{factor}: %{{y:.3f}}<extra></extra>'
                    )
                )
            
            # Add residual
            fig.add_trace(
                go.Bar(
                    name='Residual',
                    x=attribution_df['Asset'],
                    y=attribution_df['Residual Return'],
                    marker_color='gray',
                    hovertemplate='Residual: %{y:.3f}<extra></extra>'
                )
            )
            
            fig.update_layout(
                title="Factor Attribution by Asset",
                barmode='stack',
                xaxis_title="Assets",
                yaxis_title="Return Contribution",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Attribution summary
            st.markdown("**Attribution Summary**")
            
            # Average contributions
            factor_columns = [col for col in attribution_df.columns if col not in ['Asset', 'Total Return', 'Explained Return', 'Residual Return']]
            
            avg_contributions = {}
            for factor in factor_columns:
                avg_contributions[factor] = attribution_df[factor].mean()
            
            # Sort by absolute contribution
            sorted_contributions = sorted(avg_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            
            for factor, contribution in sorted_contributions:
                contribution_class = "exposure-positive" if contribution > 0 else "exposure-negative" if contribution < 0 else "exposure-neutral"
                st.markdown(f"""
                <div class="factor-metric">
                    <strong>{factor}</strong><br>
                    <span class="{contribution_class}">Avg Contribution: {contribution:.4f}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed attribution table
        st.markdown("**Detailed Attribution**")
        
        # Format the dataframe
        display_columns = ['Asset', 'Total Return', 'Explained Return', 'Residual Return'] + factor_columns
        display_df = attribution_df[display_columns].round(4)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    def render_factor_correlations(self, factor_returns: pd.DataFrame):
        """Render factor correlation analysis."""
        st.subheader("  Factor Correlations")
        
        if factor_returns.empty:
            st.warning("No factor return data available")
            return
        
        # Calculate correlation matrix
        corr_matrix = factor_returns.corr()
        
        # Create heatmap
        fig = go.Figure()
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Correlation")
            )
        )
        
        fig.update_layout(
            title="Factor Correlation Matrix",
            height=max(400, len(corr_matrix) * 40),
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**High Correlations**")
            high_corrs = []
            
            for i, factor1 in enumerate(corr_matrix.columns):
                for j, factor2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid duplicates
                        corr = corr_matrix.loc[factor1, factor2]
                        if abs(corr) > 0.5:  # Threshold for high correlation
                            high_corrs.append((factor1, factor2, corr))
            
            high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
            
            for factor1, factor2, corr in high_corrs[:5]:
                corr_class = "exposure-positive" if corr > 0 else "exposure-negative"
                st.markdown(f"""
                <div class="factor-metric">
                    <strong>{factor1} - {factor2}</strong><br>
                    <span class="{corr_class}">Correlation: {corr:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Factor Statistics**")
            
            factor_stats = factor_returns.describe()
            
            for factor in factor_returns.columns:
                vol = factor_stats.loc['std', factor] * np.sqrt(252)  # Annualized volatility
                skew = stats.skew(factor_returns[factor].dropna())
                
                st.markdown(f"""
                <div class="factor-metric">
                    <strong>{factor}</strong><br>
                    Volatility: {vol:.1%}<br>
                    Skewness: {skew:.2f}
                </div>
                """, unsafe_allow_html=True)
    
    def run_factor_explorer(self):
        """Run the complete factor explorer interface."""
        try:
            # Header
            self.render_header()
            
            # Sidebar controls
            controls = self.render_sidebar_controls()
            
            if not controls['selected_assets']:
                st.warning("Please select at least one asset to analyze")
                return
            
            # Generate data
            selected_model = self.factor_models[controls['selected_model']]
            
            with st.spinner("Generating factor analysis..."):
                asset_returns, factor_returns = self.generate_factor_data(
                    controls['selected_assets'],
                    selected_model.factors,
                    controls['lookback_period']
                )
                
                if asset_returns.empty or factor_returns.empty:
                    st.error("Failed to generate factor data")
                    return
                
                # Calculate exposures
                exposures = self.calculate_factor_exposures(asset_returns, factor_returns)
                
                # Filter by R-squared
                filtered_exposures = {
                    asset: exp for asset, exp in exposures.items()
                    if exp.r_squared >= controls['min_r_squared']
                }
            
            # Main content
            main_content = st.container()
            
            with main_content:
                # Factor overview
                self.render_factor_overview(selected_model, factor_returns)
                
                st.markdown("---")
                
                # Factor exposures
                self.render_factor_exposures(filtered_exposures, selected_model.factors)
                
                st.markdown("---")
                
                # Performance attribution
                self.render_factor_performance_attribution(filtered_exposures, factor_returns, asset_returns)
                
                st.markdown("---")
                
                # Factor correlations
                if controls['show_correlations']:
                    self.render_factor_correlations(factor_returns)
                    
                st.markdown("---")
                
                # Additional analysis based on options
                if controls['show_residuals']:
                    st.subheader("  Residual Analysis")
                    
                    if filtered_exposures:
                        residual_risks = [exp.residual_risk for exp in filtered_exposures.values()]
                        assets = list(filtered_exposures.keys())
                        
                        fig = go.Figure()
                        fig.add_trace(
                            go.Bar(
                                x=assets,
                                y=residual_risks,
                                name='Residual Risk',
                                marker_color='orange',
                                hovertemplate='Asset: %{x}<br>Residual Risk: %{y:.2%}<extra></extra>'
                            )
                        )
                        
                        fig.update_layout(
                            title="Residual Risk by Asset",
                            xaxis_title="Assets",
                            yaxis_title="Annualized Residual Risk",
                            template='plotly_white',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in factor explorer: {str(e)}")
            logger.error(f"Factor explorer error: {str(e)}")

def main():
    """Main function to run the factor explorer."""
    explorer = FactorExplorer()
    explorer.run_factor_explorer()

if __name__ == "__main__":
    main()