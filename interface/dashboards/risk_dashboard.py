"""
Risk Dashboard for Quantum Forge Trading Platform
================================================

Comprehensive risk management dashboard for real-time risk monitoring,
portfolio risk analysis, and risk limit management.

Features:
- Real-time portfolio risk metrics (VaR, Expected Shortfall, Beta)
- Position-level risk analysis and stress testing
- Risk factor attribution and decomposition
- Concentration risk and correlation analysis
- Risk limit monitoring and alerting
- Scenario analysis and stress testing
- Liquidity risk assessment

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
from scipy.stats import norm

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
class RiskMetrics:
    """Container for risk dashboard metrics."""
    portfolio_var_95: float
    portfolio_var_99: float
    expected_shortfall_95: float
    portfolio_beta: float
    sharpe_ratio: float
    max_drawdown: float
    volatility_annualized: float
    tracking_error: float
    concentration_risk: float
    liquidity_score: float
    stress_test_pnl: float
    risk_limit_utilization: float

@dataclass
class PositionRisk:
    """Container for position-level risk metrics."""
    symbol: str
    market_value: float
    weight: float
    var_contribution: float
    beta: float
    volatility: float
    liquidity_days: float
    sector: str
    correlation_to_portfolio: float

class RiskDashboard:
    """
    Risk management dashboard for comprehensive portfolio risk monitoring.
    
    Provides real-time risk analytics, stress testing, and risk limit monitoring
    with advanced visualization and alerting capabilities.
    """
    
    def __init__(self):
        """Initialize risk dashboard."""
        self.last_update = None
        self.update_frequency = 5  # seconds
        self._setup_page_config()
        
    def _setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Quantum Forge - Risk Dashboard",
            page_icon=" ️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for risk dashboard
        st.markdown("""
        <style>
        .risk-header {
            background: linear-gradient(90deg, #dc2626 0%, #ef4444 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        .risk-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #dc2626;
            margin-bottom: 1rem;
        }
        .risk-low {
            color: #10b981;
            font-weight: bold;
        }
        .risk-medium {
            color: #f59e0b;
            font-weight: bold;
        }
        .risk-high {
            color: #ef4444;
            font-weight: bold;
        }
        .limit-ok {
            background: #ecfdf5;
            border-left: 4px solid #10b981;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 5px;
        }
        .limit-warning {
            background: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 5px;
        }
        .limit-breach {
            background: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 5px;
        }
        .stress-scenario {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def generate_risk_data(self) -> Tuple[RiskMetrics, List[PositionRisk], Dict[str, Any]]:
        """Generate risk data based on real-time market conditions."""
        try:
            cache = get_data_cache()
            
            # Portfolio-level risk metrics
            portfolio_value = 10000000  # $10M portfolio
            
            # Generate position-level risk data from real market data
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT']
            sectors = ['Crypto'] * len(symbols)
            
            positions = []
            remaining_value = portfolio_value
            
            # Fetch historical data for all symbols to calculate portfolio metrics
            histories = {}
            for symbol in symbols:
                hist = cache.get_historical_data(symbol, days=90)
                if not hist.empty:
                    histories[symbol] = hist['close'].pct_change().dropna()
            
            # Calculate portfolio returns if we have data
            if histories:
                # Align dates
                df_returns = pd.DataFrame(histories).dropna()
                if not df_returns.empty:
                    # Assume equal weight for simplicity of metric calculation initially
                    port_returns = df_returns.mean(axis=1)
                    
                    volatility = port_returns.std() * np.sqrt(365)
                    var_95 = np.percentile(port_returns, 5) * portfolio_value
                    var_99 = np.percentile(port_returns, 1) * portfolio_value
                    es_95 = port_returns[port_returns <= np.percentile(port_returns, 5)].mean() * portfolio_value
                    max_dd = (1 - (1 + port_returns).cumprod() / (1 + port_returns).cumprod().cummax()).min()
                else:
                    volatility = 0.5
                    var_95 = -portfolio_value * 0.05
                    var_99 = -portfolio_value * 0.10
                    es_95 = -portfolio_value * 0.07
                    max_dd = -0.2
            else:
                volatility = 0.5
                var_95 = -portfolio_value * 0.05
                var_99 = -portfolio_value * 0.10
                es_95 = -portfolio_value * 0.07
                max_dd = -0.2

            risk_metrics = RiskMetrics(
                portfolio_var_95=var_95,
                portfolio_var_99=var_99,
                expected_shortfall_95=es_95,
                portfolio_beta=1.0, # Crypto portfolio beta to crypto market is 1
                sharpe_ratio=1.5 if volatility > 0 else 0, # Simplified
                max_drawdown=max_dd,
                volatility_annualized=volatility,
                tracking_error=0.05,
                concentration_risk=0.3,
                liquidity_score=0.85,
                stress_test_pnl=var_99 * 1.5,
                risk_limit_utilization=abs(var_95) / (portfolio_value * 0.1) # Assume 10% VaR limit
            )
            
            # Create positions
            btc_returns = histories.get('BTCUSDT', pd.Series())
            
            for i, (symbol, sector) in enumerate(zip(symbols, sectors)):
                if i == len(symbols) - 1:
                    market_value = remaining_value
                else:
                    max_position = remaining_value * 0.3
                    market_value = min(max_position, 1500000) # Distribute $10M
                    remaining_value -= market_value
                
                if remaining_value <= 0:
                    break
                
                weight = market_value / portfolio_value
                
                # Calculate specific risk metrics
                sym_returns = histories.get(symbol, pd.Series())
                if not sym_returns.empty:
                    sym_vol = sym_returns.std() * np.sqrt(365)
                    sym_var = np.percentile(sym_returns, 5) * market_value
                    
                    # Beta to BTC
                    if not btc_returns.empty:
                        common_idx = sym_returns.index.intersection(btc_returns.index)
                        if len(common_idx) > 10:
                            cov = np.cov(sym_returns[common_idx], btc_returns[common_idx])[0][1]
                            var = np.var(btc_returns[common_idx])
                            beta = cov / var if var > 0 else 1.0
                        else:
                            beta = 1.0
                    else:
                        beta = 1.0
                else:
                    sym_vol = 0.5
                    sym_var = -market_value * 0.05
                    beta = 1.0

                position = PositionRisk(
                    symbol=symbol,
                    market_value=market_value,
                    weight=weight,
                    var_contribution=sym_var,
                    beta=beta,
                    volatility=sym_vol,
                    liquidity_days=market_value / (cache.get_ticker(symbol).get('quoteVolume', 10000000) * 0.01), # 1% of daily volume
                    sector=sector,
                    correlation_to_portfolio=0.7 # Simplified
                )
                positions.append(position)
            
            # Generate additional risk analytics data
            risk_analytics = {
                'factor_exposures': self._generate_factor_exposures(),
                'correlation_matrix': self._generate_correlation_matrix(positions, histories),
                'stress_scenarios': self._generate_stress_scenarios(portfolio_value),
                'var_decomposition': self._generate_var_decomposition(positions),
                'liquidity_analysis': self._generate_liquidity_analysis(positions),
                'sector_exposures': self._generate_sector_exposures(positions)
            }
            
            return risk_metrics, positions, risk_analytics
            
        except Exception as e:
            logger.error(f"Error generating risk data: {str(e)}")
            # Return default data
            default_metrics = RiskMetrics(
                portfolio_var_95=0, portfolio_var_99=0, expected_shortfall_95=0,
                portfolio_beta=1, sharpe_ratio=0, max_drawdown=0,
                volatility_annualized=0, tracking_error=0, concentration_risk=0,
                liquidity_score=0, stress_test_pnl=0, risk_limit_utilization=0
            )
            return default_metrics, [], {}
    
    def _generate_factor_exposures(self) -> Dict[str, float]:
        """Generate factor exposure data."""
        # In a real system, we'd regress portfolio returns against factor returns
        # Here we'll just use placeholders but they represent the concept
        factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Low Vol', 'Profitability']
        exposures = {factor: 0.0 for factor in factors}
        exposures['Market'] = 1.0
        exposures['Momentum'] = 0.5 # Crypto is high momentum
        exposures['Vol'] = 1.2 # High vol
        return exposures
    
    def _generate_correlation_matrix(self, positions: List[PositionRisk], histories: Dict[str, pd.Series] = None) -> pd.DataFrame:
        """Generate real correlation matrix for positions."""
        if not histories:
            return pd.DataFrame()
            
        df = pd.DataFrame(histories)
        return df.corr()
    
    def _generate_stress_scenarios(self, portfolio_value: float = 10000000) -> Dict[str, Dict[str, float]]:
        """Generate stress test scenarios."""
        scenarios = {
            'Crypto Crash (-20%)': {
                'equity_shock': -0.20,
                'volatility_shock': 2.0,
                'correlation_shock': 0.3,
                'portfolio_impact': portfolio_value * -0.25 # Higher beta
            },
            'Stablecoin Depeg': {
                'rate_shock': 0.02,
                'duration_impact': -0.05,
                'credit_spread': 0.01,
                'portfolio_impact': portfolio_value * -0.15
            },
            'Exchange Hack': {
                'credit_spread': 0.05,
                'liquidity_shock': -0.3,
                'volatility_shock': 1.5,
                'portfolio_impact': portfolio_value * -0.10
            },
            'Regulatory Ban': {
                'fx_shock': -0.15,
                'emerging_market_shock': -0.25,
                'flight_to_quality': 0.1,
                'portfolio_impact': portfolio_value * -0.30
            }
        }
        return scenarios
    
    def _generate_var_decomposition(self, positions: List[PositionRisk]) -> Dict[str, float]:
        """Generate VaR decomposition by position."""
        total_var = sum(pos.var_contribution for pos in positions)
        decomposition = {}
        
        for position in positions[:10]:  # Top 10 contributors
            decomposition[position.symbol] = position.var_contribution / total_var if total_var > 0 else 0
        
        return decomposition
    
    def _generate_liquidity_analysis(self, positions: List[PositionRisk]) -> Dict[str, Any]:
        """Generate liquidity analysis data."""
        liquidity_buckets = {
            'Highly Liquid (< 1 day)': sum(1 for pos in positions if pos.liquidity_days < 1),
            'Liquid (1-3 days)': sum(1 for pos in positions if 1 <= pos.liquidity_days < 3),
            'Medium Liquidity (3-7 days)': sum(1 for pos in positions if 3 <= pos.liquidity_days < 7),
            'Low Liquidity (> 7 days)': sum(1 for pos in positions if pos.liquidity_days >= 7)
        }
        
        total_value = sum(pos.market_value for pos in positions)
        liquidity_weighted = {
            bucket: sum(pos.market_value for pos in positions 
                       if (bucket == 'Highly Liquid (< 1 day)' and pos.liquidity_days < 1) or
                          (bucket == 'Liquid (1-3 days)' and 1 <= pos.liquidity_days < 3) or
                          (bucket == 'Medium Liquidity (3-7 days)' and 3 <= pos.liquidity_days < 7) or
                          (bucket == 'Low Liquidity (> 7 days)' and pos.liquidity_days >= 7)
                       ) / total_value if total_value > 0 else 0
            for bucket in liquidity_buckets.keys()
        }
        
        return {
            'count_by_bucket': liquidity_buckets,
            'value_weighted': liquidity_weighted,
            'avg_liquidity_days': np.mean([pos.liquidity_days for pos in positions])
        }
    
    def _generate_sector_exposures(self, positions: List[PositionRisk]) -> Dict[str, float]:
        """Generate sector exposure analysis."""
        sector_exposures = {}
        total_value = sum(pos.market_value for pos in positions)
        
        for position in positions:
            if position.sector not in sector_exposures:
                sector_exposures[position.sector] = 0
            sector_exposures[position.sector] += position.market_value / total_value if total_value > 0 else 0
        
        return sector_exposures
    
    def render_header(self):
        """Render risk dashboard header."""
        st.markdown("""
        <div class="risk-header">
            <h1> ️ Quantum Forge Risk Management</h1>
            <p>Real-time Portfolio Risk Monitoring & Control System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk controls
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("  Refresh Risk", key="refresh_risk"):
                st.rerun()
                
        with col2:
            if st.button("  Stress Test", key="stress_test"):
                st.info("Running comprehensive stress test...")
                
        with col3:
            if st.button("  Risk Report", key="risk_report"):
                st.info("Generating detailed risk report...")
                
        with col4:
            if st.button("  Risk Alert", key="risk_alert"):
                st.warning("Risk alert system activated")
    
    def render_risk_metrics(self, metrics: RiskMetrics):
        """Render key risk metrics."""
        st.subheader("  Portfolio Risk Metrics")
        
        cols = st.columns(6)
        
        with cols[0]:
            var_class = "risk-low" if metrics.portfolio_var_95 > -100000 else "risk-medium" if metrics.portfolio_var_95 > -200000 else "risk-high"
            st.markdown(f"""
            <div class="risk-card">
                <h4>VaR (95%)</h4>
                <h2 class="{var_class}">${metrics.portfolio_var_95:,.0f}</h2>
                <small>Daily Value at Risk</small>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            es_class = "risk-low" if metrics.expected_shortfall_95 > -200000 else "risk-medium" if metrics.expected_shortfall_95 > -300000 else "risk-high"
            st.markdown(f"""
            <div class="risk-card">
                <h4>Expected Shortfall</h4>
                <h2 class="{es_class}">${metrics.expected_shortfall_95:,.0f}</h2>
                <small>95% Expected Shortfall</small>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[2]:
            beta_class = "risk-medium" if 0.8 <= metrics.portfolio_beta <= 1.2 else "risk-high"
            st.markdown(f"""
            <div class="risk-card">
                <h4>Portfolio Beta</h4>
                <h2 class="{beta_class}">{metrics.portfolio_beta:.2f}</h2>
                <small>Market sensitivity</small>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[3]:
            vol_class = "risk-low" if metrics.volatility_annualized < 0.15 else "risk-medium" if metrics.volatility_annualized < 0.20 else "risk-high"
            st.markdown(f"""
            <div class="risk-card">
                <h4>Volatility</h4>
                <h2 class="{vol_class}">{metrics.volatility_annualized:.1%}</h2>
                <small>Annualized volatility</small>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[4]:
            conc_class = "risk-low" if metrics.concentration_risk < 0.25 else "risk-medium" if metrics.concentration_risk < 0.35 else "risk-high"
            st.markdown(f"""
            <div class="risk-card">
                <h4>Concentration</h4>
                <h2 class="{conc_class}">{metrics.concentration_risk:.1%}</h2>
                <small>Top 5 positions</small>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[5]:
            liq_class = "risk-high" if metrics.liquidity_score < 0.7 else "risk-medium" if metrics.liquidity_score < 0.85 else "risk-low"
            st.markdown(f"""
            <div class="risk-card">
                <h4>Liquidity Score</h4>
                <h2 class="{liq_class}">{metrics.liquidity_score:.1%}</h2>
                <small>Portfolio liquidity</small>
            </div>
            """, unsafe_allow_html=True)
    
    def render_position_risk_analysis(self, positions: List[PositionRisk]):
        """Render position-level risk analysis."""
        if not positions:
            st.warning("No position data available")
            return
            
        st.subheader("  Position Risk Analysis")
        
        # Position risk table
        position_data = []
        for pos in positions[:15]:  # Show top 15 positions
            risk_level = "High" if pos.var_contribution > 50000 else "Medium" if pos.var_contribution > 20000 else "Low"
            
            position_data.append({
                'Symbol': pos.symbol,
                'Sector': pos.sector,
                'Market Value': f"${pos.market_value:,.0f}",
                'Weight': f"{pos.weight:.1%}",
                'VaR Contribution': f"${pos.var_contribution:,.0f}",
                'Beta': f"{pos.beta:.2f}",
                'Volatility': f"{pos.volatility:.1%}",
                'Liquidity (Days)': f"{pos.liquidity_days:.1f}",
                'Risk Level': risk_level
            })
        
        position_df = pd.DataFrame(position_data)
        
        # Style the dataframe
        def style_risk_level(val):
            if val == 'High':
                return 'color: #ef4444; font-weight: bold'
            elif val == 'Medium':
                return 'color: #f59e0b; font-weight: bold'
            else:
                return 'color: #10b981; font-weight: bold'
        
        styled_df = position_df.style.applymap(style_risk_level, subset=['Risk Level'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Position risk charts
        col1, col2 = st.columns(2)
        
        with col1:
            # VaR contribution by position
            top_var_positions = sorted(positions, key=lambda x: x.var_contribution, reverse=True)[:8]
            
            fig = px.bar(
                x=[pos.symbol for pos in top_var_positions],
                y=[pos.var_contribution for pos in top_var_positions],
                title='VaR Contribution by Position',
                labels={'x': 'Symbol', 'y': 'VaR Contribution ($)'},
                color=[pos.var_contribution for pos in top_var_positions],
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk vs Return scatter
            # Fetch real 24h change as proxy for return
            cache = get_data_cache()
            returns = []
            for pos in positions[:10]:
                ticker = cache.get_ticker(pos.symbol)
                ret = float(ticker.get('priceChangePercent', 0)) / 100 if ticker else 0
                returns.append(ret)
            
            fig = px.scatter(
                x=[pos.volatility for pos in positions[:10]],
                y=returns,
                size=[pos.market_value for pos in positions[:10]],
                color=[pos.beta for pos in positions[:10]],
                hover_name=[pos.symbol for pos in positions[:10]],
                title='Risk-Return Profile (24h)',
                labels={'x': 'Volatility (Annualized)', 'y': '24h Return', 'color': 'Beta'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_factor_exposure(self, factor_exposures: Dict[str, float]):
        """Render factor exposure analysis."""
        if not factor_exposures:
            st.warning("Factor exposure data unavailable")
            return
            
        st.subheader("  Factor Exposure Analysis")
        
        factors = list(factor_exposures.keys())
        exposures = list(factor_exposures.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=factors,
                y=exposures,
                marker_color=['#10b981' if x > 0 else '#ef4444' for x in exposures],
                text=[f"{x:.2f}" for x in exposures],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Portfolio Factor Exposures",
            xaxis_title="Risk Factor",
            yaxis_title="Exposure",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_stress_testing(self, stress_scenarios: Dict[str, Dict[str, float]]):
        """Render stress testing results."""
        if not stress_scenarios:
            st.warning("Stress testing data unavailable")
            return
            
        st.subheader("  Stress Testing Results")
        
        # Stress test summary
        scenario_names = list(stress_scenarios.keys())
        portfolio_impacts = [scenario['portfolio_impact'] for scenario in stress_scenarios.values()]
        
        fig = go.Figure(data=[
            go.Bar(
                x=scenario_names,
                y=[impact * 100 for impact in portfolio_impacts],  # Convert to percentage
                marker_color=['#ef4444' if x < -0.15 else '#f59e0b' if x < -0.08 else '#10b981' for x in portfolio_impacts],
                text=[f"{x:.1%}" for x in portfolio_impacts],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Stress Test Impact on Portfolio",
            xaxis_title="Stress Scenario",
            yaxis_title="Portfolio Impact (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed scenario analysis
        st.markdown("**Detailed Scenario Analysis**")
        
        for scenario_name, scenario_data in stress_scenarios.items():
            impact = scenario_data['portfolio_impact']
            impact_class = "limit-breach" if impact < -0.15 else "limit-warning" if impact < -0.08 else "limit-ok"
            
            st.markdown(f"""
            <div class="{impact_class}">
                <strong>{scenario_name}</strong> - Portfolio Impact: {impact:.1%}<br>
                <small>Scenario parameters: {', '.join([f'{k}: {v:.1%}' if isinstance(v, float) and abs(v) < 1 else f'{k}: {v}' for k, v in scenario_data.items() if k != 'portfolio_impact'])}</small>
            </div>
            """, unsafe_allow_html=True)
    
    def render_correlation_analysis(self, correlation_matrix: pd.DataFrame):
        """Render correlation analysis."""
        if correlation_matrix.empty:
            st.warning("Correlation data unavailable")
            return
            
        st.subheader("  Correlation Analysis")
        
        fig = px.imshow(
            correlation_matrix,
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            title="Position Correlation Matrix",
            aspect='auto'
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_liquidity_analysis(self, liquidity_data: Dict[str, Any]):
        """Render liquidity analysis."""
        if not liquidity_data:
            st.warning("Liquidity analysis data unavailable")
            return
            
        st.subheader("  Liquidity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Liquidity buckets by count
            buckets = list(liquidity_data['count_by_bucket'].keys())
            counts = list(liquidity_data['count_by_bucket'].values())
            
            fig = px.pie(
                values=counts,
                names=buckets,
                title='Positions by Liquidity Bucket'
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Liquidity buckets by value
            values = list(liquidity_data['value_weighted'].values())
            
            fig = px.pie(
                values=values,
                names=buckets,
                title='Portfolio Value by Liquidity Bucket'
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Average liquidity metric
        avg_liquidity = liquidity_data.get('avg_liquidity_days', 0)
        liquidity_class = "risk-low" if avg_liquidity < 2 else "risk-medium" if avg_liquidity < 5 else "risk-high"
        
        st.markdown(f"""
        <div class="risk-card">
            <h4>Portfolio Average Liquidity</h4>
            <h2 class="{liquidity_class}">{avg_liquidity:.1f} days</h2>
            <small>Weighted average time to liquidate</small>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sector_exposure(self, sector_exposures: Dict[str, float]):
        """Render sector exposure analysis."""
        if not sector_exposures:
            st.warning("Sector exposure data unavailable")
            return
            
        st.subheader("  Sector Exposure Analysis")
        
        sectors = list(sector_exposures.keys())
        exposures = list(sector_exposures.values())
        
        fig = px.pie(
            values=exposures,
            names=sectors,
            title='Portfolio Sector Allocation'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_risk_limits_monitoring(self, metrics: RiskMetrics):
        """Render risk limits monitoring."""
        st.subheader(" ️ Risk Limits Monitoring")
        
        # Define risk limits (these would normally come from configuration)
        limits = {
            'Daily VaR Limit': {'current': abs(metrics.portfolio_var_95), 'limit': 200000, 'unit': '$'},
            'Portfolio Beta Limit': {'current': abs(metrics.portfolio_beta - 1), 'limit': 0.3, 'unit': ''},
            'Concentration Limit': {'current': metrics.concentration_risk, 'limit': 0.4, 'unit': '%'},
            'Volatility Limit': {'current': metrics.volatility_annualized, 'limit': 0.25, 'unit': '%'},
            'Max Drawdown Limit': {'current': metrics.max_drawdown, 'limit': 0.15, 'unit': '%'}
        }
        
        for limit_name, limit_data in limits.items():
            current = limit_data['current']
            limit_val = limit_data['limit']
            unit = limit_data['unit']
            
            utilization = current / limit_val if limit_val > 0 else 0
            
            if utilization < 0.7:
                status_class = "limit-ok"
                status_text = "  OK"
            elif utilization < 0.9:
                status_class = "limit-warning"
                status_text = " ️ WARNING"
            else:
                status_class = "limit-breach"
                status_text = "  BREACH"
            
            if unit == '%':
                current_display = f"{current:.1%}"
                limit_display = f"{limit_val:.1%}"
            elif unit == '$':
                current_display = f"${current:,.0f}"
                limit_display = f"${limit_val:,.0f}"
            else:
                current_display = f"{current:.2f}"
                limit_display = f"{limit_val:.2f}"
            
            st.markdown(f"""
            <div class="{status_class}">
                <strong>{limit_name}</strong> {status_text}<br>
                Current: {current_display} | Limit: {limit_display} | Utilization: {utilization:.1%}
            </div>
            """, unsafe_allow_html=True)
    
    def render_sidebar_controls(self):
        """Render risk dashboard sidebar controls."""
        # Navigation
        if st.sidebar.button("← Back to Main Dashboard"):
            st.session_state.current_dashboard = 'main'
            st.rerun()
            
        st.sidebar.markdown("##  ️ Risk Controls")
        
        # Risk monitoring mode
        risk_mode = st.sidebar.selectbox(
            "Risk Monitoring Mode",
            options=["Real-time", "End-of-Day", "Intraday Snapshots"],
            index=0
        )
        
        # Confidence levels
        st.sidebar.markdown("###   VaR Settings")
        
        confidence_level = st.sidebar.selectbox(
            "VaR Confidence Level",
            options=["95%", "99%", "99.9%"],
            index=0
        )
        
        holding_period = st.sidebar.selectbox(
            "Holding Period",
            options=["1 Day", "5 Days", "10 Days", "1 Month"],
            index=0
        )
        
        # Risk limits
        st.sidebar.markdown("###  ️ Risk Limits")
        
        var_limit = st.sidebar.number_input(
            "Daily VaR Limit ($)",
            min_value=50000,
            max_value=1000000,
            value=200000,
            step=25000
        )
        
        concentration_limit = st.sidebar.slider(
            "Concentration Limit (%)",
            min_value=10,
            max_value=50,
            value=40,
            step=5
        )
        
        beta_limit = st.sidebar.slider(
            "Beta Limit (±)",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.1
        )
        
        # Alert settings
        st.sidebar.markdown("###   Alert Settings")
        
        enable_email_alerts = st.sidebar.checkbox("Email Alerts", value=True)
        enable_sms_alerts = st.sidebar.checkbox("SMS Alerts", value=False)
        enable_slack_alerts = st.sidebar.checkbox("Slack Alerts", value=True)
        
        # Stress testing
        st.sidebar.markdown("###   Stress Testing")
        
        if st.sidebar.button("Run All Scenarios"):
            st.info("Running comprehensive stress test scenarios...")
        
        if st.sidebar.button("Custom Scenario"):
            st.info("Opening custom stress test builder...")
    
    def render_risk_dashboard(self):
        """Render complete risk dashboard."""
        try:
            # Header
            self.render_header()
            
            # Sidebar controls
            self.render_sidebar_controls()
            
            # Generate risk data
            metrics, positions, risk_analytics = self.generate_risk_data()
            self.last_update = datetime.now()
            
            # Main content
            main_content = st.container()
            
            with main_content:
                # Risk metrics overview
                self.render_risk_metrics(metrics)
                
                st.markdown("---")
                
                # Risk limits monitoring
                self.render_risk_limits_monitoring(metrics)
                
                st.markdown("---")
                
                # Position risk analysis
                self.render_position_risk_analysis(positions)
                
                st.markdown("---")
                
                # Risk analytics in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'factor_exposures' in risk_analytics:
                        self.render_factor_exposure(risk_analytics['factor_exposures'])
                    
                    if 'liquidity_analysis' in risk_analytics:
                        self.render_liquidity_analysis(risk_analytics['liquidity_analysis'])
                
                with col2:
                    if 'sector_exposures' in risk_analytics:
                        self.render_sector_exposure(risk_analytics['sector_exposures'])
                    
                    if 'correlation_matrix' in risk_analytics:
                        self.render_correlation_analysis(risk_analytics['correlation_matrix'])
                
                st.markdown("---")
                
                # Stress testing
                if 'stress_scenarios' in risk_analytics:
                    self.render_stress_testing(risk_analytics['stress_scenarios'])
            
        except Exception as e:
            st.error(f"Error rendering risk dashboard: {str(e)}")
            logger.error(f"Risk dashboard error: {str(e)}")

def main():
    """Main function to run the risk dashboard."""
    dashboard = RiskDashboard()
    dashboard.render_risk_dashboard()

if __name__ == "__main__":
    main()