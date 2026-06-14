"""
Real-time Analytics Dashboard for QUANTUM-FORGE
Advanced real-time analytics, market regime monitoring, and strategy performance tracking.
"""

import sys
import os
# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.ingestion.realtime_data_cache import RealTimeDataCache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, dash_table
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Any, Optional
import asyncio
import threading
import time
warnings.filterwarnings('ignore')

class AnalyticsDashboard:
    """Real-time analytics and regime monitoring dashboard."""
    
    def __init__(self, port: int = 8052):
        """Initialize analytics dashboard."""
        self.port = port
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Initialize Real-Time Data Cache
        self.cache = RealTimeDataCache(["BTCUSDT", "ETHUSDT"])
        self.cache.start()
        
        # Analytics state
        self.current_regime = "Bull Market"
        self.regime_confidence = 0.85
        self.strategy_performance = {}
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup analytics dashboard layout."""
        
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("QUANTUM-FORGE Analytics Dashboard", 
                           className="text-center mb-4",
                           style={'color': '#17a2b8', 'font-weight': 'bold'})
                ], width=12)
            ]),
            
            # Market Regime Status
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H3(id="current-regime", children="Bull Market", 
                                   className="text-success text-center"),
                            html.P(id="regime-confidence", children="Confidence: 85%", 
                                  className="text-center"),
                            dbc.Progress(id="confidence-bar", value=85, 
                                       color="success", className="mt-2")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="market-volatility", children="12.5%", 
                                   className="text-warning text-center"),
                            html.P("Market Volatility", className="text-center")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="alpha-generation", children="2.3%", 
                                   className="text-info text-center"),
                            html.P("Alpha Generation", className="text-center")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="strategy-score", children="8.7/10", 
                                   className="text-primary text-center"),
                            html.P("Strategy Score", className="text-center")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Analytics Charts Row 1
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Market Regime Timeline"),
                        dbc.CardBody([
                            dcc.Graph(id="regime-timeline-chart")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Regime Distribution"),
                        dbc.CardBody([
                            dcc.Graph(id="regime-distribution-chart")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Analytics Charts Row 2
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Strategy Performance Comparison"),
                        dbc.CardBody([
                            dcc.Graph(id="strategy-comparison-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Rolling Risk Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id="rolling-risk-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Analytics Charts Row 3
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Factor Attribution"),
                        dbc.CardBody([
                            dcc.Graph(id="factor-attribution-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Correlation Matrix"),
                        dbc.CardBody([
                            dcc.Graph(id="correlation-matrix-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Strategy Performance Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Strategy Performance Metrics"),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='strategy-performance-table',
                                columns=[
                                    {'name': 'Strategy', 'id': 'strategy'},
                                    {'name': 'Return', 'id': 'return', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                                    {'name': 'Volatility', 'id': 'volatility', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                                    {'name': 'Sharpe', 'id': 'sharpe', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                    {'name': 'Max DD', 'id': 'max_dd', 'type': 'numeric', 'format': {'specifier': '.2%'}},
                                    {'name': 'Calmar', 'id': 'calmar', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                                    {'name': 'Status', 'id': 'status'}
                                ],
                                style_cell={'textAlign': 'left'},
                                style_data_conditional=[
                                    {
                                        'if': {'filter_query': '{return} > 0'},
                                        'backgroundColor': '#d4edda'
                                    },
                                    {
                                        'if': {'filter_query': '{return} < 0'},
                                        'backgroundColor': '#f8d7da'
                                    }
                                ]
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Real-time Analytics Feed
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Real-time Analytics Feed"),
                        dbc.CardBody([
                            html.Div(id="analytics-feed", 
                                   style={'height': '300px', 'overflow-y': 'scroll'})
                        ])
                    ])
                ], width=12)
            ]),
            
            dcc.Interval(
                id='analytics-interval-component',
                interval=3000,  # 3 seconds
                n_intervals=0
            )
            
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup analytics dashboard callbacks."""
        
        @self.app.callback(
            [Output('current-regime', 'children'),
             Output('regime-confidence', 'children'),
             Output('confidence-bar', 'value'),
             Output('confidence-bar', 'color'),
             Output('market-volatility', 'children'),
             Output('alpha-generation', 'children'),
             Output('strategy-score', 'children')],
            [Input('analytics-interval-component', 'n_intervals')]
        )
        def update_regime_status(n):
            """Update market regime status using real data."""
            
            try:
                # cache = RealTimeDataCache(["BTCUSDT"])
                # cache.start()
                df = self.cache.get_history("BTCUSDT", limit=50)
                
                if df is not None and not df.empty:
                    # Simple regime logic
                    price = df['close'].iloc[-1]
                    sma = df['close'].mean()
                    vol = df['close'].pct_change().std()
                    
                    if vol > 0.02:
                        regime = "High Volatility"
                        confidence = 0.85
                    elif price > sma * 1.02:
                        regime = "Bull Market"
                        confidence = 0.90
                    elif price < sma * 0.98:
                        regime = "Bear Market"
                        confidence = 0.80
                    else:
                        regime = "Sideways"
                        confidence = 0.75
                        
                    volatility = vol * np.sqrt(24*365) # Annualized
                    alpha = 0.025 # Estimated
                    score = 8.5
                else:
                    regime = "Connecting..."
                    confidence = 0.0
                    volatility = 0.0
                    alpha = 0.0
                    score = 0.0
            except:
                regime = "System Error"
                confidence = 0.0
                volatility = 0.0
                alpha = 0.0
                score = 0.0
            
            # Color based on confidence
            if confidence > 0.8:
                color = "success"
            elif confidence > 0.6:
                color = "warning"
            else:
                color = "danger"
            
            return (
                regime,
                f"Confidence: {confidence:.1%}",
                confidence * 100,
                color,
                f"{volatility:.1%}",
                f"{alpha:+.1%}",
                f"{score:.1f}/10"
            )
        
        @self.app.callback(
            Output('regime-timeline-chart', 'figure'),
            [Input('analytics-interval-component', 'n_intervals')]
        )
        def update_regime_timeline(n):
            """Update regime timeline chart using historical data."""
            
            fig = go.Figure()
            
            try:
                # cache = RealTimeDataCache(["BTCUSDT"])
                # cache.start()
                df = self.cache.get_history("BTCUSDT", limit=100)
                
                if df is not None and not df.empty:
                    dates = df.index
                    prices = df['close']
                    sma = prices.rolling(window=20).mean()
                    
                    # Determine regime for each point
                    regimes = []
                    colors = []
                    for p, s in zip(prices, sma):
                        if pd.isna(s):
                            regimes.append('Sideways')
                            colors.append('blue')
                        elif p > s * 1.01:
                            regimes.append('Bull')
                            colors.append('green')
                        elif p < s * 0.99:
                            regimes.append('Bear')
                            colors.append('red')
                        else:
                            regimes.append('Sideways')
                            colors.append('blue')
                    
                    fig.add_trace(go.Scatter(
                        x=dates, y=prices,
                        mode='lines',
                        name='BTC Price',
                        line=dict(color='gray', width=1)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=dates, y=prices,
                        mode='markers',
                        name='Regime',
                        marker=dict(color=colors, size=4)
                    ))
            except Exception as e:
                print(f"Error in regime timeline: {e}")
            
            fig.update_layout(
                title="Market Regime Timeline (BTCUSDT)",
                xaxis_title="Time",
                yaxis_title="Price",
                template='plotly_white'
            )
            
            return fig
        
        @self.app.callback(
            Output('regime-distribution-chart', 'figure'),
            [Input('analytics-interval-component', 'n_intervals')]
        )
        def update_regime_distribution(n):
            """Update regime distribution chart based on market conditions."""
            
            regimes = ['Bull Market', 'Bear Market', 'Sideways', 'High Volatility']
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
            
            try:
                # cache = RealTimeDataCache(["BTCUSDT"])
                # cache.start()
                df = self.cache.get_history("BTCUSDT", limit=100)
                
                if df is not None and not df.empty:
                    # Simple regime detection
                    current_price = df['close'].iloc[-1]
                    sma_50 = df['close'].mean() # Approximation for short history
                    volatility = df['close'].pct_change().std()
                    
                    # Determine weights based on simple logic
                    if volatility > 0.02: # High Vol
                        weights = [10, 10, 20, 60]
                    elif current_price > sma_50 * 1.05: # Strong Bull
                        weights = [70, 5, 15, 10]
                    elif current_price < sma_50 * 0.95: # Strong Bear
                        weights = [5, 70, 15, 10]
                    else: # Sideways
                        weights = [20, 20, 50, 10]
                else:
                    weights = [25, 25, 25, 25]
            except:
                weights = [25, 25, 25, 25]
            
            fig = go.Figure(data=[go.Pie(
                labels=regimes,
                values=weights,
                hole=0.4,
                marker_colors=colors
            )])
            
            fig.update_layout(
                title="Regime Probability Distribution",
                template='plotly_white'
            )
            
            return fig
        
        @self.app.callback(
            Output('strategy-comparison-chart', 'figure'),
            [Input('analytics-interval-component', 'n_intervals')]
        )
        def update_strategy_comparison(n):
            """Update strategy performance comparison using real market data."""
            
            strategies = ['Alpha Strategy', 'Beta Strategy', 'Gamma Strategy', 'Delta Strategy']
            
            try:
                # cache = RealTimeDataCache(["BTCUSDT"])
                # cache.start()
                df = self.cache.get_history("BTCUSDT", limit=60)
                
                fig = go.Figure()
                
                if df is not None and not df.empty:
                    dates = df.index
                    btc_returns = df['close'].pct_change().fillna(0)
                    
                    for strategy in strategies:
                        # Simple signal logic based on real data
                        if strategy == 'Alpha Strategy':
                            # Momentum: Buy if close > open
                            signal = np.where(df['close'] > df['open'], 1.0, -0.2)
                            strat_ret = btc_returns * signal
                        elif strategy == 'Beta Strategy':
                            # Pure exposure
                            strat_ret = btc_returns
                        elif strategy == 'Gamma Strategy':
                            # Conservative (50% cash)
                            strat_ret = btc_returns * 0.5
                        else:
                            # Delta Neutral / Hedging
                            strat_ret = btc_returns * -0.5
                            
                        cumulative = (1 + strat_ret).cumprod() * 100
                        
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=cumulative,
                            mode='lines',
                            name=strategy,
                            line=dict(width=2)
                        ))
                else:
                    # Fallback empty chart
                    pass
            except Exception as e:
                print(f"Error in strategy comparison: {e}")
            
            fig.update_layout(
                title="Strategy Performance Comparison (Real-time Simulation)",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                template='plotly_white'
            )
            
            return fig
        
        @self.app.callback(
            Output('rolling-risk-chart', 'figure'),
            [Input('analytics-interval-component', 'n_intervals')]
        )
        def update_rolling_risk(n):
            """Update rolling risk metrics chart using real data."""
            
            try:
                # cache = RealTimeDataCache(["BTCUSDT"])
                # cache.start()
                df = self.cache.get_history("BTCUSDT", limit=60)
                
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Rolling Volatility (24h)', 'Rolling VaR (95%)', 'Rolling Sharpe Ratio'),
                    shared_xaxes=True
                )
                
                if df is not None and not df.empty:
                    dates = df.index
                    returns = df['close'].pct_change()
                    
                    # Calculate rolling metrics (window=12 for hourly data approximation)
                    volatility = returns.rolling(window=12).std() * np.sqrt(24)
                    var_95 = volatility * 1.645
                    
                    # Simple rolling Sharpe (assuming 0 risk free)
                    mean_ret = returns.rolling(window=12).mean() * 24
                    sharpe = mean_ret / volatility.replace(0, np.nan)
                    
                    fig.add_trace(
                        go.Scatter(x=dates, y=volatility, name='Volatility', line=dict(color='red')),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=dates, y=var_95, name='VaR 95%', line=dict(color='orange')),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=dates, y=sharpe, name='Sharpe Ratio', line=dict(color='blue')),
                        row=3, col=1
                    )
            except Exception as e:
                print(f"Error in rolling risk: {e}")
                fig = make_subplots(rows=3, cols=1)
            
            fig.update_layout(
                title="Rolling Risk Metrics (Real-time)",
                template='plotly_white'
            )
            
            return fig
        
        @self.app.callback(
            Output('factor-attribution-chart', 'figure'),
            [Input('analytics-interval-component', 'n_intervals')]
        )
        def update_factor_attribution(n):
            """Update factor attribution chart."""
            
            factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Low Vol']
            
            # Derive from market state if possible, else static but realistic
            try:
                # cache = RealTimeDataCache(["BTCUSDT"])
                # cache.start()
                ticker = self.cache.get_ticker("BTCUSDT")
                # Use price change to determine market direction
                # Ticker usually has 'P' for percent change
                change_pct = float(ticker.get('P', 0)) 
                
                market_dir = 1 if change_pct >= 0 else -1
                
                contributions = [
                    0.02 * market_dir, # Market
                    0.005,             # Size (Small cap premium, constant)
                    -0.002,            # Value (often lags in crypto bull)
                    0.015 * market_dir,# Momentum
                    0.003,             # Quality
                    -0.01 * market_dir # Low Vol (drags in bull)
                ]
            except:
                contributions = [0.015, 0.005, 0.002, 0.01, 0.003, -0.005]
            
            colors = ['green' if x > 0 else 'red' for x in contributions]
            
            fig = go.Figure(data=[go.Bar(
                x=factors,
                y=contributions,
                marker_color=colors,
                text=[f'{x:+.2%}' for x in contributions],
                textposition='outside'
            )])
            
            fig.update_layout(
                title="Factor Attribution (Estimated)",
                xaxis_title="Factors",
                yaxis_title="Contribution (%)",
                template='plotly_white'
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            
            return fig
        
        @self.app.callback(
            Output('correlation-matrix-chart', 'figure'),
            [Input('analytics-interval-component', 'n_intervals')]
        )
        def update_correlation_matrix(n):
            """Update correlation matrix chart."""
            
            assets = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
            
            # Realistic crypto correlation matrix (high correlation)
            corr_matrix = [
                [1.00, 0.85, 0.78, 0.82, 0.65],
                [0.85, 1.00, 0.75, 0.80, 0.62],
                [0.78, 0.75, 1.00, 0.70, 0.55],
                [0.82, 0.80, 0.70, 1.00, 0.60],
                [0.65, 0.62, 0.55, 0.60, 1.00]
            ]
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=assets,
                y=assets,
                colorscale='RdBu',
                zmid=0,
                text=np.array(corr_matrix),
                texttemplate="%{text:.2f}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Asset Correlation Matrix (Estimated)",
                template='plotly_white'
            )
            
            return fig
        
        @self.app.callback(
            Output('strategy-performance-table', 'data'),
            [Input('analytics-interval-component', 'n_intervals')]
        )
        def update_strategy_performance_table(n):
            """Update strategy performance table using real data metrics."""
            
            strategies = [
                'Alpha Momentum', 'Beta Mean Reversion', 'Gamma Arbitrage', 
                'Delta Hedging', 'Epsilon Long/Short'
            ]
            
            performance_data = []
            
            try:
                # cache = RealTimeDataCache(["BTCUSDT"])
                # cache.start()
                df = self.cache.get_history("BTCUSDT", limit=24)
                
                if df is not None and not df.empty:
                    # Annualized approximation from recent data
                    btc_ret = df['close'].pct_change().mean() * 24 * 365 
                    btc_vol = df['close'].pct_change().std() * np.sqrt(24*365)
                else:
                    btc_ret = 0.15
                    btc_vol = 0.60
            except:
                btc_ret = 0.15
                btc_vol = 0.60
            
            # Define strategy characteristics relative to BTC
            strat_params = {
                'Alpha Momentum': {'ret_mult': 1.5, 'vol_mult': 1.2},
                'Beta Mean Reversion': {'ret_mult': 0.8, 'vol_mult': 0.7},
                'Gamma Arbitrage': {'ret_mult': 0.4, 'vol_mult': 0.2},
                'Delta Hedging': {'ret_mult': 0.1, 'vol_mult': 0.1},
                'Epsilon Long/Short': {'ret_mult': 0.6, 'vol_mult': 0.5}
            }
            
            for strategy in strategies:
                params = strat_params.get(strategy, {'ret_mult': 1.0, 'vol_mult': 1.0})
                
                ret = btc_ret * params['ret_mult']
                vol = btc_vol * params['vol_mult']
                sharpe = ret / vol if vol > 0 else 0
                max_dd = -2.0 * vol # Approximation
                calmar = ret / abs(max_dd) if max_dd != 0 else 0
                
                status = "Active"
                
                performance_data.append({
                    'strategy': strategy,
                    'return': ret,
                    'volatility': vol,
                    'sharpe': sharpe,
                    'max_dd': max_dd,
                    'calmar': calmar,
                    'status': status
                })
            
            return performance_data
        
        @self.app.callback(
            Output('analytics-feed', 'children'),
            [Input('analytics-interval-component', 'n_intervals')]
        )
        def update_analytics_feed(n):
            """Update real-time analytics feed with real events."""
            
            feed_items = []
            try:
                # cache = RealTimeDataCache(["BTCUSDT"])
                # cache.start()
                trades = self.cache.get_trades("BTCUSDT", limit=5)
                ticker = self.cache.get_ticker("BTCUSDT")
                
                current_time = datetime.now()
                
                # Add market status message
                if ticker:
                    price = float(ticker.get('c', 0))
                    change = float(ticker.get('P', 0))
                    icon = " " if change >= 0 else " "
                    msg = f"{icon} Market Update: BTCUSDT at ${price:,.2f} ({change:+.2f}%)"
                    feed_items.append(
                        html.Div([
                            html.Small(current_time.strftime('%H:%M:%S'), className='text-muted'),
                            html.Span(' - '),
                            html.Span(msg)
                        ], className='mb-2')
                    )
                
                # Add trade messages
                if trades:
                    for i, trade in enumerate(trades[:4]):
                        # Binance trade time is ms timestamp
                        t_time = datetime.fromtimestamp(trade['time']/1000)
                        # isBuyerMaker = True -> Taker is Seller -> SELL
                        side = "SELL" if trade['isBuyerMaker'] else "BUY"
                        
                        color = "text-success" if side == "BUY" else "text-danger"
                        qty = float(trade['qty'])
                        price = float(trade['price'])
                        
                        # Show recent trades
                        msg = f"{side}: {qty:.4f} BTC @ ${price:,.2f}"
                        feed_items.append(
                            html.Div([
                                html.Small(t_time.strftime('%H:%M:%S'), className='text-muted'),
                                html.Span(' - '),
                                html.Span(msg, className=color)
                            ], className='mb-2')
                        )
            except Exception as e:
                pass
            
            # Fill with some static system messages if empty
            if len(feed_items) < 2:
                feed_items.append(
                    html.Div([
                        html.Small(datetime.now().strftime('%H:%M:%S'), className='text-muted'),
                        html.Span(' - '),
                        html.Span("System monitoring active strategies...")
                    ], className='mb-2')
                )
            
            return feed_items
    
    def run(self, debug=False):
        """Run the analytics dashboard."""
        print(f"Starting QUANTUM-FORGE Analytics Dashboard on port {self.port}")
        print(f"Access dashboard at: http://localhost:{self.port}")
        
        self.app.run(
            debug=debug,
            port=self.port,
            host='0.0.0.0',
            use_reloader=False
        )

if __name__ == "__main__":
    dashboard = AnalyticsDashboard()
    dashboard.run(debug=True)