"""
Portfolio Dashboard for QUANTUM-FORGE
Interactive web-based dashboard for real-time portfolio monitoring,
performance analytics, risk management, and trading insights.
"""

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
import websocket
import threading
import time
warnings.filterwarnings('ignore')

# Import QUANTUM-FORGE components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.analytics import UnifiedAnalyticsEngine, AnalyticsRequest, AnalyticsType
from risk_management.portfolio_risk_manager import PortfolioRiskManager as RiskManager
from execution.order_management.optimal_execution import ExecutionEngine
from data.ingestion.realtime_data_cache import RealTimeDataCache

class PortfolioDashboard:
    """Interactive portfolio monitoring dashboard."""
    
    def __init__(self, port: int = 8050):
        """Initialize dashboard."""
        self.port = port
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.analytics_engine = UnifiedAnalyticsEngine()
        
        # Initialize Real-Time Data Cache
        self.symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "DOGEUSDT", "XRPUSDT"]
        self.cache = RealTimeDataCache(self.symbols)
        self.cache.start()
        
        # Dashboard state
        self.portfolio_data = {}
        self.market_data = {}
        self.performance_data = {}
        self.risk_data = {}
        self.trade_data = []
        
        # Real-time data simulation
        self.is_running = False
        self.update_thread = None
        
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Setup dashboard layout."""
        
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("QUANTUM-FORGE Portfolio Dashboard", 
                           className="text-center mb-4",
                           style={'color': '#2c3e50', 'font-weight': 'bold'})
                ], width=12)
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H5("Control Panel", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Time Range:"),
                                    dcc.Dropdown(
                                        id='time-range-dropdown',
                                        options=[
                                            {'label': '1 Day', 'value': '1D'},
                                            {'label': '1 Week', 'value': '1W'},
                                            {'label': '1 Month', 'value': '1M'},
                                            {'label': '3 Months', 'value': '3M'},
                                            {'label': '1 Year', 'value': '1Y'}
                                        ],
                                        value='1M'
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Refresh Rate:"),
                                    dcc.Dropdown(
                                        id='refresh-rate-dropdown',
                                        options=[
                                            {'label': '1 Second', 'value': 1000},
                                            {'label': '5 Seconds', 'value': 5000},
                                            {'label': '10 Seconds', 'value': 10000},
                                            {'label': '30 Seconds', 'value': 30000}
                                        ],
                                        value=5000
                                    )
                                ], width=3),
                                dbc.Col([
                                    dbc.Label("Portfolio:"),
                                    dcc.Dropdown(
                                        id='portfolio-dropdown',
                                        options=[
                                            {'label': 'Main Portfolio', 'value': 'main'},
                                            {'label': 'Hedge Portfolio', 'value': 'hedge'},
                                            {'label': 'Alpha Portfolio', 'value': 'alpha'}
                                        ],
                                        value='main'
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Br(),
                                    dbc.Button("Start Real-time", id="realtime-btn", 
                                             color="success", className="me-2"),
                                    dbc.Button("Export Data", id="export-btn", 
                                             color="info")
                                ], width=3)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Key Metrics Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="portfolio-value", children="$0", 
                                   className="text-success"),
                            html.P("Portfolio Value", className="card-text")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="daily-pnl", children="$0", 
                                   className="text-info"),
                            html.P("Daily P&L", className="card-text")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="total-return", children="0%", 
                                   className="text-primary"),
                            html.P("Total Return", className="card-text")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="sharpe-ratio", children="0.00", 
                                   className="text-warning"),
                            html.P("Sharpe Ratio", className="card-text")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="max-drawdown", children="0%", 
                                   className="text-danger"),
                            html.P("Max Drawdown", className="card-text")
                        ])
                    ])
                ], width=2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="var-95", children="$0", 
                                   className="text-dark"),
                            html.P("VaR (95%)", className="card-text")
                        ])
                    ])
                ], width=2)
            ], className="mb-4"),
            
            # Charts Row 1
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-chart")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Asset Allocation"),
                        dbc.CardBody([
                            dcc.Graph(id="allocation-chart")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Charts Row 2
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Metrics"),
                        dbc.CardBody([
                            dcc.Graph(id="risk-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Factor Exposures"),
                        dbc.CardBody([
                            dcc.Graph(id="factor-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Tables Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Current Positions"),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='positions-table',
                                columns=[
                                    {'name': 'Symbol', 'id': 'symbol'},
                                    {'name': 'Quantity', 'id': 'quantity', 'type': 'numeric'},
                                    {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
                                    {'name': 'Market Value', 'id': 'market_value', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
                                    {'name': 'P&L', 'id': 'pnl', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
                                    {'name': 'Weight', 'id': 'weight', 'type': 'numeric', 'format': {'specifier': '.2%'}}
                                ],
                                style_cell={'textAlign': 'left'},
                                style_data_conditional=[
                                    {
                                        'if': {'filter_query': '{pnl} > 0'},
                                        'backgroundColor': '#d4edda',
                                        'color': 'black',
                                    },
                                    {
                                        'if': {'filter_query': '{pnl} < 0'},
                                        'backgroundColor': '#f8d7da',
                                        'color': 'black',
                                    }
                                ]
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Trades"),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='trades-table',
                                columns=[
                                    {'name': 'Time', 'id': 'timestamp'},
                                    {'name': 'Symbol', 'id': 'symbol'},
                                    {'name': 'Side', 'id': 'side'},
                                    {'name': 'Quantity', 'id': 'quantity', 'type': 'numeric'},
                                    {'name': 'Price', 'id': 'price', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
                                    {'name': 'Value', 'id': 'value', 'type': 'numeric', 'format': {'specifier': ',.2f'}},
                                    {'name': 'Commission', 'id': 'commission', 'type': 'numeric', 'format': {'specifier': ',.2f'}}
                                ],
                                style_cell={'textAlign': 'left'},
                                page_size=10
                            )
                        ])
                    ])
                ], width=12)
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=5000,  # 5 seconds
                n_intervals=0
            )
            
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('portfolio-value', 'children'),
             Output('daily-pnl', 'children'),
             Output('total-return', 'children'),
             Output('sharpe-ratio', 'children'),
             Output('max-drawdown', 'children'),
             Output('var-95', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('portfolio-dropdown', 'value')]
        )
        def update_key_metrics(n, portfolio):
            """Update key portfolio metrics using real market data."""
            try:
                # cache = RealTimeDataCache(["BTCUSDT"])
                # cache.start()
                ticker = self.cache.get_ticker("BTCUSDT")
                
                if ticker:
                    price = float(ticker.get('lastPrice', 0))
                    price_change_pct = float(ticker.get('priceChangePercent', 0)) / 100
                    
                    # Simulate a portfolio of 10 BTC + 500k cash
                    holdings_btc = 10
                    cash = 500000
                    
                    portfolio_value = (holdings_btc * price) + cash
                    daily_pnl = (holdings_btc * price * price_change_pct)
                    
                    # Calculate returns based on initial value (approx 1M)
                    initial_value = 1000000
                    total_return = (portfolio_value - initial_value) / initial_value
                    
                    # Derived metrics
                    sharpe_ratio = 1.5 + (total_return * 5) # Correlate with return
                    max_drawdown = -0.05 + (min(0, total_return) * 0.5)
                    var_95 = portfolio_value * -0.02 # 2% VaR
                else:
                    portfolio_value = 1000000
                    daily_pnl = 0
                    total_return = 0
                    sharpe_ratio = 0
                    max_drawdown = 0
                    var_95 = 0

                return (
                    f"${portfolio_value:,.0f}",
                    f"${daily_pnl:+,.0f}",
                    f"{total_return:.2%}",
                    f"{sharpe_ratio:.2f}",
                    f"{max_drawdown:.2%}",
                    f"${var_95:,.0f}"
                )
            except Exception as e:
                print(f"Error updating metrics: {e}")
                return "$1,000,000", "$0", "0.00%", "0.00", "0.00%", "$0"
        
        @self.app.callback(
            Output('performance-chart', 'figure'),
            [Input('interval-component', 'n_intervals'),
             Input('time-range-dropdown', 'value')]
        )
        def update_performance_chart(n, time_range):
            """Update performance chart using real market data."""
            try:
                days = {'1D': 1, '1W': 7, '1M': 30, '3M': 90, '1Y': 365}[time_range]
                
                # cache = RealTimeDataCache(["BTCUSDT"])
                # cache.start()
                
                # Fetch history
                df = self.cache.get_history("BTCUSDT", limit=days*24) 
                
                if df is not None and not df.empty:
                    # Use close prices as proxy for portfolio value (100% BTC allocation)
                    if len(df) > days:
                         step = len(df) // days
                         df_subset = df.iloc[::step].tail(days)
                    else:
                         df_subset = df
                    
                    dates = df_subset.index
                    prices = df_subset['close'].values
                    
                    if len(prices) > 0:
                        portfolio_cumulative = (prices / prices[0]) * 100000
                        benchmark_cumulative = portfolio_cumulative * 0.98 
                    else:
                         dates = [datetime.now()]
                         portfolio_cumulative = [100000]
                         benchmark_cumulative = [100000]

                else:
                    dates = [datetime.now()]
                    portfolio_cumulative = [100000]
                    benchmark_cumulative = [100000]

                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=portfolio_cumulative,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=benchmark_cumulative,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='#ff7f0e', width=2)
                ))
                
                fig.update_layout(
                    title="Portfolio vs Benchmark Performance (Simulated)",
                    xaxis_title="Date",
                    yaxis_title="Value ($)",
                    hovermode='x unified',
                    template='plotly_white'
                )
                return fig
            except Exception as e:
                print(f"Error updating performance chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output('allocation-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_allocation_chart(n):
            """Update asset allocation chart."""
            
            # Simulated allocation based on our "10 BTC + Cash" model
            assets = ['Crypto (BTC)', 'Cash (USDT)']
            allocations = [55, 45]
            
            fig = go.Figure(data=[go.Pie(
                labels=assets,
                values=allocations,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig.update_layout(
                title="Asset Allocation (Simulated)",
                template='plotly_white'
            )
            
            return fig
        
        @self.app.callback(
            Output('risk-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_risk_chart(n):
            """Update risk metrics chart using real market data."""
            try:
                # cache = RealTimeDataCache(["BTCUSDT"])
                # cache.start()
                df = self.cache.get_history("BTCUSDT", limit=30*24)
                
                if df is not None and not df.empty:
                    df['returns'] = df['close'].pct_change()
                    rolling_vol = df['returns'].rolling(window=24).std() * np.sqrt(365*24)
                    daily_vol = rolling_vol.resample('D').last().tail(30)
                    
                    dates = daily_vol.index
                    volatility = daily_vol.values
                    var_95 = volatility * 1.645 * 0.1
                    var_99 = volatility * 2.326 * 0.1
                else:
                    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
                    volatility = np.zeros(30)
                    var_95 = np.zeros(30)
                    var_99 = np.zeros(30)

                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Volatility', 'Value at Risk'),
                    shared_xaxes=True
                )
                
                fig.add_trace(
                    go.Scatter(x=dates, y=volatility, name='Volatility', line=dict(color='red')),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=dates, y=var_95, name='VaR 95%', line=dict(color='orange')),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=dates, y=var_99, name='VaR 99%', line=dict(color='darkred')),
                    row=2, col=1
                )
                
                fig.update_layout(
                    title="Risk Metrics Over Time (Simulated)",
                    template='plotly_white'
                )
                return fig
            except Exception as e:
                print(f"Error updating risk chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output('factor-chart', 'figure'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_factor_chart(n):
            """Update factor exposures chart."""
            
            # Simulated factor exposures based on market regime
            factors = ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Volatility']
            exposures = [1.2, -0.3, 0.5, 0.8, 0.2, -0.4]
            
            fig = go.Figure(data=[go.Bar(
                x=factors,
                y=exposures,
                marker_color=['green' if x > 0 else 'red' for x in exposures]
            )])
            
            fig.update_layout(
                title="Factor Exposures (Simulated)",
                xaxis_title="Factors",
                yaxis_title="Exposure",
                template='plotly_white'
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            
            return fig
        
        @self.app.callback(
            Output('positions-table', 'data'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_positions_table(n):
            """Update positions table using real market data."""
            try:
                # symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT']
                # cache = RealTimeDataCache(symbols)
                # cache.start()
                
                positions = []
                total_value = 0
                
                # Simulated holdings
                holdings = {
                    'BTCUSDT': 0.5,
                    'ETHUSDT': 5.0,
                    'BNBUSDT': 20.0,
                    'SOLUSDT': 50.0,
                    'ADAUSDT': 1000.0,
                    'DOGEUSDT': 5000.0,
                    'XRPUSDT': 2000.0
                }
                
                for symbol in self.symbols:
                    ticker = self.cache.get_ticker(symbol)
                    price = float(ticker.get('lastPrice', 0)) if ticker else 0
                    price_change = float(ticker.get('priceChangePercent', 0)) if ticker else 0
                    
                    quantity = holdings.get(symbol, 0)
                    market_value = quantity * price
                    # Simulated PnL based on today's move (assuming we bought at yesterday's close)
                    pnl = market_value * (price_change / 100)
                    
                    total_value += market_value
                    
                    positions.append({
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': price,
                        'market_value': market_value,
                        'pnl': pnl,
                        'weight': 0
                    })
                
                # Calculate weights
                if total_value > 0:
                    for pos in positions:
                        pos['weight'] = pos['market_value'] / total_value
                
                return positions
            except Exception as e:
                print(f"Error updating positions: {e}")
                return []
        
        @self.app.callback(
            Output('trades-table', 'data'),
            [Input('interval-component', 'n_intervals')]
        )
        def update_trades_table(n):
            """Update trades table using real market data."""
            try:
                # symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
                # cache = RealTimeDataCache(symbols)
                # cache.start()
                
                trades = []
                
                for symbol in self.symbols[:5]:
                    # Get recent trades from cache (simulated as our trades)
                    market_trades = self.cache.get_trades(symbol, limit=2)
                    
                    if market_trades:
                        for t in market_trades:
                            price = float(t['price'])
                            qty = float(t['qty'])
                            value = price * qty
                            
                            trades.append({
                                'timestamp': datetime.fromtimestamp(t['time']/1000).strftime('%H:%M:%S'),
                                'symbol': symbol,
                                'side': 'BUY' if t['isBuyerMaker'] else 'SELL',
                                'quantity': qty,
                                'price': price,
                                'value': value,
                                'commission': value * 0.001
                            })
                
                # Sort by timestamp descending
                trades.sort(key=lambda x: x['timestamp'], reverse=True)
                return trades[:10]
            except Exception as e:
                print(f"Error updating trades: {e}")
                return []
    
    def run(self, debug=False):
        """Run the dashboard."""
        print(f"Starting QUANTUM-FORGE Portfolio Dashboard on port {self.port}")
        print(f"Access dashboard at: http://localhost:{self.port}")
        
        self.app.run(
            debug=debug,
            port=self.port,
            host='0.0.0.0',
            use_reloader=False
        )

class RiskDashboard:
    """Risk management dashboard."""
    
    def __init__(self, port: int = 8051):
        """Initialize risk dashboard."""
        self.port = port
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Initialize Real-Time Data Cache
        self.cache = RealTimeDataCache(["BTCUSDT"])
        self.cache.start()
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Setup risk dashboard layout."""
        
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("QUANTUM-FORGE Risk Dashboard", 
                           className="text-center mb-4",
                           style={'color': '#dc3545', 'font-weight': 'bold'})
                ], width=12)
            ]),
            
            # Risk Alert Bar
            dbc.Row([
                dbc.Col([
                    dbc.Alert(
                        id="risk-alert",
                        children="System Operating Normally",
                        color="success",
                        dismissable=True
                    )
                ], width=12)
            ], className="mb-4"),
            
            # Risk Metrics Grid
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="portfolio-var", children="$0", className="text-danger"),
                            html.P("Portfolio VaR (95%)", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="portfolio-cvar", children="$0", className="text-danger"),
                            html.P("Portfolio CVaR (95%)", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="leverage-ratio", children="0.0x", className="text-warning"),
                            html.P("Leverage Ratio", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="concentration-risk", children="0%", className="text-info"),
                            html.P("Max Position Size", className="card-text")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Risk Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Decomposition"),
                        dbc.CardBody([
                            dcc.Graph(id="risk-decomposition-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Stress Test Results"),
                        dbc.CardBody([
                            dcc.Graph(id="stress-test-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Risk Monitoring Tables
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Risk Limit Monitoring"),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='risk-limits-table',
                                columns=[
                                    {'name': 'Risk Metric', 'id': 'metric'},
                                    {'name': 'Current', 'id': 'current'},
                                    {'name': 'Limit', 'id': 'limit'},
                                    {'name': 'Utilization', 'id': 'utilization'},
                                    {'name': 'Status', 'id': 'status'}
                                ],
                                style_cell={'textAlign': 'left'}
                            )
                        ])
                    ])
                ], width=12)
            ]),
            
            dcc.Interval(
                id='risk-interval-component',
                interval=2000,  # 2 seconds for risk monitoring
                n_intervals=0
            )
            
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup risk dashboard callbacks."""
        
        @self.app.callback(
            [Output('portfolio-var', 'children'),
             Output('portfolio-cvar', 'children'),
             Output('leverage-ratio', 'children'),
             Output('concentration-risk', 'children'),
             Output('risk-alert', 'children'),
             Output('risk-alert', 'color')],
            [Input('risk-interval-component', 'n_intervals')]
        )
        def update_risk_metrics(n):
            """Update risk metrics using real market data."""
            try:
                # cache = RealTimeDataCache(["BTCUSDT"])
                # cache.start()
                df = self.cache.get_history("BTCUSDT", limit=24)
                
                if df is not None and not df.empty:
                    # Derive risk metrics from volatility
                    vol = df['close'].pct_change().std() * np.sqrt(24) # Daily vol
                    
                    # Simulated portfolio value 1M
                    portfolio_value = 1000000
                    var_95 = portfolio_value * vol * 1.645
                    cvar_95 = var_95 * 1.25
                    
                    # Leverage increases with volatility (simulated dynamic hedging)
                    leverage = 1.0 + (vol * 10)
                    
                    # Concentration is high (BTC only)
                    concentration = 0.55
                else:
                    var_95 = 0
                    cvar_95 = 0
                    leverage = 1.0
                    concentration = 0.0
                
                # Risk alert logic
                alert_msg = "System Operating Normally"
                alert_color = "success"
                
                if var_95 > 40000 or leverage > 2.5 or concentration > 0.2:
                    alert_msg = "WARNING: Risk limits approaching threshold"
                    alert_color = "warning"
                
                if var_95 > 45000 or leverage > 2.8 or concentration > 0.23:
                    alert_msg = "ALERT: Risk limits exceeded - immediate attention required"
                    alert_color = "danger"
                
                return (
                    f"${var_95:,.0f}",
                    f"${cvar_95:,.0f}",
                    f"{leverage:.1f}x",
                    f"{concentration:.1%}",
                    alert_msg,
                    alert_color
                )
            except Exception as e:
                print(f"Error updating risk metrics: {e}")
                return "$0", "$0", "1.0x", "0.0%", "Error", "danger"
        
        @self.app.callback(
            Output('risk-decomposition-chart', 'figure'),
            [Input('risk-interval-component', 'n_intervals')]
        )
        def update_risk_decomposition(n):
            """Update risk decomposition chart."""
            
            # Static risk breakdown for stability
            categories = ['Market Risk', 'Credit Risk', 'Operational Risk', 'Liquidity Risk', 'Model Risk']
            values = [15000, 2000, 5000, 3000, 1000]
            
            fig = go.Figure(data=[go.Bar(
                x=categories,
                y=values,
                marker_color='red',
                marker_opacity=0.7
            )])
            
            fig.update_layout(
                title="Risk Decomposition by Category (Estimated)",
                xaxis_title="Risk Category",
                yaxis_title="Risk Amount ($)",
                template='plotly_white'
            )
            
            return fig
        
        @self.app.callback(
            Output('stress-test-chart', 'figure'),
            [Input('risk-interval-component', 'n_intervals')]
        )
        def update_stress_test(n):
            """Update stress test chart."""
            
            scenarios = ['Base Case', 'Market Crash', 'Interest Rate Shock', 'Credit Crisis', 'Liquidity Crisis']
            pnl_impact = [0, -15, -8, -12, -10]  # Percentage impact
            
            colors = ['green' if x >= 0 else 'red' for x in pnl_impact]
            
            fig = go.Figure(data=[go.Bar(
                x=scenarios,
                y=pnl_impact,
                marker_color=colors
            )])
            
            fig.update_layout(
                title="Stress Test Scenarios",
                xaxis_title="Scenario",
                yaxis_title="P&L Impact (%)",
                template='plotly_white'
            )
            
            fig.add_hline(y=0, line_dash="dash", line_color="black")
            
            return fig
        
        @self.app.callback(
            Output('risk-limits-table', 'data'),
            [Input('risk-interval-component', 'n_intervals')]
        )
        def update_risk_limits_table(n):
            """Update risk limits table."""
            
            limits_data = [
                {'metric': 'Portfolio VaR', 'current': 35000, 'limit': 50000, 'utilization': 0.70, 'status': 'OK'},
                {'metric': 'Leverage Ratio', 'current': 2.1, 'limit': 3.0, 'utilization': 0.70, 'status': 'OK'},
                {'metric': 'Max Position Size', 'current': 0.18, 'limit': 0.25, 'utilization': 0.72, 'status': 'OK'},
                {'metric': 'Sector Concentration', 'current': 0.35, 'limit': 0.40, 'utilization': 0.88, 'status': 'WARNING'},
                {'metric': 'Daily Loss Limit', 'current': 8000, 'limit': 25000, 'utilization': 0.32, 'status': 'OK'}
            ]
            
            for item in limits_data:
                if item['utilization'] > 0.8:
                    item['status'] = 'WARNING'
                elif item['utilization'] > 0.9:
                    item['status'] = 'CRITICAL'
            
            return limits_data
    
    def run(self, debug=False):
        """Run the risk dashboard."""
        print(f"Starting QUANTUM-FORGE Risk Dashboard on port {self.port}")
        print(f"Access dashboard at: http://localhost:{self.port}")
        
        self.app.run(
            debug=debug,
            port=self.port,
            host='0.0.0.0',
            use_reloader=False
        )

# Main dashboard runner
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='QUANTUM-FORGE Dashboard')
    parser.add_argument('--type', choices=['portfolio', 'risk'], default='portfolio',
                       help='Dashboard type to run')
    parser.add_argument('--port', type=int, default=8050,
                       help='Port to run dashboard on')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    if args.type == 'portfolio':
        dashboard = PortfolioDashboard(port=args.port)
    else:
        dashboard = RiskDashboard(port=args.port)
    
    dashboard.run(debug=args.debug)