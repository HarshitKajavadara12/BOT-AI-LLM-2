"""
Configuration Interface for QUANTUM-FORGE
Web-based configuration management, parameter tuning, and system settings.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, dash_table, State
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Any, Optional
import yaml
import os
warnings.filterwarnings('ignore')

class ConfigurationInterface:
    """Web-based configuration management interface."""
    
    def __init__(self, port: int = 8054):
        """Initialize configuration interface."""
        self.port = port
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Configuration state
        self.current_config = self._load_default_config()
        self.config_history = []
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _load_default_config(self):
        """Load default configuration."""
        return {
            'risk_management': {
                'max_portfolio_var': 0.02,
                'position_limit': 0.05,
                'sector_limit': 0.15,
                'leverage_limit': 3.0,
                'stop_loss_threshold': 0.03
            },
            'execution': {
                'default_venue': 'SMART',
                'order_timeout': 30,
                'max_order_size': 10000,
                'min_fill_rate': 0.95,
                'slippage_tolerance': 2.0
            },
            'strategies': {
                'alpha_strategy': {
                    'enabled': True,
                    'allocation': 0.4,
                    'lookback_period': 20,
                    'rebalance_frequency': 'daily'
                },
                'beta_strategy': {
                    'enabled': True,
                    'allocation': 0.3,
                    'momentum_threshold': 0.02,
                    'volatility_filter': True
                },
                'gamma_strategy': {
                    'enabled': False,
                    'allocation': 0.2,
                    'correlation_threshold': 0.8,
                    'hedge_ratio': 0.5
                }
            },
            'market_data': {
                'primary_feed': 'Reuters',
                'backup_feed': 'Bloomberg',
                'update_frequency': 1,
                'data_retention_days': 365
            },
            'system': {
                'log_level': 'INFO',
                'max_cpu_usage': 80,
                'max_memory_usage': 16,
                'backup_frequency': 'hourly'
            }
        }
    
    def _setup_layout(self):
        """Setup configuration interface layout."""
        
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("QUANTUM-FORGE Configuration", 
                           className="text-center mb-4",
                           style={'color': '#6c757d', 'font-weight': 'bold'})
                ], width=12)
            ]),
            
            # Configuration Tabs
            dbc.Row([
                dbc.Col([
                    dbc.Tabs(id="config-tabs", active_tab="risk-tab", children=[
                        dbc.Tab(label="Risk Management", tab_id="risk-tab"),
                        dbc.Tab(label="Execution", tab_id="execution-tab"),
                        dbc.Tab(label="Strategies", tab_id="strategies-tab"),
                        dbc.Tab(label="Market Data", tab_id="data-tab"),
                        dbc.Tab(label="System", tab_id="system-tab"),
                        dbc.Tab(label="Advanced", tab_id="advanced-tab")
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Configuration Content
            html.Div(id="config-content"),
            
            # Action Buttons
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("Save Configuration", id="save-btn", 
                                 color="success", className="me-2"),
                        dbc.Button("Load Configuration", id="load-btn", 
                                 color="primary", className="me-2"),
                        dbc.Button("Reset to Default", id="reset-btn", 
                                 color="warning", className="me-2"),
                        dbc.Button("Export Config", id="export-btn", 
                                 color="info", className="me-2"),
                        dbc.Button("Validate Config", id="validate-btn", 
                                 color="secondary")
                    ])
                ], width=12, className="text-center")
            ], className="mb-4"),
            
            # Status and Alerts
            html.Div(id="config-alerts"),
            
            # Configuration History
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Configuration History"),
                        dbc.CardBody([
                            dash_table.DataTable(
                                id='config-history-table',
                                columns=[
                                    {'name': 'Timestamp', 'id': 'timestamp'},
                                    {'name': 'User', 'id': 'user'},
                                    {'name': 'Section', 'id': 'section'},
                                    {'name': 'Changes', 'id': 'changes'},
                                    {'name': 'Status', 'id': 'status'}
                                ],
                                style_cell={'textAlign': 'left'},
                                page_size=10
                            )
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Configuration Validation Results
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Validation Results"),
                        dbc.CardBody([
                            html.Div(id="validation-results")
                        ])
                    ])
                ], width=12)
            ]),
            
            dcc.Store(id='config-store', data=self.current_config),
            dcc.Interval(
                id='config-interval-component',
                interval=10000,  # 10 seconds
                n_intervals=0
            )
            
        ], fluid=True)
    
    def _setup_callbacks(self):
        """Setup configuration interface callbacks."""
        
        @self.app.callback(
            Output('config-content', 'children'),
            [Input('config-tabs', 'active_tab'),
             Input('config-store', 'data')]
        )
        def render_config_content(active_tab, config_data):
            """Render configuration content based on active tab."""
            
            if active_tab == "risk-tab":
                return self._render_risk_config(config_data.get('risk_management', {}))
            elif active_tab == "execution-tab":
                return self._render_execution_config(config_data.get('execution', {}))
            elif active_tab == "strategies-tab":
                return self._render_strategies_config(config_data.get('strategies', {}))
            elif active_tab == "data-tab":
                return self._render_data_config(config_data.get('market_data', {}))
            elif active_tab == "system-tab":
                return self._render_system_config(config_data.get('system', {}))
            elif active_tab == "advanced-tab":
                return self._render_advanced_config(config_data)
            
            return html.Div("Select a configuration section")
    
    def _render_risk_config(self, risk_config):
        """Render risk management configuration."""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Portfolio Risk Limits"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Max Portfolio VaR"),
                                dbc.Input(
                                    id="max-portfolio-var",
                                    type="number",
                                    value=risk_config.get('max_portfolio_var', 0.02),
                                    step=0.001,
                                    min=0,
                                    max=0.1
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Position Limit (% of Portfolio)"),
                                dbc.Input(
                                    id="position-limit",
                                    type="number",
                                    value=risk_config.get('position_limit', 0.05),
                                    step=0.01,
                                    min=0,
                                    max=1
                                )
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Sector Limit (% of Portfolio)"),
                                dbc.Input(
                                    id="sector-limit",
                                    type="number",
                                    value=risk_config.get('sector_limit', 0.15),
                                    step=0.01,
                                    min=0,
                                    max=1
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Maximum Leverage"),
                                dbc.Input(
                                    id="leverage-limit",
                                    type="number",
                                    value=risk_config.get('leverage_limit', 3.0),
                                    step=0.1,
                                    min=1,
                                    max=10
                                )
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Stop Loss Threshold"),
                                dbc.Input(
                                    id="stop-loss-threshold",
                                    type="number",
                                    value=risk_config.get('stop_loss_threshold', 0.03),
                                    step=0.005,
                                    min=0,
                                    max=0.2
                                )
                            ], width=6)
                        ])
                    ])
                ])
            ], width=12)
        ])
    
    def _render_execution_config(self, exec_config):
        """Render execution configuration."""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Order Execution Settings"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Default Venue"),
                                dbc.Select(
                                    id="default-venue",
                                    options=[
                                        {"label": "SMART", "value": "SMART"},
                                        {"label": "NYSE", "value": "NYSE"},
                                        {"label": "NASDAQ", "value": "NASDAQ"},
                                        {"label": "BATS", "value": "BATS"}
                                    ],
                                    value=exec_config.get('default_venue', 'SMART')
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Order Timeout (seconds)"),
                                dbc.Input(
                                    id="order-timeout",
                                    type="number",
                                    value=exec_config.get('order_timeout', 30),
                                    min=1,
                                    max=300
                                )
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Max Order Size"),
                                dbc.Input(
                                    id="max-order-size",
                                    type="number",
                                    value=exec_config.get('max_order_size', 10000),
                                    min=1,
                                    max=1000000
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Min Fill Rate"),
                                dbc.Input(
                                    id="min-fill-rate",
                                    type="number",
                                    value=exec_config.get('min_fill_rate', 0.95),
                                    step=0.01,
                                    min=0,
                                    max=1
                                )
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Slippage Tolerance (bps)"),
                                dbc.Input(
                                    id="slippage-tolerance",
                                    type="number",
                                    value=exec_config.get('slippage_tolerance', 2.0),
                                    step=0.1,
                                    min=0,
                                    max=50
                                )
                            ], width=6)
                        ])
                    ])
                ])
            ], width=12)
        ])
    
    def _render_strategies_config(self, strategies_config):
        """Render strategies configuration."""
        strategies_cards = []
        
        for strategy_name, strategy_config in strategies_config.items():
            card = dbc.Card([
                dbc.CardBody([
                    html.H5(strategy_name.replace('_', ' ').title()),
                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                id=f"{strategy_name}-enabled",
                                options=[{"label": "Enabled", "value": "enabled"}],
                                value=["enabled"] if strategy_config.get('enabled', False) else []
                            )
                        ], width=3),
                        dbc.Col([
                            dbc.Label("Allocation"),
                            dbc.Input(
                                id=f"{strategy_name}-allocation",
                                type="number",
                                value=strategy_config.get('allocation', 0.0),
                                step=0.01,
                                min=0,
                                max=1
                            )
                        ], width=3)
                    ])
                ])
            ], className="mb-3")
            strategies_cards.append(card)
        
        return dbc.Row([
            dbc.Col(strategies_cards, width=12)
        ])
    
    def _render_data_config(self, data_config):
        """Render market data configuration."""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Market Data Settings"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Primary Data Feed"),
                                dbc.Select(
                                    id="primary-feed",
                                    options=[
                                        {"label": "Reuters", "value": "Reuters"},
                                        {"label": "Bloomberg", "value": "Bloomberg"},
                                        {"label": "Refinitiv", "value": "Refinitiv"},
                                        {"label": "IEX", "value": "IEX"}
                                    ],
                                    value=data_config.get('primary_feed', 'Reuters')
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Backup Data Feed"),
                                dbc.Select(
                                    id="backup-feed",
                                    options=[
                                        {"label": "Bloomberg", "value": "Bloomberg"},
                                        {"label": "Reuters", "value": "Reuters"},
                                        {"label": "Refinitiv", "value": "Refinitiv"},
                                        {"label": "IEX", "value": "IEX"}
                                    ],
                                    value=data_config.get('backup_feed', 'Bloomberg')
                                )
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Update Frequency (seconds)"),
                                dbc.Input(
                                    id="update-frequency",
                                    type="number",
                                    value=data_config.get('update_frequency', 1),
                                    min=0.1,
                                    max=60,
                                    step=0.1
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Data Retention (days)"),
                                dbc.Input(
                                    id="data-retention",
                                    type="number",
                                    value=data_config.get('data_retention_days', 365),
                                    min=1,
                                    max=3650
                                )
                            ], width=6)
                        ])
                    ])
                ])
            ], width=12)
        ])
    
    def _render_system_config(self, system_config):
        """Render system configuration."""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("System Settings"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Log Level"),
                                dbc.Select(
                                    id="log-level",
                                    options=[
                                        {"label": "DEBUG", "value": "DEBUG"},
                                        {"label": "INFO", "value": "INFO"},
                                        {"label": "WARNING", "value": "WARNING"},
                                        {"label": "ERROR", "value": "ERROR"}
                                    ],
                                    value=system_config.get('log_level', 'INFO')
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Backup Frequency"),
                                dbc.Select(
                                    id="backup-frequency",
                                    options=[
                                        {"label": "Every 15 minutes", "value": "15min"},
                                        {"label": "Hourly", "value": "hourly"},
                                        {"label": "Daily", "value": "daily"},
                                        {"label": "Weekly", "value": "weekly"}
                                    ],
                                    value=system_config.get('backup_frequency', 'hourly')
                                )
                            ], width=6)
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Max CPU Usage (%)"),
                                dbc.Input(
                                    id="max-cpu-usage",
                                    type="number",
                                    value=system_config.get('max_cpu_usage', 80),
                                    min=10,
                                    max=100
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Max Memory Usage (GB)"),
                                dbc.Input(
                                    id="max-memory-usage",
                                    type="number",
                                    value=system_config.get('max_memory_usage', 16),
                                    min=1,
                                    max=128
                                )
                            ], width=6)
                        ])
                    ])
                ])
            ], width=12)
        ])
    
    def _render_advanced_config(self, config_data):
        """Render advanced configuration as JSON editor."""
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Advanced Configuration (JSON)"),
                        dbc.Textarea(
                            id="json-config-editor",
                            value=json.dumps(config_data, indent=2),
                            style={"height": "500px", "font-family": "monospace"}
                        )
                    ])
                ])
            ], width=12)
        ])
    
    def run(self, debug=False):
        """Run the configuration interface."""
        print(f"Starting QUANTUM-FORGE Configuration Interface on port {self.port}")
        print(f"Access interface at: http://localhost:{self.port}")
        
        self.app.run(
            debug=debug,
            port=self.port,
            host='0.0.0.0',
            use_reloader=False
        )

if __name__ == "__main__":
    interface = ConfigurationInterface()
    interface.run(debug=True)