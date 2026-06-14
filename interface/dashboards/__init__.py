"""
Interface Dashboards Module Initialization
Unified entry point for all QUANTUM-FORGE dashboard interfaces.
"""

from .portfolio_dashboard import PortfolioDashboard, RiskDashboard
from .analytics_dashboard import AnalyticsDashboard
from .trading_dashboard import TradingDashboard
from .config_interface import ConfigurationInterface

import threading
import time
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class DashboardManager:
    """Unified dashboard management system for QUANTUM-FORGE."""
    
    def __init__(self):
        """Initialize dashboard manager."""
        self.dashboards = {
            'portfolio': None,
            'risk': None,
            'analytics': None,
            'trading': None,
            'config': None
        }
        
        self.dashboard_threads = {}
        self.base_port = 8050
        
    def initialize_dashboards(self) -> Dict[str, Any]:
        """Initialize all dashboard instances."""
        try:
            # Initialize Portfolio Dashboard
            self.dashboards['portfolio'] = PortfolioDashboard(port=self.base_port)
            
            # Initialize Risk Dashboard  
            self.dashboards['risk'] = RiskDashboard(port=self.base_port + 1)
            
            # Initialize Analytics Dashboard
            self.dashboards['analytics'] = AnalyticsDashboard(port=self.base_port + 2)
            
            # Initialize Trading Dashboard
            self.dashboards['trading'] = TradingDashboard(port=self.base_port + 3)
            
            # Initialize Configuration Interface
            self.dashboards['config'] = ConfigurationInterface(port=self.base_port + 4)
            
            print("  All QUANTUM-FORGE dashboards initialized successfully")
            return self.dashboards
            
        except Exception as e:
            print(f"  Error initializing dashboards: {e}")
            return {}
    
    def start_dashboard(self, dashboard_name: str, debug: bool = False) -> bool:
        """Start a specific dashboard."""
        if dashboard_name not in self.dashboards:
            print(f"  Dashboard '{dashboard_name}' not found")
            return False
        
        if self.dashboards[dashboard_name] is None:
            print(f"  Dashboard '{dashboard_name}' not initialized")
            return False
        
        try:
            dashboard = self.dashboards[dashboard_name]
            
            # Start dashboard in separate thread
            thread = threading.Thread(
                target=dashboard.run,
                args=(debug,),
                daemon=True,
                name=f"{dashboard_name}_dashboard"
            )
            
            thread.start()
            self.dashboard_threads[dashboard_name] = thread
            
            print(f"  {dashboard_name.title()} Dashboard started successfully")
            return True
            
        except Exception as e:
            print(f"  Error starting {dashboard_name} dashboard: {e}")
            return False
    
    def start_all_dashboards(self, debug: bool = False) -> Dict[str, bool]:
        """Start all dashboards."""
        results = {}
        
        # Initialize dashboards if not already done
        if not any(self.dashboards.values()):
            self.initialize_dashboards()
        
        # Start each dashboard
        for dashboard_name in self.dashboards.keys():
            results[dashboard_name] = self.start_dashboard(dashboard_name, debug)
            time.sleep(1)  # Small delay between dashboard starts
        
        return results
    
    def get_dashboard_urls(self) -> Dict[str, str]:
        """Get URLs for all running dashboards."""
        urls = {}
        
        for dashboard_name, dashboard in self.dashboards.items():
            if dashboard is not None:
                port = dashboard.port
                urls[dashboard_name] = f"http://localhost:{port}"
        
        return urls
    
    def get_dashboard_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all dashboards."""
        status = {}
        
        for dashboard_name, dashboard in self.dashboards.items():
            if dashboard is not None:
                thread = self.dashboard_threads.get(dashboard_name)
                status[dashboard_name] = {
                    'initialized': True,
                    'running': thread is not None and thread.is_alive(),
                    'port': dashboard.port,
                    'url': f"http://localhost:{dashboard.port}"
                }
            else:
                status[dashboard_name] = {
                    'initialized': False,
                    'running': False,
                    'port': None,
                    'url': None
                }
        
        return status
    
    def print_dashboard_summary(self):
        """Print summary of all dashboards."""
        print("\n" + "="*80)
        print("QUANTUM-FORGE DASHBOARD SUMMARY")
        print("="*80)
        
        status = self.get_dashboard_status()
        
        for dashboard_name, info in status.items():
            status_icon = " " if info['running'] else " "
            print(f"{status_icon} {dashboard_name.title()} Dashboard:")
            
            if info['running']:
                print(f"     URL: {info['url']}")
                print(f"     Port: {info['port']}")
            else:
                print(f"     Status: Not running")
            print()
        
        print("="*80)
        print("Dashboard Features:")
        print("  Portfolio Dashboard - Real-time portfolio monitoring and performance tracking")
        print(" ️  Risk Dashboard - Comprehensive risk management and VaR analysis")
        print("  Analytics Dashboard - Market regime detection and strategy analytics")
        print("  Trading Dashboard - Live trading operations and execution monitoring")
        print(" ️  Configuration Interface - System settings and parameter management")
        print("="*80)

def create_master_dashboard():
    """Create a master dashboard launcher page."""
    import dash
    from dash import dcc, html
    import dash_bootstrap_components as dbc
    
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("QUANTUM-FORGE Dashboard Hub", 
                       className="text-center mb-5",
                       style={'color': '#17a2b8', 'font-weight': 'bold'})
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardImg(src="/assets/portfolio-icon.png", top=True, 
                               style={'height': '100px', 'object-fit': 'contain'}),
                    dbc.CardBody([
                        html.H4("Portfolio Dashboard", className="card-title"),
                        html.P("Real-time portfolio monitoring, performance tracking, and asset allocation visualization.",
                               className="card-text"),
                        dbc.Button("Launch", href="http://localhost:8050", 
                                 color="primary", external_link=True)
                    ])
                ])
            ], width=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardImg(src="/assets/risk-icon.png", top=True,
                               style={'height': '100px', 'object-fit': 'contain'}),
                    dbc.CardBody([
                        html.H4("Risk Dashboard", className="card-title"),
                        html.P("Comprehensive risk management, VaR analysis, stress testing, and limit monitoring.",
                               className="card-text"),
                        dbc.Button("Launch", href="http://localhost:8051", 
                                 color="danger", external_link=True)
                    ])
                ])
            ], width=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardImg(src="/assets/analytics-icon.png", top=True,
                               style={'height': '100px', 'object-fit': 'contain'}),
                    dbc.CardBody([
                        html.H4("Analytics Dashboard", className="card-title"),
                        html.P("Market regime detection, strategy analytics, and performance attribution analysis.",
                               className="card-text"),
                        dbc.Button("Launch", href="http://localhost:8052", 
                                 color="info", external_link=True)
                    ])
                ])
            ], width=4, className="mb-4")
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardImg(src="/assets/trading-icon.png", top=True,
                               style={'height': '100px', 'object-fit': 'contain'}),
                    dbc.CardBody([
                        html.H4("Trading Dashboard", className="card-title"),
                        html.P("Live trading operations, order management, execution monitoring, and venue analysis.",
                               className="card-text"),
                        dbc.Button("Launch", href="http://localhost:8053", 
                                 color="success", external_link=True)
                    ])
                ])
            ], width=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardImg(src="/assets/config-icon.png", top=True,
                               style={'height': '100px', 'object-fit': 'contain'}),
                    dbc.CardBody([
                        html.H4("Configuration Interface", className="card-title"),
                        html.P("System settings, parameter management, strategy configuration, and validation.",
                               className="card-text"),
                        dbc.Button("Launch", href="http://localhost:8054", 
                                 color="secondary", external_link=True)
                    ])
                ])
            ], width=4, className="mb-4"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("System Status", className="card-title text-center"),
                        html.Div(id="system-status", className="text-center"),
                        html.Hr(),
                        html.P("All dashboards operational", className="text-success text-center"),
                        html.Small("Last updated: " + str(time.strftime("%Y-%m-%d %H:%M:%S")), 
                                 className="text-muted")
                    ])
                ], style={'border': '2px solid #28a745'})
            ], width=4, className="mb-4")
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P("QUANTUM-FORGE Advanced HFT Trading System v1.0", 
                       className="text-center text-muted"),
                html.P("Real-time quantitative trading platform with comprehensive risk management", 
                       className="text-center text-muted small")
            ], width=12)
        ])
        
    ], fluid=True)
    
    return app

# Dashboard instances for easy import
portfolio_dashboard = None
risk_dashboard = None
analytics_dashboard = None
trading_dashboard = None
config_interface = None

def get_dashboard_manager() -> DashboardManager:
    """Get or create dashboard manager instance."""
    if not hasattr(get_dashboard_manager, '_manager'):
        get_dashboard_manager._manager = DashboardManager()
    return get_dashboard_manager._manager

def launch_all_dashboards(debug: bool = False):
    """Launch all QUANTUM-FORGE dashboards."""
    manager = get_dashboard_manager()
    results = manager.start_all_dashboards(debug=debug)
    
    # Wait a moment for dashboards to start
    time.sleep(3)
    
    # Print summary
    manager.print_dashboard_summary()
    
    # Launch Master Dashboard in a separate thread
    master_thread = threading.Thread(
        target=launch_master_dashboard,
        args=(8049, debug),
        daemon=True,
        name="master_dashboard"
    )
    master_thread.start()
    print("  Master Dashboard started successfully")
    print(f"     URL: http://localhost:8049")
    
    return results

def launch_master_dashboard(port: int = 8049, debug: bool = False):
    """Launch master dashboard hub."""
    app = create_master_dashboard()
    print(f"Starting QUANTUM-FORGE Master Dashboard on port {port}")
    print(f"Access master dashboard at: http://localhost:{port}")
    
    app.run(
        debug=debug,
        port=port,
        host='0.0.0.0',
        use_reloader=False
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "master":
        # Launch master dashboard only
        launch_master_dashboard(debug=True)
    else:
        # Launch all dashboards
        launch_all_dashboards(debug=True)
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Shutting down QUANTUM-FORGE dashboards...")
            sys.exit(0)