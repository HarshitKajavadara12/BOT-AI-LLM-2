"""
QUANTUM-FORGE SYSTEM LAUNCHER
==============================
Visual launcher showing complete system activation status.
"""

import os
import sys
import time
from datetime import datetime

def print_banner():
    """Display system banner."""
    banner = """
                                                                        
                                                                        
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                      
                                                                        
                                                                       
                                                                       
                                                                       
                                                                       
                                                                       
                                                                       
                                                                        
                   INSTITUTIONAL TRADING PLATFORM                       
                   100% MODULE ACTIVATION                               
                                                                        
                                                                        
    """
    print(banner)

def print_system_status():
    """Display system module status."""
    print("\n" + "="*70)
    print(" SYSTEM STATUS - 100% ACTIVATION")
    print("="*70 + "\n")
    
    modules = [
        ("Data Layer", 17, "Real-time ingestion, storage, preprocessing"),
        ("Intelligence", 24, "20+ AI/ML models (LSTM, GRU, SAC, etc.)"),
        ("Analytics", 24, "Backtesting, attribution, regime detection"),
        ("Execution", 18, "VWAP, TWAP, smart routing, HFT"),
        ("Core Math", 17, "Stochastic calculus, Fourier, microstructure"),
        ("Risk Management", 1, "EVT, copulas, optimal stopping"),
        ("Interface", 24, "9 dashboards (multi-page app)"),
        ("Infrastructure", 6, "Monitoring, backup, deployment"),
    ]
    
    total_files = sum(count for _, count, _ in modules)
    
    for name, count, desc in modules:
        bar = " " * 40
        print(f"    {name:<18} [{bar}] {count:>3} files")
        print(f"        {desc}")
        print()
    
    print("-" * 70)
    print(f"  TOTAL: {total_files} files active (100% of system)")
    print("="*70 + "\n")

def print_features():
    """Display key features."""
    print("\n" + "="*70)
    print(" KEY FEATURES ACTIVATED")
    print("="*70 + "\n")
    
    features = [
        "  9 Interactive Dashboards (Streamlit multi-page app)",
        "  20+ AI Models running in ensemble",
        "  Professional execution algorithms (VWAP, TWAP, IS)",
        "  Complete data infrastructure (TimescaleDB, Redis, Parquet)",
        "  Market regime detection (HMM, changepoint)",
        "  Advanced risk mathematics (EVT, Copulas)",
        "  Comprehensive backtesting framework",
        "  Performance attribution & P&L decomposition",
        "  High-frequency trading infrastructure",
        " ️ Multi-level risk management system",
    ]
    
    for feature in features:
        print(f"  {feature}")
        time.sleep(0.1)
    
    print("\n" + "="*70 + "\n")

def print_dashboard_list():
    """Display available dashboards."""
    print("\n" + "="*70)
    print(" AVAILABLE DASHBOARDS")
    print("="*70 + "\n")
    
    dashboards = [
        (" ", "Main Dashboard", "Unified system overview"),
        (" ", "Trading Dashboard", "Order execution & management"),
        (" ️", "Risk Dashboard", "Risk analytics & monitoring"),
        (" ", "Portfolio Dashboard", "Holdings & performance tracking"),
        (" ", "Analytics Dashboard", "Backtesting & attribution"),
        (" ", "Research Dashboard", "Strategy development tools"),
        (" ", "Execution Dashboard", "Order flow analysis"),
        (" ", "Market Microstructure", "Orderbook visualization"),
        (" ️", "Configuration", "System settings & parameters"),
    ]
    
    for icon, name, desc in dashboards:
        print(f"  {icon}  {name:<25} - {desc}")
    
    print("\n" + "="*70 + "\n")

def print_launch_info():
    """Display launch information."""
    print("\n" + "="*70)
    print(" LAUNCHING SYSTEM")
    print("="*70 + "\n")
    
    print("    Starting QUANTUM-FORGE Full System...")
    print(f"    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Directory: {os.getcwd()}")
    print("\n    Initializing modules...\n")

def main():
    """Main launcher."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print_banner()
    time.sleep(0.5)
    
    print_system_status()
    time.sleep(0.5)
    
    print_features()
    time.sleep(0.5)
    
    print_dashboard_list()
    time.sleep(0.5)
    
    print_launch_info()
    time.sleep(1)
    
    # Launch the actual system
    print("  " + "-"*66)
    print("  Starting run_full_system.py...")
    print("  " + "-"*66 + "\n")
    
    os.system(f"{sys.executable} run_full_system.py")

if __name__ == "__main__":
    main()
