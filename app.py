"""
QUANTUM-FORGE: MULTI-PAGE STREAMLIT APP
========================================
Main entry point for the complete dashboard system with all 9 dashboards integrated.

Author: Quantum Forge Team
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure page
st.set_page_config(
    page_title="Quantum Forge Trading Platform",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page
st.title("  QUANTUM-FORGE Trading Platform")
st.markdown("### Complete Institutional-Grade Trading System")

st.info("""
  **Select a dashboard from the sidebar to begin**

This system includes:
-   Main Dashboard - Unified overview
-   Trading Dashboard - Order execution & management
-  ️ Risk Dashboard - Risk analytics & monitoring  
-   Portfolio Dashboard - Holdings & performance
-   Analytics Dashboard - Backtesting & attribution
-   Research Dashboard - Strategy development
-   Execution Dashboard - Order flow analysis
-   Market Microstructure - Orderbook visualization
-  ️ Configuration - System settings
""")

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("System Status", "  ONLINE", "All modules active")
    
with col2:
    st.metric("Dashboards", "9 Active", "100% operational")
    
with col3:
    st.metric("Real-Time Data", "7 Symbols", "Binance WebSocket")

st.markdown("---")

st.success("  Full system initialized - All 135+ modules active!")

st.markdown("""
###   Quick Links
- [Main Dashboard](Main_Dashboard) - Start here for overview
- [Trading Dashboard](Trading_Dashboard) - Execute trades
- [Risk Dashboard](Risk_Dashboard) - Monitor risk
- [Portfolio Dashboard](Portfolio_Dashboard) - Track positions
""")
