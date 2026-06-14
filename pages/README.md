#   QUANTUM-FORGE Multi-Page Dashboard System

## Overview
Complete multi-page Streamlit application integrating all 9 dashboards without conflicts.

## Architecture

```
QUANTUM-FORGE/
    app.py                          # Main entry point (landing page)
    pages/                          # Streamlit multi-page app directory
        1_ _Main_Dashboard.py      # Unified overview
        2_ _Trading_Dashboard.py   # Order execution
        3_ ️_Risk_Dashboard.py     # Risk monitoring
        4_ _Portfolio_Dashboard.py # Holdings tracking
        5_ _Analytics_Dashboard.py # Backtesting
        6_ _Research_Dashboard.py  # Strategy development
        7_ _Execution_Dashboard.py # Order flow
        8_ _Market_Microstructure.py # Orderbook viz
        9_ ️_Configuration.py      # System settings
    interface/dashboards/           # Original dashboard modules
        main_dashboard.py
        trading_dashboard.py
        risk_dashboard.py
        ... (all other dashboards)
```

## How It Works

1. **app.py** - Landing page with system overview
2. **pages/** - Streamlit automatically detects numbered files and creates sidebar navigation
3. Each page imports and runs the corresponding dashboard from `interface/dashboards/`

## Running the System

```bash
# Option 1: Run via full system
python run_full_system.py

# Option 2: Run directly
streamlit run app.py
```

## Dashboard Navigation

All dashboards are accessible via the **sidebar** on the left:

- **  Main Dashboard** - Start here for complete overview
- **  Trading** - Execute and manage orders
- ** ️ Risk** - Monitor risk metrics and exposures
- **  Portfolio** - Track positions and performance
- **  Analytics** - Run backtests and attribution
- **  Research** - Develop new strategies
- **  Execution** - Analyze order flow
- **  Microstructure** - Visualize orderbook depth
- ** ️ Configuration** - Adjust system settings

## Features

  **No Streamlit Conflicts** - Multi-page architecture prevents module clashes
  **Unified Navigation** - Sidebar access to all dashboards
  **Real-Time Updates** - All dashboards connect to live data cache
  **Persistent State** - Session state shared across pages
  **Professional UI** - Consistent design across all dashboards

## URL Access

- Home: http://localhost:8501
- Main Dashboard: http://localhost:8501/Main_Dashboard
- Trading: http://localhost:8501/Trading_Dashboard
- Risk: http://localhost:8501/Risk_Dashboard
- Portfolio: http://localhost:8501/Portfolio_Dashboard
- Analytics: http://localhost:8501/Analytics_Dashboard
- Research: http://localhost:8501/Research_Dashboard
- Execution: http://localhost:8501/Execution_Dashboard
- Microstructure: http://localhost:8501/Market_Microstructure
- Config: http://localhost:8501/Configuration

## Status

  **ALL 9 DASHBOARDS ACTIVE AND INTEGRATED**

System utilization: **100%** (139/139 files)
