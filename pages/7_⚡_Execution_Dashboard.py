"""Page 7: Execution Dashboard"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
st.set_page_config(page_title="Execution Dashboard", page_icon="⚡", layout="wide")

# Auto-refresh every 30 seconds
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30_000, key="auto_refresh_7")
except ImportError:
    pass

# System status in sidebar
with st.sidebar:
    try:
        import requests
        resp = requests.get("http://localhost:8000/health", timeout=2)
        if resp.status_code == 200:
            st.success("System: Online")
        else:
            st.warning("System: Degraded")
    except Exception:
        st.info("System: Status unavailable")

try:
    from interface.dashboards.execution_dashboard import main
    main()
except ImportError as e:
    st.error(f"Dashboard module not found: {e}")
    st.info("Run the system with `python scripts/paper_trade.py` first.")
except Exception as e:
    st.error(f"Dashboard error: {e}")
    st.exception(e)
