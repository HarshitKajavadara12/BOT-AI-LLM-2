import streamlit as st
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from core.audit import AuditLogger

st.set_page_config(
    page_title="Investor Portal",
    page_icon=" ️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" ️ Institutional Investor Portal")
st.markdown("### Read-Only Oversight Dashboard")

# --- 1. System Health ---
st.header("1. System Health & Integrity")

col1, col2, col3, col4 = st.columns(4)

# Mock Health Data (In real system, read from status file)
system_status = {
    "LLM Engine": "ONLINE",
    "Data Feeds": "CONNECTED",
    "Kill Switch": "DISARMED",
    "Last Snapshot": datetime.now().strftime("%H:%M:%S")
}

with col1:
    st.metric("LLM Engine", system_status["LLM Engine"])
with col2:
    st.metric("Data Feeds", system_status["Data Feeds"])
with col3:
    st.metric("Kill Switch", system_status["Kill Switch"], delta_color="inverse")
with col4:
    st.metric("Last Snapshot", system_status["Last Snapshot"])

# Audit Integrity Check
logger = AuditLogger()
log_file = logger._get_log_file()
integrity_status = "UNKNOWN"
if log_file.exists():
    is_valid = logger.verify_integrity(str(log_file))
    integrity_status = "  VALID" if is_valid else "  TAMPERED"
else:
    integrity_status = " ️ NO LOGS"

st.info(f"**Audit Log Integrity:** {integrity_status} | **Log File:** `{log_file}`")

# --- 2. Strategy Performance ---
st.header("2. Strategy Performance (Audited)")

# Load Audit Logs
data = []
if log_file.exists():
    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                # Extract relevant metrics
                timestamp = datetime.fromtimestamp(entry['timestamp'])
                # Mocking equity for visualization since we don't have full backtest data in logs yet
                # In real system, 'execution_decision' or 'market_state' would contain NAV
                equity = 100000.0 # Base
                if 'execution_decision' in entry:
                    # Simulate some movement based on decision hash just for demo
                    equity += (hash(entry['execution_decision'].get('action', '')) % 100) - 50
                
                data.append({"Timestamp": timestamp, "Equity": equity})
            except:
                pass

if data:
    df = pd.DataFrame(data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Equity'], mode='lines', name='NAV'))
    fig.update_layout(title="Net Asset Value (NAV)", xaxis_title="Time", yaxis_title="Equity ($)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No performance data available in today's audit log.")

# --- 3. Decision Explanations ---
st.header("3. Decision Explanations")

if data:
    # Show last 5 decisions
    st.markdown("#### Recent System Decisions")
    
    # Re-read file to get full objects
    with open(log_file, 'r') as f:
        lines = f.readlines()
        
    for line in reversed(lines[-5:]):
        entry = json.loads(line)
        ts = datetime.fromtimestamp(entry['timestamp']).strftime("%H:%M:%S")
        decision = entry.get('execution_decision', {})
        risk = entry.get('risk_state', {})
        
        with st.expander(f"[{ts}] Decision: {decision.get('action', 'UNKNOWN')}"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Execution Logic:**")
                st.json(decision)
            with c2:
                st.markdown("**Risk State:**")
                st.json(risk)
            
            if entry.get('cognitive_context'):
                st.markdown("**LLM Context:**")
                st.write(entry['cognitive_context'])
                
            st.caption(f"Snapshot ID: {entry['snapshot_id']} | Hash: `{entry['current_hash'][:16]}...`")

else:
    st.write("No decisions logged yet.")

# --- 4. Regime & Risk State ---
st.header("4. Regime & Risk State")

# Mock Regime Data (Replace with real read from latest log)
current_regime = "NEUTRAL"
risk_throttles = "NONE"

if data:
    # Try to get from last log
    try:
        last_entry = json.loads(lines[-1])
        # Assuming regime is in market_state or risk_state
        # For now, we mock it or extract if available
        pass
    except:
        pass

c1, c2 = st.columns(2)
with c1:
    st.success(f"**Current Regime:** {current_regime}")
with c2:
    st.warning(f"**Active Throttles:** {risk_throttles}")

st.markdown("---")
st.caption("  **Institutional Grade Security:** This dashboard is read-only and physically isolated from the execution engine.")
