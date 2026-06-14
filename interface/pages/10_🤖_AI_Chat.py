"""
AI Chat Interface for QUANTUM-FORGE
Natural language trading queries
"""

import streamlit as st
import requests
import time
from datetime import datetime


# Page config
st.set_page_config(
    page_title="AI Chat - Quantum Forge",
    page_icon=" ",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"


def check_api_status():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def query_ai(query: str, symbol: str = None):
    """Send query to AI"""
    try:
        payload = {
            "query": query,
            "symbol": symbol,
            "max_tokens": 512
        }
        
        response = requests.post(
            f"{API_URL}/api/v1/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"error": "Request timeout (>30s)"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}


def get_portfolio():
    """Get portfolio status"""
    try:
        response = requests.get(f"{API_URL}/api/v1/portfolio", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_system_status():
    """Get system status"""
    try:
        response = requests.get(f"{API_URL}/api/v1/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'api_online' not in st.session_state:
    st.session_state.api_online = check_api_status()


# Header
st.title("  Quantum Forge AI Chat")
st.markdown("*Ask me anything about your trading, portfolio, or market analysis*")

# Sidebar
with st.sidebar:
    st.header("System Status")
    
    # Check API status
    api_status = check_api_status()
    st.session_state.api_online = api_status
    
    if api_status:
        st.success("  API Online")
        
        # Get system status
        sys_status = get_system_status()
        if sys_status:
            st.metric("Status", sys_status['status'].upper())
            
            with st.expander("Components"):
                for component, active in sys_status['components'].items():
                    icon = " " if active else " ️"
                    st.write(f"{icon} {component}")
            
            with st.expander("Statistics"):
                stats = sys_status['stats']
                st.metric("Tracked Symbols", stats.get('tracked_symbols', 0))
                st.metric("Total Trades", stats.get('total_trades', 0))
                st.metric("Active Positions", stats.get('active_positions', 0))
                st.metric("Cash Balance", f"${stats.get('cash_balance', 0):,.2f}")
    else:
        st.error(" ️ API Offline")
        st.info("Start API server:\n```bash\npython llm_integration/api.py\n```")
    
    st.divider()
    
    # Quick queries
    st.header("Quick Queries")
    
    quick_queries = [
        "What's my portfolio status?",
        "Show recent trades",
        "Analyze my performance",
        "What are my best positions?",
        "System metrics summary",
        "Bitcoin trading history",
        "Risk assessment"
    ]
    
    for query in quick_queries:
        if st.button(query, key=f"quick_{query}", use_container_width=True):
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now()
            })
            st.rerun()
    
    st.divider()
    
    # Settings
    st.header("Settings")
    symbol_filter = st.selectbox(
        "Symbol Filter",
        ["None", "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"],
        index=0
    )
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main chat area
if not st.session_state.api_online:
    st.warning(" ️ API server is offline. Start the API server to use AI chat.")
    st.code("python llm_integration/api.py", language="bash")
    
    st.info("""
    **Alternative:** Use template responses (no LLM required)
    
    The system will work with basic template responses if the API is unavailable.
    Full AI capabilities require:
    1. API server running
    2. Llama 3.2 8B model installed
    3. Integration bridge active
    """)

else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                with st.expander("Details"):
                    meta = message["metadata"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Latency", f"{meta.get('latency_ms', 0):.1f}ms")
                    with col2:
                        st.metric("Model", meta.get('model', 'Unknown'))
                    with col3:
                        st.metric("Context Items", meta.get('context_items', 0))
    
    # Chat input
    user_query = st.chat_input("Ask me anything...")
    
    if user_query:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
            "timestamp": datetime.now()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_query)
        
        # Query AI
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                symbol = None if symbol_filter == "None" else symbol_filter
                result = query_ai(user_query, symbol)
                
                if "error" in result:
                    response = f"  {result['error']}"
                    st.error(response)
                else:
                    response = result['response']
                    st.write(response)
                    
                    # Show metadata
                    with st.expander("Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Latency", f"{result['latency_ms']:.1f}ms")
                        with col2:
                            st.metric("Model", result['model'])
                        with col3:
                            context = result.get('context', {}).get('relevant_items', {})
                            total_items = sum(context.values())
                            st.metric("Context Items", total_items)
                        
                        if result.get('context', {}).get('analytics'):
                            st.json(result['context']['analytics'])
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now(),
            "metadata": {
                "latency_ms": result.get('latency_ms', 0),
                "model": result.get('model', 'Unknown'),
                "context_items": sum(result.get('context', {}).get('relevant_items', {}).values())
            } if "error" not in result else {}
        })
        
        st.rerun()


# Footer
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Sample Queries")
    st.markdown("""
    - Portfolio and positions
    - Trade history and performance
    - Market analysis and trends
    - Risk assessment
    - System metrics
    """)

with col2:
    st.subheader("Features")
    st.markdown("""
    - Natural language queries
    - Real-time portfolio data
    - RAG-powered context
    - AI-generated insights
    - 60-250ms response time
    """)

with col3:
    st.subheader("Architecture")
    st.markdown("""
    - Llama 3.2 8B LLM
    - Qdrant vector search
    - DuckDB analytics cache
    - Redis event streams
    - FastAPI backend
    """)
