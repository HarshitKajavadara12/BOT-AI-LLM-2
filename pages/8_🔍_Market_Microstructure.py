"""Page 8: Market Microstructure"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
st.set_page_config(page_title="Market Microstructure", page_icon=" ", layout="wide")

from interface.dashboards.market_microstructure_viz import main
main()
