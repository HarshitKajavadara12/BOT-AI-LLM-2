"""Page 9: Configuration Interface"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
st.set_page_config(page_title="Configuration", page_icon=" ️", layout="wide")

from interface.dashboards.config_interface import main
main()
