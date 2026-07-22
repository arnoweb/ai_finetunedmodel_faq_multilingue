from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Architecture", layout="wide")

html_path = Path(__file__).resolve().parent.parent / "docs" / "architecture.html"
html_content = html_path.read_text(encoding="utf-8")
components.html(html_content, height=6000, scrolling=True)
