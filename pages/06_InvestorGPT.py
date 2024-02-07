import streamlit as st

from src.utils import load_file

markdown_file = load_file("./markdowns/investor_gpt.md")
st.set_page_config(
    page_title="InvestorGPT",
    page_icon="📈",
)
st.markdown(markdown_file)
