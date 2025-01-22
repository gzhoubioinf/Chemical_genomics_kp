import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_csv(filepath, **kwargs):
    """Load a CSV file with caching."""
    return pd.read_csv(filepath, **kwargs)

@st.cache_data(show_spinner=False)
def load_excel(filepath, **kwargs):
    """Load an Excel file with caching."""
    return pd.read_excel(filepath, **kwargs)