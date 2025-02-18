import time
import streamlit as st
from joblib import load
import pandas as pd

@st.cache_resource(show_spinner=True)
def load_pca_and_unitigs(pca_model_file, unitig_to_index_file):
    start_time = time.time()
    pca = load(pca_model_file)
    unitig_to_index = pd.read_pickle(unitig_to_index_file)
    elapsed_time = time.time() - start_time
    st.write(f"Loaded PCA model and unitig_to_index in {elapsed_time:.2f} seconds.")  # (Remove after testing)
    return pca, unitig_to_index

@st.cache_resource(show_spinner=False)
def load_models_for_condition(algorithm, condition, MODEL_PATHS):
    """
    Loads the models for a given condition and algorithm.
    
    If the MODEL_PATHS[condition] dictionary contains keys for different algorithms 
    (e.g., "XGBoost" or "TabNet"), it will select the sub-dictionary for the chosen algorithm.
    Otherwise, it assumes the mapping is direct.
    """
    condition_paths = MODEL_PATHS.get(condition)
    if not condition_paths:
        st.error(f"No model paths available for condition: {condition}")
        return None, None, None
    # If algorithm key exists, use it; otherwise, use the direct mapping.
    if algorithm in condition_paths:
        model_files = condition_paths[algorithm]
    else:
        model_files = condition_paths
    model_opacity = load(model_files["opacity"])
    model_circ = load(model_files["circularity"])
    model_size = load(model_files["size"])
    return model_opacity, model_circ, model_size

@st.cache_data(show_spinner=False)
def load_distributions(circularity_data_file, opacity_data_file, size_data_file):
    df_circ = pd.read_csv(circularity_data_file)
    df_opa = pd.read_csv(opacity_data_file)
    df_siz = pd.read_csv(size_data_file)
    return df_circ, df_opa, df_siz