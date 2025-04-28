import sys, os, yaml

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from app.colony_picker import colonypicker
from app.ml_prediction import app_fasta_prediction
import streamlit as st


def load_config(config_path='/ibex/project/c2205/Natalie/Project_File/config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load configuration
    config = load_config()

    # Sidebar for navigation
    st.sidebar.title("Choose an Option")
    mode = st.sidebar.radio(
        "Select what you want to do:",
        [
            "ColonyPicker",
            "Get Predictions from ML Model",
        ]
    )

    if mode == "ColonyPicker":
        colonypicker(config)
    else:
        app_fasta_prediction(config)

if __name__ == "__main__":
    main()