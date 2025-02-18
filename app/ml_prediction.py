import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import glob
import io
import cv2
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore, gaussian_kde
from Bio import SeqIO
from joblib import load
import json
import shap
from collections import defaultdict
import warnings
import time
from Bio.Seq import Seq

# Utils imports 
from app.utils.data_loading import load_csv, load_excel
from app.utils.image_handling import crop_img, extract_colony
from app.utils.ml_models import load_pca_and_unitigs, load_models_for_condition, load_distributions
from app.utils.gbff_processing import process_gbff

warnings.filterwarnings("ignore")

def fasta_to_unitig_vector(fasta_contents, unitig_index, total_unitigs, k=31):
    """Convert a FASTA file into a presence/absence vector of 31-mers."""
    fasta_str = fasta_contents.getvalue().decode("utf-8")
    fasta_io = io.StringIO(fasta_str)
    binary_vector = np.zeros(total_unitigs, dtype=int)
    for record in SeqIO.parse(fasta_io, "fasta"):
        sequence = str(record.seq).upper()
        seq_len = len(sequence)
        for i in range(seq_len - k + 1):
            kmer = sequence[i:i+31]
            if kmer in unitig_index:
                binary_vector[unitig_index[kmer]] = 1
    return binary_vector

def compute_stats(df, prediction):
    """Compute mean, S-score, percentile, and % difference from mean."""
    col_name = df.columns[-1]
    data = df[col_name].dropna()
    mean_val = data.mean()
    std_val = data.std()
    s_score = (prediction - mean_val) / std_val if std_val != 0 else np.nan
    percentile = (data <= prediction).sum() / len(data) * 100
    pct_diff = ((prediction - mean_val) / mean_val) * 100 if mean_val != 0 else np.nan
    return mean_val, s_score, percentile, pct_diff

def app_fasta_prediction(config):
    st.title("ML Prediction")

    MODEL_PATHS = config['MODEL_PATHS']
    condition = st.selectbox("Select Condition for ML models:", list(MODEL_PATHS.keys()))
    algorithm = st.selectbox("Select ML Algorithm:", ["XGBoost", "TabNet"])
    
    with st.spinner("Loading PCA model and unitig_to_index mapping..."):
        pca, unitig_to_index = load_pca_and_unitigs(config['pca_model_file'], config['unitig_to_index_file'])
    
    # Prepare reverse index for unitigs
    index_to_unitig = [None] * len(unitig_to_index)
    for u, i in unitig_to_index.items():
        index_to_unitig[i] = u
    num_unitigs = len(unitig_to_index)
    
    with st.spinner(f"Loading {algorithm} models for {condition}..."):
        model_opacity, model_circ, model_size = load_models_for_condition(algorithm, condition, MODEL_PATHS)
    
    with st.spinner("Loading distribution data..."):
        df_circ, df_opa, df_siz = load_distributions(
            config['files']['circularity_data'],
            config['files']['opacity_data'],
            config['files']['size_data']
        )
    
    uploaded_file = st.file_uploader("Upload your FASTA file:", type=["fasta"])
    if uploaded_file is not None and model_opacity and model_circ and model_size:
        st.write("Processing your file...")
        
        # Check against known resistance/virulence genes
        user_fasta_str = uploaded_file.getvalue().decode("utf-8")
        user_fasta_str_no_ws = "".join(user_fasta_str.split()).upper()
        matched_genes = []
        matched_virulence_genes = []
        
        try:
            for record in SeqIO.parse(config['files']['all_fsa'], "fasta"):
                ref_seq = str(record.seq).upper()
                if ref_seq in user_fasta_str_no_ws:
                    matched_genes.append(record.id)
        except Exception as e:
            st.warning(f"Could not process resistance_genes_seq.fsa for reference genes. Error: {e}")
        
        try:
            for record in SeqIO.parse(config['files']['vfdb_fas'], "fasta"):
                ref_seq = str(record.seq).upper()
                if ref_seq in user_fasta_str_no_ws:
                    all_parens = re.findall(r"\(([^)]*)\)", record.description)
                    if len(all_parens) >= 2:
                        matched_virulence_genes.append(all_parens[1])
                    elif len(all_parens) == 1:
                        matched_virulence_genes.append(all_parens[0])
                    else:
                        matched_virulence_genes.append(record.id)
        except Exception as e:
            st.warning(f"Could not process virulence_genes_seq.fas for virulence genes. Error: {e}")
        
        st.write("### Matched Resistance Genes:")
        if matched_genes:
            for gene in matched_genes:
                st.write(gene)
        else:
            st.write("No matched resistance genes found in your FASTA.")
        
        st.write("### Matched Virulence Genes:")
        if matched_virulence_genes:
            for vir_gene in matched_virulence_genes:
                st.write(vir_gene)
        else:
            st.write("No matched virulence genes found in your FASTA.")
        
        # Convert to unitig vector and transform with the shared PCA model
        bin_vector = fasta_to_unitig_vector(uploaded_file, unitig_to_index, num_unitigs)
        bin_vector_pca = pca.transform([bin_vector])
        
        pred_opacity = float(model_opacity.predict(bin_vector_pca)[0])
        pred_circ = float(model_circ.predict(bin_vector_pca)[0])
        pred_size = float(model_size.predict(bin_vector_pca)[0])
        
        st.write("### Predictions")
        st.write(f"**Opacity:** {pred_opacity:.4f}")
        st.write(f"**Circularity:** {pred_circ:.4f}")
        st.write(f"**Size:** {pred_size:.4f}")
        
        c_mean, c_s, c_pct, c_diff = compute_stats(df_circ, pred_circ)
        o_mean, o_s, o_pct, o_diff = compute_stats(df_opa, pred_opacity)
        s_mean, s_s, s_pct, s_diff = compute_stats(df_siz, pred_size)
        
        # Circularity distribution plot
        st.write("### Circularity Distribution")
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
        sns.kdeplot(
            df_circ[df_circ.columns[-1]], 
            fill=True, 
            color='skyblue', 
            alpha=0.7, 
            bw_method='scott',
            ax=ax
        )
        ax.axvline(pred_circ, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Prediction')
        ax.axvline(c_mean, color='gray', linestyle=':', linewidth=2, alpha=0.9, label='Dataset Mean')
        ax.set_xlabel("Circularity")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
        
        st.write(
            f"**Dataset Mean:** {c_mean:.2f}\n"
            f"**Prediction:** {pred_circ:.2f} (Percentile: {c_pct:.1f}%, Diff: {c_diff:.1f}%, S-score: {c_s:.2f})"
        )
        
        # Opacity distribution plot
        st.write("### Opacity Distribution")
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
        sns.kdeplot(
            df_opa[df_opa.columns[-1]], 
            fill=True, 
            color='lightcoral', 
            alpha=0.7, 
            bw_method='scott',
            ax=ax
        )
        ax.axvline(pred_opacity, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Prediction')
        ax.axvline(o_mean, color='gray', linestyle=':', linewidth=2, alpha=0.9, label='Dataset Mean')
        ax.set_xlabel("Opacity")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
        
        st.write(
            f"**Dataset Mean:** {o_mean:.2f}\n"
            f"**Prediction:** {pred_opacity:.2f} (Percentile: {o_pct:.1f}%, Diff: {o_diff:.1f}%, S-score: {o_s:.2f})"
        )
        
        # Size distribution plot
        st.write("### Size Distribution")
        fig, ax = plt.subplots(figsize=(7, 4), facecolor='white')
        sns.kdeplot(
            df_siz[df_siz.columns[-1]], 
            fill=True, 
            color='mediumseagreen', 
            alpha=0.7, 
            bw_method='scott',
            ax=ax
        )
        ax.axvline(pred_size, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Prediction')
        ax.axvline(s_mean, color='gray', linestyle=':', linewidth=2, alpha=0.9, label='Dataset Mean')
        ax.set_xlabel("Size")
        ax.set_ylabel("Density")
        ax.legend()
        st.pyplot(fig)
        
        st.write(
            f"**Dataset Mean:** {s_mean:.2f}\n"
            f"**Prediction:** {pred_size:.2f} (Percentile: {s_pct:.1f}%, Diff: {s_diff:.1f}%, S-score: {s_s:.2f})"
        )
        
        # PCA & SHAP Analysis
        if algorithm == "XGBoost":
            st.write("---")
            st.header("PCA and SHAP Analysis")
            num_pcs = st.number_input(
                "Select number of top PCs to display (max 50):",
                min_value=1,
                max_value=50,
                value=10,
                step=1
            )
            if num_pcs > 50:
                st.error("Maximum = 50.")
            else:
                with st.spinner("Computing SHAP values..."):
                    explainer_opacity = shap.TreeExplainer(model_opacity)
                    shap_values_opacity = explainer_opacity.shap_values(bin_vector_pca)
                    shap_vals_for_sample = shap_values_opacity[0]

                abs_shap = np.abs(shap_vals_for_sample)
                top_pc_indices = np.argsort(abs_shap)[::-1][:num_pcs]

                n_top_unitigs = 5
                pc_to_unitigs = {}
                for pc_idx in top_pc_indices:
                    loadings = pca.components_[pc_idx, :]
                    abs_loadings = np.abs(loadings)
                    top_idxs = np.argsort(abs_loadings)[::-1][:n_top_unitigs]
                    top_unitigs = [index_to_unitig[i] for i in top_idxs]
                    pc_to_unitigs[pc_idx] = top_unitigs

                with st.spinner("Processing reference genome..."):
                    matches = process_gbff(
                        config['files']['reference_gbff'],
                        pc_to_unitigs,
                        pca,
                        index_to_unitig,
                        unitig_to_index
                    )

                pc_to_genes = defaultdict(set)
                for pc_num, unitig, gene in matches:
                    if gene != "Unknown":
                        pc_to_genes[pc_num].add(gene)

                pc_numbers = [
                    f"PC{pc_idx + 1} => {', '.join(list(pc_to_genes.get(pc_idx + 1, []))[:3]) or 'noGene'}"
                    for pc_idx in top_pc_indices
                ]
                shap_values_top = shap_vals_for_sample[top_pc_indices]

                st.write("### SHAP Values with PC-to-Gene Mappings")
                plt.figure(figsize=(10, max(6, num_pcs * 0.4)), facecolor='white')
                bar_colors = sns.color_palette("coolwarm", n_colors=len(shap_values_top))
                plt.barh(pc_numbers, shap_values_top, color=bar_colors)
                plt.axvline(x=0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
                plt.xlabel("SHAP value (impact on model output)", fontsize=12)
                plt.title("Opacity Model - SHAP with Top PC-to-Gene Mappings", fontsize=14)
                plt.tight_layout()
                st.pyplot(plt)
        else:
            st.write("---")
            st.header("SHAP Analysis Not Implemented for TabNet")
            st.write(
                "Currently, the SHAP explanation is only demonstrated for XGBoost. "
                "SHAP’s TreeExplainer is specifically optimized for tree models (such as XGB) and works with them. "
                "TabNet is not a tree-based model."
            )

