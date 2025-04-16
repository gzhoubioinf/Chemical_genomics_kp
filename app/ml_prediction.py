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
import matplotlib.colors as colors 
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
from app.utils.ml_models import load_pca_and_unitigs, load_distributions
from app.utils.gbff_processing import process_gbff, load_panaroo_csv
from app.utils.go import run_go_enrichment, load_kp_go_tsv, plot_go_dag

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
            kmer = sequence[i:i+k]
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

    # ------------------------------------------------------------------
    # Load PCA and Unitig Index (shared across models)
    # ------------------------------------------------------------------
    with st.spinner("Loading PCA model and unitig_to_index mapping..."):
        pca, unitig_to_index = load_pca_and_unitigs(
            config['pca_model_file'], 
            config['unitig_to_index_file']
        )
    
    # Prepare reverse index for unitigs
    index_to_unitig = [None] * len(unitig_to_index)
    for u, i in unitig_to_index.items():
        index_to_unitig[i] = u
    num_unitigs = len(unitig_to_index)
    
    # ------------------------------------------------------------------
    # Dynamically select ML algorithm and condition by scanning the models directory
    # ------------------------------------------------------------------
    models_dir = config["models_directory"]
    
    # List available algorithms (each subfolder under models_dir)
    available_algorithms = sorted(
        [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    )
    if not available_algorithms:
        st.error("No ML algorithm directories found in the models directory.")
        return
    algorithm = st.selectbox("Select ML Algorithm:", available_algorithms)
    
    # List conditions available under the selected algorithm
    algorithm_dir = os.path.join(models_dir, algorithm)
    available_conditions = sorted(
        [d for d in os.listdir(algorithm_dir) if os.path.isdir(os.path.join(algorithm_dir, d))]
    )
    if not available_conditions:
        st.error(f"No condition folders found for {algorithm} in {algorithm_dir}.")
        return
    condition = st.selectbox("Select Condition:", available_conditions)
    
    # ------------------------------------------------------------------
    # Load models dynamically by searching for model files by metric
    # ------------------------------------------------------------------
    def load_model(metric_keyword):
        """Search for a model file in the designated folder that matches the metric keyword in a case-insensitive manner."""
        pattern = os.path.join(algorithm_dir, condition, "*")
        # Search all files in the directory and filter using a case-insensitive match
        model_files = [
            f for f in glob.glob(pattern)
            if metric_keyword.lower() in os.path.basename(f).lower()
        ]
        if not model_files:
            st.error(f"No {metric_keyword} model found for {algorithm} under condition '{condition}'.")
            return None
        # If more than one file matches, take the first (or modify as needed)
        return load(model_files[0])
    
    with st.spinner(f"Loading {algorithm} models for condition '{condition}'..."):
        model_opacity = load_model("opacity")
        model_circ = load_model("circularity")
        model_size = load_model("size")
    
    if model_opacity is None or model_circ is None or model_size is None:
        st.error("Error loading one or more models. Please check your model files.")
        return

    # ------------------------------------------------------------------
    # Load Distribution Data
    # ------------------------------------------------------------------
    with st.spinner("Loading distribution data..."):
        df_circ, df_opa, df_siz = load_distributions(
            config['files']['circularity_data'],
            config['files']['opacity_data'],
            config['files']['size_data']
        )
    
    # ------------------------------------------------------------------
    # (Optional) Load additional annotation files for GO analysis, Panaroo, etc.
    # ------------------------------------------------------------------
    id_to_anno = {}
    if 'panaroo_csv' in config['files']:
        with st.spinner("Loading Panaroo gene_presence_absence.csv..."):
            try:
                panaroo_csv_path = config['files']['panaroo_csv']
                id_to_anno = load_panaroo_csv(panaroo_csv_path)
            except Exception as e:
                st.warning(f"Could not load Panaroo CSV file: {e}")
                id_to_anno = {}

    # For GO analysis, load cugene->GO mapping if available
    gene2gos = {}
    if 'kp_go_tsv' in config['files']:
        try:
            gene2gos = load_kp_go_tsv(config['files']['kp_go_tsv'])
        except Exception as e:
            st.warning(f"Could not load KP GO TSV: {e}")
            gene2gos = {}
    
    go_obo_file = config['files'].get('go_obo_file', None)

    # ------------------------------------------------------------------
    # FASTA File Uploader and Gene Matching
    # ------------------------------------------------------------------
    uploaded_file = st.file_uploader("Upload your FASTA file:", type=["fasta"])
    if uploaded_file is not None:
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
        
        st.write("---")
        st.subheader("Key Genes:")
        st.write("**Matched Resistance Genes:**")
        if matched_genes:
            df_matched_res = pd.DataFrame({"Genes": matched_genes})
            df_matched_res.index = df_matched_res.index + 1
            st.table(df_matched_res)
        else:
            st.write("No matched resistance genes found in your FASTA.")
    
        st.write("**Matched Virulence Genes:**")
        if matched_virulence_genes:
            df_matched_vir = pd.DataFrame({"Genes": matched_virulence_genes})
            df_matched_vir.index = df_matched_vir.index + 1
            st.table(df_matched_vir)
        else:
            st.write("**No matched virulence genes found in your FASTA.**")
        
        # ------------------------------------------------------------------
        # Unitig vector creation, PCA transformation, and predictions
        # ------------------------------------------------------------------
        bin_vector = fasta_to_unitig_vector(
            uploaded_file, 
            unitig_to_index, 
            num_unitigs
        )
        bin_vector_pca = pca.transform([bin_vector])
        
        pred_opacity = float(model_opacity.predict(bin_vector_pca)[0])
        pred_circ = float(model_circ.predict(bin_vector_pca)[0])
        pred_size = float(model_size.predict(bin_vector_pca)[0])
        
        st.write("---")
        st.subheader("Predicted Metrics")
        st.write(f"**Opacity:** {pred_opacity:.4f}")
        st.write(f"**Circularity:** {pred_circ:.4f}")
        st.write(f"**Size:** {pred_size:.4f}")
        
        # ------------------------------------------------------------------
        # Distribution Visualization, Statistics and PCA SHAP Analysis
        # ------------------------------------------------------------------
        st.write("---")
        st.subheader("Distribution Visualization, Statistics and PCA SHAP Analysis")
        st.write(
            "Note: The reference distributions are derived solely "
            "from our internal lab dataset and may not fully represent the variability "
            "seen in external samples."
        )
        
        # (Optional) If using XGBoost, get number of top PCs to display for each parameter
        if algorithm == "XGBoost":
            num_pcs = st.number_input(
                "Select number of top PCs to display for each parameter (max 50):",
                min_value=1,
                max_value=50,
                value=10,
                step=1
            )
        
        # Initialize a global set for gene names across parameters (for GO analysis)
        combined_genes = set()
        # Initialize a dictionary to store matched genes per parameter
        matched_genes_by_param = {}

        # Iterate through each parameter
        for param, df, pred in zip(
            ["Circularity", "Size", "Opacity"],
            [df_circ, df_siz, df_opa],
            [pred_circ, pred_size, pred_opacity]
        ):
            st.markdown(
                f"<h5 style='font-size:17px; margin-bottom: 0;'>{param}</h5>", 
                unsafe_allow_html=True
            )
            data_arr = df[df.columns[-1]].dropna().values
            mean_val = np.mean(data_arr)
            std_dev = np.std(data_arr)
            rep_mean = pred  # the prediction for the parameter
            n = 1
            s_score = (rep_mean - mean_val) / (std_dev / np.sqrt(n)) if std_dev > 0 else None

            fig, ax = plt.subplots(figsize=(5, 3))
    
            kde_fn = gaussian_kde(data_arr)
            x_vals = np.linspace(min(data_arr), max(data_arr), 1000)
            y_vals = kde_fn(x_vals)
            ax.plot(x_vals, y_vals, color='blue', label=f"{param} KDE")
    
            ax.axvline(mean_val, color='gray', linestyle='-.', linewidth=1.5,
                       label=f"Dataset Mean: {mean_val:.2f}")
    
            label_str = f"Prediction: {rep_mean:.2f}"
            if s_score is not None:
                label_str += f" (S-score: {round(s_score, 2)})"
            ax.axvline(rep_mean, color='black', linestyle='--', linewidth=1.5, label=label_str)
    
            percentile_val = percentileofscore(data_arr, rep_mean)
            pct_diff = ((rep_mean - mean_val) / mean_val) * 100 if mean_val != 0 else 0
            y_val = kde_fn(rep_mean)
            ax.plot(
                rep_mean, y_val, 'x',
                color='green', markersize=10,
                label=f"Prediction: {rep_mean:.2f} (Pctile: {percentile_val:.1f}%, Diff: {pct_diff:.1f}%)"
            )
    
            ax.set_xlabel("Value", fontsize='x-small')
            ax.set_ylabel("Density", fontsize='x-small')
            ax.set_title(f"{param} Distribution", fontsize='small')
            ax.tick_params(axis='x', labelsize='x-small')
            ax.tick_params(axis='y', labelsize='x-small')
            ax.legend(
                loc='center left', 
                bbox_to_anchor=(1, 0.5),
                fontsize='small', 
                frameon=False, 
                handletextpad=0.5
            )
            st.pyplot(fig)
            
            # ------------------------------------------------------------------
            # PCA SHAP Analysis for each parameter (only if using XGBoost)
            # ------------------------------------------------------------------
            if algorithm == "XGBoost":
                # Select the correct model for the parameter
                if param == "Opacity":
                    model = model_opacity
                elif param == "Circularity":
                    model = model_circ
                elif param == "Size":
                    model = model_size
                
                with st.spinner(f"Computing SHAP values for {param}..."):
                    explainer = shap.TreeExplainer(model)
                    shap_values_param = explainer.shap_values(bin_vector_pca)
                    shap_vals_for_sample = shap_values_param[0]
                
                abs_shap = np.abs(shap_vals_for_sample)
                top_pc_indices = np.argsort(abs_shap)[::-1][:num_pcs]
                n_top_unitigs = 5
                pc_to_unitigs = {}
                for pc_idx in top_pc_indices:
                    loadings = pca.components_[pc_idx, :]
                    # Multiply loadings by the binary unitig vector to zero out unitigs not present
                    relevant_score = loadings * bin_vector
                    abs_score = np.abs(relevant_score)
                    sorted_idx = np.argsort(abs_score)[::-1]
                    filtered_top_idx = [idx for idx in sorted_idx if bin_vector[idx] == 1][:n_top_unitigs]
                    top_unitigs = [index_to_unitig[i] for i in filtered_top_idx]
                    pc_to_unitigs[pc_idx] = top_unitigs
                
                # Process the reference genome to map PCs to genes
                matches = process_gbff(
                    config['files']['reference_gbff'],
                    pc_to_unitigs,
                    pca,
                    index_to_unitig,
                    unitig_to_index
                )
                
                # Create a mapping from PC (using PC number: pc_idx+1) to genes and collect genes
                pc_to_genes = defaultdict(set)
                for pc_num, unitig, gene in matches:
                    if gene != "Unknown":
                        pc_to_genes[pc_num].add(gene)
                for genes in pc_to_genes.values():
                    combined_genes.update(genes)
                
                # Build label strings for the SHAP bar chart
                pc_labels = []
                for i, pc_idx in enumerate(top_pc_indices):
                    pcnum = pc_idx + 1  # using 1-indexed for display
                    genes_found = list(pc_to_genes.get(pcnum, []))
                    annotated_list = [id_to_anno.get(gene_id, gene_id) for gene_id in genes_found]
                    short_label_list = annotated_list[:3]
                    if not short_label_list:
                        label_str = f"PC{pcnum} => noGene"
                    else:
                        label_str = f"PC{pcnum} => {', '.join(short_label_list)}"
                    pc_labels.append(label_str)
                shap_values_top = shap_vals_for_sample[top_pc_indices]

                # Use a diverging colormap centered at zero:
                cmap = plt.cm.RdBu_r  # Red to Blue, reversed; negative values will be blue, positive red.
                vmin = shap_values_top.min()
                vmax = shap_values_top.max()
                norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                bar_colors = [cmap(norm(val)) for val in shap_values_top]

                # Plot the SHAP bar chart with the new color mapping
                fig_shap, ax_shap = plt.subplots(figsize=(5, max(3, num_pcs * 0.4)))
                ax_shap.barh(pc_labels, shap_values_top, color=bar_colors)
                ax_shap.axvline(x=0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
                ax_shap.set_xlabel("SHAP value (impact on model output)", fontsize='small')
                ax_shap.set_title(f"{param} Model - SHAP with Top PC-to-Gene Mappings", fontsize='small')
                ax_shap.tick_params(axis='both', labelsize='x-small')
                plt.tight_layout()
                st.pyplot(fig_shap, use_container_width=True)
                
                # -----------------------------------------------------------------
                # Store matched genes for printing later (per parameter)
                # -----------------------------------------------------------------
                temp_mapping = {}
                for pc_idx in top_pc_indices:
                    pcnum = pc_idx + 1  # 1-indexed for display
                    cluster_ids = pc_to_genes.get(pcnum, set())
                    # Convert cluster IDs to annotated gene names (if available)
                    annotated_list = [id_to_anno.get(cid, cid) for cid in cluster_ids]
                    temp_mapping[f"PC{pcnum}"] = annotated_list
                matched_genes_by_param[param] = temp_mapping
            else:
                st.info("SHAP analysis not implemented for non-XGBoost algorithms.")
        
        # # ------------------------------------------------------------------
        # # Matched Genes from shap - for debugging mostly
        # # ------------------------------------------------------------------
        # if algorithm == "XGBoost" and matched_genes_by_param:
        #     st.write("---")
        #     st.subheader("Matched Genes from SHAP Analysis (per parameter)")
        #     for param, gene_mapping in matched_genes_by_param.items():
        #         st.markdown(f"### {param}")
        #         for pc_label, genes in gene_mapping.items():
        #             if genes:
        #                 gene_str = ", ".join(genes)
        #                 st.write(f"**{pc_label}** -> {gene_str}")
        #             else:
        #                 st.write(f"**{pc_label}** -> No genes matched.")
        
        # ------------------------------------------------------------------
        # GO Enrichment Analysis (Unified for all parameters)
        # ------------------------------------------------------------------
        st.write("---")
        st.subheader("GO Enrichment Analysis")
    
        if not combined_genes:
            st.write("No gene names identified for GO Enrichment.")
        else:
            # Convert combined_genes to a list
            all_matched_gene_names = list(combined_genes)
            
            # Perform flexible gene name matching
            with st.spinner("Performing flexible gene name matching..."):
                flexible_study_genes = set()
                for user_gene in all_matched_gene_names:
                    user_gene_str = user_gene.strip().lower()
                    # Create a mapping from lowercased, stripped keys to original keys
                    dict_keys_lower = {k.lower().strip(): k for k in gene2gos.keys()}
                    if user_gene_str in dict_keys_lower:
                        # Exact match after lowercasing and stripping
                        matched_key = dict_keys_lower[user_gene_str]
                        flexible_study_genes.add(matched_key)
                    else:
                        # Check for partial matches (substring in either direction)
                        for dict_gene_name in gene2gos.keys():
                            dict_gene_str = dict_gene_name.lower().strip()
                            if (user_gene_str in dict_gene_str) or (dict_gene_str in user_gene_str):
                                flexible_study_genes.add(dict_gene_name)
                study_genes = list(flexible_study_genes)
            
            if not study_genes:
                st.write("None of the matched genes have GO annotations (even with flexible matching).")
            else:
                with st.spinner("Running GO Enrichment Analysis..."):
                    results = run_go_enrichment(study_genes, gene2gos, go_obo_file, alpha=0.05)
                filtered_results = [r for r in results if r.study_count > 0 and r.p_fdr_bh < 1]
                enriched_terms = [r for r in filtered_results if r.p_fdr_bh <= 0.05]

                if not enriched_terms:
                    st.write("No enriched GO terms found with these genes (FDR-BH <= 0.05).")
                else:
                    rows = []
                for r in enriched_terms:
                    rows.append({
                        "GO Term": r.goterm.name,
                        "GO ID": r.goterm.id,
                        "p_uncorrected": r.p_uncorrected,
                        "p_fdr_bh": r.p_fdr_bh,
                        "pop_count": r.pop_count,
                    })
                df_go = pd.DataFrame(rows).sort_values("p_fdr_bh")
                st.write("**GO Enrichment Results** (FDR-BH <= 0.05)")
                st.table(df_go)
    
                st.write("---")
                st.subheader("GO DAG Diagram")
                fig_dag = plot_go_dag(enriched_terms, go_obo_file)
                if fig_dag:
                    st.pyplot(fig_dag)
                else:
                    st.write("Unable to generate DAG diagram for enriched terms.")