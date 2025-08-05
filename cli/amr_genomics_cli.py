#!/usr/bin/env python3
"""
CLI for AMR Genomics KP Analysis.
Transforms the previously Streamlit-based application into
a terminal application that outputs results to AMR_Genomics_KP_Results.



1) Colony Picker:
- colony picker - with row/column coordinates

python cli/amr_genomics_cli.py     colony_picker     --config config/config.yaml     --row 31     --col 48     --condition "Colistin-0.8ugml"

- - colony picker - with strain names - refer to strain_names.txt

python cli/amr_genomics_cli.py colony_picker \
    --config config/config.yaml  \
    --strain B42 \
    --condition "Colistin-0.8ugml"

or 
python cli/amr_genomics_cli.py colony_picker \
    --config config/config.yaml \
    --row 21 \
    --col 22 \
    --condition "Colistin-0.8ugml"

2) ML Prediction:

 --model_type choices are "XGBoost" or "TabNet" 

python cli/amr_genomics_cli.py ml_prediction \
  --config config/config.yaml \
  --fasta /path/to/your/fasta.fasta \
  --condition "Colistin_0.8ugml" \
  --model_type TabNet
"""

#!/usr/bin/env python3
"""
CLI for AMR Genomics KP Analysis.
Transforms the previously Streamlit-based application into
a terminal application that outputs results to AMR_Genomics_KP_Results.

Modifications:
--------------
1. All debug/info/warning prints have been removed.
2. Numerical/statistical output is written to a .out file (text file) in the same
   results directory instead of being printed directly to the terminal.
3. Results are saved to /ibex/user/hinkovn/AMR_Genomics_KP_Results.
4. In `colony_picker`, users can input the strain name instead of row and column coordinates.
5. Circularity, size, and opacity plots are combined into one image.
6. The ML model loader now extracts the XGBoost model paths from a nested config.

Usage Examples:

1) Colony Picker:
   (a) By row/column:
       python cli/amr_genomics_cli.py colony_picker --config config/config.yaml --row 11 --col 22 --condition "Colistin-0.8ugml"
   (b) By strain name:
       python cli/amr_genomics_cli.py colony_picker --config config/config.yaml --strain H150 --condition "Colistin-0.8ugml"

2) ML Prediction:
       python cli/amr_genomics_cli.py ml_prediction --config config/config.yaml --fasta /path/to/fasta.fasta --condition "Colistin_0.8ugml"
"""
import argparse
import os
import sys
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # For headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import cv2
from collections import defaultdict
from scipy.stats import percentileofscore, gaussian_kde
from Bio import SeqIO
from Bio.Seq import Seq
from joblib import load
import re
import traceback

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def load_config(config_path):
    """Load and parse configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config file: {e}")
        sys.exit(1)
    return raw_config

def ensure_dir(dir_path):
    """Create directory if it does not exist."""
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Could not create directory {dir_path}: {e}")
        sys.exit(1)

def load_csv(filepath):
    """Load CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load CSV file {filepath}: {e}")
        sys.exit(1)

def load_excel(filepath):
    """Load Excel file into a pandas DataFrame."""
    try:
        df = pd.read_excel(filepath)
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load Excel file {filepath}: {e}")
        sys.exit(1)

def crop_img(image_path):
    """Read an image from disk using OpenCV."""
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Failed to read image: {image_path}")
    return img

def extract_colony(img, row, col, num_rows=32, num_cols=48):
    """Improved colony extraction with dynamic grid calculations"""
    img_height, img_width = img.shape[:2]
    cell_height = img_height / num_rows
    cell_width = img_width / num_cols
    x = col * cell_width
    y = row * cell_height
    x_start = int(round(x))
    y_start = int(round(y))
    x_end = int(round(x + cell_width))
    y_end = int(round(y + cell_height))
    x_start = max(0, min(x_start, img_width - 2))
    y_start = max(0, min(y_start, img_height - 2))
    x_end = max(x_start + 1, min(x_end, img_width))
    y_end = max(y_start + 1, min(y_end, img_height))
    cell = img[y_start:y_end, x_start:x_end]
    if cell.size == 0:
        return None
    lower_bound = np.array([90, 160, 30])  # Using adjusted bounds as in original logic
    upper_bound = np.array([270, 255, 220])
    lower_bound = np.clip(lower_bound, 0, 255)
    upper_bound = np.clip(upper_bound, 0, 255)
    mask = cv2.inRange(cell, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cell
    colony_contour = max(contours, key=cv2.contourArea)
    xC, yC, wC, hC = cv2.boundingRect(colony_contour)
    cX = xC + wC // 2
    cY = yC + hC // 2
    sub_center_x = cell.shape[1] // 2
    sub_center_y = cell.shape[0] // 2
    shiftX = cX - sub_center_x
    shiftY = cY - sub_center_y
    new_x = x_start + shiftX
    new_y = y_start + shiftY
    new_x = max(0, min(new_x, img_width - cell.shape[1]))
    new_y = max(0, min(new_y, img_height - cell.shape[0]))
    centered_cell = img[new_y:new_y+cell.shape[0], new_x:new_x+cell.shape[1]]
    return centered_cell

def load_pca_and_unitigs(pca_model_file, unitig_to_index_file):
    """Load PCA model and unitig index mapping."""
    try:
        pca = load(pca_model_file)
    except Exception as e:
        print(f"[ERROR] Failed to load PCA model from {pca_model_file}: {e}")
        sys.exit(1)
    try:
        unitig_to_index = pd.read_pickle(unitig_to_index_file)
    except Exception as e:
        print(f"[ERROR] Failed to load unitig to index mapping from {unitig_to_index_file}: {e}")
        sys.exit(1)
    return pca, unitig_to_index

def load_models_for_condition(condition, MODEL_PATHS, model_type):
    """Load models for the given condition and model type (e.g., XGBoost or TabNet)."""
    model_files = MODEL_PATHS.get(condition)
    if not model_files:
        print(f"[ERROR] No model paths available for condition: {condition}")
        sys.exit(1)
    # Unwrap the dictionary using the selected model_type key.
    if isinstance(model_files, dict) and model_type in model_files:
        model_files = model_files[model_type]
    else:
        print(f"[ERROR] No model paths available for {model_type} for condition: {condition}")
        sys.exit(1)
    models = {}
    for param, path in model_files.items():
        try:
            models[param] = load(path)
        except Exception as e:
            print(f"[ERROR] Failed to load {param} model from {path}: {e}")
            sys.exit(1)
    return models

def fasta_to_unitig_vector(fasta_path, unitig_to_index, total_unitigs, k=31):
    """Convert a FASTA file into a presence/absence vector of k-mers."""
    binary_vector = np.zeros(total_unitigs, dtype=int)
    try:
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequence = str(record.seq).upper()
            seq_len = len(sequence)
            for i in range(seq_len - k + 1):
                kmer = sequence[i:i+k]
                if kmer in unitig_to_index:
                    binary_vector[unitig_to_index[kmer]] = 1
    except Exception as e:
        print(f"[ERROR] Failed to convert FASTA to unitig vector for {fasta_path}: {e}")
        sys.exit(1)
    return binary_vector

def compute_stats(df, prediction, param_name):
    """Compute mean, S-score, percentile, and % difference from mean."""
    col_name = df.columns[-1]
    data = df[col_name].dropna()
    mean_val = data.mean()
    std_val = data.std()
    s_score = (prediction - mean_val) / std_val if std_val != 0 else np.nan
    percentile = percentileofscore(data, prediction)
    pct_diff = ((prediction - mean_val) / mean_val) * 100 if mean_val != 0 else np.nan
    return mean_val, s_score, percentile, pct_diff

def find_all_occurrences(sequence, subseq):
    """Yield starting index of each occurrence of subseq in sequence."""
    start = 0
    while True:
        pos = sequence.find(subseq, start)
        if pos == -1:
            break
        yield pos
        start = pos + 1

def get_overlapping_genes(features, start_pos, end_pos):
    """Return a list of gene-locus_tag pairs overlapping with [start_pos, end_pos]."""
    hits = []
    for feat in features:
        if feat.type in ["gene", "CDS"]:
            f_start = int(feat.location.start)
            f_end   = int(feat.location.end)
            if not (f_end < start_pos or f_start > end_pos):
                locus_tag = feat.qualifiers.get("locus_tag", ["?"])[0]
                gene = feat.qualifiers.get("gene", ["?"])[0]
                product = feat.qualifiers.get("product", ["?"])[0]
                hits.append((feat.type, f_start, f_end, locus_tag, gene, product))
    return hits

def extract_genes(overlap_hits):
    """Extract gene information from overlap hits."""
    genes = []
    for (ftype, fs, fe, lt, gene, prod) in overlap_hits:
        if gene != "?":
            genes.append(gene)
        elif lt != "?":
            genes.append(lt)
        else:
            genes.append(prod if prod != "?" else "Unknown")
    return genes

def process_gbff(gbff_path, target_unitigs, pca, index_to_unitig, unitig_to_index):
    """
    Process .gbff file to find genes overlapping with specified unitigs.
    Returns list of (PC_number, Unitig, Gene).
    """
    matches = []
    try:
        all_ref_records = list(SeqIO.parse(gbff_path, "genbank"))
    except Exception as e:
        return matches
    if not all_ref_records:
        return matches
    for rec in all_ref_records:
        forward_seq_str = str(rec.seq).upper()
        revcomp_seq_str = str(rec.seq.reverse_complement()).upper()
        rec_feats = rec.features
        for pc_idx, unitigs in target_unitigs.items():
            for utg in unitigs:
                utg_revcomp = str(Seq(utg).reverse_complement())
                for pos in find_all_occurrences(forward_seq_str, utg):
                    overlap_hits = get_overlapping_genes(rec_feats, pos, pos + len(utg) - 1)
                    genes = extract_genes(overlap_hits)
                    for gene in genes:
                        matches.append((pc_idx + 1, utg, gene))
                for pos in find_all_occurrences(forward_seq_str, utg_revcomp):
                    overlap_hits = get_overlapping_genes(rec_feats, pos, pos + len(utg_revcomp) - 1)
                    genes = extract_genes(overlap_hits)
                    for gene in genes:
                        matches.append((pc_idx + 1, utg, gene))
                for pos in find_all_occurrences(revcomp_seq_str, utg):
                    L = len(forward_seq_str)
                    fwd_start = L - (pos + len(utg))
                    fwd_end   = L - pos - 1
                    overlap_hits = get_overlapping_genes(rec_feats, fwd_start, fwd_end)
                    genes = extract_genes(overlap_hits)
                    for gene in genes:
                        matches.append((pc_idx + 1, utg, gene))
                for pos in find_all_occurrences(revcomp_seq_str, utg_revcomp):
                    L = len(forward_seq_str)
                    fwd_start = L - (pos + len(utg_revcomp))
                    fwd_end   = L - pos - 1
                    overlap_hits = get_overlapping_genes(rec_feats, fwd_start, fwd_end)
                    genes = extract_genes(overlap_hits)
                    for gene in genes:
                        matches.append((pc_idx + 1, utg, gene))
    return matches

def get_color(score, max_score):
    """Map numeric score to an RGB color from green to red for reporting."""
    if score is None or max_score == 0:
        return "#FFFFFF"
    color_val = int((score / max_score) * 255)
    color_val = min(max(color_val, 0), 255)
    return f"rgb({color_val}, {255 - color_val}, 0)"

# ---------------------------
# COLONY PICKER LOGIC
# ---------------------------

def colony_picker(args, config):
    try:
        output_dir = "AMR_Genomics_KP_Results"
        ensure_dir(output_dir)
        strain_file_path = config["files"]["strain_file"]
        try:
            strains_df = load_excel(strain_file_path)
        except Exception:
            strains_df = load_csv(strain_file_path)
        if args.strain:
            strain_identifier = args.strain
            strain_row_col = strains_df[strains_df['ID'] == args.strain]
            if strain_row_col.empty:
                print(f"ERROR: Strain {args.strain} not found in strain file.")
                sys.exit(1)
            row = int(strain_row_col.iloc[0]['Row'])
            col = int(strain_row_col.iloc[0]['Column'])
        elif args.row is not None and args.col is not None:
            row = args.row
            col = args.col
            strain_identifier = f"r{row}_c{col}"
        else:
            print("ERROR: Please provide either --strain or both --row and --col.")
            sys.exit(1)
        summary_filename = os.path.join(
            output_dir,
            f"colony_picker_{strain_identifier}_{args.condition}.out"
        )
        summary_file = open(summary_filename, "w")
        circularity_data_file = config["files"]["circularity_data"]
        size_data_file = config["files"]["size_data"]
        opacity_data_file = config["files"]["opacity_data"]
        amr_data_file_path = config["files"]["amr_data"]
        gene_antibiotics_file = config["files"]["gene_antibiotics"]
        image_directory = config["directories"]["image_directory"]
        iris_directory = config["directories"]["iris_directory"]
        strain = args.strain if args.strain else strain_identifier
        circ_data_flat = load_csv(circularity_data_file).values.flatten()
        size_data_flat = load_csv(size_data_file).values.flatten()
        opa_data_flat  = load_csv(opacity_data_file).values.flatten()
        amr_df = pd.read_csv(amr_data_file_path, sep='\t', low_memory=False)
        try:
            with open(gene_antibiotics_file, 'r') as f:
                if gene_antibiotics_file.endswith(('.yaml', '.yml')):
                    gene_abx_map = yaml.safe_load(f)
                elif gene_antibiotics_file.endswith('.json'):
                    gene_abx_map = json.load(f)
                else:
                    gene_abx_map = {}
        except Exception as e:
            summary_file.write(f"ERROR loading gene_antibiotics_file: {e}\n")
            summary_file.close()
            sys.exit(1)
        ybtq_index = amr_df.columns.get_loc('ybtQ.x') if 'ybtQ.x' in amr_df.columns else None
        if ybtq_index is None:
            summary_file.write("ERROR: 'ybtQ.x' column not found in AMR data.\n")
            summary_file.close()
            sys.exit(1)
        all_res_genes = amr_df.columns[3:ybtq_index]
        virulence_cols = [
            c for c in amr_df.columns[ybtq_index:]
            if not any(ex in c for ex in ['1536_Position_column', 'virulence_score'])
        ]
        mutation_pattern = re.compile(r'_[A-Z]\d+[A-Z]|_STOP|_D')
        just_res_genes = [g for g in all_res_genes if not mutation_pattern.search(g)]
        condition = args.condition
        replicates = ["A", "B", "C", "D", "E"]
        def replicate_available(cond, rep):
            fpath = os.path.join(image_directory, f"{cond}-1-1_{rep}.JPG.grid.jpg")
            return os.path.exists(fpath)
        available_reps = [rep for rep in replicates if replicate_available(condition, rep)]
        subset = amr_df[(amr_df['Row'] == row) & (amr_df['Column'] == col)]
        if subset.empty:
            species, st_type, origin = None, None, None
            vir_score, res_score = None, None
            std_res_genes, present_vir_genes, res_muts_ = [], [], []
        else:
            species = subset.iloc[0].get('species')
            st_type = subset.iloc[0].get('ST')
            origin  = subset.iloc[0].get('origin')
            vir_score = subset.iloc[0].get('virulence_score')
            res_score = subset.iloc[0].get('resistance_score')
            std_res_genes = [g for g in just_res_genes if subset.iloc[0][g] == 1]
            res_muts_ = [
                g for g in all_res_genes
                if (subset.iloc[0][g] == 1 and mutation_pattern.search(g))
            ]
            present_vir_genes = []
            for g in virulence_cols:
                if subset.iloc[0][g] == 1:
                    if g.endswith(('.x', '.y')):
                        present_vir_genes.append(g[:-2])
                    else:
                        present_vir_genes.append(g)
        summary_file.write(f"=== ColonyPicker Summary ===\n")
        summary_file.write(f"Strain: {strain}, Condition: {condition}\n")
        if species:
            summary_file.write(f"Species: {species}\n")
        if st_type:
            summary_file.write(f"MLST: {st_type}\n")
        if origin:
            summary_file.write(f"Origin: {origin}\n")
        if vir_score is not None:
            summary_file.write(f"Virulence Score: {vir_score}\n")
        if res_score is not None:
            summary_file.write(f"Resistance Score: {res_score}\n")
        if std_res_genes:
            summary_file.write("Resistance Genes:\n")
            abx_to_genes = defaultdict(list)
            for g in std_res_genes:
                abx = gene_abx_map.get(g, 'Unknown')
                abx_to_genes[abx].append(g)
            for abx_group, genes_list in abx_to_genes.items():
                summary_file.write(f"  {abx_group}: {', '.join(genes_list)}\n")
        else:
            summary_file.write("No resistance genes found.\n")
        if res_muts_:
            summary_file.write("Resistance Mutations:\n")
            summary_file.write(f"  {', '.join(res_muts_)}\n")
        else:
            summary_file.write("No resistance mutations found.\n")
        if present_vir_genes:
            summary_file.write("Virulence Genes:\n")
            summary_file.write(f"  {', '.join(present_vir_genes)}\n")
        else:
            summary_file.write("No virulence genes found.\n")
        rep_values = {"circularity": [], "size": [], "opacity": []}
        replicate_images = []
        for rep in available_reps:
            image_filename = f"{condition}-1-1_{rep}.JPG.grid.jpg"
            image_path = os.path.join(image_directory, image_filename)
            plate_img = crop_img(image_path)
            if plate_img is None:
                continue
            cell_width, cell_height = 104, 90
            adj_row = row - 1
            adj_col = col - 1
            extracted = extract_colony(plate_img, adj_row, adj_col)
            if extracted is not None:
                rgb_img = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)
                replicate_images.append((rep, rgb_img))
            else:
                summary_file.write(f"WARNING: No colony extracted for replicate {rep}\n")
            iris_file = os.path.join(iris_directory, f"{condition}-1-1_{rep}.JPG.iris")
            circ, siz, opa = None, None, None
            if os.path.exists(iris_file):
                try:
                    iris_df = pd.read_csv(
                        iris_file,
                        sep='\s+',
                        comment='#',
                        skip_blank_lines=True,
                        header=0,
                        engine='python'
                    )
                    matching = iris_df[
                        (iris_df['row'] == row) & (iris_df['column'] == col)
                    ]
                    if not matching.empty:
                        circ = float(matching['circularity'].iloc[0])
                        siz  = float(matching['size'].iloc[0])
                        opa  = float(matching['opacity'].iloc[0])
                    else:
                        summary_file.write(f"WARNING: No matching IRIS data for replicate {rep}\n")
                except Exception as e:
                    summary_file.write(f"WARNING: Could not parse IRIS file {iris_file}: {e}\n")
            rep_values["circularity"].append(circ)
            rep_values["size"].append(siz)
            rep_values["opacity"].append(opa)
            summary_file.write(
                f"Replicate {rep}: Circularity={circ}, Size={siz}, Opacity={opa}\n"
            )
        if replicate_images:
            fig, axes = plt.subplots(
                1, len(replicate_images),
                figsize=(5 * len(replicate_images), 5)
            )
            if len(replicate_images) == 1:
                axes = [axes]
            for ax, (rep_label, rgb_img) in zip(axes, replicate_images):
                ax.imshow(rgb_img)
                ax.set_title(f"Replicate {rep_label}")
                ax.axis("off")
            plt.tight_layout()
            out_colony_reps = os.path.join(
                output_dir,
                f"{strain}_{condition}_replicates.png"
            )
            plt.savefig(out_colony_reps, dpi=150)
            plt.close()
        param_list = ["Circularity", "Size", "Opacity"]
        data_arrays = [circ_data_flat, size_data_flat, opa_data_flat]
        rep_arrays = [rep_values["circularity"], rep_values["size"], rep_values["opacity"]]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, (param, data_arr, vals) in enumerate(zip(param_list, data_arrays, rep_arrays)):
            ax = axes[i]
            sns.kdeplot(data_arr, fill=True, color='blue', label=f"{param} KDE", ax=ax)
            mean_val = np.mean(data_arr)
            std_dev = np.std(data_arr)
            ax.axvline(mean_val, color='gray', linestyle='-.', linewidth=1.5,
                       label=f"Dataset Mean: {mean_val:.2f}")
            valid_vals = [v for v in vals if v is not None]
            if valid_vals:
                rep_mean = np.mean(valid_vals)
                n = len(valid_vals)
                s_score = (rep_mean - mean_val) / (std_dev / np.sqrt(n)) if std_dev > 0 and n > 0 else None
                label_str = f"Rep Mean: {rep_mean:.2f}"
                if s_score is not None:
                    label_str += f" (S-score: {s_score:.2f})"
                ax.axvline(rep_mean, color='black', linestyle='--', linewidth=1.5, label=label_str)
                colors = ['green', 'red', 'purple', 'orange', 'cyan']
                for v, rp, clr in zip(vals, available_reps, colors):
                    if v is not None:
                        pctile = percentileofscore(data_arr, v)
                        pct_diff = ((v - mean_val) / mean_val) * 100 if mean_val != 0 else 0
                        yv = gaussian_kde(data_arr)(v)
                        ax.plot(v, yv, 'x', color=clr, markersize=10,
                                label=f"{rp}: {v:.2f} (Pctile: {pctile:.1f}%, Diff:{pct_diff:.1f}%)")
            ax.set_xlabel(param, fontsize='large')
            ax.set_ylabel("Density", fontsize='large')
            ax.set_title(f"{param} Distribution", fontsize='large')
            ax.legend(loc='best', fontsize='small')
        plt.tight_layout()
        out_plot_path = os.path.join(output_dir, f"{strain}_{condition}_distributions.png")
        plt.savefig(out_plot_path, dpi=150)
        plt.close()
        summary_file.write("\n[INFO] ColonyPicker finished successfully.\n")
        summary_file.close()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

# ---------------------------
# ML PREDICTION LOGIC
# ---------------------------

def ml_prediction(args, config):
    """
    Perform ML-based predictions for one or more FASTA files.
    Outputs info on matched genes, predictions, distribution plots,
    and SHAP-based PCA analysis. All results are saved to /ibex/user/hinkovn/AMR_Genomics_KP_Results.
    Numerical/statistics are also written to a .out file in the same directory.
    """
    try:
        output_dir = "AMR_Genomics_KP_Results"
        ensure_dir(output_dir)
        pca_model_file = config["pca_model_file"]
        unitig_to_index_file = config["unitig_to_index_file"]
        MODEL_PATHS = config["MODEL_PATHS"]
        condition = args.condition
        circularity_data_file = config["files"]["circularity_data"]
        opacity_data_file = config["files"]["opacity_data"]
        size_data_file = config["files"]["size_data"]
        all_fsa_file = config["files"]["all_fsa"]
        vfdb_fas_file = config["files"]["vfdb_fas"]
        reference_gbff_file = config["files"]["reference_gbff"]
        df_circ = load_csv(circularity_data_file)
        df_opa  = load_csv(opacity_data_file)
        df_siz  = load_csv(size_data_file)
        pca, unitig_to_index = load_pca_and_unitigs(pca_model_file, unitig_to_index_file)
        index_to_unitig = [None] * len(unitig_to_index)
        for u, i in unitig_to_index.items():
            index_to_unitig[i] = u
        num_unitigs = len(unitig_to_index)
        # Load models based on the selected model type (XGBoost or TabNet)
        models = load_models_for_condition(condition, MODEL_PATHS, args.model_type)
        model_opacity = models.get("opacity")
        model_circ    = models.get("circularity")
        model_size    = models.get("size")
        summary_filename = os.path.join(output_dir, f"ml_prediction_{condition}.out")
        summary_file = open(summary_filename, "w")
        for fasta_path in args.fasta:
            fasta_basename = os.path.splitext(os.path.basename(fasta_path))[0]
            summary_file.write(f"=== ML Prediction for {fasta_basename} ===\n")
            try:
                with open(fasta_path, 'r') as f:
                    user_fasta_str = f.read().upper()
                user_fasta_str_no_ws = "".join(user_fasta_str.split())
            except Exception as e:
                summary_file.write(f"ERROR reading FASTA file {fasta_path}: {e}\n")
                continue
            matched_genes = []
            matched_virulence_genes = []
            try:
                for record in SeqIO.parse(all_fsa_file, "fasta"):
                    ref_seq = str(record.seq).upper()
                    if ref_seq in user_fasta_str_no_ws:
                        matched_genes.append(record.id)
            except Exception as e:
                summary_file.write(f"WARNING: Could not process {all_fsa_file} for resistance genes: {e}\n")
            try:
                for record in SeqIO.parse(vfdb_fas_file, "fasta"):
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
                summary_file.write(f"WARNING: Could not process {vfdb_fas_file} for virulence genes: {e}\n")
            bin_vector = fasta_to_unitig_vector(fasta_path, unitig_to_index, num_unitigs)
            bin_vector_pca = pca.transform([bin_vector])
            pred_opacity = model_opacity.predict(bin_vector_pca)[0]
            pred_circ    = model_circ.predict(bin_vector_pca)[0]
            pred_size    = model_size.predict(bin_vector_pca)[0]
            summary_file.write(f"Matched Resistance Genes: {matched_genes if matched_genes else 'None'}\n")
            summary_file.write(f"Matched Virulence Genes: {matched_virulence_genes if matched_virulence_genes else 'None'}\n")
            summary_file.write(f"Predicted Opacity: {pred_opacity:.4f}\n")
            summary_file.write(f"Predicted Circularity: {pred_circ:.4f}\n")
            summary_file.write(f"Predicted Size: {pred_size:.4f}\n")
            c_mean, c_s, c_pct, c_diff = compute_stats(df_circ, pred_circ, "Circularity")
            o_mean, o_s, o_pct, o_diff = compute_stats(df_opa,  pred_opacity, "Opacity")
            s_mean, s_s, s_pct, s_diff = compute_stats(df_siz,  pred_size,    "Size")
            summary_file.write(
                f"Circularity Stats => Mean: {c_mean:.4f}, S-score: {c_s:.4f}, "
                f"Percentile: {c_pct:.2f}, %Diff: {c_diff:.2f}\n"
            )
            summary_file.write(
                f"Opacity Stats => Mean: {o_mean:.4f}, S-score: {o_s:.4f}, "
                f"Percentile: {o_pct:.2f}, %Diff: {o_diff:.2f}\n"
            )
            summary_file.write(
                f"Size Stats => Mean: {s_mean:.4f}, S-score: {s_s:.4f}, "
                f"Percentile: {s_pct:.2f}, %Diff: {s_diff:.2f}\n"
            )
            try:
                param_list = ["Circularity", "Size", "Opacity"]
                data_arrays = [df_circ.iloc[:, -1].dropna(), df_siz.iloc[:, -1].dropna(), df_opa.iloc[:, -1].dropna()]
                pred_values = [pred_circ, pred_size, pred_opacity]
                means = [c_mean, s_mean, o_mean]
                std_devs = [df_circ.iloc[:, -1].std(), df_siz.iloc[:, -1].std(), df_opa.iloc[:, -1].std()]
                diffs = [c_diff, s_diff, o_diff]
                pc_scores = [c_s, s_s, o_s]
                percentiles = [c_pct, s_pct, o_pct]
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                for i, (param, data, pred, mean, std, pct, diff, s_score) in enumerate(zip(
                        param_list, data_arrays, pred_values, means, std_devs, percentiles, diffs, [c_s, s_s, o_s])):
                    ax = axes[i]
                    sns.kdeplot(data, fill=True, color='blue', label=f"{param} KDE", ax=ax)
                    ax.axvline(mean, color='gray', linestyle='-.', linewidth=1.5, label=f"Dataset Mean: {mean:.2f}")
                    ax.axvline(pred, color='red', linestyle='--', linewidth=1.5, label=f"Prediction: {pred:.2f}")
                    ax.set_xlabel(param, fontsize='large')
                    ax.set_ylabel("Density", fontsize='large')
                    ax.set_title(f"{param} Distribution", fontsize='large')
                    ax.legend(loc='best', fontsize='small')
                plt.tight_layout()
                combined_plot_path = os.path.join(output_dir, f"{fasta_basename}_distributions.png")
                plt.savefig(combined_plot_path, dpi=150)
                plt.close()
                summary_file.write(f"Saved combined distributions plot: {combined_plot_path}\n")
            except Exception as e:
                summary_file.write(f"ERROR creating combined distribution plots: {e}\n")
                traceback.print_exc()
            try:
                explainer_opacity = shap.TreeExplainer(model_opacity)
                shap_values_opacity = explainer_opacity.shap_values(bin_vector_pca)
                shap_vals_for_sample = shap_values_opacity[0]
                abs_shap = np.abs(shap_vals_for_sample)
                num_pcs_to_show = 10
                top_pc_indices = np.argsort(abs_shap)[::-1][:num_pcs_to_show]
                pc_to_unitigs = {}
                for pc_idx in top_pc_indices:
                    loadings = pca.components_[pc_idx, :]
                    abs_loadings = np.abs(loadings)
                    top_idxs = np.argsort(abs_loadings)[::-1][:5]
                    top_utgs = [index_to_unitig[i] for i in top_idxs]
                    pc_to_unitigs[pc_idx] = top_utgs
                matches = process_gbff(reference_gbff_file, pc_to_unitigs, pca, index_to_unitig, unitig_to_index)
                pc_to_genes = defaultdict(set)
                for pc_num, unitig, gene in matches:
                    if gene != "Unknown":
                        pc_to_genes[pc_num].add(gene)
                pc_numbers = [
                    f"PC{pc_idx+1} => {', '.join(list(pc_to_genes[pc_idx+1])[:3]) or 'noGene'}"
                    for pc_idx in top_pc_indices
                ]
                shap_values_top = shap_vals_for_sample[top_pc_indices]
                plt.figure(figsize=(12, max(6, num_pcs_to_show * 0.5)))
                plt.barh(pc_numbers, shap_values_top, color='mediumpurple')
                plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
                plt.xlabel("SHAP value (impact on model output)", fontsize='large')
                plt.title(f"Opacity Model - SHAP (Top {num_pcs_to_show} PCs)", fontsize='x-large')
                plt.tight_layout()
                shap_out_path = os.path.join(output_dir, f"{fasta_basename}_shap_opacity.png")
                plt.savefig(shap_out_path, dpi=150)
                plt.close()
            except Exception as e:
                summary_file.write(f"ERROR during SHAP computation: {e}\n")
                traceback.print_exc()
            summary_file.write(f"=== ML Prediction for {fasta_basename} Completed ===\n\n")
        summary_file.write("\n[INFO] ML Prediction finished successfully.\n")
        summary_file.close()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

# ---------------------------
# MAIN - ARGPARSE
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="AMR Genomics KP Terminal Application")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")
    
    # Colony Picker sub-command.
    parser_colony = subparsers.add_parser("colony_picker", help="Extract colony images and data.")
    parser_colony.add_argument("--config", required=True, help="Path to YAML config file.")
    parser_colony.add_argument("--strain", help="Strain name (e.g., H150).")
    parser_colony.add_argument("--row", type=int, help="Row coordinate of colony.")
    parser_colony.add_argument("--col", type=int, help="Column coordinate of colony.")
    parser_colony.add_argument("--condition", required=True, help="Condition name for images.")
    
    # ML Prediction sub-command.
    parser_ml = subparsers.add_parser("ml_prediction", help="Run ML predictions for one or more FASTA files.")
    parser_ml.add_argument("--config", required=True, help="Path to YAML config file.")
    parser_ml.add_argument("--fasta", nargs="+", required=True, help="One or more FASTA files.")
    parser_ml.add_argument("--condition", required=True, help="Condition for ML models (e.g. Colistin_0.8ugml).")
    parser_ml.add_argument("--model_type", choices=["XGBoost", "TabNet"], default="XGBoost",
                           help="Model type to use for predictions (default: XGBoost).")
    
    args = parser.parse_args()
    
    if args.command == "colony_picker":
        config = load_config(args.config)
        colony_picker(args, config)
    elif args.command == "ml_prediction":
        config = load_config(args.config)
        ml_prediction(args, config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()