import streamlit as st

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loading import load_csv, load_excel

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from app.utils.data_loading import load_csv, load_excel
from app.utils.image_handling import crop_img, extract_colony
from app.utils.gbff_processing import process_gbff
from app.utils.ml_models import load_pca_and_unitigs, load_distributions
import json
import re
from scipy.stats import percentileofscore, gaussian_kde
import cv2
import numpy as np

def colonypicker(config):
    st.title("ColonyPicker")

    # CSS to make the table background transparent and reduce spacing. just for aesthetics
    st.markdown("""
        <style>
            ul {
                margin-top: 0px;
                margin-bottom: 0px;
                padding-left: 20px;
            }
            li {
                margin-bottom: 2px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                background-color: transparent;
            }
            th, td {
                background-color: transparent;
                text-align: left;
                padding: 5px;
                vertical-align: top;
                border: none;
            }
        </style>
        """, unsafe_allow_html=True
    )

    try:
        strains_df = load_excel(config['files']['strain_file'])
    except:
        strains_df = load_csv(config['files']['strain_file'])

    circ_data_flat = load_csv(config['files']['circularity_data']).values.flatten()
    size_data_flat = load_csv(config['files']['size_data']).values.flatten()
    opa_data_flat = load_csv(config['files']['opacity_data']).values.flatten()

    amr_df = pd.read_csv(config['files']['amr_data'], sep='\t')

    with open(config['files']['gene_antibiotics'], 'r') as f:
        gene_abx_map = json.load(f)

    ybtq_index = amr_df.columns.get_loc('ybtQ.x')
    all_res_genes = amr_df.columns[3:ybtq_index]
    virulence_cols = [
        c for c in amr_df.columns[ybtq_index:]
        if not any(ex in c for ex in ['1536_Position_column', 'virulence_score'])
    ]

    mutation_pattern = re.compile(r'_[A-Z]\d+[A-Z]|_STOP|_D')
    just_res_genes = [g for g in all_res_genes if not mutation_pattern.search(g)]
    res_mutations = [g for g in all_res_genes if mutation_pattern.search(g)]

    image_directory = config['directories']['image_directory']
    iris_directory = config['directories']['iris_directory']

    conditions = get_conditions(image_directory)
    replicates = ["A", "B", "C", "D", "E"]

    method = st.sidebar.radio("Select colony extraction method:",
                              ("By Strain Name", "By Row and Column"))

    if method == "By Strain Name":
        strain_name = st.sidebar.selectbox("Select Strain", strains_df['ID'])
        r_c_vals = strains_df.loc[strains_df['ID'] == strain_name, ['Row', 'Column']].values[0]
        row, col = int(r_c_vals[0]), int(r_c_vals[1])
        st.markdown(
            f"<p style='font-size:18px;'>Selected Strain: <b>{strain_name}</b> "
            f"(Row {row}, Column {col})</p>",
            unsafe_allow_html=True
        )
    else:
        row = st.sidebar.number_input("Enter Row:", min_value=0, value=0, step=1)
        col = st.sidebar.number_input("Enter Column:", min_value=0, value=0, step=1)

        # Check for row and column limits
        if row > 32:
            st.sidebar.error("Maximum number of rows is 32.")
        if col > 48:
            st.sidebar.error("Maximum number of columns is 48.")

        strain_entry = strains_df[(strains_df['Row'] == row) & (strains_df['Column'] == col)]
        if not strain_entry.empty:
            strain_id = strain_entry.iloc[0]['ID']
            st.markdown(
                f"<p style='font-size:18px;'>Strain for Row {row}, Column {col}: "
                f"<b>{strain_id}</b></p>",
                unsafe_allow_html=True
            )
        else:
            st.write(f"No strain found for Row {row}, Column {col}.")

    selected_condition = st.sidebar.selectbox("Select a condition", conditions)
# Add the Submit button to the sidebar
    submit_button = st.sidebar.button("Submit")

    available_reps = [rep for rep in replicates if replicate_available(selected_condition, rep, image_directory)]
    missing_reps = [rep for rep in replicates if rep not in available_reps]
    if missing_reps:
        st.warning(f"Missing replicates for {selected_condition}: {', '.join(missing_reps)}")

    if submit_button:
        if method == "By Strain Name":
            pass  # Do nothing for strain name method since it's already displayed
        else:
            st.write(f"Colonies for Row: **{row}**, Column: **{col}**")


        (
            std_res_genes,
            res_muts_,
            vir_genes,
            sp,
            st_type,
            orig,
            vir_score,
            res_score
        ) = get_colony_data(row, col, amr_df, just_res_genes, res_mutations, virulence_cols)
        st.write("---")
        st.subheader("Strain Information:")
        if sp:
            st.write(f"**Species**: {sp}")
        if st_type:
            st.write(f"**MLST**: {st_type}")
        if orig:
            st.write(f"**Origin**: {orig}")

        if vir_score is not None:
            vir_color = get_color(vir_score, 5)
            st.markdown(
                f"""
                <div style="display: flex; align-items: center;">
                    <div style="flex: 1;">
                        <strong>Virulence Score:</strong> <span style="color:{vir_color};">{vir_score}</span>
                    </div>
                    <div>
                        <details style="cursor: pointer; font-size: 16px; margin-left: 5px;">
                            <summary style="list-style: none; display: inline;">ℹ️</summary>
                            <div style="margin-top: 5px; font-size: 14px;">
                                <strong>Virulence Score Explanation:</strong><br>
                                - <strong>0</strong>: negative for all of yersiniabactin (ybt), colibactin (clb), aerobactin (iuc)<br>
                                - <strong>1</strong>: yersiniabactin only<br>
                                - <strong>2</strong>: yersiniabactin and colibactin (or colibactin only)<br>
                                - <strong>3</strong>: aerobactin (without yersiniabactin or colibactin)<br>
                                - <strong>4</strong>: aerobactin with yersiniabactin (without colibactin)<br>
                                - <strong>5</strong>: yersiniabactin, colibactin and aerobactin<br> 
                                <strong>Source:</strong> <a href="https://usegalaxy.eu/?tool_id=kleborate" target="_blank">https://usegalaxy.eu/?tool_id=kleborate</a>
                            </div>
                        </details>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if res_score is not None:
            res_color = get_color(res_score, 3)
            st.markdown(
                f"""
                <div style="display: flex; align-items: center;">
                    <div style="flex: 1;">
                        <strong>Resistance Score:</strong> <span style="color:{res_color};">{res_score}</span>
                    </div>
                    <div>
                        <details style="cursor: pointer; font-size: 16px; margin-left: 5px;">
                            <summary style="list-style: none; display: inline;">ℹ️</summary>
                            <div style="margin-top: 5px; font-size: 14px;">
                                <strong>Resistance Score Explanation:</strong><br>
                                - <strong>0</strong>: no ESBL, no carbapenemase (regardless of colistin resistance)<br>
                                - <strong>1</strong>: ESBL, no carbapenemase (regardless of colistin resistance)<br>
                                - <strong>2</strong>: Carbapenemase without colistin resistance (regardless of ESBL genes or OmpK mutations)<br>
                                - <strong>3</strong>: Carbapenemase with colistin resistance (regardless of ESBL genes or OmpK mutations)<br>
                                <strong>Source:</strong> <a href="https://usegalaxy.eu/?tool_id=kleborate" target="_blank">https://usegalaxy.eu/?tool_id=kleborate</a>
                            </div>
                        </details>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.write("---")
        st.subheader("Key Genes:")
        if std_res_genes:
            st.markdown("**Resistance Genes:**")
            abx_to_genes = defaultdict(list)
            for g in std_res_genes:
                abx = gene_abx_map.get(g, 'Unknown')
                abx_to_genes[abx].append(g)

            table_data = defaultdict(list)
            for abx_group, genes in abx_to_genes.items():
                table_data['Antibiotic Class'].append(abx_group)
                table_data['Resistance Genes'].append(", ".join(genes))
            
            df_table = pd.DataFrame(table_data)
            # Make the index start at 1
            df_table.index = df_table.index + 1
            st.table(df_table)
        else:
            st.write("No resistance genes found.")

        if res_muts_:
            st.markdown("**Resistance Mutations:**")
            df_mutations = pd.DataFrame({"Resistance Mutation": res_muts_})
            df_mutations.index = df_mutations.index + 1
            st.table(df_mutations)
        else:
            st.write("No resistance mutations found.")

        if vir_genes:
            st.markdown("**Virulence Genes:**")
            df_virulence = pd.DataFrame({"Virulence Gene": vir_genes})
            df_virulence.index = df_virulence.index + 1
            st.table(df_virulence)
        else:
            st.write("No virulence genes found.")

        st.write("---")
        st.subheader("Colony Images and Metrics")

        rep_values = {"circularity": [], "size": [], "opacity": []}
        if len(available_reps) == 0:
            st.warning("No replicates are available.")
        else:
            cols = st.columns(len(available_reps))
            for i, rep in enumerate(available_reps):
                path = os.path.join(image_directory, f"{selected_condition}-1-1_{rep}.JPG.grid.jpg")
                plate_img = crop_img(path)
                if plate_img is None:
                    continue

                # Fixed cell dimensions
                cell_width, cell_height = 104, 90

                adj_row = row - 1
                adj_col = col - 1
                extracted = extract_colony(plate_img, adj_row, adj_col)

                if extracted is not None:
                    rgb_img = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)
                    with cols[i]:
                        st.image(rgb_img, caption=f"Replicate {rep}", use_container_width=True)

                # IRIS-based lookup
                iris_file = os.path.join(
                    iris_directory, f"{selected_condition}-1-1_{rep}.JPG.iris"
                )

                circ, siz, opa = None, None, None
                if os.path.exists(iris_file):
                    try:
                        iris_df = pd.read_csv(
                            iris_file,
                            delim_whitespace=True,
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
                            siz = float(matching['size'].iloc[0])
                            opa = float(matching['opacity'].iloc[0])
                    except Exception as e:
                        st.warning(f"Could not parse IRIS file: {iris_file}. Error: {e}")

                rep_values["circularity"].append(circ)
                rep_values["size"].append(siz)
                rep_values["opacity"].append(opa)

                with cols[i]:
                    if circ is not None:
                        st.write(f"Circularity: {circ:.3f}")
                    else:
                        st.write("Circularity: N/A (Missing IRIS log)")

                    if siz is not None:
                        st.write(f"Size: {int(siz)} px")
                    else:
                        st.write("Size: N/A Missing IRIS log")

                    if opa is not None:
                        st.write(f"Opacity: {int(opa)}")
                    else:
                        st.write("Opacity: N/A Missing IRIS log")


        st.write("---")
        st.subheader("Distribution Visualization and Statistics")
        # Distribution Plots
        for param, data_arr, vals in zip(
            ["Circularity", "Size", "Opacity"],
            [circ_data_flat, size_data_flat, opa_data_flat],
            [rep_values["circularity"], rep_values["size"], rep_values["opacity"]]
        ):
            st.markdown(f"<h5 style='font-size:17px; margin-bottom: 0;'>{param}</h5>", unsafe_allow_html=True)
            mean_val = np.mean(data_arr)
            std_dev = np.std(data_arr)
            valid_vals = [v for v in vals if v is not None]

            if valid_vals:
                rep_mean = np.mean(valid_vals)
                n = len(valid_vals)
                if std_dev > 0 and n > 0:
                    s_score = (rep_mean - mean_val) / (std_dev / np.sqrt(n))
                else:
                    s_score = None
            else:
                rep_mean = None
                s_score = None

            fig, ax = plt.subplots()
            fig.set_size_inches(5, 3)

            kde_fn = gaussian_kde(data_arr)
            x_vals = np.linspace(min(data_arr), max(data_arr), 1000)
            y_vals = kde_fn(x_vals)
            ax.plot(x_vals, y_vals, color='blue', label=f"{param} KDE")

            ax.axvline(mean_val, color='gray', linestyle='-.', linewidth=1.5,
                    label=f"Dataset Mean: {mean_val:.2f}")

            if rep_mean is not None:
                label_str = (
                    f"Replicates Mean: {rep_mean:.2f}"
                    f"{' (S-score: ' + str(round(s_score,2)) + ')' if s_score is not None else ''}"
                )
                ax.axvline(rep_mean, color='black', linestyle='--', linewidth=1.5, label=label_str)

            colors = ['green', 'blue', 'purple', 'orange', 'red']
            for v, rp, clr in zip(vals, available_reps, colors):
                if v is not None:
                    percentile_val = percentileofscore(data_arr, v)
                    if mean_val != 0:
                        pct_diff = ((v - mean_val) / mean_val) * 100
                    else:
                        pct_diff = 0
                    y_val = kde_fn(v)
                    ax.plot(
                        v, y_val, 'x',
                        color=clr, markersize=10,
                        label=(
                            f"{rp}: {v:.2f} "
                            f"(Pctile: {percentile_val:.1f}%, Diff: {pct_diff:.1f}%)"
                        )
                    )

            ax.set_xlabel("Value", fontsize='x-small')
            ax.set_ylabel("Density", fontsize='x-small')
            ax.set_title(f"{param} Distribution", fontsize='small')
            ax.tick_params(axis='x', labelsize='x-small')
            ax.tick_params(axis='y', labelsize='x-small')
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                    fontsize='small', frameon=False, handletextpad=0.5)
            st.pyplot(fig)

def get_conditions(directory):
    """Collect names of conditions from all *.JPG.grid.jpg files."""
    files = glob.glob(os.path.join(directory, "*.JPG.grid.jpg"))
    condition_names = set()
    for f in files:
        base_name = os.path.basename(f)
        # remove last 19 chars: e.g. conditionX-1-1_A.JPG.grid.jpg - makes it easier to look through the conditions
        cond = base_name[:-19]
        condition_names.add(cond)
    return sorted(condition_names)

def replicate_available(condition, rep, image_directory):
    """Check if an image file exists for the given condition+replicate."""
    fpath = os.path.join(image_directory, f"{condition}-1-1_{rep}.JPG.grid.jpg")
    return os.path.exists(fpath)

def get_colony_data(row, col, amr_df, just_res_genes, res_mutations, vir_genes):
    """Retrieve species, ST, etc. from the AMR dataframe."""
    subset = amr_df[(amr_df['Row'] == row) & (amr_df['Column'] == col)]
    if subset.empty:
        return (None,) * 8

    species = subset.iloc[0]['species']
    ST = subset.iloc[0]['ST']
    origin = subset.iloc[0]['origin']
    vir_score = subset.iloc[0].get('virulence_score')
    res_score = subset.iloc[0].get('resistance_score')

    standard_res_genes = [
        g for g in just_res_genes if subset.iloc[0][g] == 1
    ]
    res_muts = [
        g for g in just_res_genes
        if subset.iloc[0][g] == 1 and re.search(r'_[A-Z]\d+[A-Z]|_STOP|_D', g)
    ]

    present_vir_genes = []
    for g in vir_genes:
        if subset.iloc[0][g] == 1:
            # remove trailing .x / .y
            if g.endswith(('.x', '.y')):
                present_vir_genes.append(g[:-2])
            else:
                present_vir_genes.append(g)

    return (
        standard_res_genes,
        res_muts,
        present_vir_genes,
        species,
        ST,
        origin,
        vir_score,
        res_score
    )

def get_color(score, max_score):
    """Map numeric score to an RGB color from green to red."""
    if score is None or max_score == 0:
        return "rgb(255,255,255)"
    color_val = int((score / max_score) * 255)
    color_val = min(max(color_val, 0), 255)
    return f"rgb({color_val}, {255 - color_val}, 0)"