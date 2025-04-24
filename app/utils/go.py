import pandas as pd
import streamlit as st
try:
    from goatools import obo_parser
    from goatools.go_enrichment import GOEnrichmentStudy
except ImportError:
    raise ImportError("Please install goatools via 'pip install goatools'.")
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict




def load_kp_go_tsv(tsv_path):
    """
    Reads a TSV file downloaded from uniprot for K. Pneumoniae containing the following:
      - 'Gene Names' (e.g., the gene identifier)
      - 'Gene Ontology IDs' (semicolon-separated)

    Returns dict: gene_name -> set of GO IDs
    """
    gene2gos = defaultdict(set)

    df = pd.read_csv(tsv_path, sep='\t')
    if 'Gene Ontology IDs' not in df.columns:
        raise ValueError("Could not find 'Gene Ontology IDs' column in the TSV!")
    if 'Gene Names' not in df.columns:
        raise ValueError("Could not find 'Gene Names' column in the TSV!")

    for _, row in df.iterrows():
        gene_name = str(row['Gene Names']).strip()
        if not gene_name or gene_name.lower() in ['nan', '']:
            continue
        go_ids_str = str(row['Gene Ontology IDs']).strip()
        if go_ids_str.lower() in ['nan', '']:
            continue
        go_ids = [x.strip() for x in go_ids_str.split(';') if x.strip()]
        for goid in go_ids:
            if goid.startswith("GO:"):
                gene2gos[gene_name].add(goid)




    return dict(gene2gos)


def run_go_enrichment(study_genes, gene2gos, obo_file, alpha=0.05):
    """
    Runs GO enrichment with GOATOOLS, using **all genes in 'gene2gos'** as the background.
    This is effectively "all genes in the .tsv" for the background population. The study set are the genes 
    identified by PCA/SHAP analysis. 
    """
    obodag = obo_parser.GODag(obo_file)

    # Use every gene in 'gene2gos' as the background
    population_genes = list(gene2gos.keys())

    goea_obj = GOEnrichmentStudy(
        population_genes,
        gene2gos,
        obodag,
        propagate_counts=False,
        alpha=alpha,
        methods=['fdr_bh']
    )
    results = goea_obj.run_study(study_genes)
    return results

def plot_go_dag(filtered_results, go_obo_file):
    """
    Build and return a matplotlib figure that displays a DAG of the enriched
    GO terms based on their parent-child relationships.
    """
    obodag = obo_parser.GODag(go_obo_file)
    enriched_goids = [r.goterm.id for r in filtered_results]
    
    G = nx.DiGraph()
    for goid in enriched_goids:
        term = obodag[goid]
        G.add_node(goid, label=term.name)

    for goid in enriched_goids:
        term = obodag[goid]
        for parent in term.parents:
            if parent.id in enriched_goids:
                G.add_edge(parent.id, goid)
                
    if len(G.nodes) == 0:
        return None

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        pos = nx.spring_layout(G)
    
    fig, ax = plt.subplots(figsize=(10,8))
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(
        G, pos,
        with_labels=True,
        labels=labels,
        node_color='lightblue',
        font_size=8,
        ax=ax,
        arrows=True
    )
    ax.set_title("GO DAG for Enriched Terms")
    return fig