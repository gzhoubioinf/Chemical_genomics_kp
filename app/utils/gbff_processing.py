import re
import io
import streamlit as st
from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict

def find_all_occurrences(sequence, subseq):
    """
    Generator to find all occurrences of subseq in sequence.
    Yields the starting index of each occurrence.
    """
    start = 0
    while True:
        pos = sequence.find(subseq, start)
        if pos == -1:
            break
        yield pos
        start = pos + 1

def get_overlapping_genes(features, start_pos, end_pos):
    """
    Return a list of gene-locus_tag pairs overlapping with [start_pos, end_pos].
    Coordinates are 0-based in this function.
    """
    hits = []
    for feat in features:
        if feat.type in ["gene", "CDS"]:
            f_start = int(feat.location.start)
            f_end   = int(feat.location.end)
            # Overlap if they share any region
            if not (f_end < start_pos or f_start > end_pos):
                locus_tag = feat.qualifiers.get("locus_tag", ["?"])[0]
                gene = feat.qualifiers.get("gene", ["?"])[0]
                product = feat.qualifiers.get("product", ["?"])[0]
                hits.append((feat.type, f_start, f_end, locus_tag, gene, product))
    return hits

def extract_genes(overlap_hits):
    """
    Extract gene information from overlap_hits.
    Returns a list of gene descriptions.
    """
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
    Process a single .gbff file and return a list of tuples:
    (PC_number, Unitig, Gene)
    """
    matches = []
    try:
        all_ref_records = list(SeqIO.parse(gbff_path, "genbank"))
    except Exception as e:
        st.warning(f"  [WARNING] Could not parse {gbff_path} as GenBank: {e}")
        return matches

    if not all_ref_records:
        st.warning(f"  [WARNING] No GenBank records in {gbff_path}, skipping.")
        return matches

    for rec in all_ref_records:
        rec_id = rec.id
        forward_seq_str = str(rec.seq).upper()
        revcomp_seq_str = str(rec.seq.reverse_complement()).upper()
        rec_feats = rec.features

        for pc_idx, unitigs in target_unitigs.items():
            for utg in unitigs:
                utg_revcomp = str(Seq(utg).reverse_complement())
                # (A) forward ref, forward unitig
                for pos in find_all_occurrences(forward_seq_str, utg):
                    overlap_hits = get_overlapping_genes(rec_feats, pos, pos + len(utg) - 1)
                    genes = extract_genes(overlap_hits)
                    for gene in genes:
                        matches.append((pc_idx + 1, utg, gene))

                # (B) forward ref, revcomp unitig
                for pos in find_all_occurrences(forward_seq_str, utg_revcomp):
                    overlap_hits = get_overlapping_genes(rec_feats, pos, pos + len(utg_revcomp) - 1)
                    genes = extract_genes(overlap_hits)
                    for gene in genes:
                        matches.append((pc_idx + 1, utg, gene))

                # (C) reverse-comp ref, forward unitig
                for pos in find_all_occurrences(revcomp_seq_str, utg):
                    L = len(forward_seq_str)
                    fwd_start = L - (pos + len(utg))
                    fwd_end   = L - pos - 1
                    overlap_hits = get_overlapping_genes(rec_feats, fwd_start, fwd_end)
                    genes = extract_genes(overlap_hits)
                    for gene in genes:
                        matches.append((pc_idx + 1, utg, gene))

                # (D) reverse-comp ref, revcomp unitig
                for pos in find_all_occurrences(revcomp_seq_str, utg_revcomp):
                    L = len(forward_seq_str)
                    fwd_start = L - (pos + len(utg_revcomp))
                    fwd_end   = L - pos - 1
                    overlap_hits = get_overlapping_genes(rec_feats, fwd_start, fwd_end)
                    genes = extract_genes(overlap_hits)
                    for gene in genes:
                        matches.append((pc_idx + 1, utg, gene))
    return matches