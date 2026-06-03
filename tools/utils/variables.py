#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 07:58:44 2024

@author: charmainechia
"""
try:
    import pandas as pd
    pandas_imported = True
except ImportError as e:
    pandas_imported = False
import os
import numpy as np
import platform
from pathlib import Path
opsys = platform.system()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / 'tools'
data_folder = str(PROJECT_ROOT / 'data') + '/'

subfolders = {
    'sequences': 'sequences/',
    'msa': 'msa/',
    'blast': 'blast/',
    'hmm': 'hmm/',
    'conservation_and_distance': 'conservation_and_distance/',
    'aggregation': 'aggregation/',
    'stability': 'stability/',
    'ml_prediction': 'ml_prediction/',
    'ml_prediction_input': 'ml_prediction/Input/',
    'yasara': 'yasara/',
    'pdb': 'pdb/',
    'sce': 'sce/',
    'protein_embeddings': 'protein_embeddings/',
    'expdata': 'expdata/',
    'mutagenesis_proposal': 'mutagenesis_proposal/',
    'generative_design': 'generative_design/',
    'feature_extraction': 'feature_extraction/'
}


aaList = list('ARNDCQEGHILKMFPSTWYV')
aaList_with_X = list('ARNDCQEGHILKMFPSTWYVX')
mapping = {
    'A': 'Ala',
    'H': 'His',
    'Y': 'Tyr',
    'R': 'Arg',
    'T': 'Thr',
    'K': 'Lys',
    'M': 'Met',
    'D': 'Asp',
    'N': 'Asn',
    'C': 'Cys',
    'Q': 'Gln',
    'E': 'Glu',
    'G': 'Gly',
    'I': 'Ile',
    'L': 'Leu',
    'F': 'Phe',
    'P': 'Pro',
    'S': 'Ser',
    'W': 'Trp',
    'V': 'Val'
    }

mapping_inv = {v.upper():k for k,v in mapping.items()}

struct_dist_list = [
    # 'min_distance_to_HEM',
    # 'avg_distance_HEM',
    'min_distance_to_UNK',
    'avg_distance_UNK'
]

variant_proposal_scores_base = [
    'Position', 'mutations', 'num_mutation_sites',
    'min_distance_to_UNK',
    'sift_avg', 'shanID', 'shanMS',
    'Pythia_DDGstability', 'FoldX_DDGstability',
    'avg_PLM_score', 'ESM-2_LLR(-ive)', 'ProtT5_LLR(-ve)', 'PoET_LLR(-ve)', 'PoET2_LLR(-ve)', 'avg_PLM_score_bypos'
]

amino_acid_groups = {
    "np": ["F", "L", "I", "V", "M", "A", "W", "G", "P"],
    "p~": ["Y", "C", "T", "S", "H", "Q", "N"],
    "p-": ["E", "D"],  # Acidic
    "p+": ["K", "R"]   # Basic
}

struct_properties_list = [
    'secondary_structure',
    'relative ASA',
    'phi',
    'psi',
    'NH_O_1_relidx',
    'NH_O_1_energy',
    'O_NH_1_relidx',
    'O_NH_1_energy',
    'NH_O_2_relidx',
    'NH_O_2_energy',
    'O_NH_2_relidx',
    'O_NH_2_energy',
    'min_distance_to_HEM',
    'avg_distance_HEM',
    'min_distance_to_UNK',
    'avg_distance_UNK'
]
