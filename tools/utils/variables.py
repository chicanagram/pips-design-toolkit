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
opsys = platform.system()


data_folder = '../data/'

subfolders = {
    'sequences': 'sequences/',
    'msa': 'msa/',
    'blast': 'blast/',
    'hmm': 'hmm/',
    'conservation_analysis': 'conservation_and_distance/',
    'aggregation': 'aggregation/',
    'stability': 'stability/',
    'ml_prediction': 'ml_prediction/',
    'yasara': 'yasara/',
    'pdb': 'pdb/',
    'sce': 'sce/',
    'protein_embeddings': 'protein_embeddings/',
    'expdata': 'expdata/',
    'mutagenesis_proposal': 'mutagenesis_proposal/',
    'generative_design': 'generative_design'
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
