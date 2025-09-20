import sys
import pandas as pd
import numpy as np
from scipy.stats import entropy
from collections import defaultdict

def parseArgs():
    """Parse command line arguments"""
    import argparse
    import traceback
    try:
        parser = argparse.ArgumentParser(
            description='Compute per base/residue Shannon entropy of a Multiple Sequence Alignment.')
        parser.add_argument('msa_fname',
                            '--msa',
                            action='store',
                            required=True)
        parser.add_argument('data_folder',
                            '--dir',
                            action='store',
                            required=False,
                            default='../../../../PIPS2-UPOs-data/')
        parser.add_argument('msa_subfolder',
                            '-msa_dir',
                            action='store',
                            required=False,
                            default='msa/')
        parser.add_argument('conservation_analysis_subfolder',
                            '-con_dir',
                            action='store',
                            required=False,
                            default='conservation_and_distance/')
    except:
        print("An exception occurred with argument parsing. Check your provided options.")
        traceback.print_exc()

    return parser.parse_args()

def readFasta(msa_fname, mydesc):
    """Read fasta msa_fname of MSA"""
    myid = ""
    myseqs = {}
    with open(msa_fname, 'r') as f:
        for line in f:
            line = line.strip()
            if mydesc == 0 and line.startswith(">"):
                myid = line
                myseqs[myid] = ""
            elif mydesc == 1 and line.startswith(">"):
                myid = line.split()[0][1:]
                myseqs[myid] = ""
            elif mydesc == 2 and line.startswith(">"):
                myid = line.split("|")[1]
                myseqs[myid] = ""
            elif not line.startswith(">"):
                line = ''.join([c for c in line if c.isalpha() or c in "-."])
                myseqs[myid] += line.upper()
    return myseqs

def getAlnLen(myseqs):
    alnlen = 0
    for seq in myseqs.values():
        if len(seq) > alnlen:
            alnlen = len(seq)
    return alnlen

def calculate_shannon_entropy(
        myseqs,
        alnlen,
        AAdict,
):
    AAdict_zeroindexed = {aa:col_idx-1 for aa, col_idx in AAdict.items()}
    AAcount = np.max(np.array(list(AAdict.values())))
    probs = np.zeros((alnlen, AAcount)) # [[0 for _ in range(7)] for _ in range(alnlen + 2)]
    numseqs = len(myseqs)
    # count the occurrences of each amino acid in each position of the alignment
    for seq in myseqs.values():
        for i in range(alnlen):
            aa = seq[i]
            if aa in AAdict_zeroindexed:
                j = AAdict_zeroindexed[aa]
                probs[i, j] += 1
    probs /= numseqs
    shantropy = np.zeros((alnlen,))
    for i in range(alnlen):
        probs_i = probs[i,:]
        probs_i_nonzero = probs_i[probs_i>0]
        shantropy[i] = np.sum(probs_i_nonzero*np.log(probs_i_nonzero) / np.log(2))
    shantropy += np.log(AAcount)/np.log(2)
    return shantropy, probs

def alfa2cons(msa_fpath, output_fpath, save_csv=False):
    myseqs = readFasta(msa_fpath, 0)
    alnlen = getAlnLen(myseqs)
    aaList = list("ACDEFGHIKLMNPQRSTVWY")
    AAdict_shanID = {aa: i+1 for i, aa in enumerate(aaList)}
    AAdict_shanMS = {"A": 1, "C": 1, "V": 1, "I": 1, "L": 1, "M": 1,
                     "F": 2, "H": 2, "W": 2, "Y": 2,
                     "N": 3, "Q": 3, "S": 3, "T": 3,
                     "K": 4, "R": 4,
                     "D": 5, "E": 5,
                     "G": 6, "P": 6,
                     }
    shanMS_AA_groups = {
        1: 'ACILMV',
        2: 'EFHY',
        3: 'NQST',
        4: 'KR',
        5: 'DE',
        6: 'GP'
    }
    shanID, probs_ID = calculate_shannon_entropy(myseqs, alnlen, AAdict_shanID)
    shanMS, probs_MS = calculate_shannon_entropy(myseqs, alnlen, AAdict_shanMS)
    df = pd.DataFrame({
        'Position':list(np.arange(len(shanID))+1),
        'shanID': shanID,
        'shanMS': shanMS,
    })
    # concatenate probsID df
    probsID_df = pd.DataFrame(probs_ID, columns=[aa+'-ID' for aa in aaList])
    df = pd.concat([df,probsID_df], axis=1)
    # concatenate probsMS df
    probsMS_df = pd.DataFrame(probs_MS, columns=[aaGrp+'-MS' for aaGrp in shanMS_AA_groups.values()])
    df = pd.concat([df,probsMS_df], axis=1)
    # save full dataframe
    if save_csv:
        df.to_csv(output_fpath, index=None)
    return df

def main(args=None):
    # Parse arguments
    if args is None:
        args = parseArgs()
        data_folder = args.data_folder
        msa_subfolder = args.msa_subfolder
        conservation_analysis_subfolder = args.conservation_analysis_subfolder
        msa_fname = args.msa_fname
        # derive input & out put fpaths
        msa_fpath = data_folder + msa_subfolder + msa_fname
        output_dir = data_folder + conservation_analysis_subfolder
        output_fpath = output_dir+msa_fname.split('.')[0]+'_ShanEntropy-python.csv'
    else:
        data_folder = args['data_folder']
        msa_subfolder = args['msa_subfolder']
        conservation_analysis_subfolder = args['conservation_analysis_subfolder']
        msa_fname = args['msa_fname']
        # derive input & out put fpaths
        msa_fpath = data_folder + msa_subfolder + msa_fname
        output_dir = data_folder + conservation_analysis_subfolder
        output_fpath = output_dir+msa_fname.split('.')[0]+'_ShanEntropy-python.csv'

    df = alfa2cons(msa_fpath, output_fpath)
    return df

if __name__ == "__main__":
    args = {
        'data_folder': '../../../PIPS2-UPOs-data/',
        'msa_subfolder': 'msa/',
        'conservation_analysis_subfolder': 'conservation_and_distance/',
        'msa_fname': 'UPO_aligned_clustalo',
    }
    df = main(args)