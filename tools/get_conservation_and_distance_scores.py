import os
import numpy as np
import pandas as pd
from utils.utils import fetch_sequences_from_fasta, write_sequence_to_fasta, get_ref_seq_idxs_aa_from_msa, compute_entropy
from utils.variables import data_folder, mapping_inv

def insert_position_col_offset(df, offset, pos_col='RealPos'):
    pos = df[pos_col].to_numpy()
    pos_woffset = pos + offset
    df.insert(0,'Position', pos_woffset)
    return df

def ShanEntropy(msa_fname, output_fname, ref_seq_idxs, ref_seq, position_offset=None):
    from conservation_and_distance.alfa2cons import alfa2cons
    csv = alfa2cons(data_folder+'msa/'+msa_fname+'.fasta', data_folder+'conservation_and_distance/'+output_fname+'.csv', save_csv=False)
    csv_filt = csv[csv['Position'].isin(ref_seq_idxs)].copy()
    ref_seq_parsed = list(ref_seq) if ref_seq is not None else None
    if ref_seq_parsed is not None:
        csv_filt['AA'] = ref_seq_parsed
    csv_filt['RealPos'] = list(np.arange(len(ref_seq_parsed)) + 1)
    csv_filt = csv_filt.rename(columns={'Position': 'PositionMSA'})
    csv_filt = csv_filt[['AA', 'RealPos', 'PositionMSA', 'shanID', 'shanMS']]
    if position_offset is not None:
        csv_filt = insert_position_col_offset(csv_filt, position_offset)
    # write dataframe to file
    output_fpath = data_folder+'conservation_and_distance/'+output_fname+'_shanms.csv'
    csv_filt.to_csv(output_fpath)
    print(f'Saved Shannon Entropy scores to: {output_fpath}')


def SIFT(msa_fname, output_fname, ref_seq_name, position_offset=None):
    from conservation_and_distance import access_sift_webserver
    msa_path = data_folder+'msa/'+msa_fname+'.fasta'
    msa_seqs, msa_names, _ = fetch_sequences_from_fasta(msa_path)

    msa_idx = msa_names.index(ref_seq_name)
    msa_names_rearranged = [msa_names[msa_idx]] + msa_names[:msa_idx] + msa_names[msa_idx+1:]
    msa_seqs_rearranged = [msa_seqs[msa_idx]] + msa_seqs[:msa_idx] + msa_seqs[msa_idx+1:]
    msa_path_rearranged = data_folder+'msa/'+msa_fname + f'_{ref_seq_name}.fasta'
    write_sequence_to_fasta(msa_seqs_rearranged, msa_names_rearranged, msa_fname+f'_{ref_seq_name}', data_folder+'msa/')
    csv = access_sift_webserver.main(
      {'-a': os.path.abspath(msa_path_rearranged),
       '--fname': data_folder+'conservation_and_distance/'+msa_fname+f'_sift.csv'
       }
    )
    # get probabilities pre-normalization
    probs_matrix_norm = np.transpose(csv.iloc[:, -20:].to_numpy()).astype(float)
    csv.iloc[:,-20:] = np.transpose(probs_matrix_norm)
    prob_col = csv['prob'].to_numpy()
    probs_matrix = probs_matrix_norm * prob_col
    # get average sift score for each residue
    probs_mean = np.mean(probs_matrix_norm, axis=0)
    csv.insert(2, 'sift_avg', list(probs_mean))
    # calculate entropy for each position
    ent = compute_entropy(probs_matrix)
    csv.insert(2, 'entropy', list(ent))
    # write dataframe to file
    csv = csv.rename(columns={'pos': 'RealPos', 'wt': 'AA'})
    if position_offset is not None:
        csv = insert_position_col_offset(csv, position_offset)
    output_fpath = data_folder+'conservation_and_distance/'+output_fname+'_sift.csv'
    csv.to_csv(output_fpath)
    os.remove(msa_path_rearranged)
    os.remove(data_folder+'conservation_and_distance/'+msa_fname+f'_sift.csv')
    print(f'Saved SIFT scores to: {output_fpath}')


def DistResSub(pdb_fpath, output_fname, chain_id='A', ligand_resname='UNK', position_offset=None):
    from Bio.PDB import PDBParser, is_aa

    # Parse PDB structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_fpath)
    model = structure[0]

    # Select ligand atoms
    ligand_atoms = []
    model = structure[0]
    for chain in model:
        for residue in chain:
            # select target (ligand, heme, metal ion) atoms only, excluding H
            if residue.get_resname().strip() == ligand_resname:
                ligand_atoms.extend([atom for atom in residue if atom.element != 'H'])

    # Compute distances between ligand atoms and heavy atoms of each residue
    distance_results = []
    for chain in model:
        # match the
        if chain.id == chain_id:
            for residue in chain:
                # select only the protein residues
                if is_aa(residue, standard=True):
                    res_id = (residue.get_resname(), residue.get_id()[1])
                    aa = mapping_inv[res_id[0]]
                    min_distances = []
                    avg_distances = []
                    # iterate through atoms in the protein residue, excluding H atoms
                    for atom in residue:
                        dist_res_atom_to_all_target_atoms = []
                        if atom.element != 'H':  # exclude hydrogen atoms
                            # calculate distances between all residue atoms and target atoms
                            for lig_atom in ligand_atoms:
                                dist_atom2atom = np.linalg.norm(atom.coord - lig_atom.coord)
                                dist_res_atom_to_all_target_atoms.append(dist_atom2atom)
                            avg_dist_res_atom_to_target = np.mean(np.array(dist_res_atom_to_all_target_atoms))
                            avg_distances.append(avg_dist_res_atom_to_target)
                            min_distances.append(min(dist_res_atom_to_all_target_atoms))
                    distance_results.append({'RealPos':res_id[1], 'aa': aa, f'min_distance_to_{ligand_resname}':min(min_distances), f'avg_distance_{ligand_resname}':min(avg_distances)})

    # create dataframe and add offset residue numbering
    distance_results = pd.DataFrame(distance_results)
    if position_offset is not None:
        distance_results = insert_position_col_offset(distance_results, position_offset)

    # save results
    output_fpath = data_folder+'conservation_and_distance/'+output_fname+'_distance.csv'
    distance_results.to_csv(output_fpath)
    print(f'Saved distance scores to: {output_fpath}')
    return distance_results
    

def get_conservation_and_distance_scores(
        msa_fname,
        ref_seq_name,
        ref_seq,
        output_fname,
        pdb_fname,
        chain_id='A',
        ligand_resname='UNK',
        position_offset=0
):
    if ref_seq is None:
        _, ref_seq, ref_seq_idxs = get_ref_seq_idxs_aa_from_msa(data_folder+'msa/'+msa_fname+'.fasta', [ref_seq_name])

    # run ShanEntropy
    ShanEntropy(msa_fname, output_fname, ref_seq_idxs[0], ref_seq[0], position_offset)
    # run SIFT
    SIFT(msa_fname, output_fname, ref_seq_name, position_offset)
    # run YASARA distance calculation
    DistResSub(data_folder+'pdb/'+pdb_fname+'.pdb', output_fname, chain_id=chain_id, ligand_resname=ligand_resname, position_offset=position_offset)

if __name__ == "__main__":

    # set inputs
    msa_fname = 'GOh1052_msa'
    ref_seq_name = 'FGGALOX'
    ref_seq = None
    output_fname = 'GOh1052'
    pdb_fname = 'S152_preOpt_2EIE_GOh1001b_preOpt'
    chain_id = 'A'
    ligand_resname = 'UNK'
    position_offset = 23 # 0

    # run conservation analysis
    get_conservation_and_distance_scores(msa_fname, ref_seq_name, ref_seq, output_fname, pdb_fname, chain_id, ligand_resname, position_offset)