from feature_extraction.binding import get_yasara_binding_features
from feature_extraction.stability import  get_yasara_foldx_stability_features
from feature_extraction.aggregation import  AggWaltz, AggTango
from utils.variables import data_folder
from utils.utils import fetch_sequences_from_fasta, get_mutations

if __name__ == "__main__":

    # set inputs and parameters
    input_fname = 'GOh1052_mutPos_DomainIII.txt'
    struct_fname = 'S152_1GOG_GOh1001b_postOpt'
    sequence_fname = 'GOh1052'
    input_dir = data_folder + 'feature_extraction/Input/'
    struct_dir = input_dir
    output_dir = data_folder + 'feature_extraction/'
    nrep = 5
    features_to_extract = [
        'binding',
        'stability',
        'aggregation-Waltz',
        'aggregation-Tango'
    ]
    
    # get mutations
    seqs, seq_names, seq_description = fetch_sequences_from_fasta(f'{data_folder}sequences/{sequence_fname}.fasta')
    seq_base = seqs[0]
    seq_name = seq_names[0]
    mutatePos = [aa+str(i+1) for i,aa in enumerate(seq_base)]
    mutations = [None] + get_mutations(mutatePos)

    # get binding ddG and energy features
    if 'binding' in features_to_extract:
        get_yasara_binding_features(
            input_fname,
            struct_fname,
            input_dir,
            struct_dir,
            output_dir,
            nrep=5
        )

    # get stability ddG and energy featrues
    if 'stability' in features_to_extract:
        get_yasara_foldx_stability_features()

    # get base sequence for aggregation calculation
    sequence_fpath = data_folder + 'sequences/' + seq_name + '.fasta'
    seqs, seq_names, seq_description = fetch_sequences_from_fasta(sequence_fpath)
    
    # get Waltz aggregation features
    if 'aggregation-Waltz' in features_to_extract:
        AggWaltz(mutations, seq_base, seq_name, input_dir, output_dir, f'AggregationScore_{sequence_fname}')    

    # get Tango aggregation features
    if 'aggregation-Tango' in features_to_extract:
        AggTango(mutations, seq_base, seq_name, input_dir, output_dir, f'AggregationScore_{sequence_fname}')
