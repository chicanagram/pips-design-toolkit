from pathlib import Path
import numpy as np
import torch
import esm
from scipy.special import softmax
from utils.variables import aaList_with_X, data_folder, subfolders
from utils.utils import fetch_sequences_from_fasta, compose_prob_entropy_PPPL_outputs

# code adapted from
# https://www.kaggle.com/code/mnicatavares/enzyme-mutation-effect-prediction-using-esm2
def get_device():
    if torch.cuda.is_available():
        device = 'cuda'
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    else:
        device = 'cpu'
    print('Device:', device)
    return device

class ESM2:
    def __init__(self,
                 data_fbase,
                 data_folder,
                 embeddings_subfolder='protein_embeddings/',
                 conservation_analysis_subfolder='conservation_and_distance/',
                 model_name='esm2_t36_3B_UR50D', # 'esm2_t33_650M_UR50D',
                 model_type='esm2',
                 remove_gaps=True
                 ):
        # model details
        self.model_type = model_type
        self.model_name = model_name
        # get device
        self.device = torch.device(get_device())
        # Load ESM-2 model
        if model_name=='esm2_t33_650M_UR50D':
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif model_name=='esm2_t36_3B_UR50D':
            self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        # disables dropout for deterministic results
        self.model.eval()
        self.model.to(self.device)
        print('Loaded model.')

        # set folders
        self.data_folder = data_folder
        self.data_fbase = data_fbase
        self.embeddings_subfolder = embeddings_subfolder
        self.conservation_analysis_subfolder = conservation_analysis_subfolder

        # other settings
        self.remove_gaps = remove_gaps

    def tokenize_input_sequences(self, seq_labels, seqs):
        # get batch tokens
        if self.remove_gaps:
            input_data = [(seq_label, seq.replace('-','')) for seq_label, seq in zip(seq_labels, seqs)]
        else:
            input_data = [(seq_label, seq) for seq_label, seq in zip(seq_labels, seqs)]
        batch_labels, batch_seqs, batch_tokens = self.batch_converter(input_data)
        # Move batch_tokens to the same device as the model
        batch_tokens = batch_tokens.to(self.device)
        return batch_labels, batch_seqs, batch_tokens

    def forward_prediction(self, batch_tokens, repr_layers):
        with torch.no_grad():
            output_results = self.model(batch_tokens, repr_layers=repr_layers)
        logits_batch = [output_results['logits'].detach().cpu().numpy()[i,:,:] for i in range(len(self.batch_seqs))]
        representations_batch = output_results["representations"]
        return logits_batch, representations_batch

    def get_mutation_logits_probs(self, logits_batch):
        # get logits
        probs_matrix_list = []
        pppl_list = []
        for logits, seq in zip(logits_batch, self.batch_seqs):
            # apply softmax to calculate probabilities
            probs = softmax(logits, axis=-1)
            # Initialize matrix for the probabilities
            seq_len = len(seq)
            probs_matrix = np.zeros((len(aaList_with_X), seq_len))
            # Populate the matrix with probabilities
            for i, token in enumerate(aaList_with_X):
                if token in self.alphabet.tok_to_idx:
                    index = self.alphabet.tok_to_idx[token]
                    probs_matrix[i] = probs[1:seq_len+1, index]
            probs_matrix /= np.sum(probs_matrix, axis=0)
            probs_matrix_list.append(probs_matrix)
        return probs_matrix_list

    def get_embeddings(self, representations_batch, get_full_representation=True, get_mean_representation=True, save_torch_embeddings=True):

        # get representations of each layer
        representations = {layer: layer_embeddings.detach().to('cpu') for layer, layer_embeddings in representations_batch.items()}

        # iterate through sequences in batch
        for i, (seq_label, seq, mutations_seq) in enumerate(zip(self.batch_labels, self.batch_seqs, self.batch_mutations)):
            seq_len = len(seq)
            if isinstance(mutations_seq, str):
                seq_name_wmut = seq_label+'_'+mutations_seq
            else:
                seq_name_wmut = seq_label
            torch_emb_fpath = Path(self.embeddings_dir) / f"{seq_name_wmut}_{self.model_type}-{self.layer_str}.pt"
            result_torch = {'entry_id': seq_name_wmut, 'full_representations':{}, 'mean_representations':{}}
            # iterate through embedding layers to get
            for layer, layer_embeddings in representations.items():
                if get_full_representation:
                    result_torch['full_representations'][layer] = layer_embeddings[i, 1:len(seq)+1].clone()
                if get_mean_representation:
                    result_torch["mean_representations"][layer] = layer_embeddings[i, 1:len(seq)+1].mean(0).clone()
            # save torch embeddings
            if save_torch_embeddings:
                print('Saved torch embeddings:', torch_emb_fpath)
                torch.save(result_torch, torch_emb_fpath)

    def run_prediction_pipeline(self,
                                sequence_labels,
                                sequence_strs,
                                mutations=None,
                                tokens_per_batch=20, # 4096 #1
                                get_mutation_logits_probs='mask',
                                get_residue_embeddings=True,
                                get_sequence_embeddings=True,
                                repr_layers=[33],
                                save_torch_embeddings=True,
                                print_time_taken=False,
                                msa_fpath=None
                                ):

        # settings
        self.layer_str = '-'.join([str(l) for l in repr_layers])
        self.print_time_taken = print_time_taken
        # get directories
        self.embeddings_dir = self.data_folder + self.embeddings_subfolder + self.data_fbase + '/' + self.model_type + '-' + self.layer_str + '/'
        print('Embeddings dir:', self.embeddings_dir)
        self.conservation_analysis_dir = self.data_folder + self.conservation_analysis_subfolder + self.data_fbase + '/' + self.model_type + '-' + self.layer_str + '/'
        print('Conservation analysis dir:', self.conservation_analysis_dir)

        # process batches of sequences
        num_batches = int(np.ceil(len(sequence_strs)/tokens_per_batch))
        for batch_idx in range(num_batches):
            print(f'Processing batch {batch_idx+1}/{num_batches}...')
            # get sequences in batch
            seq_idx_start = batch_idx*tokens_per_batch
            seq_idx_end = min((batch_idx+1)*tokens_per_batch, len(sequence_strs))
            mutations_batch = mutations[seq_idx_start:seq_idx_end]
            seq_labels_batch = sequence_labels[seq_idx_start:seq_idx_end]
            seqs_batch = sequence_strs[seq_idx_start:seq_idx_end]
            self.batch_mutations = mutations[seq_idx_start:seq_idx_end]
            self.batch_labels, self.batch_seqs, batch_tokens = self.tokenize_input_sequences(seq_labels_batch, seqs_batch)

            # pass input tokens to the model
            if get_mutation_logits_probs is not None or get_residue_embeddings or get_sequence_embeddings:
                print('Performing forward pass...')
                logits_batch, representations_batch = self.forward_prediction(batch_tokens, repr_layers=repr_layers)

            # get mutation logits, probabilities, PPPL
            if get_mutation_logits_probs is not None:
                print('Obtaining MutProbs...')
                # create subfolder for storing probabilities
                Path(self.conservation_analysis_dir).mkdir(parents=True, exist_ok=True)
                # get logits and probs
                if get_mutation_logits_probs == 'mask':
                    probs_matrix_list = self.get_mutation_logits_probs(logits_batch)
                # save results to CSV
                df_list = compose_prob_entropy_PPPL_outputs(probs_matrix_list, self.batch_seqs, self.batch_labels,
                                                              out_dir=self.conservation_analysis_dir,
                                                              fname_suffix=f'_{self.model_type}-{self.layer_str}_MutProbs')

            # get residue embeddings
            if get_residue_embeddings or get_sequence_embeddings:
                print('Obtaining embeddings...')
                # create subfolder for storing torch embeddings
                Path(self.embeddings_dir).mkdir(parents=True, exist_ok=True)
                # get embeddings
                self.get_embeddings(representations_batch,
                                   get_full_representation=get_residue_embeddings,
                                   get_mean_representation=get_sequence_embeddings,
                                   save_torch_embeddings=save_torch_embeddings)


if __name__ == "__main__":
    data_fbase =  'lipases' # 'ET096'  # fasta_file.split('.')[0]
    sequences_folder = data_folder + subfolders['sequences']
    embeddings_folder = data_folder + subfolders['protein_embeddings']
    fasta_file = 'RML.fasta' # 'ET096.fasta'  # 'GOh1052_mutagenesis.fasta' #
    fasta_fpath = sequences_folder + data_fbase + '/' + fasta_file
    output_dir = embeddings_folder + data_fbase + '/'
    get_mutation_logits_probs = 'mask'
    mutations = None # [[10, 12]]  #
    print_time_taken = False

    # get sequences
    seqs, seq_names, _ = fetch_sequences_from_fasta(fasta_fpath)

    # initialize class
    esm2 = ESM2(data_fbase, data_folder)
    esm2.run_prediction_pipeline(seq_names, seqs, mutations=mutations,
                                 tokens_per_batch=2,
                                 get_mutation_logits_probs=get_mutation_logits_probs,
                                 get_residue_embeddings=False,
                                 get_sequence_embeddings=False,
                                 repr_layers=[33],
                                 save_torch_embeddings=True,
                                 print_time_taken=print_time_taken
                                 )