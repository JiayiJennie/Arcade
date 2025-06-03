"""
This script fetches the steering vectors for sequences.
"""
from Bio import SeqIO
import sys
import torch
from transformers import BertForPreTraining
import pandas as pd
import numpy as np
from tqdm import tqdm
from peft import PeftConfig, PeftModel
from finetune_token_cls import load_model
import os

sys.path.append("/mnt/disk90/user/jiayili/CodonBERT/benchmarks/utils")
from tokenizer import mytok, get_tokenizer

model_path = '../chckpoints'

def encode_sequences(sequences, tokenizer, max_length=1024):
    tokenized_sequences = [" ".join(mytok(seq, 3, 3)) for seq in sequences]
    input_ids = tokenizer(tokenized_sequences, padding="max_length", truncation=True, max_length=max_length, return_tensors='pt').input_ids
    return input_ids

def fetch_steering_vector(data_type, model, tokenizer, device, sequences=None, percent=None, min_seq_len=300, max_seq_len=3072-6, top_cai_threshold=0.4, \
                          bottom_cai_threshold=0.2, lambda_=None, high_fa_path=None, low_fa_path=None):
    """
    Visualize sequence embeddings colored by MFE or CAI values.
    
    Args:
        data_type (str): Type of data to analyze - 'mfe' or 'cai'
    """
    # Read CSV file
    if data_type.lower() == 'mfe' or data_type.lower() == 'cai' or data_type.lower() == 'mfe_cai':
        assert sequences is not None and mfe_cai_table_path is not None
        sequences = load_data(data_path)
        print(f"Loaded {len(sequences)} sequences") 
        df = pd.read_csv(mfe_cai_table_path)
        df['MFE_normalized'] = df['MFE'] / df['Length']
        df['log_CAI'] = np.log(df['CAI'])
    else:
        pass

    if data_type.lower() == 'mfe':
        value_col = 'MFE_normalized'
    elif data_type.lower() == 'cai':
        value_col = 'CAI'
    elif data_type.lower() == 'mfe_cai':
        value_col = 'score'  #TODO: change this to the correct column name
        if lambda_ is None:
            raise ValueError("Must provide 'lambda_' for 'mfe_cai' data type.")
        # df['score'] = - df['MFE_normalized'] + lambda_ * df['CAI']
        df['score'] = - df['MFE_normalized'] + lambda_ * df['log_CAI']
    elif data_type.lower() == 'fasta':
        assert high_fa_path is not None and low_fa_path is not None
        high_fa = SeqIO.parse(high_fa_path, "fasta")
        low_fa = SeqIO.parse(low_fa_path, "fasta")
    else:
        raise ValueError("data_type must be 'mfe' or 'cai' or 'mfe_cai' or 'fasta'.")
    

    # Select sequences based on percentile or CAI threshold
    if data_type.lower() == 'fasta':
        top_seqs = [str(record.seq) for record in high_fa]
        low_seqs = [str(record.seq) for record in low_fa]
    elif percent is not None:
        # Apply sequence length filter
        df['seq_length'] = df['ID'].apply(lambda x: len(sequences[x]))  # Assuming sequences dict is available
        df = df[(df['seq_length'] >= min_seq_len) & (df['seq_length'] <= max_seq_len)]
        # Get percentile-based selection
        n = int(len(df) * percent)
        top_seqs = df.nlargest(n, value_col)
        bottom_seqs = df.nsmallest(n, value_col)
        
        top_seqs = top_seqs['ID'].values
        top_seqs = [sequences[seq] for seq in top_seqs]

        low_seqs = bottom_seqs['ID'].values
        low_seqs = [sequences[seq] for seq in low_seqs]
    # elif top_cai_threshold is not None and data_type.lower() == 'cai':
    #     # Select based on absolute CAI threshold
    #     top_seqs = df[df['CAI'] >= top_cai_threshold]
    #     bottom_seqs = df[df['CAI'] <= bottom_cai_threshold]
    #     print(f"Number of top sequences: {len(top_seqs)}")
    #     print(f"Number of bottom sequences: {len(bottom_seqs)}")
    else:
        raise ValueError("Must provide either 'percent' or 'cai_threshold' (only for 'cai').")

    # Set up model
    model.eval()

    # make equences into a tensor with shape (n, max_len, emb_size)
    max_length = 1024
    top_seqs = encode_sequences(top_seqs, tokenizer, max_length).to(device)
    low_seqs = encode_sequences(low_seqs, tokenizer, max_length).to(device)
    assert top_seqs.size(1) == low_seqs.size(1)

    # get output hiddenstates for each encoder layer
    print("Computing steering vectors...")
    with torch.no_grad():
        batch_size = 32
        n = top_seqs.size(0)
        top_steering_vectors = []
        for i in tqdm(range(0, n, batch_size), desc="processing top sequences"):
            # outputs = model(top_seqs[i:i+batch_size], labels=top_seqs[i:i+batch_size], output_hidden_states=True)
            outputs = model(top_seqs[i:i+batch_size], output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            encoder_layer_outputs = hidden_states[1:]
            top_steering_vectors.append(torch.stack(encoder_layer_outputs, dim=0).cpu())
        top_steering_vectors = torch.cat(top_steering_vectors, dim=1)
        top_steering_vectors = top_steering_vectors.numpy()  # shape (n_layers, n, max_len, emb_size)
        top_steering_vectors = top_steering_vectors.mean(axis=1, keepdims=True)     

        low_steering_vectors = []
        n = low_seqs.size(0)
        for i in tqdm(range(0, n, batch_size), desc="processing low sequences"):
            # outputs = model(low_seqs[i:i+batch_size], labels=low_seqs[i:i+batch_size], output_hidden_states=True)
            outputs = model(low_seqs[i:i+batch_size], output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            encoder_layer_outputs = hidden_states[1:]
            low_steering_vectors.append(torch.stack(encoder_layer_outputs, dim=0).cpu())
        low_steering_vectors = torch.cat(low_steering_vectors, dim=1)
        low_steering_vectors = low_steering_vectors.numpy()  # shape (n_layers, n, max_len, emb_size)
        low_steering_vectors = low_steering_vectors.mean(axis=1, keepdims=True)

        steering_vectors = top_steering_vectors - low_steering_vectors

    if data_type.lower() == 'mfe':
        steering_vectors = - steering_vectors

    return steering_vectors

def load_data(data_path):
    sequences = {}

    with open(data_path) as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            sequences[record.id] = str(record.seq)
    
    return sequences

if __name__ == "__main__":
    lambda_ = 0
    percent = 0.01
    high_fa_path = '/mnt/disk90/user/jiayili/codon_optimization_steering/data/for_steering/high_upa.fa'
    low_fa_path = '/mnt/disk90/user/jiayili/codon_optimization_steering/data/for_steering/low_upa.fa'
    save_name = 'upa'
    data_type = 'fasta'
    save_dir = '/mnt/disk90/user/jiayili/codon_optimization_steering/data'

    tokenizer = get_tokenizer()
    device = torch.device("cuda:0")
    config = PeftConfig.from_pretrained(model_path)
    model, bert_tokenizer_fast, _, _ = load_model(model_path=config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, model_path)
    model=model.merge_and_unload()
    model.to(device)

    steering_vectors = fetch_steering_vector(data_type, model, bert_tokenizer_fast, device, high_fa_path=high_fa_path, low_fa_path=low_fa_path)

    print(steering_vectors.shape) # n_layers, 1, max_len, emb_size
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'steering_vectors_'+save_name+'.npy'), steering_vectors)
    print("Saved steering vectors to", os.path.join(save_dir, 'steering_vectors_'+save_name+'.npy'))