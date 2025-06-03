"""
This script is used to steer the freezed model towards a specific optimization goal given input sequences.
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch

from utils.load_utils import tokenize_and_align_labels, load_model, load_data
from transformers import Trainer, TrainingArguments
from utils.mapping import aa_to_codon, codon_to_aa
from safetensors import safe_open
from functools import partial
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import argparse
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from peft import PeftModel, PeftConfig

import sys
sys.path.append("/mnt/disk90/user/jiayili/calculator")
from calculator import mfe_calculation, cai_calculation, cpb_calculation
from calculator import CpG_density as cpg_calculation
from calculator import UpA_density as upa_calculation

sys.path.append("/mnt/disk90/user/jiayili/CodonBERT/benchmarks/utils")
from tokenizer import get_tokenizer, mytok

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default='/mnt/disk90/user/jiayili/codon_optimization_steering/saved_model/2025-03-24-23-38-02/best_model', help="Path to the pretrained model directory.")
parser.add_argument("--data_path", type=str, default='/mnt/disk90/user/jiayili/GENCODE/gencode.v47.pc_transcripts_cds_3072_test.fa', help="Path to the FASTA data.")
parser.add_argument("--cai_mfe_table_path", type=str, default='/mnt/disk90/user/jiayili/GENCODE/gencode.v47.pc_transcripts_cds_3072_test.csv', help="Path to the table of CAI and MFE values.")
parser.add_argument("--mfe_steering_vectors_path", type=str, default='/mnt/disk90/user/jiayili/codon_optimization_steering/data/steering_vectors_mfe.npy', help="Path to the MFE steering vectors .npy file.")
parser.add_argument("--cai_steering_vectors_path", type=str, default='/mnt/disk90/user/jiayili/codon_optimization_steering/data/steering_vectors_cai.npy', help="Path to the CAI steering vectors .npy file.")
parser.add_argument("--gc_steering_vectors_path", type=str, default='/mnt/disk90/user/jiayili/codon_optimization_steering/data/steering_vectors_gc.npy', help="Path to the GC content steering vectors .npy file.")
parser.add_argument("--cpb_steering_vectors_path", type=str, default='/mnt/disk90/user/jiayili/codon_optimization_steering/data/steering_vectors_cpb.npy', help="Path to the CPB steering vectors .npy file.")
parser.add_argument("--liver_steering_vectors_path", type=str, default='/mnt/disk90/user/jiayili/codon_optimization_steering/data/steering_vectors_liver_normal.npy', help="Path to the liver steering vectors .npy file.")
parser.add_argument("--hepatocellular_vectors_path", type=str, default='/mnt/disk90/user/jiayili/codon_optimization_steering/data/steering_vectors_hepatocellular.npy', help="Path to the Hepatocellular vector .npy file.")
parser.add_argument("--mRFP_steering_vectors_path", type=str, default='/mnt/disk90/user/jiayili/codon_optimization_steering/data/steering_vectors_mRFP.npy', help="Path to the mRFP steering vectors .npy file.")
parser.add_argument("--cpg_steering_vectors_path", type=str, default='/mnt/disk90/user/jiayili/codon_optimization_steering/data/steering_vectors_cpg.npy', help="Path to the CpG steering vectors .npy file.")
parser.add_argument("--upa_steering_vectors_path", type=str, default='/mnt/disk90/user/jiayili/codon_optimization_steering/data/steering_vectors_upa.npy', help="Path to the UpA steering vectors .npy file.")
parser.add_argument("--steering_strength", type=float, default=1, help="Scaling factor for the steering vectors.")
parser.add_argument("--lambda_mfe", type=float, default=0, help="Weight for the MFE steering vectors.")
parser.add_argument("--lambda_cai", type=float, default=0, help="Weight for the CAI steering vectors.")
parser.add_argument("--lambda_gc", type=float, default=0, help="Weight for the GC content steering vectors.")
parser.add_argument("--lambda_cpb", type=float, default=0, help="Weight for the CPB steering vectors.")
parser.add_argument("--lambda_liver", type=float, default=0, help="Weight for the liver steering vectors.")
parser.add_argument("--lambda_hepa", type=float, default=0, help="Weight for the Hepatocellular steering vectors.")
parser.add_argument("--lambda_mRFP", type=float, default=0, help="Weight for the mRFP steering vectors.")
parser.add_argument("--lambda_cpg", type=float, default=0, help="Weight for the CpG steering vectors.")
parser.add_argument("--lambda_upa", type=float, default=0, help="Weight for the UpA steering vectors.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
parser.add_argument("--single_example", action="store_true", help="Whether to evaluate a single example.")
parser.add_argument("--input_example", type=str, default=None, help="Input example to evaluate.")
parser.add_argument("--save_dir", type=str, default="/mnt/disk90/user/jiayili/codon_optimization_steering/results/", help="Directory to save the optimized sequences.")
parser.add_argument("--save_file_name", type=str, default="results")
args = parser.parse_args()

model_path = args.model_path
data_path = args.data_path
cai_mfe_table_path = args.cai_mfe_table_path
MFE_steering_vectors_path = args.mfe_steering_vectors_path
CAI_steering_vectors_path = args.cai_steering_vectors_path
GC_steering_vectors_path = args.gc_steering_vectors_path
CPB_steering_vectors_path = args.cpb_steering_vectors_path
liver_steering_vectors_path = args.liver_steering_vectors_path
hepatocellular_vectors_path = args.hepatocellular_vectors_path
mRFP_steering_vectors_path = args.mRFP_steering_vectors_path

steering_strength = args.steering_strength
lambda1 = args.lambda_mfe
lambda2 = args.lambda_cai
lambda_gc = args.lambda_gc
lambda_cpb = args.lambda_cpb
lambda_liver = args.lambda_liver
lambda_hepatocellular = args.lambda_hepa
lambda_mRFP = args.lambda_mRFP
lambda_cpg = args.lambda_cpg
lambda_upa = args.lambda_upa
print("Lambda for MFE:", lambda1)
print("Lambda for CAI:", lambda2)
print("Lambda for GC content:", lambda_gc)
print("Lambda for CPB:", lambda_cpb)
print("Lambda for liver:", lambda_liver)
print("Lambda for Hepatocellular:", lambda_hepatocellular)
print("Lambda for mRFP:", lambda_mRFP)
print("Lambda for CpG:", lambda_cpg)
print("Lambda for UpA:", lambda_upa)
batch_size = args.batch_size
single_example = args.single_example
print("Single example: ", single_example)
input_example = args.input_example
print("Input example: ", input_example)
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

start = time.time()


# load model and tokenizer
config = PeftConfig.from_pretrained(model_path)
model, bert_tokenizer_fast, label2id, id2label = load_model(model_path=config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, model_path)
model=model.merge_and_unload()
model.to(device)
model.eval()

tokenize_and_align_labels_fn = partial(tokenize_and_align_labels, bert_tokenizer_fast=bert_tokenizer_fast, label2id=label2id)

def build_dataset(data_path):
    """
    Build dataset from fasta file and split it into training, validation and test sets.
    Args:
        data_path: str, pathc to the fasta file.
    Returns:
        Dataset object.
    """
    X, headers = load_data(data_path, save_header=True)
    if single_example:
        print("Single example mode.")
        if input_example:
            lst_tok = mytok(input_example, 3, 3)
            tok_seq = " ".join(lst_tok)
            return Dataset.from_list([{"seq": tok_seq}]), None
        else:
            return Dataset.from_list([{"seq": seq} for seq in X][:1]), headers[:1]
    return Dataset.from_list([{"seq": seq} for seq in X]), headers

# load data
ds_test, headers = build_dataset(data_path)
tokenized_test = ds_test.map(tokenize_and_align_labels_fn, batched=True)

# load steering vectors
mfe_steering_vectors = torch.tensor(np.load(MFE_steering_vectors_path)).to(device)
cai_steering_vectors = torch.tensor(np.load(CAI_steering_vectors_path)).to(device)
gc_steering_vectors = torch.tensor(np.load(GC_steering_vectors_path)).to(device)
cpb_steering_vectors = torch.tensor(np.load(CPB_steering_vectors_path)).to(device)
liver_steering_vectors = torch.tensor(np.load(liver_steering_vectors_path)).to(device)
mRFP_steering_vectors = torch.tensor(np.load(mRFP_steering_vectors_path)).to(device)
hepatocellular_steering_vectors = torch.tensor(np.load(hepatocellular_vectors_path)).to(device)
cpg_steering_vectors = torch.tensor(np.load(args.cpg_steering_vectors_path)).to(device)
upa_steering_vectors = torch.tensor(np.load(args.upa_steering_vectors_path)).to(device)
steering_vectors = - lambda1 * mfe_steering_vectors + lambda2 * cai_steering_vectors \
    + lambda_gc * gc_steering_vectors + lambda_cpb * cpb_steering_vectors \
    + lambda_liver * liver_steering_vectors + lambda_hepatocellular * hepatocellular_steering_vectors\
    + lambda_mRFP * mRFP_steering_vectors\
    + lambda_cpg * cpg_steering_vectors + lambda_upa * upa_steering_vectors


def create_synonymous_mask(logits, labels, label2id=label2id, id2label=id2label):
    """
    Creates a mask for MLM that allows only synonymous codons for each token.

    Args:
        logits (torch.Tensor): The logits output from the model.
        labels (torch.Tensor): The labels for the input sequences.
        label2id (dict): Mapping from token labels to IDs.
        id2label (dict): Mapping from token IDs to labels.

    Returns:
        mask (torch.Tensor): A mask tensor of the same shape as logits.
                             For each non-special token, positions corresponding to
                             non-synonymous codons are set to -inf; allowed positions are 0.
    """
    batch_size, seq_length, vocab_size = logits.shape

    mask = torch.full_like(logits, float('-inf')).to(logits.device)

    for i in range(batch_size):
        for j in range(seq_length):
            token_id = labels[i, j].item()
            if token_id == -100:
                mask[i, j, :] = 0
                continue

            codon = id2label.get(token_id, None)
            if codon is None or codon not in codon_to_aa:
                mask[i, j, :] = 0
                continue

            aa = codon_to_aa[codon]
            assert aa in aa_to_codon, f"AA {aa} not found in aa_to_codon mapping."
            allowed_codons = aa_to_codon[aa]
            allowed_token_ids = [label2id[c] for c in allowed_codons if c in label2id]
            mask[i, j, allowed_token_ids] = 0

    return mask

def custom_compute_metrics(eval_preds):
    logits, labels = eval_preds
    # logits = torch.tensor(logits).to(device)
    # labels = torch.tensor(labels).to(device)
    logits = torch.tensor(logits).cpu()
    labels = torch.tensor(labels).cpu()

    synonymous_mask = create_synonymous_mask(logits, labels)

    logits += synonymous_mask
    final_predictions = torch.argmax(logits, dim=-1).cpu().numpy()

    return {"predictions": final_predictions}

def hook(module, input, output, steering_vector):
    # Extract hidden state from the output tuple if needed.
    if steering_vector is None:
        return output
    hidden_state = output[0] if isinstance(output, tuple) else output
    steering_vector = steering_vector.to(hidden_state.dtype)
    # Return modified output.
    new_hidden_state = hidden_state + steering_vector
    if isinstance(output, tuple):
        return (new_hidden_state,) + output[1:]
    return new_hidden_state

if steering_vectors is not None:
    print("Applying steering vectors...")
    handles = []
    for i, layer in enumerate(model.bert.encoder.layer):
        handle = layer.register_forward_hook(partial(hook, steering_vector=steering_vectors[i]*steering_strength))           
        handles.append(handle)

training_args = TrainingArguments(
    output_dir="./results",   
    per_device_eval_batch_size=batch_size, 
    do_predict=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_test,
    tokenizer=bert_tokenizer_fast,
    compute_metrics=custom_compute_metrics,
)

predictions = trainer.predict(tokenized_test, metric_key_prefix="test")
logits = torch.tensor(predictions.predictions) # shape of logits: (n, max_len, vocab_size)
print("shape of logits:", predictions.predictions.shape)
pred_labels = predictions.metrics["test_predictions"]


if single_example:
    table = '/mnt/disk90/user/jiayili/calculator/data/codon_adaptiveness.json'
    mtable = '/mnt/disk90/user/jiayili/calculator/data/max_codon_adaptiveness.json'
    cps_table_path = '/mnt/disk90/user/jiayili/calculator/CPS/codon_pair_scores_homo_sapiens.csv'

    seq_index = 0  #2,3,4
    input = tokenized_test["seq"][seq_index].split(" ")
    seq = "".join(input)
    mfe, structure = mfe_calculation(seq)
    cai = cai_calculation(seq, table, mtable)
    gc_content = (seq.count('G') + seq.count('C')) / len(seq)
    # cpb = cpb_calculation(seq, cps_table_path)
    cpg = cpg_calculation(seq)
    upa = upa_calculation(seq)
    print("Input:")
    print("".join(input)[:50])
    print("MFE:", mfe)
    print("CAI:", cai)
    print("GC content:", gc_content)
    # print("CPB: ", cpb)
    print("CpG density", cpg)
    print("UpA density", upa)

    
    p = pred_labels[seq_index][1:len(input)+1]
    p = [id2label[i] for i in p]
    p_str = "".join(p)
    mfe, structure = mfe_calculation(p_str)
    cai = cai_calculation(p_str, table, mtable)
    gc_content = (p_str.count('G') + p_str.count('C')) / len(p_str)
    # cpb = cpb_calculation(p_str, cps_table_path)
    cpg = cpg_calculation(p_str)
    upa = upa_calculation(p_str)
    print("Predictions:")
    print("".join(p)[:50])
    print("MFE:", mfe)
    print("CAI:", cai)
    print("GC content:", gc_content)
    # print("CPB: ", cpb)
    print("CpG density", cpg)
    print("UpA density", upa)
    
else:
    preds = []
    for i in tqdm(range(len(pred_labels)), desc="Decoding predictions"):
        input = tokenized_test["seq"][i].split(" ")
        p = pred_labels[i][1:len(input)+1]
        p = [id2label[pred] for pred in p]
        preds.append("".join(p))

    # df = pd.read_csv(cai_mfe_table_path)
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # save_path = os.path.join(save_dir, f"gencode.v47.pc_transcripts_cds_test_pred_{time_str}_{lambda1}_{lambda2}_gc{lambda_gc}_cpb{lambda_cpb}_liver{lambda_liver}_hepa{lambda_hepatocellular}.fa")
    save_file_name = args.save_file_name
    save_path = os.path.join(save_dir, f"{time_str}_{save_file_name}.fa")

    records = []
    for i, seq in tqdm(enumerate(preds), total=len(preds)):
        # header = df['ID'][i]
        header = headers[i]
        records.append(SeqRecord(Seq(seq), id=header, description=""))

    SeqIO.write(records, save_path, "fasta")
    
if steering_vectors is not None:
    for handle in handles:
        handle.remove()

end = time.time()
print("Time taken: ", end - start)
