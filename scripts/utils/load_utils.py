from transformers import BertForTokenClassification
from transformers import (
    PreTrainedTokenizerFast, BertForTokenClassification
    )
from Bio import SeqIO

import sys, os
# print('hi',os.path.abspath('../../benchmarks/utils'))
# assert 1==0
sys.path.append(os.path.abspath('../benchmarks/utils'))
# sys.path.append("/mnt/disk90/user/jiayili/CodonBERT_sanofi/benchmarks/utils")
from tokenizer import get_tokenizer, mytok

def tokenize_and_align_labels(examples, bert_tokenizer_fast, label2id, max_length=1024):
    """
    Tokenize the input sequences and align the labels.
    Args:
        examples: dict, input sequences.
    Returns:
        tokenized_inputs: dict, tokenized sequences.
    """
    tokenized_inputs = bert_tokenizer_fast(examples["seq"], truncation=True, padding="max_length", max_length=max_length)
    
    all_labels = []
    assert len(tokenized_inputs["input_ids"]) == len(examples["seq"])
    for i in range(len(tokenized_inputs["input_ids"])):
        input_ids = tokenized_inputs["input_ids"][i]
        tokens = bert_tokenizer_fast.convert_ids_to_tokens(input_ids)
        labels = []
        for token in tokens:
            if token.startswith("##"):
                print("token starts with ##:", token)
                token = token[2:]
            if token in bert_tokenizer_fast.all_special_tokens:
                labels.append(-100)
            else:
                labels.append(label2id.get(token, -100))
        all_labels.append(labels)
    tokenized_inputs["labels"] = all_labels

    return tokenized_inputs

def load_data(data_path, max_length=1024, save_header=False):
    """
    Load data from fasta file and tokenize it.
    Args:
        data_path: str, path to the fasta file.
    Returns:
        seqs: list of str, tokenized sequences.
    """
    headers = []
    seqs = []

    raw_seqs = []
    with open(data_path) as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            header = str(record.id)
            seq = str(record.seq)
            headers.append(header)
            raw_seqs.append(seq)

    skipped = 0
    # total = len(raw_seqs)
    for seq in raw_seqs:
        lst_tok = mytok(seq, 3, 3)
        if lst_tok:
            if len(lst_tok) > max_length - 2:
                skipped += 1
                # print("Skip one sequence with length", len(lst_tok), \
                #     "codons. Skipped %d seqs from total %d seqs." % (skipped, total))
                continue
            seqs.append(" ".join(lst_tok))

    if save_header:
        return seqs, headers
    return seqs

def load_model(model_path=None):
    """
    Load pretrained CodonBERT model and Tokenizer.
    """
    global label2id, id2label, bert_tokenizer_fast

    tokenizer = get_tokenizer()
    vocab = tokenizer.get_vocab() # includes special tokens

    label2id = {k: v-5 for k, v in vocab.items() if v > 4}
    id2label = {v: k for k, v in label2id.items()}

    bert_tokenizer_fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        do_lower_case=False, clean_text=False,
        tokenize_chinese_chars=False, strip_accents=False,
        unk_token='[UNK]', sep_token='[SEP]',
        pad_token='[PAD]', cls_token='[CLS]',
        mask_token='[MASK]'
        )
    
    if model_path is None:
        raise ValueError("Please provide a model path.")
    else: 
        model = BertForTokenClassification.from_pretrained(model_path, num_labels=len(label2id), 
                                                       label2id=label2id,id2label=id2label)
    
    return model, bert_tokenizer_fast, label2id, id2label
