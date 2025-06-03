"""
This script fine-tunes a pre-trained model for token classification tasks.
"""
import os
import time
import torch
import numpy as np
import evaluate
from transformers import (
    TrainingArguments, Trainer, 
    )
from datasets import Dataset
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model, TaskType

from functools import partial
import argparse
from peft import PeftModel
from utils.load_utils import tokenize_and_align_labels, load_model, load_data


def build_dataset(data_path, max_length=1024):
    """
    Build dataset from fasta file and split it into training, validation and test sets.
    Args:
        data_path: str, path to the fasta file.
    Returns:
        ds_train: Dataset, training dataset.
        ds_valid: Dataset, validation dataset.
        ds_test: Dataset, test dataset.
    """
    X = load_data(data_path, max_length=max_length)
    print("Loaded", len(X), "sequences")

    X_train, X_test = train_test_split(X, test_size=0.02, random_state=42)
    X_train, X_valid = train_test_split(X_train, test_size=0.01, random_state=42)

    print("Train size:", len(X_train), "Validation size:", len(X_valid), "Test size:", len(X_test))

    ds_train = Dataset.from_list([{"seq": seq} for seq in X_train]) #TODO: change to full dataset
    ds_valid = Dataset.from_list([{"seq": seq} for seq in X_valid])
    ds_test = Dataset.from_list([{"seq": seq} for seq in X_test])

    return ds_train, ds_valid, ds_test

def encode_string(data):
    """
    Tokenize the input string using the bert_tokenizer_fast.
    """
    return bert_tokenizer_fast(data['seq'],
                               truncation=True,
                               padding="max_length",
                               max_length=max_length,
                               return_special_tokens_mask=True)


def compute_metrics(pred):
    """
    Compute the metrics.
    """
    metric = evaluate.load("evaluate/metrics/accuracy/accuracy.py")  
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)  
    
    true_labels = []
    true_preds = []
    
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] != -100:  # ignore special tokens
                true_labels.append(labels[i][j])
                true_preds.append(predictions[i][j])
    
    return metric.compute(predictions=true_preds, references=true_labels) 





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    parser = argparse.ArgumentParser(description="Fine-tune CodonBERT for token classification.")
    parser.add_argument('--model_dir', type=str, default='/mnt/disk90/user/jiayili/codon_optimization_steering/saved_model', help='Directory to save the fine-tuned model.')
    parser.add_argument('--data_path', type=str, default='/mnt/disk90/user/jiayili/GENCODE/gencode.v47.pc_transcripts_cds_3072_train.fa',
                        help='Path to the FASTA data for training.')
    parser.add_argument('--model_path', type=str, default='/mnt/disk90/user/jiayili/CodonBERT/ckpt',
                        help='Path to the pretrained model directory. This should be a CodonBERT checkpoint.')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length for tokenization. Default is 1024.')
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of the model.')
    parser.add_argument('--inter_size', type=int, default=3072, help='Intermediate size of the model.')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads.')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers.')
    parser.add_argument('--lr', type=float, default=0.00005, help='Learning rate for the optimizer.')
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of training steps.')
    parser.add_argument('--bs_train', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--bs_test', type=int, default=64, help='Batch size for testing.')
    parser.add_argument('--save_steps', type=int, default=100, help='Steps to save the model during training.')
    parser.add_argument('--eval_steps', type=int, default=100, help='Steps to evaluate the model during training.')
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA rank.')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha.')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout rate.')   
    parser.add_argument('--use_lora', action='store_true', help='Whether to use LoRA for fine-tuning. If set, will apply LoRA to the model.')
    parser.add_argument('--freeze_backbone', action='store_true', help='Whether to freeze the backbone model during training. Default is False.')
    parser.add_argument('--load_lora_checkpoint', action='store_true', help='Whether to load a LoRA checkpoint if provided.')

    args = parser.parse_args()

    model_dir = args.model_dir
    data_path = args.data_path
    model_path = args.model_path
    max_length = args.max_length
    hidden_size = args.hidden_size
    inter_size = args.inter_size
    num_heads = args.num_heads
    num_layers = args.num_layers
    lr = args.lr
    num_steps = args.num_steps
    bs_train = args.bs_train
    bs_test = args.bs_test
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    use_lora = args.use_lora
    freeze_backbone = args.freeze_backbone
    load_lora_checkpoint = args.load_lora_checkpoint

    if model_dir is None:
        raise ValueError("Please provide a model directory.")

    model, bert_tokenizer_fast, _, _ = load_model(model_path=model_path)
    if load_lora_checkpoint:
        model = PeftModel.from_pretrained(model, model_path)
        model=model.merge_and_unload()
    if freeze_backbone:
        for param in model.bert.parameters():
            param.requires_grad = False

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=lora_r, 
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_rslora=True,
            # modules_to_save = ["classifier"],
        )
        model = get_peft_model(model, lora_config)
        print("Model is using LoRA for fine-tuning.")
        model.print_trainable_parameters()
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.to(device)
    print(f"Model loaded from {model_path} and moved to {device}")

    ds_train, ds_eval, ds_test = build_dataset(data_path, max_length=max_length)

    tokenize_and_align_labels_fn = partial(tokenize_and_align_labels, bert_tokenizer_fast=bert_tokenizer_fast, label2id=label2id, max_length=max_length)
    tokenized_train = ds_train.map(tokenize_and_align_labels_fn, batched=True)
    tokenized_eval = ds_eval.map(tokenize_and_align_labels_fn, batched=True)
    tokenized_test = ds_test.map(tokenize_and_align_labels_fn, batched=True)

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    model_dir = model_dir + '/' + current_time
    os.makedirs(model_dir, exist_ok=True)

    training_args = TrainingArguments(
        optim='adamw_torch',
        learning_rate=lr,                    # learning rate
        output_dir=model_dir,                # output directory to where save model checkpoint
        evaluation_strategy="steps",         # evaluate each `logging_steps` steps
        overwrite_output_dir=True,
        # num_train_epochs=num_epoches,         # number of training epochs, feel free to tweak
        max_steps=num_steps,                  # number of training steps
        per_device_train_batch_size=bs_train, # the training batch size, put it as high as your GPU memory fits
        per_device_eval_batch_size=bs_test,   # evaluation batch size
        save_strategy="steps",
        save_steps=save_steps,                # save model
        eval_steps=eval_steps,
        load_best_model_at_end=True,          # whether to load the best model (in terms of loss) at the end of training
        save_total_limit = 5,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=bert_tokenizer_fast,
        compute_metrics=compute_metrics
    )

    trainer.train()

    best_model_path = model_dir + '/best_model'
    # trainer.save_model(best_model_path)
    model.save_pretrained(best_model_path)  # save model and adapter
    print("Best model saved to", best_model_path)   

    eval_result = trainer.evaluate(tokenized_test)
    print("Evaluation result on test set:", eval_result)