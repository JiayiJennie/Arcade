# Arcade: Activation Engineering for Controllable Codon Design

This repository extends CodonBERT[*CodonBERT: large language models for mRNA design and optimization*](https://www.biorxiv.org/content/10.1101/2023.09.09.556981v1) with our framework — Arcade — for controllable codon design using activation engineering and semantic steering.

### Environment Setup

We recommend starting from a clean Conda environment:

```bash
conda create -n arcade python=3.10
conda activate arcade
```
You can install dependencies in one of two ways:

**Option 1: Using pip and a `requirements.txt` file**
```bash
pip install -r requirements.txt
```

**Option 2: Using Poetry (TODO)** 
Dependency management can also be done via [Poetry](https://python-poetry.org/).  
This project extends the setup from the original CodonBERT repository, with updated dependencies defined in `pyproject.toml`:

```bash
pip install poetry
poetry install
```
Ensure you have CUDA drivers if you plan on using a GPU.

### Download Pretrained Model CodonBERT

The CodonBERT Pytorch model can be downloaded [here](https://cdn.prod.accelerator.sanofi/llm/CodonBERT.zip). The artifact is under a [license](ARTIFACT_LICENSE.md).
The code and repository are under a [software license](SOFTWARE_LICENSE.md).

## Arcade
Arcade (Activation Engineering for Controllable Codon Design) builds on CodonBERT and introduces a framework for:
- Controllable codon design
- Steering on a metric with continuous values
- Steering on the Encoder-only foundation model

### Project Structure

```
Arcade/
├── scripts/               # Main scripts 
    ├── recons_token_cls.py         # script for codon design
    ├── finetune_token_cls.py       # script for finetuning the pretrained base model with token classification head
    ├── fetch_steering_vectors.py   # script for making steering vectors
├── data/                  # Data for finetuning and steering vectors
├── checkpoints/           # fine-tuned models
    ├── arcade/            # Arcade checkpoint
    ├── codonbert/         # CodonBERT checkpoint
├── results/               # Model outputs
├── calculator/            # Calculator for metrics (e.g. CAI, MFE, GC content)
``` 

## Quick Start

We provide a checkpoint with a token classification head, along with precomputed steering vectors.

### Codon Design with Different Targets
```bash
cd checkpoint
wget https://cdn.prod.accelerator.sanofi/llm/CodonBERT.zip
unzip CodonBERT.zip
sed -i 's|"base_model_name_or_path": *".*"|"base_model_name_or_path": "'"$(realpath arcade/)"'"|g' arcade/adapter_config.json
```

```bash
cd ../scripts
```

If you only want to see one sequence:
```bash
CUDA_VISIBLE_DEVICES=0 python -u recons_token_cls.py \
  --lambda_gc 1 \
  --single_example \
  --input_example ATGCCA
```

Use the test data at `data/GENCODE/gencode.v47.pc_transcripts_cds_test.fa`
```bash
CUDA_VISIBLE_DEVICES=0 python -u recons_token_cls.py \
  --lambda_cai 1 \
  --save_file_name 'cai' 
```

With multiple targets:
```bash
CUDA_VISIBLE_DEVICES=0 python -u recons_token_cls.py \
  --lambda_cai 1 \
  --lambda_gc 1 \
  --save_file_name 'cai+gc' 
```



If you want to use your own data and base model to get started, you can train the token classification head and generate steering vectors using the following commands:

### Fine-Tune Token Classification Model

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/finetune_token_cls.py
  --model_path path/to/downloaded/base/model (e.g., CodonBERT checkpoints) \
  --data_path path/to/your/training/data \
  --use_lora
```

### Fetch Steering Vectors

This script computes **steering vectors** between two groups of mutated sequences (e.g., high vs. low expression) using a frozen CodonBERT model.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/fetch_steering_vectors.py \
  --data_type fasta \
  --high_fa_path data/for_steering/high_gc.fa \
  --low_fa_path data/for_steering/low_gc.fa \
  --model_path checkpoint/arcade \
  --save_name 'gc' \
  --save_dir data/steering_vectors
```

