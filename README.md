# Arcade: Activation Engineering for Controllable Codon Design

This repository extends CodonBERT[*CodonBERT: large language models for mRNA design and optimization*](https://www.biorxiv.org/content/10.1101/2023.09.09.556981v1) with our framework — Arcade — for controllable codon design using activation engineering and semantic steering.

### Environment Setup

Dependency management is done via [poetry](https://python-poetry.org/).
```
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
    ├── fetch_steering_vectors      # script for making steering vectors
├── data/                  # Data for finetuning and steering vectors
├── checkpoints/           # fine-tuned models
├── outputs/               # Model outputs
```

## Quick Start

We provide a checkpoint with a token classification head, along with precomputed steering vectors.

### Codon Design with Different Targets

```bash
cd scripts
```
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

If you only want to see one sequence:
```bash
CUDA_VISIBLE_DEVICES=0 python -u recons_token_cls.py \
  --lambda_gc 0 \
  --single_example \
  --input_example ATGCCA
```

If you want to use your own data and base model to get started, you can train the token classification head and generate steering vectors using the following commands:

### Fine-Tune Token Classification Model

```bash
  --model_path path/to/downloaded/base/model (e.g., CodonBERT checkpoints) \
  --data_path path/to/your/training/data \
  --use_lora
```

### Fetch Steering Vectors

This script computes **steering vectors** between two groups of mutated sequences (e.g., high vs. low expression) using a frozen CodonBERT model.

```bash
python scripts/fetch_steering_vectors.py \
  --data_type fasta \
  --high_fa_path data/for_steering/high_gc.fa \
  --low_fa_path data/for_steering/low_gc.fa \
  --model_path checkpoint \
  --save_name gc \
  --save_dir data/steering_vectors
```

