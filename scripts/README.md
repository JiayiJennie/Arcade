# Arcade: CodonBERT-Based Steering for Codon Optimization

Arcade is a toolkit built on top of CodonBERT to steer codon sequence optimization through latent space manipulation. It allows for guided fine-tuning, steering vector extraction, and codon-level sequence editing.

---

## 🚀 Quick Start

### Fine-Tune Token Classification Model

```bash
python scripts/finetune_codonbert_token_cls.py \
  --model_path checkpoints/codonbert_base \
  --data_path data/GENCODE/gencode.v47.pc_transcripts_cds_train.fa \
  --output_dir outputs/finetuned_model \
  --use_lora
```

### Fetch Steering Vectors

This script computes **steering vectors** between two groups of sequences (e.g., high vs. low expression) using a frozen CodonBERT model.

```bash
python scripts/fetch_steering_vectors.py \
  --data_type fasta \
  --high_fa_path data/for_steering/high_upa.fa \
  --low_fa_path data/for_steering/low_upa.fa \
  --model_path checkpoints/codonbert_peft \
  --save_name upa \
  --save_dir data/steering_vectors
```

---

## 📁 Project Structure

```
Arcade/
├── scripts/               # Main training and analysis scripts
├── data/                  # Input data and FASTA files
├── checkpoints/           # Pretrained or fine-tuned models
├── outputs/               # Model outputs and steering vectors
├── utils/                 # Helper modules (e.g., tokenizer, metrics)
```

---

## 🧪 Dataset Notes

1. CDS sequences longer than 3072 nt are excluded due to CodonBERT’s input length limit.
2. The dataset is split 9:1 for training and testing. All transcript variants of the same gene are grouped to avoid information leakage.
3. Test set path:  
   `/mnt/disk90/user/jiayili/Arcade/data/GENCODE/gencode.v47.pc_transcripts_cds_test.fa`

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 📄 License

MIT License. See [LICENSE](./LICENSE) for details.