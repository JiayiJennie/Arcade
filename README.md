# Arcade: ctivation Engineering for Controllable Codon Design

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

