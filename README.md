# HARP

This repository contains an implementation of ["High-Layer Attention Pruning with Rescaling"](https://openreview.net/forum?id=jkPBIxYmWE), which is a framework for evaluating pruned Large Language Models on standard and long-context benchmarks.

## Project Structure

```
.
├── models/          # Modified LLM model files adapted for HARP
├── lib/             # FLAP pruning algorithm implementation
├── config/          # Configuration files for LongBench
├── main.py          # Standard benchmark evaluation
├── main_longbench.py # Long-context benchmark evaluation
├── metrics.py       # LongBench evaluation metrics
└── ex.sh            # Shell script for running benchmarks
```

## Setup

### Prerequisites

Download the required LLM model files and save them to the directory specified by `to_save_dir` in both `main.py` and `main_longbench.py`.

## Usage

### Running Benchmarks

Execute the benchmark script:

```bash
bash ex.sh
```

This will run both standard and long-context benchmarks.

### Standard Benchmark

```bash
python main.py
```

### Long-Context Benchmark (LongBench)

```bash
python main_longbench.py
```

## Components

### Models

The `models/` directory contains modified versions of Hugging Face Transformers model files, adapted to work with the HARP framework.

### FLAP Pruning

The `lib/` directory implements the FLAP (Fast Language Model Pruning) algorithm used for model compression.

### Configuration

Benchmark configurations for LongBench are stored in the `config/` directory. Evaluation metrics are defined in `metrics.py`.

## Citation
```
@article{liu2025high,
  title={High-Layer Attention Pruning with Rescaling},
  author={Liu, Songtao and Liu, Peng},
  journal={Transactions on Machine Learning Research},
  year={2026}
}
```