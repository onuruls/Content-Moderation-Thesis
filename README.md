# Content Moderation Thesis: Evaluation & Training Pipeline

This repository contains the official codebase for the proposed B.Sc. thesis on Content Moderation at Flensburg University of Applied Sciences. It provides a standardized, reproducible pipeline for evaluating and training state-of-the-art vision models (CLIP, SigLIP, EVA-02) on content safety benchmarks.

**Purpose**: To benchmark existing safety classifiers and train new lightweight probes for efficient, high-performance content moderation.

## 🔗 Project Links
- **Demo Space**: [Hugging Face Space](https://huggingface.co/spaces/onullusoy/content_moderation_demo)
- **Dataset**: [Hugging Face Dataset](https://huggingface.co/datasets/onullusoy/harmful-contents)

## 📦 What's Inside
- **Unified Evaluation Runner**: Run evaluations across NudeNet, LSPD, UnsafeBench, and internal datasets with a single config.
- **Training Pipeline**: Train lightweight Linear Heads (Probes) on top of frozen backbones (CLIP, SigLIP).
- **Standardized Interfaces**: Common `Model` protocol and `Dataset` registry for easy extension.
- **Metrics & Calibration**: Automated threshold selection (F1-optimal) and prevalence alignment.

## 🚀 Quickstart

### 1. Install
```bash
# Clone repository
git clone https://github.com/onullusoy/Content-Moderation-Thesis.git
cd Content-Moderation-Thesis

# Install dependencies (requires Python 3.10+)
pip install -e .
```

### 2. Prepare Data

**Open-source datasets** — automatically loaded from Hugging Face, no manual setup needed:
- 🤗 **UnsafeBench**: [`yiting/UnsafeBench`](https://huggingface.co/datasets/yiting/UnsafeBench)
- 🤗 **Internal (Harmful-Contents)**: [`onullusoy/harmful-contents`](https://huggingface.co/datasets/onullusoy/harmful-contents)

**Closed-source datasets** — not publicly available. If you have access, place them under `src/data/` or symlink accordingly:
- `src/data/nudenet/` — [NudeNet Dataset
- `src/data/lspd/` — Large-Scale Pornography Detection Dataset


### 3. Run Evaluation
Evaluate a pre-trained CLIP Multi-label head on Internal Dataset:
```bash
python scripts/eval.py --config configs/eval/clip_multilabel.yaml
```
Results will be saved to `results/<run_name>.json`.

### 4. Train a Probe
Train a linear head on extracted features (e.g. for a new internal dataset):
```bash
python scripts/train.py --config configs/train/sample_head.yaml
```

## 📂 Project Structure
```
content_moderation_thesis/
├── configs/              # Experiment configurations
│   ├── eval/             # Evaluation configs (models + datasets)
│   ├── train/            # Training configs
│   └── probe/            # Linear probe specific configs
├── scripts/              # CLI Entrypoints
│   ├── eval.py           # Main evaluation script
│   ├── train.py          # Main training script
│   └── probe.py          # Alias for linear probing
├── src/
│   └── cmt/              # Main package (Content Moderation Thesis)
│       ├── data/         # Dataset adapters & registry
│       ├── eval/         # Metrics, runner, calibration
│       ├── models/       # Model implementations (CLIP, SigLIP, EVA)
│       ├── train/        # Training loop & losses
│       └── utils/        # Shared utilities
├── results/              # Output directory
└── third_party/          # Vendored dependencies (HySAC)
```

## ⚙️ Configuration
Configs are YAML files controlling every aspect of a run. Key fields:

- **`model_name`**: `clip_multilabel`, `clip_lp`, `siglip_lp`, `wdtagger`, `hysac`.
- **`dataset`**: `nudenet`, `lspd`, `internal`, `unsafebench`.
- **`val_split` / `test_split`**: Standard slice syntax supported (e.g., `train[:10%]`, `validation`).
- **`dtype`**: `float16`, `bfloat16`, `float32`.
- **`batch_size`**: Inference batch size.

## 📊 Results

The pipeline produces detailed evaluation reports in `results/`. Below is a summary of the performance for key models on the **NudeNet** dataset (Testing split):

| Model Architecture | Method | F1 Score | Accuracy | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CLIP ViT-L/14** | Linear Probe | **0.9637** | 0.9640 | 0.9510 | 0.9767 |
| **SigLIP So400M** | Linear Probe | **0.9631** | 0.9644 | 0.9764 | 0.9502 |
| **AnimeTIMM** | Fine-tuned | **0.9504** | 0.9526 | 0.9725 | 0.9293 |
| **CLIP Multilabel** | Linear Probe | **0.9489** | 0.9506 | 0.9587 | 0.9394 |
| **WD-EVA-02** | Fine-tuned | **0.9270** | 0.9309 | 0.9584 | 0.8976 |

### Multi-label Capabilities
Models like `CLIP Multilabel` and `AnimeTIMM` also support specialized categories. Performance on the **Internal** dataset for selected categories:

| Category | CLIP ML (FT) F1 | AnimeTIMM (FT) F1 | WD-EVA (FT) F1 |
| :--- | :---: | :---: | :---: |
| **Nudity** | 0.9806 | 0.9615 | 0.9542 |
| **Sexy** | 0.9447 | 0.9485 | 0.9167 |
| **Drugs** | 0.9600 | 0.8493 | 0.8493 |
| **Violence** | 0.9372 | 0.9300 | 0.9100 |

Detailed results, including ROC-AUC, PR-AUC, and per-category thresholds, are stored in `results/post_train/`.

## ⚠️ Disclaimer & License
**Usage**: The weights and code provided are for research and non-commercial use only.
**Commercial Use**: Requires independent rights verification for the underlying models (e.g., CLIP, SigLIP) and datasets.
**Content Warning**: This repository deals with content moderation of unsafe/NSFW material. The datasets and samples may contain sensitive content.

**License**: MIT License (Codebase). Model weights are subject to their respective original licenses.
