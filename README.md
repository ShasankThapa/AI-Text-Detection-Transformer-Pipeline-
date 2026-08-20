# AI Text Detection: Classical vs Transformer Pipeline

A research pipeline comparing classical machine learning and transformer 
models for detecting AI-generated text, evaluated across in-domain, 
cross-domain, and combined-domain settings.

## Overview

This project investigates how well classical ML baselines and transformer 
models generalize when detecting AI-generated text across different data 
sources. Beyond standard train/test evaluation, the pipeline includes 
cross-domain testing (training on one dataset, evaluating on another) and, 
for the transformer models, additional learning curve and threshold tuning 
experiments to assess robustness and calibration.

## Key Features

- **Classical baselines**: Logistic Regression and Linear SVC with TF-IDF 
  vectorization
- **Transformer model**: DistilBert and DistilRoBERTa
- **In-domain evaluation**: models trained and tested on the same dataset
- **Cross-domain evaluation**: models trained on one dataset, tested on another, 
  to measure generalization
- **Combined-domain evaluation**: models trained on merged datasets (domain 
  adaptation setting)
- **Learning curve experiments** (transformer only): performance across 
  varying training set sizes
- **Threshold tuning** (transformer only): optimizing classification 
  thresholds beyond the default 0.5 cutoff

## Tech Stack

- Python
- scikit-learn (classical models, TF-IDF)
- Hugging Face Transformers & Datasets
- pandas, numpy
## Datasets

This project uses two data sources. Neither is redistributed in this
repository per their respective licensing terms — see below for access,
terms, and citation.

### 1. M-DAIGT Shared Task Dataset
Used for `task1-train.csv` (News Article Detection) and `task2-train.csv`
(Academic Writing Detection).

- Source: M-DAIGT (Multi-Domain Detection of AI-Generated Text) shared task
- Paper: https://arxiv.org/abs/2511.11340v1
- Citation: Lamsiyah, S. et al. (2025) 'M-DAIGT: A shared task on
  multi-domain detection of AI-generated text', arXiv:2511.11340.
- Copyright: M-DAIGT Organizers
- Access: Available at the organizers' request/download link (see paper).
  Usage is restricted to academic research purposes; data may not be
  redistributed or re-published without organizer consent. This repository
  contains only code, not the dataset itself, in compliance with these terms.

### 2. Human vs AI-Generated Text Dataset
- Source: Hugging Face — `ahmadreza13/human-vs-Ai-generated-dataset`
- Access: Loaded automatically via 
  `datasets.load_dataset("ahmadreza13/human-vs-Ai-generated-dataset")`

## Results

Transformer models achieved near-perfect accuracy in-domain but degraded 
significantly in cross-domain testing, in one case collapsing into 
single-class prediction despite a misleadingly high F1-score. Classical 
models scored lower overall but stayed more stable under domain shift. 
Full results are available in the dissertation report.

## How to Run

### 1. Clone the repo and set up environment
```bash
git clone https://github.com/ShasankThapa/AI-Text-Detection-Transformer-Pipeline-.git
cd AI-Text-Detection-Transformer-Pipeline
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

#### Create Requirements.txt & include
pandas
numpy
scikit-learn
scipy
datasets
transformers
torch

### 2. Get the datasets
- **Hugging Face dataset**: downloads automatically when the notebook runs (requires internet access).
- **Codabench dataset**: not included in this repo due to size/licensing.

### 3. Run the notebooks
```bash
jupyter notebook
\`\`\`
- **`CLASSICAL-MODELS-AI-TEXT-DETECTION-PIPELINE.ipynb`** — classical models (Logistic Regression, LinearSVC). Runs on CPU in a few minutes.
- **`TRANSFORMERS-MODELS-AI-TEXT-DETECTION-PIPELINE.ipynb`** — fine-tunes DistilBERT/DistilRoBERTa. **A GPU is strongly recommended** 

Run cells top to bottom — each notebook is organised sequentially (dataset loading → preprocessing → train/test splits → training → evaluation).
```

## License

This project's code is licensed under the MIT License — see LICENSE for details. 
Note: this license applies to the code only, not the datasets described above, which remain under their original owners' terms.
