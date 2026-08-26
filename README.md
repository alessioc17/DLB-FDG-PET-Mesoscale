# DLB-FDG-PET-Mesoscale

Computational pipeline for multiscale analysis of clinical heterogeneity in dementia with Lewy bodies using FDG-PET.

This repository contains the analysis code accompanying the study and is intended to facilitate reproducibility of the reported computational analyses.

## Overview

This repository contains the Python implementation used to perform the analyses described in:

"Mesoscale metabolic connectivity captures clinical heterogeneity in dementia with Lewy bodies"

The pipeline evaluates FDG-PET data at three complementary levels of brain organisation:

- Regional: regional FDG-PET uptake with XGBoost classification
- Mesoscale: metabolic connectomes analysed using PCA, RFE and connectome-based predictive modelling (CPM)
- Whole-brain: global graph-theoretical metrics with XGBoost classification

All predictive analyses use leave-one-out cross-validation, with preprocessing and feature selection performed within the training set of each fold. Statistical significance is assessed using permutation testing.

## Methods

The metabolic connectome is constructed from pairwise log-ratio relationships between regional FDG-PET uptake values:

C_ij = |log(x_i / x_j)|

For N regions, this results in N(N-1)/2 unique undirected edges.

The pipeline implements:

1. Regional FDG-PET uptake → XGBoost + LOOCV
2. Metabolic connectome → PCA + XGBoost + LOOCV
3. Metabolic connectome → RFE + XGBoost + LOOCV
4. Global graph metrics → XGBoost + LOOCV
5. Metabolic connectome → CPM + logistic regression + LOOCV

Permutation testing is applied to the complete predictive pipeline.

## Requirements

Python 3.10

Required packages:

- numpy
- pandas
- scipy
- scikit-learn
- xgboost
- networkx
- joblib
- openpyxl

## Installation

```bash
# Clone the repository:

git clone https://github.com/alessioc17/DLB-FDG-PET-Mesoscale.git

cd DLB-FDG-PET-Mesoscale

# Install dependencies:

pip install -r requirements.txt
```

## Input data

The pipeline requires an Excel workbook containing:

1. An uptake sheet with one row per subject and one column per ROI.
2. A clinical outcome sheet containing the same subject identifiers and binary clinical outcomes.

No patient-level data are distributed with this repository.

## Usage

```bash
# Example:

python run_analysis.py \
    --input-excel data.xlsx \
    --uptake-sheet FDGPET \
    --labels-sheet clinical \
    --id-column subject_id \
    --labels "UPDRS=UPDRS_binary,MMSE=MMSE_binary,Hallucinations=hallucinations" \
    --output-dir results
```

## Main parameters

- `--n-permutations`: number of permutations for permutation testing (default: 500)
- `--n-jobs`: number of parallel jobs (default: -1)
- `--random-state`: random seed (default: 42)
- `--graph-density`: proportional graph threshold (default: 0.20)
- `--pca-variance`: cumulative variance retained by PCA (default: 0.70)
- `--rfe-features`: number of features selected by RFE (default: 20)
- `--rfe-stability`: minimum selection frequency for RFE consensus features (default: 0.50)
- `--cpm-p-thresholds`: CPM feature-selection thresholds (default: 0.05,0.01,0.001)
- `--cpm-consensus`: minimum selection frequency for CPM consensus edges (default: 0.50)

The complete analysis configuration is automatically saved to `analysis_configuration.json`.

## Reproducibility

The pipeline is designed to minimise information leakage by performing scaling, dimensionality reduction and feature selection within each training fold of the LOOCV procedure.

Permutation testing repeats the complete predictive pipeline after randomisation of the outcome labels.

The random seed and all analysis parameters are recorded in the output configuration file.

## Output

For each clinical outcome, the pipeline generates:

- classification performance metrics
- ROC curves
- permutation-test results
- feature-importance summaries
- consensus feature/edge information
- PCA summaries
- graph-theoretical metrics
- analysis metadata

A master summary table is also generated across outcomes.

## Data availability

Patient-level imaging and clinical data are not included in this repository.

The code is intended to be used with appropriately governed datasets for which the user has the necessary ethical and institutional permissions.

## Citation

If you use this software, please cite:

```markdown
Cirone A, et al. *Mesoscale metabolic connectivity captures clinical heterogeneity in dementia with Lewy bodies*. Manuscript submitted for publication.
```

The citation will be updated with the final publication details upon acceptance.

## License

MIT License
