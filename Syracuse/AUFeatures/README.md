# Pain Level Classification using AU and PSPI Features

This directory contains scripts for pain level classification using facial Action Unit (AU) features and PSPI (Prkachin and Solomon Pain Intensity) scores.

## Overview

Two main scripts are provided:

1. `au_pain_level_classification.py`: Uses individual AU features for pain level classification
2. `pspi_pain_level_classification.py`: Uses PSPI scores (derived from AU features) for pain level classification

Both scripts support 3-class and 5-class pain level classification:

- **3-Class**: Low (0-2), Medium (3-5), High (6-10)
- **5-Class**: Class 0 (0-1), Class 1 (2-3), Class 2 (4-5), Class 3 (6-7), Class 4 (8-10)

## Usage

### AU Features Classification

```bash
python au_pain_level_classification.py [--use_pspi] [--n_splits N_SPLITS] [--random_state RANDOM_STATE]
```

Options:
- `--use_pspi`: Use PSPI score instead of individual AU features
- `--n_splits`: Number of folds for cross-validation (default: 5)
- `--random_state`: Random seed (default: 42)

### PSPI Classification

```bash
python pspi_pain_level_classification.py [--n_splits N_SPLITS] [--random_state RANDOM_STATE] [--output_dir OUTPUT_DIR]
```

Options:
- `--n_splits`: Number of folds for cross-validation (default: 5)
- `--random_state`: Random seed (default: 42)
- `--output_dir`: Output directory for results (default: results/pspi_pain_classification)

## Features

### AU Features

The scripts use the top 5 AU features based on effect size:
- AU12_r (lip corner puller)
- AU07_r (lid tightener)
- AU05_r (upper lid raiser)
- AU01_r (inner brow raiser)
- AU02_r (outer brow raiser)

### PSPI Score

The PSPI score is calculated as:
```
PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10)
```

Where:
- AU4: Brow lowerer
- AU6: Cheek raiser
- AU7: Lid tightener
- AU9: Nose wrinkler
- AU10: Upper lip raiser

## Evaluation

The scripts perform k-fold cross-validation and report the following metrics:
- Accuracy
- Quadratic Weighted Kappa (QWK)
- Mean Absolute Error (MAE)
- Confusion Matrix

Results are saved as JSON files and confusion matrices are visualized as PNG images.

## Output

Results are saved in the following directories:
- AU Features: `results/au_pain_classification/`
- PSPI: `results/pspi_pain_classification/`

Each directory contains:
- JSON file with metrics for each model and classification problem
- PNG files with confusion matrices

## Models

The following models are evaluated:
- Logistic Regression
- Ridge Classifier (PSPI only)
- SVM with linear kernel (PSPI only)
- Random Forest (PSPI only)

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Data Paths

The scripts use the following paths:
- Meta data: `/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2/meta_with_outcomes.xlsx`
- Feature directory: `/Users/hd927/Documents/syracuse_pain_research/multimodal_marlin_base 2`
- AU features directory: `/Users/hd927/Documents/syracuse_pain_research/AUFeatures/processed`

These paths may need to be updated for your environment. 