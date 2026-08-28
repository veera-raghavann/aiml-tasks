# Messy Dataset, Baseline Model

**MLSA SRM Technical Recruitment — AI/ML**

This submission explores the deliberately messy `recruitment_engagement.csv` dataset and builds a reproducible baseline for predicting whether an applicant completed the recruitment task.

## What was messy?

The raw data contains missing values, inconsistent representations of categorical values, impossible numeric values, duplicate records, and numerical outliers. Examples include `Technical`/`technical`/`TECHNICAL`, `First`/`1`/`1st Year`, `Yes`/`yes`/`Y`, missing subdomains and years, negative preparation hours, negative days since signup, and unusually large preparation-hour values.

## Cleaning decisions

- Exact duplicate rows are removed because they do not represent additional observations.
- Categorical representations are normalized rather than treated as separate categories.
- Numeric columns are explicitly converted to numeric values.
- Negative preparation hours and negative days since signup are treated as missing because they are logically impossible in this context.
- Quiz scores outside 0–100 are treated as missing.
- Remaining missing numeric values are median-imputed; categorical values use most-frequent imputation. Imputation is fitted inside the scikit-learn pipeline to avoid test-set leakage.
- Unusually large preparation-hour values are **flagged but retained**. They are unusual, but not automatically impossible, so deleting them without evidence would throw away potentially valid observations. Robust scaling is used for the linear model.
- `applicant_id` and `name` are excluded from model features because they identify applicants rather than provide useful behavioral information.

## Models

The notebook compares two intentionally lightweight approaches:

1. **Logistic Regression** — a strong, interpretable linear baseline.
2. **Random Forest** — a modest nonlinear baseline that can capture interactions without requiring feature scaling.

Both models use the same train/test split and the same imputation/encoding pipeline so the comparison is meaningful. The notebook reports accuracy, precision, recall, F1, confusion matrices, and the individual misclassified test rows.

For the second-year extension, model selection should be based on performance on the training data using cross-validation rather than selecting a model from the test set. The held-out test set is reserved for the final comparison. If the simpler model performs similarly, it is the preferred choice because this dataset is small and interpretability matters.

## How to run

### Google Colab

Open `messy_dataset_baseline.ipynb` in Colab and run the cells from top to bottom. The notebook loads the official starter CSV directly from the MLSA-SRM repository.

### Local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook messy_dataset_baseline.ipynb
```

## Limitations

This is a small recruitment dataset, so a single train/test split can produce noisy estimates. The model is a baseline, not a production recruitment system. With more time, I would use repeated cross-validation, inspect feature stability, test calibration, and evaluate the model on a genuinely unseen future batch.

## Source

Starter dataset: https://github.com/MLSA-SRM/recruit-task-messy-dataset
