# Product Review Sentiment Analysis

A reproducible NLP project for classifying English-language reviews as **positive** or **negative**. The repository is designed as an internship portfolio project and demonstrates data validation, preprocessing, TF-IDF feature extraction, model training, evaluation, and inference.

> **Important:** This repository contains a runnable implementation and test suite. It does not include a trained model or claim measured performance until you run the training/evaluation commands locally. The included demo trains a model on a small synthetic dataset for smoke testing only; use a properly licensed, representative dataset for meaningful evaluation.

## Features

- Validates input data and expected schema
- Conservative text normalization that preserves negation
- Scikit-learn pipeline: TF-IDF word n-grams + logistic regression
- Majority-class baseline and evaluation metrics
- Reproducible train/validation/test split
- Saves a versioned model artifact locally
- Predicts sentiment for one or more text inputs
- Unit tests for preprocessing and inference
- Clear separation between code, data, models, and reports

## Project structure

```text
product-review-sentiment-analysis/
├── data/
│   ├── README.md
│   └── sample_reviews.csv
├── models/                 # generated locally; not committed
├── reports/                # generated evaluation JSON/CSV
├── src/
│   └── sentiment_nlp/
│       ├── __init__.py
│       ├── data.py
│       ├── preprocessing.py
│       ├── train.py
│       └── predict.py
├── tests/
│   ├── test_preprocessing.py
│   └── test_smoke.py
├── .gitignore
├── requirements.txt
└── pyproject.toml
```

## Quick start

Python 3.10+ is recommended.

### 1. Create an environment

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Train and evaluate the example pipeline

```bash
python -m sentiment_nlp.train --data data/sample_reviews.csv --output-dir models --report-dir reports
```

This command uses the small included sample dataset for a software smoke test. The resulting metrics are **not** a reliable estimate of real-world performance. Replace it with a licensed, representative dataset before drawing conclusions.

Expected input CSV columns:

- `text`: review text
- `label`: `positive` or `negative`

### 3. Run predictions

After training:

```bash
python -m sentiment_nlp.predict --model models/sentiment_pipeline.joblib --text "The battery lasts all day and the screen is excellent."
```

Example output structure (the score will depend on the trained model):

```json
{
  "text": "The battery lasts all day and the screen is excellent.",
  "label": "positive",
  "score": 0.73,
  "model_version": "tfidf-logreg-v1"
}
```

The example score above is illustrative only; it is not a benchmark result.

## Use a public dataset responsibly

A public dataset such as the [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) can be useful for learning the sentiment-analysis workflow, but it contains **movie reviews**, not product reviews. For a product-review use case, choose a dataset that matches the intended product domain and verify its license, permitted use, provenance, language, label definitions, and privacy implications.

Before using external data:
1. Read and comply with the source's license and terms.
2. Record the source URL, access date, version, and attribution.
3. Preserve an immutable raw copy and document cleaning decisions.
4. Deduplicate before splitting where feasible.
5. Split data before fitting learned transformations.
6. Never tune repeatedly on the final test set.

## Methodology

1. Validate schema, labels, missing values, and text types.
2. Preserve raw text; normalize Unicode and whitespace conservatively.
3. Create stratified train/validation/test splits.
4. Fit TF-IDF only on training data through a scikit-learn `Pipeline`.
5. Compare against a majority-class baseline.
6. Report accuracy, precision, recall, F1, ROC AUC when calculable, and a confusion matrix.
7. Inspect errors and subgroup performance before considering deployment.

## Limitations

- The included sample data is synthetic and tiny; it is for testing code paths only.
- Sentiment classification can fail on sarcasm, mixed opinions, ambiguous wording, and domain shift.
- The `score` is a model score/probability output and should not be assumed calibrated unless calibration is separately tested.
- This project is a prototype, not a production service. Production use needs security review, privacy review, representative validation, monitoring, and operational controls.

## Testing

```bash
pytest
```

## GitHub upload checklist

1. Create a new public repository named `product-review-sentiment-analysis`.
2. Upload the contents of this ZIP (the folder contents, not the ZIP nested inside the repo).
3. Commit `README.md`, `src/`, `tests/`, `data/`, `requirements.txt`, `pyproject.toml`, and `.gitignore`.
4. Do not upload private data, secrets, virtual environments, or generated model artifacts.
5. If you run experiments, commit a clearly labeled report with dataset provenance and actual results.
