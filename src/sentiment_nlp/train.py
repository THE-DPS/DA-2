"""Train and evaluate a baseline sentiment pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from sentiment_nlp.data import load_dataset, split_dataset
from sentiment_nlp.preprocessing import normalize_text

MODEL_VERSION = "tfidf-logreg-v1"


def make_pipeline() -> Pipeline:
    """Build preprocessing + TF-IDF + logistic regression as one artifact."""
    # Normalization is applied before vectorization in a callable analyzer
    # would interfere with standard tokenization, so normalize text before fit
    # and inference using the same helper.
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )


def _metrics(y_true, y_pred, y_score=None) -> dict:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_positive": float(precision_score(y_true, y_pred, pos_label="positive", zero_division=0)),
        "recall_positive": float(recall_score(y_true, y_pred, pos_label="positive", zero_division=0)),
        "f1_positive": float(f1_score(y_true, y_pred, pos_label="positive", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix_labels_negative_positive": confusion_matrix(
            y_true, y_pred, labels=["negative", "positive"]
        ).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=["negative", "positive"], output_dict=True, zero_division=0
        ),
    }
    if y_score is not None and len(set(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score((np.asarray(y_true) == "positive").astype(int), y_score))
    return result


def train(data_path: str, output_dir: str, report_dir: str) -> dict:
    frame = load_dataset(data_path)
    # Normalize conservatively, consistently for train/validation/test.
    frame["text"] = frame["text"].map(normalize_text)
    train_df, val_df, test_df = split_dataset(frame)

    # A majority-class baseline, fitted on training data only.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(np.zeros((len(train_df), 1)), train_df["label"])
    val_baseline_pred = baseline.predict(np.zeros((len(val_df), 1)))

    model = make_pipeline()
    model.fit(train_df["text"], train_df["label"])
    val_pred = model.predict(val_df["text"])
    val_scores = model.predict_proba(val_df["text"])[:, list(model.classes_).index("positive")]

    # Freeze model choice after validation, then evaluate the test set once.
    test_pred = model.predict(test_df["text"])
    test_scores = model.predict_proba(test_df["text"])[:, list(model.classes_).index("positive")]

    output_path = Path(output_dir)
    report_path = Path(report_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "model_version": MODEL_VERSION,
        "classes": list(model.classes_),
        "normalization": "NFC + HTML entity decoding + tag removal + whitespace normalization",
        "dataset_path": str(data_path),
        "warning": "Do not treat metrics from the tiny synthetic sample dataset as real-world performance.",
    }
    model_file = output_path / "sentiment_pipeline.joblib"
    joblib.dump(artifact, model_file)

    results = {
        "model_version": MODEL_VERSION,
        "dataset_path": str(data_path),
        "dataset_rows_after_validation": int(len(frame)),
        "split_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "baseline_validation": {
            "accuracy": float(accuracy_score(val_df["label"], val_baseline_pred)),
            "macro_f1": float(f1_score(val_df["label"], val_baseline_pred, average="macro", zero_division=0)),
        },
        "validation_metrics": _metrics(val_df["label"], val_pred, val_scores),
        "test_metrics": _metrics(test_df["label"], test_pred, test_scores),
        "artifact_path": str(model_file),
        "data_warning": "Sample data is synthetic and tiny; metrics are a code smoke test, not a valid benchmark.",
    }
    report_file = report_path / "evaluation_report.json"
    report_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/sample_reviews.csv", help="CSV with text,label columns")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()
    train(args.data, args.output_dir, args.report_dir)


if __name__ == "__main__":
    main()
