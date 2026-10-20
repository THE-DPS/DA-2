from pathlib import Path

import pandas as pd
import pytest

from sentiment_nlp.data import load_dataset, split_dataset
from sentiment_nlp.predict import predict_text
from sentiment_nlp.train import train


def test_load_dataset_and_split(tmp_path):
    rows = []
    for i in range(20):
        label = "positive" if i % 2 else "negative"
        text = f"{'wonderful excellent' if label == 'positive' else 'awful disappointing'} item number {i}"
        rows.append({"text": text, "label": label})
    path = tmp_path / "reviews.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    frame = load_dataset(path)
    train_df, val_df, test_df = split_dataset(frame)
    assert len(train_df) + len(val_df) + len(test_df) == len(frame)
    assert set(train_df.label) == {"positive", "negative"}


def test_train_and_predict_smoke(tmp_path):
    rows = []
    for i in range(40):
        if i % 2:
            rows.append({"text": f"excellent wonderful useful product {i}", "label": "positive"})
        else:
            rows.append({"text": f"terrible broken useless product {i}", "label": "negative"})
    data_path = tmp_path / "reviews.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    result = train(str(data_path), str(models_dir), str(reports_dir))
    assert (models_dir / "sentiment_pipeline.joblib").exists()
    assert (reports_dir / "evaluation_report.json").exists()
    prediction = predict_text(str(models_dir / "sentiment_pipeline.joblib"), "excellent and useful")
    assert prediction["label"] in {"positive", "negative"}
    assert 0 <= prediction["score"] <= 1
