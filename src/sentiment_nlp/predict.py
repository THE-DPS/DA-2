"""Run predictions using a saved sentiment pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from sentiment_nlp.preprocessing import normalize_text


def predict_text(model_path: str, text: str) -> dict:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("text must contain at least one non-whitespace character")
    artifact = joblib.load(Path(model_path))
    model = artifact["model"]
    label = str(model.predict([normalized])[0])
    classes = list(model.classes_)
    score = float(model.predict_proba([normalized])[0][classes.index("positive")])
    return {
        "text": text,
        "label": label,
        "score": score,
        "model_version": artifact.get("model_version", "unknown"),
        "score_note": "Positive-class model probability; calibration has not been assessed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/sentiment_pipeline.joblib")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(json.dumps(predict_text(args.model, args.text), indent=2))


if __name__ == "__main__":
    main()
