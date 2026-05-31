from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_ai.config import get_settings  # noqa: E402
from legal_ai.logging_utils import configure_logging  # noqa: E402
from legal_ai.services.classifier import LegalClassifier  # noqa: E402
from legal_ai.utils.data import load_dataset  # noqa: E402


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description="Evaluate the legal classifier on a CSV split.")
    parser.add_argument("--input", default=settings.test_dataset_path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    df = load_dataset(args.input)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a label column for evaluation.")

    if args.limit:
        df = df.head(args.limit).copy()

    classifier = LegalClassifier(settings)
    rows = []
    for row in tqdm(df.to_dict(orient="records"), total=len(df), desc="Evaluating"):
        prediction = classifier.predict(row["case_text"])
        rows.append(
            {
                "case_id": row["case_id"],
                "true_label": row["label"],
                "predicted_label": prediction["predicted_label"],
                "predicted_name": prediction["predicted_name"],
                "confidence_score": prediction["confidence_score"],
            }
        )

    pred_df = pd.DataFrame(rows)
    output_path = settings.resolve_path(settings.evaluation_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)

    print("\nClassification report")
    print(classification_report(pred_df["true_label"], pred_df["predicted_label"], digits=4))
    print("Confusion matrix")
    print(confusion_matrix(pred_df["true_label"], pred_df["predicted_label"]))
    print(f"\nSaved predictions to: {output_path}")


if __name__ == "__main__":
    main()
