import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

# Project root
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import *
from src.data import make_official_splits, make_dataset


def main():

    # ---------------------------------------------------------
    # 1. Create output directory
    # ---------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 2. Load trained DP-SGD model
    #    compile=False prevents Keras from trying to restore
    #    TensorFlow Privacy's DPOptimizerClass.
    #    Optimizer is NOT required for inference/evaluation.
    # ---------------------------------------------------------
    model_path = MODEL_DIR / "lightweight_brain_tumor_dp.keras"

    print("Loading model:")
    print(model_path)

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    print("Model loaded successfully.")
    print("Total parameters:", model.count_params())

    # ---------------------------------------------------------
    # 3. Load official Train/Validation/Test split
    #
    # Official Test set remains completely untouched.
    # ---------------------------------------------------------
    _, _, (te_p, te_y), classes = make_official_splits(
        TRAIN_DIR,
        TEST_DIR,
        VALIDATION_FRACTION,
        SEED
    )

    print()
    print("Classes:", classes)
    print("Official Test Images:", len(te_y))

    # ---------------------------------------------------------
    # 4. Create test dataset
    #    Training augmentation is disabled.
    # ---------------------------------------------------------
    test_dataset = make_dataset(
        te_p,
        te_y,
        False,
        BATCH_SIZE
    )

    # ---------------------------------------------------------
    # 5. Generate predictions
    # ---------------------------------------------------------
    print()
    print("Running prediction on official test set...")

    probs = model.predict(
        test_dataset,
        verbose=1
    )

    pred = np.argmax(probs, axis=1)

    # ---------------------------------------------------------
    # 6. Calculate overall metrics
    # ---------------------------------------------------------
    acc = accuracy_score(te_y, pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        te_y,
        pred,
        average="weighted",
        zero_division=0
    )

    # ---------------------------------------------------------
    # 7. Classification report
    # ---------------------------------------------------------
    report = classification_report(
        te_y,
        pred,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = OUTPUT_DIR / "classification_report.csv"
    report_df.to_csv(report_path)

    # ---------------------------------------------------------
    # 8. Confusion matrix
    # ---------------------------------------------------------
    cm = confusion_matrix(
        te_y,
        pred
    )

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - DP Lightweight CNN")

    plt.tight_layout()

    cm_path = OUTPUT_DIR / "confusion_matrix.png"

    plt.savefig(
        cm_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------
    # 9. Save metrics
    # ---------------------------------------------------------
    metrics = {
        "accuracy": float(acc),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "parameters": int(model.count_params()),
        "test_images": int(len(te_y))
    }

    metrics_path = OUTPUT_DIR / "metrics.json"

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2
        )

    # ---------------------------------------------------------
    # 10. Print final results
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test Images        : {len(te_y)}")
    print(f"Accuracy            : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Weighted Precision  : {precision:.4f}")
    print(f"Weighted Recall     : {recall:.4f}")
    print(f"Weighted F1-score   : {f1:.4f}")
    print(f"Model Parameters    : {model.count_params():,}")

    print()
    print("Classification Report:")
    print(
        classification_report(
            te_y,
            pred,
            target_names=classes,
            zero_division=0
        )
    )

    print("=" * 60)

    print()
    print("Files generated:")
    print(f"Classification report : {report_path}")
    print(f"Confusion matrix      : {cm_path}")
    print(f"Metrics               : {metrics_path}")


if __name__ == "__main__":
    main()