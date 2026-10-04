import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import *
from src.data import make_official_splits, make_dataset
from src.model import build_lightweight_cnn


def dp_optimizer():
    try:
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
            DPKerasSGDOptimizer
        )
    except Exception as e:
        raise RuntimeError(
            "TensorFlow Privacy import failed. Install a compatible "
            "tensorflow-privacy version for your TensorFlow installation."
        ) from e

    return DPKerasSGDOptimizer(
        l2_norm_clip=L2_NORM_CLIP,
        noise_multiplier=NOISE_MULTIPLIER,

        # IMPORTANT:
        # Batch size = 32
        # Number of microbatches = 8
        # Therefore each microbatch contains 4 samples.
        num_microbatches=8,

        learning_rate=0.01
    )


def main():

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # Dataset split
    # ---------------------------------------------------------
    (
        tr_p,
        tr_y
    ), (
        va_p,
        va_y
    ), (
        te_p,
        te_y
    ), classes = make_official_splits(
        TRAIN_DIR,
        TEST_DIR,
        VALIDATION_FRACTION,
        SEED
    )

    print("Official Train:", len(tr_p) + len(va_p))
    print("Actual Train:", len(tr_p))
    print("Validation:", len(va_p))
    print("Untouched Official Test:", len(te_p))
    print("Classes:", classes)

    # ---------------------------------------------------------
    # TensorFlow datasets
    # ---------------------------------------------------------
    train_ds = make_dataset(
        tr_p,
        tr_y,
        True,
        BATCH_SIZE
    )

    val_ds = make_dataset(
        va_p,
        va_y,
        False,
        BATCH_SIZE
    )

    # ---------------------------------------------------------
    # Build lightweight CNN
    # ---------------------------------------------------------
    model = build_lightweight_cnn(len(classes))

    # ---------------------------------------------------------
    # Compile model with DP-SGD
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # TensorFlow Privacy requires the loss to return
    # one loss value per example.
    #
    # Therefore reduction=NONE is required.
    #
    model.compile(
        optimizer=dp_optimizer(),

        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
            reduction=tf.keras.losses.Reduction.NONE
        ),

        metrics=["accuracy"]
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,

        callbacks=[

            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True
            ),

            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-5
            ),

            tf.keras.callbacks.ModelCheckpoint(
                str(
                    MODEL_DIR /
                    "lightweight_brain_tumor_dp.keras"
                ),
                monitor="val_accuracy",
                save_best_only=True
            )
        ]
    )

    # ---------------------------------------------------------
    # Save training history
    # ---------------------------------------------------------
    pd.DataFrame(
        history.history
    ).to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Save class names
    # ---------------------------------------------------------
    with open(
        MODEL_DIR / "class_names.json",
        "w"
    ) as f:
        json.dump(
            classes,
            f,
            indent=2
        )

    # ---------------------------------------------------------
    # Differential Privacy accounting
    # ---------------------------------------------------------
    epsilon = None

    try:

        from tensorflow_privacy.privacy.analysis import (
            compute_dp_sgd_privacy
        )

        epsilon, _ = compute_dp_sgd_privacy(
            n=len(tr_p),
            batch_size=BATCH_SIZE,
            noise_multiplier=NOISE_MULTIPLIER,
            epochs=len(history.history["loss"]),
            delta=DELTA
        )

        print(
            f"Approximate epsilon={epsilon:.4f}, "
            f"delta={DELTA}"
        )

    except Exception as e:

        print(
            "Privacy accountant unavailable:",
            e
        )

    # ---------------------------------------------------------
    # Save privacy metadata
    # ---------------------------------------------------------
    meta = {

        "mechanism": "DP-SGD",

        "official_split":
            "80% Train / 20% Test preserved",

        "train_split":
            "90% of official Train",

        "validation_split":
            "10% of official Train",

        "test_split":
            "100% official Test, untouched",

        "batch_size":
            BATCH_SIZE,

        "num_microbatches":
            8,

        "noise_multiplier":
            NOISE_MULTIPLIER,

        "l2_norm_clip":
            L2_NORM_CLIP,

        "delta":
            DELTA,

        "epsilon":
            epsilon,

        "training_images":
            len(tr_p),

        "validation_images":
            len(va_p),

        "test_images":
            len(te_p)
    }

    with open(
        MODEL_DIR / "privacy_metadata.json",
        "w"
    ) as f:

        json.dump(
            meta,
            f,
            indent=2
        )

    # ---------------------------------------------------------
    # Final message
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print("Model saved to:")
    print(
        MODEL_DIR /
        "lightweight_brain_tumor_dp.keras"
    )
    print()
    print("Training history saved to:")
    print(
        OUTPUT_DIR /
        "training_history.csv"
    )
    print()
    print("Privacy metadata saved to:")
    print(
        MODEL_DIR /
        "privacy_metadata.json"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()