import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

from config import (
    TRAIN_DIR,
    TEST_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    SEED,
    VALIDATION_FRACTION,
)

from src.data import make_official_splits


# ============================================================
# 1. CONFIGURATION
# ============================================================

STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 20

STAGE1_LR = 1e-3
STAGE2_LR = 1e-5

# Fine-tune last 30% of MobileNetV2
FINE_TUNE_FRACTION = 0.30

MODEL_DIR = Path(MODEL_DIR)
OUTPUT_DIR = Path(OUTPUT_DIR)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = MODEL_DIR / "mobilenetv2_best.keras"
FINAL_MODEL = MODEL_DIR / "mobilenetv2_final.keras"

HISTORY_FILE = OUTPUT_DIR / "mobilenetv2_history.csv"
METRICS_FILE = OUTPUT_DIR / "mobilenetv2_metrics.json"
REPORT_FILE = OUTPUT_DIR / "mobilenetv2_classification_report.csv"
CM_FILE = OUTPUT_DIR / "mobilenetv2_confusion_matrix.png"
CURVE_FILE = OUTPUT_DIR / "mobilenetv2_training_curves.png"


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 3. PRINT ENVIRONMENT
# ============================================================

print("=" * 70)
print("MOBILENETV2 BRAIN TUMOR CLASSIFICATION")
print("=" * 70)

print("TensorFlow :", tf.__version__)
print("Train      :", TRAIN_DIR)
print("Test       :", TEST_DIR)
print("Image size :", IMG_SIZE)
print("Batch size :", BATCH_SIZE)

print("=" * 70)


# ============================================================
# 4. LOAD OFFICIAL DATASET SPLIT
# ============================================================

(
    (train_paths, train_labels),
    (val_paths, val_labels),
    (test_paths, test_labels),
    classes,
) = make_official_splits(
    TRAIN_DIR,
    TEST_DIR,
    validation_fraction=VALIDATION_FRACTION,
    seed=SEED,
)

print("\nDATASET")
print("-" * 70)

print(
    "Official Train :",
    len(train_paths) + len(val_paths)
)

print(
    "Actual Train   :",
    len(train_paths)
)

print(
    "Validation     :",
    len(val_paths)
)

print(
    "Official Test  :",
    len(test_paths)
)

print(
    "Classes        :",
    classes
)


# ============================================================
# 5. CLASS DISTRIBUTION
# ============================================================

print("\nCLASS DISTRIBUTION")
print("-" * 70)

for i, class_name in enumerate(classes):

    train_count = sum(
        np.array(train_labels) == i
    )

    val_count = sum(
        np.array(val_labels) == i
    )

    test_count = sum(
        np.array(test_labels) == i
    )

    print(
        f"{class_name:12s} "
        f"Train={train_count:4d} "
        f"Val={val_count:4d} "
        f"Test={test_count:4d}"
    )


# ============================================================
# 6. DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMG_SIZE,
        method=tf.image.ResizeMethod.BILINEAR
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, label


def make_dataset(
    paths,
    labels,
    training=False
):

    paths = np.array(paths)

    labels = np.array(
        labels,
        dtype=np.int32
    )

    ds = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:

        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    ds = ds.batch(
        BATCH_SIZE
    )

    ds = ds.prefetch(
        AUTOTUNE
    )

    return ds


train_ds = make_dataset(
    train_paths,
    train_labels,
    training=True
)

val_ds = make_dataset(
    val_paths,
    val_labels,
    training=False
)

test_ds = make_dataset(
    test_paths,
    test_labels,
    training=False
)


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [

        tf.keras.layers.RandomFlip(
            mode="horizontal"
        ),

        tf.keras.layers.RandomRotation(
            factor=0.03
        ),

        tf.keras.layers.RandomZoom(
            height_factor=0.08,
            width_factor=0.08
        ),

        tf.keras.layers.RandomContrast(
            factor=0.10
        ),

    ],
    name="mri_augmentation"
)


# ============================================================
# 8. LOAD MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2 ImageNet weights...")

base_model = tf.keras.applications.MobileNetV2(

    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),

    include_top=False,

    weights="imagenet"
)

# Stage 1: freeze backbone
base_model.trainable = False


# ============================================================
# 9. BUILD MODEL
# ============================================================

inputs = tf.keras.Input(

    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3
    ),

    name="mri_input"
)


# MRI augmentation
x = data_augmentation(inputs)


# Dataset is [0,1].
# MobileNetV2 expects [-1,1].
x = tf.keras.layers.Rescaling(

    scale=2.0,

    offset=-1.0,

    name="mobilenet_preprocessing"

)(x)


# MobileNetV2
x = base_model(
    x,
    training=False
)


# Global average pooling
x = tf.keras.layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)


# Dropout
x = tf.keras.layers.Dropout(
    0.30,
    name="dropout_1"
)(x)


# Dense layer
x = tf.keras.layers.Dense(
    128,
    activation="relu",
    name="dense_features"
)(x)


# Dropout
x = tf.keras.layers.Dropout(
    0.20,
    name="dropout_2"
)(x)


# Four-class classifier
outputs = tf.keras.layers.Dense(

    len(classes),

    activation="softmax",

    name="classifier"

)(x)


model = tf.keras.Model(

    inputs=inputs,

    outputs=outputs,

    name="MobileNetV2_BrainTumor"

)


# ============================================================
# 10. MODEL SUMMARY
# ============================================================

print("\nMODEL")
print("-" * 70)

model.summary()


# ============================================================
# 11. STAGE 1
#     FROZEN MOBILENETV2
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 1 — FROZEN MOBILENETV2")
print("=" * 70)


model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=STAGE1_LR

    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[

        tf.keras.metrics.SparseCategoricalAccuracy(

            name="accuracy"

        )

    ]

)


callbacks_stage1 = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=str(BEST_MODEL),

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1

    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=4,

        restore_best_weights=True,

        verbose=1

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=2,

        min_lr=1e-6,

        verbose=1

    ),

]


history1 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=STAGE1_EPOCHS,

    callbacks=callbacks_stage1,

    verbose=1

)


# ============================================================
# 12. LOAD BEST STAGE 1 MODEL
# ============================================================

print("\nLoading best Stage 1 model...")

model = tf.keras.models.load_model(

    str(BEST_MODEL),

    compile=False

)


# ============================================================
# 13. FIND MOBILENETV2 BACKBONE
# ============================================================

print("\nSearching for MobileNetV2 backbone...")

base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ):

        if "mobilenet" in layer.name.lower():

            base_model = layer

            break


if base_model is None:

    raise RuntimeError(
        "MobileNetV2 backbone could not be found."
    )


print(
    "MobileNetV2 backbone found:",
    base_model.name
)

print(
    "Total backbone layers:",
    len(base_model.layers)
)


# ============================================================
# 14. STAGE 2
#     FINE-TUNING
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2 — MOBILENETV2 FINE-TUNING")
print("=" * 70)


base_model.trainable = True


fine_tune_from = int(

    len(base_model.layers)
    * (1.0 - FINE_TUNE_FRACTION)

)


print(
    "Fine-tuning from layer:",
    fine_tune_from
)


for i, layer in enumerate(
    base_model.layers
):

    if i < fine_tune_from:

        layer.trainable = False

    else:

        # Batch normalization remains frozen
        # to improve fine-tuning stability.

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = False

        else:

            layer.trainable = True


trainable_count = sum(

    1
    for layer in base_model.layers
    if layer.trainable

)


print(
    "Trainable backbone layers:",
    trainable_count
)


# ============================================================
# 15. COMPILE FINE-TUNING MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=STAGE2_LR

    ),

    loss=tf.keras.losses.SparseCategoricalCrossentropy(),

    metrics=[

        tf.keras.metrics.SparseCategoricalAccuracy(

            name="accuracy"

        )

    ]

)


# ============================================================
# 16. FINE-TUNING CALLBACKS
# ============================================================

callbacks_stage2 = [

    tf.keras.callbacks.ModelCheckpoint(

        filepath=str(FINAL_MODEL),

        monitor="val_accuracy",

        mode="max",

        save_best_only=True,

        verbose=1

    ),

    tf.keras.callbacks.EarlyStopping(

        monitor="val_accuracy",

        mode="max",

        patience=5,

        restore_best_weights=True,

        verbose=1

    ),

    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.3,

        patience=2,

        min_lr=1e-7,

        verbose=1

    ),

]


# ============================================================
# 17. RUN FINE-TUNING
# ============================================================

history2 = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=STAGE2_EPOCHS,

    callbacks=callbacks_stage2,

    verbose=1

)


# ============================================================
# 18. LOAD FINAL BEST MODEL
# ============================================================

if FINAL_MODEL.exists():

    print("\nLoading best fine-tuned model...")

    model = tf.keras.models.load_model(

        str(FINAL_MODEL),

        compile=False

    )

else:

    print(
        "\nFine-tuning checkpoint not found."
    )

    model.save(
        str(FINAL_MODEL)
    )


print(
    "\nFinal model:",
    FINAL_MODEL
)


# ============================================================
# 19. COMBINE TRAINING HISTORY
# ============================================================

history = {}

all_keys = set(
    history1.history.keys()
).union(
    history2.history.keys()
)


for key in all_keys:

    history[key] = (

        history1.history.get(
            key,
            []
        )

        +

        history2.history.get(
            key,
            []
        )

    )


history_df = pd.DataFrame(
    history
)


history_df.to_csv(

    str(HISTORY_FILE),

    index=False

)


print(
    "\nTraining history saved:",
    HISTORY_FILE
)


# ============================================================
# 20. TRAINING CURVES
# ============================================================

if "accuracy" in history:

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "MobileNetV2 Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(

        str(CURVE_FILE),

        dpi=200,

        bbox_inches="tight"

    )

    plt.close()


# ============================================================
# 21. FINAL OFFICIAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL EVALUATION — OFFICIAL TEST SET")
print("=" * 70)

print(
    "Test images:",
    len(test_paths)
)


pred_prob = model.predict(

    test_ds,

    verbose=1

)


pred_labels = np.argmax(

    pred_prob,

    axis=1

)


true_labels = np.array(

    test_labels,

    dtype=np.int32

)


# ============================================================
# 22. METRICS
# ============================================================

accuracy = accuracy_score(

    true_labels,

    pred_labels

)


precision, recall, f1, _ = (

    precision_recall_fscore_support(

        true_labels,

        pred_labels,

        average="weighted",

        zero_division=0

    )

)


macro_precision, macro_recall, macro_f1, _ = (

    precision_recall_fscore_support(

        true_labels,

        pred_labels,

        average="macro",

        zero_division=0

    )

)


print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)


print(
    f"Accuracy           : {accuracy * 100:.2f}%"
)

print(
    f"Weighted Precision : {precision * 100:.2f}%"
)

print(
    f"Weighted Recall    : {recall * 100:.2f}%"
)

print(
    f"Weighted F1       : {f1 * 100:.2f}%"
)

print(
    f"Macro Precision   : {macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall      : {macro_recall * 100:.2f}%"
)

print(
    f"Macro F1          : {macro_f1 * 100:.2f}%"
)


# ============================================================
# 23. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    true_labels,

    pred_labels,

    target_names=classes,

    output_dict=True,

    zero_division=0

)


report_df = pd.DataFrame(
    report
).transpose()


report_df.to_csv(

    str(REPORT_FILE)

)


print(
    "\nClassification report saved:",
    REPORT_FILE
)


# ============================================================
# 24. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    true_labels,

    pred_labels

)


plt.figure(

    figsize=(8, 7)

)


plt.imshow(cm)


plt.title(

    "MobileNetV2 Confusion Matrix"

)


plt.xlabel(

    "Predicted Class"

)


plt.ylabel(

    "True Class"

)


plt.xticks(

    range(len(classes)),

    classes,

    rotation=45

)


plt.yticks(

    range(len(classes)),

    classes

)


for i in range(len(classes)):

    for j in range(len(classes)):

        plt.text(

            j,

            i,

            cm[i, j],

            ha="center",

            va="center"

        )


plt.tight_layout()


plt.savefig(

    str(CM_FILE),

    dpi=200,

    bbox_inches="tight"

)


plt.close()


print(
    "Confusion matrix saved:",
    CM_FILE
)


# ============================================================
# 25. SAVE METRICS JSON
# ============================================================

metrics = {

    "experiment":
        "E2_MobileNetV2_non_private",

    "architecture":
        "MobileNetV2",

    "input_size":
        list(IMG_SIZE),

    "official_train":
        int(
            len(train_paths) + len(val_paths)
        ),

    "actual_train":
        int(
            len(train_paths)
        ),

    "validation":
        int(
            len(val_paths)
        ),

    "official_test":
        int(
            len(test_paths)
        ),

    "classes":
        classes,

    "accuracy":
        float(
            accuracy
        ),

    "weighted_precision":
        float(
            precision
        ),

    "weighted_recall":
        float(
            recall
        ),

    "weighted_f1":
        float(
            f1
        ),

    "macro_precision":
        float(
            macro_precision
        ),

    "macro_recall":
        float(
            macro_recall
        ),

    "macro_f1":
        float(
            macro_f1
        ),

    "privacy":
        {

            "differential_privacy":
                False,

            "epsilon":
                None,

            "delta":
                None

        }

}


with open(

    str(METRICS_FILE),

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        metrics,

        f,

        indent=4

    )


print(
    "Metrics saved:",
    METRICS_FILE
)


# ============================================================
# 26. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("E2 COMPLETE")
print("=" * 70)

print(
    f"MobileNetV2 Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"MobileNetV2 Weighted F1: "
    f"{f1 * 100:.2f}%"
)

print(
    "\nPrevious E1 Accuracy: 54.72%"
)

print(
    f"Current E2 Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print("\nIMPORTANT:")
print(
    "The official test set was used only for final evaluation."
)

print(
    "Use validation accuracy for further model tuning."
)

print("=" * 70)