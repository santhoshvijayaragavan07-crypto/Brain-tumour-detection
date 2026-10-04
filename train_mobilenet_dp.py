# ================================================================
# E3: MobileNetV2 + Differential Privacy using DP-SGD
# Privacy-Preserving Explainable Brain Tumor Detection
# ================================================================

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
    DPKerasSGDOptimizer,
)

warnings.filterwarnings("ignore")

# ================================================================
# 1. CONFIGURATION
# ================================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------
# Dataset path
# ------------------------------------------------

DATASET_ROOT = Path(
    r"C:\Users\dell\Downloads\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset"
)

TRAIN_DIR = DATASET_ROOT / "Train"
TEST_DIR = DATASET_ROOT / "Test"

# ------------------------------------------------
# Output directories
# ------------------------------------------------

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs") / "e3_mobilenetv2_dp"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------
# Image / training configuration
# ------------------------------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

VALIDATION_FRACTION = 0.10

# Maximum number of epochs.
# EarlyStopping may stop before this.
EPOCHS = 20

# ------------------------------------------------
# Differential Privacy configuration
# ------------------------------------------------

NOISE_MULTIPLIER = 1.0
L2_NORM_CLIP = 1.0

# Privacy delta
DELTA = 1e-5

# ------------------------------------------------
# DP-SGD microbatch configuration
# ------------------------------------------------

# IMPORTANT:
# TensorFlow Privacy requires the number of microbatches
# to divide the batch size.
#
# Using 8 microbatches with batch size 32:
#
# 32 / 8 = 4 examples per microbatch

NUM_MICROBATCHES = 8

# ------------------------------------------------
# Learning rates
# ------------------------------------------------

# DP-SGD generally requires a smaller learning rate
# than the non-private E2 experiment.

STAGE1_LR = 0.005
STAGE2_LR = 0.0005

# ================================================================
# 2. CLASS DEFINITIONS
# ================================================================

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary",
]

CLASS_TO_INDEX = {
    "glioma": 0,
    "meningioma": 1,
    "no_tumor": 2,
    "pituitary": 3,
}

# Dataset folder aliases
CLASS_ALIASES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "notumor": "no_tumor",
    "no_tumor": "no_tumor",
    "no tumor": "no_tumor",
    "no_tumour": "no_tumor",
    "no tumour": "no_tumor",
}


# ================================================================
# 3. PRINT CONFIGURATION
# ================================================================

print("=" * 70)
print("E3: MobileNetV2 + Differential Privacy using DP-SGD")
print("=" * 70)

print()
print("TensorFlow version :", tf.__version__)
print()

print("Dataset")
print("-" * 70)
print("Train directory :", TRAIN_DIR)
print("Test directory  :", TEST_DIR)
print()

print("Image size      :", IMG_SIZE)
print("Batch size      :", BATCH_SIZE)
print("Validation frac :", VALIDATION_FRACTION)
print("Epochs          :", EPOCHS)
print()

print("Differential Privacy")
print("-" * 70)
print("Noise multiplier :", NOISE_MULTIPLIER)
print("L2 norm clip     :", L2_NORM_CLIP)
print("Microbatches     :", NUM_MICROBATCHES)
print("Delta            :", DELTA)
print()

print("Classes")
print("-" * 70)

for i, cls in enumerate(CLASS_NAMES):
    print(i, "->", cls)

print()


# ================================================================
# 4. CHECK DATASET DIRECTORIES
# ================================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )


# ================================================================
# 5. COLLECT IMAGE PATHS
# ================================================================

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
}


def collect_images(directory):
    """
    Collect image paths from class folders.
    """

    image_paths = []
    labels = []

    for folder_name in sorted(directory.iterdir()):

        if not folder_name.is_dir():
            continue

        raw_class = folder_name.name.lower().strip()

        if raw_class not in CLASS_ALIASES:
            print(
                f"WARNING: Unknown folder skipped: {folder_name.name}"
            )
            continue

        canonical_class = CLASS_ALIASES[raw_class]

        label = CLASS_TO_INDEX[canonical_class]

        for image_path in folder_name.rglob("*"):

            if image_path.is_file() and image_path.suffix in VALID_EXTENSIONS:

                image_paths.append(str(image_path))
                labels.append(label)

    return np.array(image_paths), np.array(labels, dtype=np.int32)


# ================================================================
# 6. LOAD OFFICIAL TRAIN AND TEST
# ================================================================

train_paths, train_labels = collect_images(TRAIN_DIR)
test_paths, test_labels = collect_images(TEST_DIR)

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print()

print("Official Train images :", len(train_paths))
print("Official Test images  :", len(test_paths))

print()

print("Training class distribution")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(train_labels == i)

    print(
        f"{class_name:12s}: {count}"
    )

print()

print("Test class distribution")

for i, class_name in enumerate(CLASS_NAMES):

    count = np.sum(test_labels == i)

    print(
        f"{class_name:12s}: {count}"
    )

print()


# ================================================================
# 7. STRATIFIED TRAIN / VALIDATION SPLIT
# ================================================================

from sklearn.model_selection import train_test_split

actual_train_paths, val_paths, actual_train_labels, val_labels = (
    train_test_split(
        train_paths,
        train_labels,
        test_size=VALIDATION_FRACTION,
        random_state=SEED,
        stratify=train_labels,
    )
)

print("=" * 70)
print("OFFICIAL SPLIT")
print("=" * 70)

print()
print("Official Train :", len(train_paths))
print("Actual Train   :", len(actual_train_paths))
print("Validation     :", len(val_paths))
print("Official Test  :", len(test_paths))
print()


# ================================================================
# 8. IMAGE LOADING FUNCTION
# ================================================================

def load_image(path, label):
    """
    Load image and resize to 224x224 RGB.
    Output range: [0, 1]
    """

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMG_SIZE,
        method=tf.image.ResizeMethod.BILINEAR,
    )

    image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(label, tf.int32)

    return image, label


# ================================================================
# 9. DATA AUGMENTATION
# ================================================================

augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip(
            mode="horizontal"
        ),

        layers.RandomRotation(
            factor=0.03
        ),

        layers.RandomZoom(
            height_factor=0.08,
            width_factor=0.08,
        ),

        layers.RandomContrast(
            factor=0.10
        ),
    ],
    name="data_augmentation",
)


# ================================================================
# 10. CREATE TF.DATA DATASETS
# ================================================================

AUTOTUNE = tf.data.AUTOTUNE


def create_dataset(
    paths,
    labels,
    training=False,
):
    """
    Create tf.data pipeline.
    """

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            labels,
        )
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE,
    )

    if training:

        dataset = dataset.map(
            lambda image, label: (
                augmentation(image, training=True),
                label,
            ),
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=training,
    )

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_ds = create_dataset(
    actual_train_paths,
    actual_train_labels,
    training=True,
)

val_ds = create_dataset(
    val_paths,
    val_labels,
    training=False,
)

test_ds = create_dataset(
    test_paths,
    test_labels,
    training=False,
)


# ================================================================
# 11. BUILD MOBILENETV2
# ================================================================

print("=" * 70)
print("BUILDING MOBILENETV2")
print("=" * 70)

print()


# ------------------------------------------------
# ImageNet MobileNetV2 backbone
# ------------------------------------------------

base_model = MobileNetV2(
    input_shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3,
    ),
    include_top=False,
    weights="imagenet",
)

# Start frozen
base_model.trainable = False


# ------------------------------------------------
# Input
# ------------------------------------------------

inputs = layers.Input(
    shape=(
        IMG_SIZE[0],
        IMG_SIZE[1],
        3,
    ),
    name="input_image",
)


# ------------------------------------------------
# Convert [0,1] -> [-1,1]
# MobileNetV2 preprocessing
# ------------------------------------------------

x = layers.Rescaling(
    scale=2.0,
    offset=-1.0,
    name="mobilenetv2_rescaling",
)(inputs)


# ------------------------------------------------
# Backbone
# ------------------------------------------------

x = base_model(
    x,
    training=False,
)


# ------------------------------------------------
# Classification head
# ------------------------------------------------

x = layers.GlobalAveragePooling2D(
    name="global_average_pooling",
)(x)

x = layers.Dropout(
    0.30,
    name="dropout_1",
)(x)

x = layers.Dense(
    128,
    activation="relu",
    name="dense_features",
)(x)

x = layers.Dropout(
    0.20,
    name="dropout_2",
)(x)

outputs = layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
    name="predictions",
)(x)


model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="MobileNetV2_DP",
)


print()
model.summary()
print()


# ================================================================
# 12. DP LOSS
# ================================================================

# IMPORTANT:
#
# DP-SGD requires per-example losses.
#
# Therefore reduction must be NONE.

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction=tf.keras.losses.Reduction.NONE,
)


# ================================================================
# 13. CREATE DP OPTIMIZER
# ================================================================

print("=" * 70)
print("CREATING DP-SGD OPTIMIZER")
print("=" * 70)

print()

dp_optimizer = DPKerasSGDOptimizer(
    l2_norm_clip=L2_NORM_CLIP,
    noise_multiplier=NOISE_MULTIPLIER,
    num_microbatches=NUM_MICROBATCHES,
    learning_rate=STAGE1_LR,
)

print("DP optimizer created successfully.")
print()


# ================================================================
# 14. COMPILE MODEL
# ================================================================

model.compile(
    optimizer=dp_optimizer,
    loss=loss_fn,
    metrics=[
        "accuracy",
    ],
)


# ================================================================
# 15. CALLBACKS
# ================================================================

best_model_path = OUTPUT_DIR / "mobilenetv2_dp_best.keras"

callbacks = [
    ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]


# ================================================================
# 16. STAGE 1 — DP TRAINING
# ================================================================

print("=" * 70)
print("STAGE 1: MOBILE NETV2 + DP-SGD")
print("=" * 70)

print()
print("Backbone frozen")
print("Learning rate:", STAGE1_LR)
print("Noise multiplier:", NOISE_MULTIPLIER)
print()

start_time = time.time()

history_stage1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

stage1_time = time.time() - start_time

print()
print("Stage 1 training time:",
      round(stage1_time / 60, 2),
      "minutes")


# ================================================================
# 17. SAVE STAGE 1 HISTORY
# ================================================================

history_df = pd.DataFrame(
    history_stage1.history
)

history_df.to_csv(
    OUTPUT_DIR / "e3_stage1_history.csv",
    index=False,
)


# ================================================================
# 18. LOAD BEST MODEL WEIGHTS
# ================================================================

if best_model_path.exists():

    print()
    print("Loading best validation model...")

    # Loading with compile=False avoids DP optimizer
    # serialization problems.

    best_loaded = tf.keras.models.load_model(
        str(best_model_path),
        compile=False,
    )

    model.set_weights(
        best_loaded.get_weights()
    )

    del best_loaded


# ================================================================
# 19. STAGE 2 — CONTROLLED FINE-TUNING
# ================================================================

print()
print("=" * 70)
print("STAGE 2: DP FINE-TUNING")
print("=" * 70)

print()
print("Unfreezing approximately last 30% of MobileNetV2")
print("BatchNormalization layers remain frozen.")
print()


# ------------------------------------------------
# Determine fine-tuning point
# ------------------------------------------------

total_layers = len(base_model.layers)

fine_tune_from = int(
    total_layers * 0.70
)

for layer_index, layer in enumerate(base_model.layers):

    if layer_index >= fine_tune_from:

        # Keep BatchNorm frozen.
        if isinstance(layer, layers.BatchNormalization):

            layer.trainable = False

        else:

            layer.trainable = True

    else:

        layer.trainable = False


trainable_backbone = sum(
    layer.trainable
    for layer in base_model.layers
)

print(
    "Total MobileNetV2 layers:",
    total_layers,
)

print(
    "Trainable backbone layers:",
    trainable_backbone,
)

print()


# ------------------------------------------------
# New DP optimizer for fine-tuning
# ------------------------------------------------

dp_optimizer_stage2 = DPKerasSGDOptimizer(
    l2_norm_clip=L2_NORM_CLIP,
    noise_multiplier=NOISE_MULTIPLIER,
    num_microbatches=NUM_MICROBATCHES,
    learning_rate=STAGE2_LR,
)


# ------------------------------------------------
# Compile again
# ------------------------------------------------

model.compile(
    optimizer=dp_optimizer_stage2,
    loss=loss_fn,
    metrics=[
        "accuracy",
    ],
)


# ------------------------------------------------
# Stage 2 callbacks
# ------------------------------------------------

best_model_stage2_path = (
    OUTPUT_DIR /
    "mobilenetv2_dp_finetuned_best.keras"
)


callbacks_stage2 = [

    ModelCheckpoint(
        filepath=str(best_model_stage2_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]


# ------------------------------------------------
# Train Stage 2
# ------------------------------------------------

start_time = time.time()

history_stage2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks_stage2,
    verbose=1,
)

stage2_time = time.time() - start_time

print()
print(
    "Stage 2 training time:",
    round(stage2_time / 60, 2),
    "minutes",
)


# ================================================================
# 20. SAVE STAGE 2 HISTORY
# ================================================================

history_stage2_df = pd.DataFrame(
    history_stage2.history
)

history_stage2_df.to_csv(
    OUTPUT_DIR / "e3_stage2_history.csv",
    index=False,
)


# ================================================================
# 21. LOAD BEST STAGE 2 MODEL
# ================================================================

if best_model_stage2_path.exists():

    print()
    print("Loading best fine-tuned model...")

    best_loaded = tf.keras.models.load_model(
        str(best_model_stage2_path),
        compile=False,
    )

    model.set_weights(
        best_loaded.get_weights()
    )

    del best_loaded


# ================================================================
# 22. SAVE FINAL MODEL
# ================================================================

final_model_path = (
    OUTPUT_DIR /
    "mobilenetv2_dp_final.keras"
)

model.save(
    str(final_model_path)
)

print()
print("Final model saved:")
print(final_model_path)


# ================================================================
# 23. FINAL OFFICIAL TEST EVALUATION
# ================================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

print()
print("IMPORTANT:")
print("Official test set is being used ONLY for final evaluation.")
print()


# ------------------------------------------------
# Predictions
# ------------------------------------------------

probabilities = model.predict(
    test_ds,
    verbose=1,
)

predicted_labels = np.argmax(
    probabilities,
    axis=1,
)


# ------------------------------------------------
# Make sure prediction count matches test labels
# ------------------------------------------------

if len(predicted_labels) != len(test_labels):

    print()
    print(
        "WARNING: prediction count:",
        len(predicted_labels),
    )

    print(
        "Expected test labels:",
        len(test_labels),
    )

    # Because test dataset does not use drop_remainder,
    # counts should normally match.


# ================================================================
# 24. CALCULATE METRICS
# ================================================================

accuracy = np.mean(
    predicted_labels == test_labels
)

weighted_precision = precision_score(
    test_labels,
    predicted_labels,
    average="weighted",
    zero_division=0,
)

weighted_recall = recall_score(
    test_labels,
    predicted_labels,
    average="weighted",
    zero_division=0,
)

weighted_f1 = f1_score(
    test_labels,
    predicted_labels,
    average="weighted",
    zero_division=0,
)

macro_precision = precision_score(
    test_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    test_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    test_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)


# ================================================================
# 25. CLASSIFICATION REPORT
# ================================================================

report = classification_report(
    test_labels,
    predicted_labels,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(
    report
).transpose()

report_path = (
    OUTPUT_DIR /
    "e3_classification_report.csv"
)

report_df.to_csv(
    report_path
)


# ================================================================
# 26. CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    test_labels,
    predicted_labels,
)


cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

cm_path = (
    OUTPUT_DIR /
    "e3_confusion_matrix.csv"
)

cm_df.to_csv(
    cm_path
)


# ================================================================
# 27. SAVE METRICS
# ================================================================

metrics = {

    "experiment": "E3",

    "model": "MobileNetV2",

    "privacy": "Differential Privacy using DP-SGD",

    "dataset": {
        "official_train": int(len(train_paths)),
        "actual_train": int(len(actual_train_paths)),
        "validation": int(len(val_paths)),
        "official_test": int(len(test_paths)),
    },

    "image_size": [
        IMG_SIZE[0],
        IMG_SIZE[1],
    ],

    "batch_size": BATCH_SIZE,

    "noise_multiplier": NOISE_MULTIPLIER,

    "l2_norm_clip": L2_NORM_CLIP,

    "num_microbatches": NUM_MICROBATCHES,

    "delta": DELTA,

    "accuracy": float(accuracy),

    "weighted_precision": float(weighted_precision),

    "weighted_recall": float(weighted_recall),

    "weighted_f1": float(weighted_f1),

    "macro_precision": float(macro_precision),

    "macro_recall": float(macro_recall),

    "macro_f1": float(macro_f1),

    "model_parameters": int(
        model.count_params()
    ),

    "stage1_training_minutes": float(
        stage1_time / 60
    ),

    "stage2_training_minutes": float(
        stage2_time / 60
    ),
}


metrics_path = (
    OUTPUT_DIR /
    "e3_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


# ================================================================
# 28. PRINT FINAL RESULTS
# ================================================================

print()
print()
print("=" * 70)
print("FINAL E3 TEST RESULTS")
print("=" * 70)

print(
    f"Accuracy           : {accuracy * 100:.2f}%"
)

print(
    f"Weighted Precision : {weighted_precision * 100:.2f}%"
)

print(
    f"Weighted Recall    : {weighted_recall * 100:.2f}%"
)

print(
    f"Weighted F1        : {weighted_f1 * 100:.2f}%"
)

print(
    f"Macro Precision    : {macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall       : {macro_recall * 100:.2f}%"
)

print(
    f"Macro F1           : {macro_f1 * 100:.2f}%"
)

print()
print("-" * 70)

print(
    "Noise multiplier   :",
    NOISE_MULTIPLIER,
)

print(
    "L2 norm clip       :",
    L2_NORM_CLIP,
)

print(
    "Delta              :",
    DELTA,
)

print(
    "Microbatches       :",
    NUM_MICROBATCHES,
)

print()

print("Classification report:")
print(report_path)

print()

print("Confusion matrix:")
print(cm_path)

print()

print("Metrics:")
print(metrics_path)

print()


# ================================================================
# 29. PRIVACY ACCOUNTING NOTICE
# ================================================================

print("=" * 70)
print("PRIVACY ACCOUNTING")
print("=" * 70)

print()
print(
    "DP-SGD was configured with:"
)

print(
    f"  Noise multiplier = {NOISE_MULTIPLIER}"
)

print(
    f"  L2 norm clip     = {L2_NORM_CLIP}"
)

print(
    f"  Batch size       = {BATCH_SIZE}"
)

print(
    f"  Microbatches     = {NUM_MICROBATCHES}"
)

print(
    f"  Delta            = {DELTA}"
)

print()

print(
    "IMPORTANT:"
)

print(
    "Epsilon must be computed using a privacy accountant"
)

print(
    "that matches the actual DP-SGD sampling and"
)

print(
    "microbatching configuration."
)

print(
    "Do NOT report an epsilon value until it has been"
)

print(
    "verified from the accountant."
)

print()


# ================================================================
# 30. COMPLETE
# ================================================================

print("=" * 70)
print("E3 COMPLETE")
print("=" * 70)

print()
print(
    f"MobileNetV2 + DP-SGD Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"MobileNetV2 + DP-SGD Weighted F1: "
    f"{weighted_f1 * 100:.2f}%"
)

print()
print(
    "Official test set was used only for final evaluation."
)

print(
    "Use validation accuracy for further model tuning."
)

print("=" * 70)