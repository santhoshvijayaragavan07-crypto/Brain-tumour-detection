import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# ================================================================
# PROJECT PATH
# ================================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import *
from src.data import make_official_splits


# ================================================================
# SETTINGS
# ================================================================

MODEL_PATH = MODEL_DIR / "lightweight_brain_tumor_dp.keras"

TARGET_LAYER_NAME = "pw_4"

OUTPUT_ROOT = OUTPUT_DIR / "gradcam_comparison"
INDIVIDUAL_DIR = OUTPUT_ROOT / "individual"
MONTAGE_DIR = OUTPUT_ROOT / "montages"

IMAGES_PER_CLASS = 2

# Heatmap settings
LOW_PERCENTILE = 5
HIGH_PERCENTILE = 98
SMOOTH_RADIUS = 2.0
OVERLAY_ALPHA = 0.42

# Minimum image intensity used to suppress pure black background
BACKGROUND_THRESHOLD = 0.05


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def ensure_directories():
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    MONTAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_image(image_path):
    """
    Load image and prepare it exactly in the same basic format
    used by the model: 224x224 RGB, values in [0,1].
    """

    img = Image.open(image_path).convert("RGB")
    original = np.array(img)

    resized = img.resize(IMG_SIZE)

    arr = np.asarray(resized).astype(np.float32) / 255.0

    return original, arr


def normalize_heatmap(cam):
    """
    Robust percentile normalization.

    This prevents a few extreme pixels from dominating the
    visualization and produces cleaner Grad-CAM maps.
    """

    cam = np.maximum(cam, 0)

    if np.max(cam) <= 0:
        return np.zeros_like(cam)

    low = np.percentile(cam, LOW_PERCENTILE)
    high = np.percentile(cam, HIGH_PERCENTILE)

    if high - low < 1e-8:
        cam = cam / (np.max(cam) + 1e-8)
    else:
        cam = np.clip((cam - low) / (high - low), 0, 1)

    return cam


def smooth_heatmap(cam):
    """
    Light Gaussian smoothing using PIL.
    """

    cam_uint8 = np.uint8(np.clip(cam, 0, 1) * 255)

    img = Image.fromarray(cam_uint8)
    img = img.filter(ImageFilter.GaussianBlur(radius=SMOOTH_RADIUS))

    return np.asarray(img).astype(np.float32) / 255.0


def apply_background_mask(cam, image):
    """
    Suppress activation in completely black image background.

    This does NOT claim to identify the tumor.
    It only removes obvious empty background pixels.
    """

    gray = np.mean(image, axis=2)

    threshold = BACKGROUND_THRESHOLD * np.max(gray)

    mask = gray > threshold

    cam = cam * mask.astype(np.float32)

    return cam


def prepare_heatmap(cam, image):
    """
    Complete heatmap cleanup pipeline.
    """

    cam = np.maximum(cam, 0)

    cam = normalize_heatmap(cam)

    cam = smooth_heatmap(cam)

    cam = normalize_heatmap(cam)

    cam = apply_background_mask(cam, image)

    cam = normalize_heatmap(cam)

    return cam


# ================================================================
# GRAD-CAM
# ================================================================

def compute_gradcam(model, image_tensor, target_layer, class_index):
    """
    Standard Grad-CAM.
    """

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            target_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(image_tensor)

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    # Global average pooling of gradients
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    cam = tf.reduce_sum(
        conv_outputs * weights[:, tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    cam = tf.nn.relu(cam)

    cam = cam[0].numpy()

    return cam


# ================================================================
# GRAD-CAM++
# ================================================================

def compute_gradcam_plus_plus(
    model,
    image_tensor,
    target_layer,
    class_index
):
    """
    Grad-CAM++ implementation.

    Uses first, second and third-order gradient terms
    to estimate pixel-level importance more precisely
    than standard Grad-CAM.
    """

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            target_layer.output,
            model.output
        ]
    )

    with tf.GradientTape(persistent=True) as tape3:

        tape3.watch(image_tensor)

        with tf.GradientTape(persistent=True) as tape2:

            tape2.watch(image_tensor)

            with tf.GradientTape() as tape1:

                conv_outputs, predictions = grad_model(
                    image_tensor
                )

                class_score = predictions[:, class_index]

            first_grad = tape1.gradient(
                class_score,
                conv_outputs
            )

        second_grad = tape2.gradient(
            first_grad,
            conv_outputs
        )

    third_grad = tape3.gradient(
        second_grad,
        conv_outputs
    )

    del tape2
    del tape3

    if second_grad is None:
        # Fallback
        return compute_gradcam(
            model,
            image_tensor,
            target_layer,
            class_index
        )

    if third_grad is None:
        third_grad = tf.zeros_like(second_grad)

    conv = conv_outputs[0]
    first = first_grad[0]
    second = second_grad[0]
    third = third_grad[0]

    # Sum over spatial dimensions
    global_sum = tf.reduce_sum(
        conv * third,
        axis=(0, 1),
        keepdims=True
    )

    denominator = (
        2.0 * second
        + global_sum
    )

    denominator = tf.where(
        tf.abs(denominator) < 1e-8,
        tf.ones_like(denominator) * 1e-8,
        denominator
    )

    alpha = second / denominator

    positive_first = tf.nn.relu(first)

    alpha = alpha * positive_first

    weights = tf.reduce_sum(
        alpha,
        axis=(0, 1)
    )

    cam = tf.reduce_sum(
        conv * weights,
        axis=-1
    )

    cam = tf.nn.relu(cam)

    return cam.numpy()


# ================================================================
# RESIZE HEATMAP
# ================================================================

def resize_heatmap(cam, target_size):
    """
    Resize CAM to original image size.
    """

    h, w = target_size[:2]

    cam_uint8 = np.uint8(
        np.clip(cam, 0, 1) * 255
    )

    cam_img = Image.fromarray(
        cam_uint8
    )

    cam_img = cam_img.resize(
        (w, h),
        Image.Resampling.BILINEAR
    )

    return np.asarray(
        cam_img
    ).astype(np.float32) / 255.0


# ================================================================
# SAVE VISUALIZATION
# ================================================================

def save_visualization(
    original_image,
    gradcam,
    gradcam_pp,
    true_class,
    predicted_class,
    confidence,
    image_path,
    output_path
):

    # Resize maps
    gradcam = resize_heatmap(
        gradcam,
        original_image.shape
    )

    gradcam_pp = resize_heatmap(
        gradcam_pp,
        original_image.shape
    )

    # Prepare maps
    gradcam = prepare_heatmap(
        gradcam,
        original_image / 255.0
    )

    gradcam_pp = prepare_heatmap(
        gradcam_pp,
        original_image / 255.0
    )

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4.8)
    )

    # Original
    axes[0].imshow(
        original_image,
        cmap="gray"
    )

    axes[0].set_title(
        f"Original MRI\nTrue: {true_class}",
        fontsize=12
    )

    axes[0].axis("off")

    # Grad-CAM
    axes[1].imshow(
        gradcam,
        cmap="jet"
    )

    axes[1].set_title(
        "Grad-CAM",
        fontsize=12
    )

    axes[1].axis("off")

    # Grad-CAM++
    axes[2].imshow(
        gradcam_pp,
        cmap="jet"
    )

    axes[2].set_title(
        "Grad-CAM++",
        fontsize=12
    )

    axes[2].axis("off")

    # Grad-CAM overlay
    axes[3].imshow(
        original_image
    )

    axes[3].imshow(
        gradcam,
        cmap="jet",
        alpha=OVERLAY_ALPHA
    )

    axes[3].set_title(
        "Grad-CAM Overlay",
        fontsize=12
    )

    axes[3].axis("off")

    # Grad-CAM++ overlay
    axes[4].imshow(
        original_image
    )

    axes[4].imshow(
        gradcam_pp,
        cmap="jet",
        alpha=OVERLAY_ALPHA
    )

    axes[4].set_title(
        "Grad-CAM++ Overlay",
        fontsize=12
    )

    axes[4].axis("off")

    # Main prediction title
    fig.suptitle(
        f"Predicted: {predicted_class} | "
        f"Confidence: {confidence * 100:.2f}%",
        fontsize=15,
        y=1.02
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# CREATE TWO-ROW MONTAGE
# ================================================================

def create_montage(image_paths, output_path, title):

    if not image_paths:
        return

    images = []

    for path in image_paths:

        img = Image.open(path).convert("RGB")

        images.append(img)

    width = 1000

    resized_images = []

    for img in images:

        ratio = width / img.width

        height = int(img.height * ratio)

        resized_images.append(
            img.resize(
                (width, height),
                Image.Resampling.LANCZOS
            )
        )

    total_height = sum(
        img.height
        for img in resized_images
    ) + 100

    canvas = Image.new(
        "RGB",
        (width, total_height),
        "white"
    )

    y = 100

    for img in resized_images:

        canvas.paste(
            img,
            (0, y)
        )

        y += img.height

    # Add title using matplotlib for clean text
    fig = plt.figure(
        figsize=(12, total_height / 100)
    )

    plt.imshow(canvas)

    plt.title(
        title,
        fontsize=16
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("       CLEAN GRAD-CAM + GRAD-CAM++ ANALYSIS")
    print("=" * 70)

    ensure_directories()

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------

    print()
    print("Model path:")
    print(MODEL_PATH)

    print()
    print("Loading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")
    print("Model parameters:", model.count_params())

    # ------------------------------------------------------------
    # Target layer
    # ------------------------------------------------------------

    print()
    print("Searching for target layer:", TARGET_LAYER_NAME)

    try:

        target_layer = model.get_layer(
            TARGET_LAYER_NAME
        )

    except ValueError:

        print()
        print("ERROR: Target layer not found.")
        print()
        print("Available convolution/depthwise layers:")

        for layer in model.layers:

            if (
                isinstance(
                    layer,
                    (
                        tf.keras.layers.Conv2D,
                        tf.keras.layers.DepthwiseConv2D
                    )
                )
            ):

                print(
                    layer.name,
                    layer.output.shape
                )

        raise

    print(
        "Target layer verified:",
        target_layer.name
    )

    # ------------------------------------------------------------
    # Load official test set
    # ------------------------------------------------------------

    print()
    print("Loading official test dataset...")

    _, _, (test_paths, test_labels), classes = (
        make_official_splits(
            TRAIN_DIR,
            TEST_DIR,
            VALIDATION_FRACTION,
            SEED
        )
    )

    print()
    print("Official Test Images:", len(test_labels))
    print("Classes:", classes)

    # ------------------------------------------------------------
    # Predict entire official test set
    # ------------------------------------------------------------

    print()
    print("Running predictions on official test set...")

    predictions = []

    for index, path in enumerate(test_paths):

        _, image = load_image(path)

        tensor = np.expand_dims(
            image,
            axis=0
        )

        probs = model.predict(
            tensor,
            verbose=0
        )[0]

        pred_index = int(
            np.argmax(probs)
        )

        confidence = float(
            probs[pred_index]
        )

        predictions.append({
            "path": path,
            "true_index": int(test_labels[index]),
            "true_class": classes[test_labels[index]],
            "pred_index": pred_index,
            "predicted_class": classes[pred_index],
            "confidence": confidence
        })

    print("Prediction completed.")

    df = pd.DataFrame(predictions)

    # ------------------------------------------------------------
    # Automatic selection
    # ------------------------------------------------------------

    selected = []

    print()
    print("=" * 70)
    print("AUTOMATIC IMAGE SELECTION")
    print("=" * 70)

    for class_name in classes:

        class_df = df[
            df["true_class"] == class_name
        ]

        # Correct predictions
        correct_df = class_df[
            class_df["true_class"]
            == class_df["predicted_class"]
        ].sort_values(
            "confidence",
            ascending=False
        )

        # Incorrect predictions
        incorrect_df = class_df[
            class_df["true_class"]
            != class_df["predicted_class"]
        ].sort_values(
            "confidence",
            ascending=False
        )

        print()
        print("Class:", class_name)

        # Best correct examples
        correct_selected = correct_df.head(
            IMAGES_PER_CLASS
        )

        print(
            "  Correct examples:",
            len(correct_selected)
        )

        for _, row in correct_selected.iterrows():

            row = row.to_dict()

            row["selection_type"] = "correct"

            selected.append(row)

            print(
                f"    Correct -> "
                f"{row['predicted_class']} "
                f"({row['confidence'] * 100:.2f}%)"
            )

        # Highest-confidence incorrect examples
        incorrect_selected = incorrect_df.head(
            IMAGES_PER_CLASS
        )

        print(
            "  Incorrect examples:",
            len(incorrect_selected)
        )

        for _, row in incorrect_selected.iterrows():

            row = row.to_dict()

            row["selection_type"] = "incorrect"

            selected.append(row)

            print(
                f"    Incorrect -> "
                f"{row['predicted_class']} "
                f"({row['confidence'] * 100:.2f}%)"
            )

    selected_df = pd.DataFrame(selected)

    # Save selection table
    selection_csv = (
        OUTPUT_ROOT /
        "selected_gradcam_cases.csv"
    )

    selected_df.to_csv(
        selection_csv,
        index=False
    )

    print()
    print("Selected cases:", len(selected_df))
    print("Selection CSV:", selection_csv)

    # ------------------------------------------------------------
    # Generate Grad-CAM
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING GRAD-CAM + GRAD-CAM++")
    print("=" * 70)

    result_rows = []

    individual_files = []

    for i, row in selected_df.iterrows():

        image_path = Path(row["path"])

        true_class = row["true_class"]
        predicted_class = row["predicted_class"]
        confidence = row["confidence"]
        selection_type = row["selection_type"]

        print()
        print(
            f"[{i + 1:02d}/{len(selected_df)}] "
            f"{selection_type.upper()} | "
            f"True: {true_class} | "
            f"Predicted: {predicted_class} | "
            f"Confidence: {confidence * 100:.2f}%"
        )

        original, image = load_image(
            image_path
        )

        tensor = np.expand_dims(
            image,
            axis=0
        )

        predicted_index = int(
            row["pred_index"]
        )

        # Grad-CAM
        cam = compute_gradcam(
            model,
            tensor,
            target_layer,
            predicted_index
        )

        # Grad-CAM++
        cam_pp = compute_gradcam_plus_plus(
            model,
            tensor,
            target_layer,
            predicted_index
        )

        output_name = (
            f"{i + 1:02d}_"
            f"{selection_type}_"
            f"true_{true_class}_"
            f"pred_{predicted_class}.png"
        )

        output_path = (
            INDIVIDUAL_DIR /
            output_name
        )

        save_visualization(
            original,
            cam,
            cam_pp,
            true_class,
            predicted_class,
            confidence,
            image_path,
            output_path
        )

        individual_files.append(
            output_path
        )

        result_rows.append({
            "image": str(image_path),
            "selection_type": selection_type,
            "true_class": true_class,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "visualization": str(output_path)
        })

    # ------------------------------------------------------------
    # Save final result table
    # ------------------------------------------------------------

    result_df = pd.DataFrame(
        result_rows
    )

    results_csv = (
        OUTPUT_ROOT /
        "gradcam_results.csv"
    )

    result_df.to_csv(
        results_csv,
        index=False
    )

    # ------------------------------------------------------------
    # Create montages
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING MONTAGES")
    print("=" * 70)

    # All
    create_montage(
        individual_files,
        MONTAGE_DIR / "all_selected_cases.png",
        "Grad-CAM and Grad-CAM++ - Selected Test Cases"
    )

    # Correct only
    correct_files = []

    for path, (_, row) in zip(
        individual_files,
        result_df.iterrows()
    ):

        if row["selection_type"] == "correct":
            correct_files.append(path)

    create_montage(
        correct_files,
        MONTAGE_DIR / "correct_predictions.png",
        "Grad-CAM and Grad-CAM++ - Correct Predictions"
    )

    # Incorrect only
    incorrect_files = []

    for path, (_, row) in zip(
        individual_files,
        result_df.iterrows()
    ):

        if row["selection_type"] == "incorrect":
            incorrect_files.append(path)

    create_montage(
        incorrect_files,
        MONTAGE_DIR / "incorrect_predictions.png",
        "Grad-CAM and Grad-CAM++ - Incorrect Predictions"
    )

    # ------------------------------------------------------------
    # Summary JSON
    # ------------------------------------------------------------

    summary = {
        "model": str(MODEL_PATH),
        "target_layer": TARGET_LAYER_NAME,
        "official_test_images": int(len(test_labels)),
        "selected_cases": int(len(selected_df)),
        "classes": classes,
        "images_per_class": IMAGES_PER_CLASS,
        "heatmap_settings": {
            "low_percentile": LOW_PERCENTILE,
            "high_percentile": HIGH_PERCENTILE,
            "smooth_radius": SMOOTH_RADIUS,
            "overlay_alpha": OVERLAY_ALPHA,
            "background_threshold": BACKGROUND_THRESHOLD
        }
    }

    summary_path = (
        OUTPUT_ROOT /
        "gradcam_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("GRAD-CAM ANALYSIS COMPLETED")
    print("=" * 70)

    print()
    print("Model parameters:", model.count_params())
    print("Official test images:", len(test_labels))
    print("Selected cases:", len(selected_df))

    print()
    print("Generated files:")

    print(
        "Individual visualizations:",
        INDIVIDUAL_DIR
    )

    print(
        "Montages:",
        MONTAGE_DIR
    )

    print(
        "Selection CSV:",
        selection_csv
    )

    print(
        "Results CSV:",
        results_csv
    )

    print(
        "Summary JSON:",
        summary_path
    )

    print()
    print("=" * 70)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()