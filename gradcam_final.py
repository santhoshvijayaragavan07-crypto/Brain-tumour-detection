import os
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/mobilenetv2_final.keras")

DATASET_ROOT = Path(
    r"C:\Users\dell\Downloads\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset"
)

TEST_DIR = DATASET_ROOT / "Test"

OUTPUT_DIR = Path("outputs/final_explainability")
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
MONTAGE_DIR = OUTPUT_DIR / "montages"

INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
MONTAGE_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

# Dataset folder names
FOLDER_NAMES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "no_tumor": "notumor",
    "pituitary": "pituitary"
}


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("=" * 60)
print("LOADING TRAINED MOBILENETV2 MODEL")
print("=" * 60)

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

print("Model loaded successfully.")
print("Model input:", model.input_shape)


# ============================================================
# FIND MOBILENETV2 BACKBONE
# ============================================================

backbone = None

for layer in model.layers:

    if isinstance(layer, tf.keras.Model):

        if "mobilenet" in layer.name.lower():

            backbone = layer
            break


if backbone is None:
    raise RuntimeError(
        "MobileNetV2 backbone could not be found."
    )


print("Backbone:", backbone.name)


# ============================================================
# TARGET CONVOLUTIONAL LAYER
# ============================================================

target_layer = backbone.get_layer("Conv_1")

print(
    "Target layer:",
    target_layer.name
)

print(
    "Target shape:",
    target_layer.output_shape
)


# ============================================================
# FIND BACKBONE POSITION
# ============================================================

backbone_index = None

for i, layer in enumerate(model.layers):

    if layer.name == backbone.name:

        backbone_index = i
        break


if backbone_index is None:

    raise RuntimeError(
        "Could not locate MobileNetV2 backbone."
    )


# ============================================================
# PREPROCESSING LAYERS
# ============================================================

pre_layers = model.layers[:backbone_index]

print("\nPreprocessing layers:")

for layer in pre_layers:

    print(
        "   ",
        layer.name
    )


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

head_layers = model.layers[backbone_index + 1:]

print("\nHead layers:")

for layer in head_layers:

    print(
        "   ",
        layer.name
    )


# ============================================================
# IMPORTANT:
# CREATE GRAPH-CONNECTED CAM MODEL
# ============================================================

print("\nCreating Grad-CAM feature model...")

cam_model = tf.keras.Model(
    inputs=backbone.input,
    outputs=[
        target_layer.output,
        backbone.output
    ],
    name="mobilenetv2_cam_model"
)

print("Grad-CAM feature model created successfully.")


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(img_path):

    img = image.load_img(
        str(img_path),
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    arr = image.img_to_array(img)

    arr = arr.astype(
        "float32"
    ) / 255.0

    return arr


# ============================================================
# APPLY MODEL PREPROCESSING
# ============================================================

def prepare_backbone_input(x):

    z = x

    for layer in pre_layers:

        if isinstance(
            layer,
            tf.keras.layers.InputLayer
        ):

            continue

        z = layer(
            z,
            training=False
        )

    return z


# ============================================================
# RUN CLASSIFICATION HEAD
# ============================================================

def run_backbone_head(features):

    z = features

    for layer in head_layers:

        z = layer(
            z,
            training=False
        )

    return z


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam(
    backbone_input,
    class_index
):

    with tf.GradientTape() as tape:

        conv_output, backbone_output = cam_model(
            backbone_input,
            training=False
        )

        predictions = run_backbone_head(
            backbone_output
        )

        class_score = predictions[
            :,
            class_index
        ]

    gradients = tape.gradient(
        class_score,
        conv_output
    )

    if gradients is None:

        raise RuntimeError(
            "Grad-CAM gradient is None."
        )

    # Global average pooling of gradients
    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Weighted feature maps
    heatmap = tf.reduce_sum(
        conv_output *
        weights[
            :,
            tf.newaxis,
            tf.newaxis,
            :
        ],
        axis=-1
    )

    # Remove negative values
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(
        heatmap,
        axis=(1, 2),
        keepdims=True
    )

    heatmap = tf.math.divide_no_nan(
        heatmap,
        max_value
    )

    return heatmap[0].numpy()


# ============================================================
# GRAD-CAM++
# ============================================================

def make_gradcam_plus_plus(
    backbone_input,
    class_index
):

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    with tf.GradientTape(
        persistent=True
    ) as tape:

        tape.watch(
            backbone_input
        )

        with tf.GradientTape(
            persistent=True
        ) as tape2:

            tape2.watch(
                backbone_input
            )

            with tf.GradientTape(
                persistent=True
            ) as tape1:

                tape1.watch(
                    backbone_input
                )

                conv_output, backbone_output = cam_model(
                    backbone_input,
                    training=False
                )

                predictions = run_backbone_head(
                    backbone_output
                )

                class_score = predictions[
                    :,
                    class_index
                ]

            first_grad = tape1.gradient(
                class_score,
                conv_output
            )

            del tape1

        second_grad = tape2.gradient(
            first_grad,
            conv_output
        )

        del tape2

    third_grad = tape.gradient(
        second_grad,
        conv_output
    )

    del tape

    # --------------------------------------------------------
    # Check gradients
    # --------------------------------------------------------

    if first_grad is None:

        raise RuntimeError(
            "Grad-CAM++ first gradient is None."
        )

    if second_grad is None:

        second_grad = tf.zeros_like(
            first_grad
        )

    if third_grad is None:

        third_grad = tf.zeros_like(
            first_grad
        )

    # First image in batch
    conv = conv_output[0]

    grad = first_grad[0]

    grad2 = second_grad[0]

    grad3 = third_grad[0]

    # --------------------------------------------------------
    # Grad-CAM++ alpha coefficients
    # --------------------------------------------------------

    activation_sum = tf.reduce_sum(
        conv,
        axis=(0, 1),
        keepdims=True
    )

    numerator = grad2

    denominator = (
        2.0 * grad2
        +
        grad3 * activation_sum
    )

    denominator = tf.where(
        tf.abs(denominator) > 1e-8,
        denominator,
        tf.ones_like(denominator)
    )

    alpha = (
        numerator /
        denominator
    )

    # Positive gradients
    positive_grad = tf.maximum(
        grad,
        0.0
    )

    # Importance weights
    weights = tf.reduce_sum(
        alpha *
        positive_grad,
        axis=(0, 1)
    )

    # Weighted feature maps
    heatmap = tf.reduce_sum(
        conv * weights,
        axis=-1
    )

    # Remove negative values
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = tf.math.divide_no_nan(
        heatmap,
        max_value
    )

    return heatmap.numpy()


# ============================================================
# SAVE INDIVIDUAL VISUALIZATION
# ============================================================

def save_visualization(
    original,
    heatmap,
    title,
    output_path
):

    # Resize heatmap
    heatmap_resized = tf.image.resize(
        heatmap[
            ...,
            np.newaxis
        ],
        IMG_SIZE
    ).numpy().squeeze()

    plt.figure(
        figsize=(12, 4)
    )

    # --------------------------------------------------------
    # Original MRI
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        1
    )

    plt.imshow(
        original
    )

    plt.axis(
        "off"
    )

    plt.title(
        "Original MRI"
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        2
    )

    plt.imshow(
        original
    )

    plt.imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.45
    )

    plt.axis(
        "off"
    )

    plt.title(
        title
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    plt.subplot(
        1,
        3,
        3
    )

    plt.imshow(
        heatmap_resized,
        cmap="jet"
    )

    plt.axis(
        "off"
    )

    plt.title(
        "Activation Map"
    )

    plt.tight_layout()

    plt.savefig(
        str(output_path),
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SELECT TEST IMAGES
# ============================================================

print("\n" + "=" * 60)
print("SELECTING TEST IMAGES")
print("=" * 60)

selected = []


for class_name in CLASS_NAMES:

    folder = (
        TEST_DIR /
        FOLDER_NAMES[class_name]
    )

    if not folder.exists():

        print(
            "WARNING: Folder not found:",
            folder
        )

        continue

    files = []

    for filename in os.listdir(folder):

        filepath = folder / filename

        if filepath.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        ]:

            files.append(
                filepath
            )

    files = sorted(
        files
    )

    # Two images per class
    selected.extend(
        [
            (class_name, p)
            for p in files[:2]
        ]
    )


print(
    "Selected images:",
    len(selected)
)


# ============================================================
# PROCESS SELECTED IMAGES
# ============================================================

results = []


for count, (
    true_class,
    img_path
) in enumerate(
    selected,
    start=1
):

    print("\n" + "-" * 60)

    print(
        f"[{count}/{len(selected)}] "
        f"Processing: {img_path.name}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    arr = load_image(
        img_path
    )

    x = np.expand_dims(
        arr,
        axis=0
    )

    x_tf = tf.convert_to_tensor(
        x,
        dtype=tf.float32
    )

    # --------------------------------------------------------
    # Normal model prediction
    # --------------------------------------------------------

    prediction = model(
        x_tf,
        training=False
    )

    probabilities = (
        prediction.numpy()[0]
    )

    pred_index = int(
        np.argmax(
            probabilities
        )
    )

    pred_class = (
        CLASS_NAMES[pred_index]
    )

    confidence = float(
        probabilities[pred_index]
    )

    print(
        "True class:",
        true_class
    )

    print(
        "Predicted:",
        pred_class
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Prepare MobileNetV2 input
    # --------------------------------------------------------

    backbone_input = (
        prepare_backbone_input(
            x_tf
        )
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    print(
        "Generating Grad-CAM..."
    )

    cam = make_gradcam(
        backbone_input,
        pred_index
    )

    cam_filename = (
        f"{count:02d}_"
        f"{true_class}_"
        f"{pred_class}_"
        f"gradcam.png"
    )

    cam_path = (
        INDIVIDUAL_DIR /
        cam_filename
    )

    save_visualization(
        arr,
        cam,
        (
            f"Grad-CAM | "
            f"{pred_class} "
            f"({confidence * 100:.2f}%)"
        ),
        cam_path
    )

    print(
        "Saved:",
        cam_path
    )

    # --------------------------------------------------------
    # Grad-CAM++
    # --------------------------------------------------------

    print(
        "Generating Grad-CAM++..."
    )

    cam_pp = make_gradcam_plus_plus(
        backbone_input,
        pred_index
    )

    cam_pp_filename = (
        f"{count:02d}_"
        f"{true_class}_"
        f"{pred_class}_"
        f"gradcam_plus_plus.png"
    )

    cam_pp_path = (
        INDIVIDUAL_DIR /
        cam_pp_filename
    )

    save_visualization(
        arr,
        cam_pp,
        (
            f"Grad-CAM++ | "
            f"{pred_class} "
            f"({confidence * 100:.2f}%)"
        ),
        cam_pp_path
    )

    print(
        "Saved:",
        cam_pp_path
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append(
        {
            "image": str(
                img_path
            ),
            "true_class": true_class,
            "predicted_class": pred_class,
            "confidence": confidence,
            "gradcam_file": str(
                cam_path
            ),
            "gradcam_plus_plus_file": str(
                cam_pp_path
            )
        }
    )


# ============================================================
# SAVE RESULTS CSV
# ============================================================

results_df = pd.DataFrame(
    results
)

csv_path = (
    OUTPUT_DIR /
    "explainability_results.csv"
)

results_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# CREATE SIMPLE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL EXPLAINABILITY COMPLETE")
print("=" * 60)

print(
    "Images processed:",
    len(results)
)

print(
    "Grad-CAM images:",
    len(results)
)

print(
    "Grad-CAM++ images:",
    len(results)
)

print(
    "\nResults CSV:"
)

print(
    csv_path
)

print(
    "\nIndividual output folder:"
)

print(
    INDIVIDUAL_DIR
)

print(
    "\nTotal PNG files expected:",
    len(results) * 2
)

print("\nDone.")