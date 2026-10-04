from pathlib import Path
import csv
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = Path("models/mobilenetv2_final.keras")

DATASET_ROOT = Path(
    r"C:\Users\dell\Downloads\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset"
)

TEST_DIR = DATASET_ROOT / "Test"

OUTPUT_DIR = Path("outputs") / "gradcam_mobilenetv2_final"
INDIVIDUAL_DIR = OUTPUT_DIR / "individual"
MONTAGE_DIR = OUTPUT_DIR / "montages"

IMG_SIZE = (224, 224)

CLASSES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

CLASS_ALIASES = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "notumor": "no_tumor",
    "no_tumor": "no_tumor",
}

INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
MONTAGE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("Loading MobileNetV2 final model")
print("=" * 70)

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

print("Model loaded:", MODEL_PATH)

# ============================================================
# FIND MOBILE NET BACKBONE
# ============================================================

backbone = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        name = layer.name.lower()

        if "mobilenet" in name:
            backbone = layer
            break

if backbone is None:
    raise RuntimeError(
        "MobileNetV2 backbone could not be found inside the model."
    )

print("Backbone:", backbone.name)

# ============================================================
# FIND LAST CONVOLUTIONAL LAYER INSIDE BACKBONE
# ============================================================

conv_layers = []

for layer in backbone.layers:

    if isinstance(
        layer,
        (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D
        )
    ):
        conv_layers.append(layer)

if not conv_layers:
    raise RuntimeError(
        "No convolutional layer found inside MobileNetV2."
    )

target_layer = conv_layers[-1]

print("Target layer:", target_layer.name)
print("Target output shape:", target_layer.output.shape)

# ============================================================
# CREATE BACKBONE FEATURE EXTRACTOR
# ============================================================

feature_extractor = tf.keras.Model(
    inputs=backbone.input,
    outputs=target_layer.output
)

print("Feature extractor created successfully.")

# ============================================================
# LOAD IMAGE
# ============================================================

def canonical_class(folder_name):

    key = folder_name.lower().strip()

    return CLASS_ALIASES.get(
        key,
        key
    )


def load_image(path):

    image = tf.keras.utils.load_img(
        path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    image = tf.keras.utils.img_to_array(image)

    image = image.astype(
        np.float32
    ) / 255.0

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image):

    tensor = tf.convert_to_tensor(
        image[None, ...],
        dtype=tf.float32
    )

    predictions = model(
        tensor,
        training=False
    ).numpy()[0]

    pred_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[pred_index]
    )

    return (
        predictions,
        pred_index,
        confidence
    )


# ============================================================
# PREPROCESS FOR BACKBONE
# ============================================================

def get_backbone_input(image):

    tensor = tf.convert_to_tensor(
        image[None, ...],
        dtype=tf.float32
    )

    # The MobileNetV2 model used in the project receives
    # [0,1] images and rescales them to [-1,1] before the
    # ImageNet MobileNetV2 backbone.

    tensor = tensor * 2.0 - 1.0

    return tensor


# ============================================================
# GET HEAD PREDICTION FROM BACKBONE FEATURES
# ============================================================

def forward_from_backbone(features):

    x = features

    # GlobalAveragePooling2D
    x = tf.reduce_mean(
        x,
        axis=[1, 2]
    )

    # Find the classification head layers.
    # The saved model has:
    #
    # MobileNetV2
    # -> GlobalAveragePooling2D
    # -> Dropout
    # -> Dense(128)
    # -> Dropout
    # -> Dense(4)

    dense_layers = [
        layer
        for layer in model.layers
        if isinstance(
            layer,
            tf.keras.layers.Dense
        )
    ]

    if len(dense_layers) < 2:
        raise RuntimeError(
            "Expected two Dense layers in classification head."
        )

    dense1 = dense_layers[-2]
    dense2 = dense_layers[-1]

    x = dense1(
        x,
        training=False
    )

    x = dense2(
        x,
        training=False
    )

    return x


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam(
    image,
    class_index
):

    backbone_input = get_backbone_input(
        image
    )

    with tf.GradientTape() as tape:

        tape.watch(
            backbone_input
        )

        conv_output = feature_extractor(
            backbone_input,
            training=False
        )

        predictions = forward_from_backbone(
            conv_output
        )

        class_score = predictions[
            :,
            class_index
        ]

    gradients = tape.gradient(
        class_score,
        conv_output
    )

    conv_output = conv_output[0]
    gradients = gradients[0]

    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1)
    )

    cam = tf.reduce_sum(
        conv_output * weights,
        axis=-1
    )

    cam = tf.nn.relu(
        cam
    )

    cam = cam.numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    cam = tf.image.resize(
        cam[..., None],
        IMG_SIZE
    ).numpy()[..., 0]

    return np.clip(
        cam,
        0,
        1
    )


# ============================================================
# GRAD-CAM++
# ============================================================

def make_gradcam_plus_plus(
    image,
    class_index
):

    backbone_input = get_backbone_input(
        image
    )

    # First-order gradients
    with tf.GradientTape(
        persistent=True
    ) as tape1:

        tape1.watch(
            backbone_input
        )

        with tf.GradientTape(
            persistent=True
        ) as tape2:

            tape2.watch(
                backbone_input
            )

            conv_output = feature_extractor(
                backbone_input,
                training=False
            )

            predictions = forward_from_backbone(
                conv_output
            )

            class_score = predictions[
                :,
                class_index
            ]

        first_grad = tape2.gradient(
            class_score,
            conv_output
        )

    # Grad-CAM++ practical approximation
    second_grad = tf.square(
        first_grad
    )

    third_grad = tf.pow(
        first_grad,
        3
    )

    denominator = (
        2.0 * second_grad
        +
        tf.reduce_sum(
            conv_output * third_grad,
            axis=(1, 2),
            keepdims=True
        )
    )

    denominator = tf.where(
        tf.abs(denominator) < 1e-7,
        tf.ones_like(denominator) * 1e-7,
        denominator
    )

    alpha = (
        second_grad /
        denominator
    )

    positive_gradients = tf.nn.relu(
        first_grad
    )

    weights = tf.reduce_sum(
        alpha * positive_gradients,
        axis=(1, 2)
    )

    cam = tf.reduce_sum(
        weights[:, None, None, :] *
        conv_output,
        axis=-1
    )

    cam = tf.nn.relu(
        cam
    )

    cam = cam[0].numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    cam = tf.image.resize(
        cam[..., None],
        IMG_SIZE
    ).numpy()[..., 0]

    return np.clip(
        cam,
        0,
        1
    )


# ============================================================
# CREATE OVERLAY
# ============================================================

def create_overlay(
    image,
    heatmap
):

    image_uint8 = np.clip(
        image * 255,
        0,
        255
    ).astype(
        np.uint8
    )

    heatmap_uint8 = np.uint8(
        heatmap * 255
    )

    heatmap_color = plt.cm.jet(
        heatmap_uint8 / 255.0
    )[:, :, :3]

    heatmap_color = np.uint8(
        heatmap_color * 255
    )

    overlay = (
        0.55 * image_uint8
        +
        0.45 * heatmap_color
    )

    return np.clip(
        overlay,
        0,
        255
    ).astype