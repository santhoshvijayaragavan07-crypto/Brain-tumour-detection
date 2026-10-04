import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import hashlib
import time
import io
import json
from pathlib import Path

from PIL import Image
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Privacy-Preserving Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

MODEL_PATH =Path("mobilenetv2_final.keras")
OUTPUT_DIR = Path("outputs/streamlit_demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "no_tumor": "No Tumor",
    "pituitary": "Pituitary"
}

# ------------------------------------------------------------
# DEMONSTRATION AES-128 KEY
# 16 bytes = 128 bits
# ------------------------------------------------------------

AES_KEY = b"BrainTumorAES12!"


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🧠 Privacy-Preserving Explainable Brain Tumor Detection")

st.markdown(
    """
    **MobileNetV2 + AES-128 + Grad-CAM + Grad-CAM++**
    
    This demonstration implements the project pipeline:

    **MRI Upload → AES-128 Encryption → Authorized Decryption → 
    Preprocessing → MobileNetV2 Prediction → Explainability**
    """
)

st.info(
    "This is a research/project demonstration and not a clinical diagnostic system."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)(
        str(MODEL_PATH),
        compile=False
    )

    return model


try:
    model = load_model()

except Exception as e:

    st.error("Unable to load the trained MobileNetV2 model.")

    st.code(str(e))

    st.stop()


# ============================================================
# FIND MOBILENETV2 BACKBONE
# ============================================================

def find_mobilenet_backbone(model):

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            name = layer.name.lower()

            if "mobilenet" in name:

                return layer

    # fallback
    for layer in model.layers:

        if "mobilenet" in layer.name.lower():

            return layer

    return None


backbone = find_mobilenet_backbone(model)


if backbone is None:

    st.error(
        "MobileNetV2 backbone could not be found inside the trained model."
    )

    st.stop()


# ============================================================
# FIND TARGET CONVOLUTIONAL LAYER
# ============================================================

def find_target_layer(backbone):

    # Your final Grad-CAM experiment used Conv_1
    preferred_names = [
        "Conv_1",
        "conv_1"
    ]

    for name in preferred_names:

        try:

            layer = backbone.get_layer(name)

            if len(layer.output.shape) == 4:

                return layer

        except Exception:

            pass

    # fallback: last 4-D layer
    for layer in reversed(backbone.layers):

        try:

            if len(layer.output.shape) == 4:

                return layer

        except Exception:

            continue

    return None


target_layer = find_target_layer(backbone)


if target_layer is None:

    st.error(
        "Could not find a suitable convolutional layer for Grad-CAM."
    )

    st.stop()


# ============================================================
# GRAD-CAM FEATURE MODEL
# ============================================================

@st.cache_resource
def create_cam_model():

    cam_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            target_layer.output,
            backbone.output
        ],
        name="mobilenetv2_cam_model"
    )

    return cam_model


cam_model = create_cam_model()


# ============================================================
# MODEL PREPROCESSING
# ============================================================

def preprocess_for_model(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE,
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def apply_model_preprocessing(image_array):

    """
    MobileNetV2 preprocessing used by the trained model.

    Input:
        [0, 1]

    Output:
        [-1, 1]
    """

    return (image_array * 2.0) - 1.0


# ============================================================
# AES-128 ENCRYPTION
# ============================================================

def encrypt_aes_128(data):

    iv = np.random.bytes(16)

    padder = padding.PKCS7(
        algorithms.AES.block_size
    ).padder()

    padded_data = padder.update(data) + padder.finalize()

    cipher = Cipher(
        algorithms.AES(AES_KEY),
        modes.CBC(iv)
    )

    encryptor = cipher.encryptor()

    encrypted = (
        encryptor.update(padded_data)
        + encryptor.finalize()
    )

    return encrypted, iv


# ============================================================
# AES-128 DECRYPTION
# ============================================================

def decrypt_aes_128(encrypted_data, iv):

    cipher = Cipher(
        algorithms.AES(AES_KEY),
        modes.CBC(iv)
    )

    decryptor = cipher.decryptor()

    padded_data = (
        decryptor.update(encrypted_data)
        + decryptor.finalize()
    )

    unpadder = padding.PKCS7(
        algorithms.AES.block_size
    ).unpadder()

    original_data = (
        unpadder.update(padded_data)
        + unpadder.finalize()
    )

    return original_data


# ============================================================
# SHA-256
# ============================================================

def calculate_sha256(data):

    return hashlib.sha256(data).hexdigest()


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(processed_image, class_index):

    processed_image = tf.convert_to_tensor(
        processed_image,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        tape.watch(processed_image)

        conv_outputs, backbone_output = cam_model(
            processed_image,
            training=False
        )

        # Reproduce the trained model head
        x = model.get_layer(
            "global_average_pooling"
        )(backbone_output)

        x = model.get_layer(
            "dropout_1"
        )(x, training=False)

        x = model.get_layer(
            "dense_features"
        )(x)

        x = model.get_layer(
            "dropout_2"
        )(x, training=False)

        predictions = model.get_layer(
            "classifier"
        )(x)

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    conv_outputs = conv_outputs[0]

    gradients = gradients[0]

    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1)
    )

    cam = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )

    cam = tf.maximum(
        cam,
        0
    )

    max_value = tf.reduce_max(cam)

    if float(max_value) > 0:

        cam = cam / max_value

    return cam.numpy()


# ============================================================
# GRAD-CAM++
# ============================================================

def generate_gradcam_plus_plus(
    processed_image,
    class_index
):

    processed_image = tf.convert_to_tensor(
        processed_image,
        dtype=tf.float32
    )

    with tf.GradientTape(
        persistent=True
    ) as tape3:

        tape3.watch(processed_image)

        with tf.GradientTape(
            persistent=True
        ) as tape2:

            tape2.watch(processed_image)

            with tf.GradientTape() as tape1:

                tape1.watch(processed_image)

                conv_outputs, backbone_output = cam_model(
                    processed_image,
                    training=False
                )

                x = model.get_layer(
                    "global_average_pooling"
                )(backbone_output)

                x = model.get_layer(
                    "dropout_1"
                )(x, training=False)

                x = model.get_layer(
                    "dense_features"
                )(x)

                x = model.get_layer(
                    "dropout_2"
                )(x, training=False)

                predictions = model.get_layer(
                    "classifier"
                )(x)

                class_score = predictions[
                    :, class_index
                ]

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

    del tape1
    del tape2
    del tape3

    conv_outputs = conv_outputs[0]

    first_grad = first_grad[0]
    second_grad = second_grad[0]
    third_grad = third_grad[0]

    alpha_num = second_grad

    alpha_denom = (
        2.0 * second_grad
        +
        conv_outputs * third_grad
    )

    alpha_denom = tf.where(
        alpha_denom != 0,
        alpha_denom,
        tf.ones_like(alpha_denom)
    )

    alphas = alpha_num / alpha_denom

    positive_gradients = tf.maximum(
        first_grad,
        0
    )

    weights = tf.reduce_sum(
        alphas * positive_gradients,
        axis=(0, 1)
    )

    cam = tf.reduce_sum(
        conv_outputs * weights,
        axis=-1
    )

    cam = tf.maximum(
        cam,
        0
    )

    max_value = tf.reduce_max(cam)

    if float(max_value) > 0:

        cam = cam / max_value

    return cam.numpy()


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap_overlay(
    original_image,
    cam,
    alpha=0.45
):

    original = np.asarray(
        original_image.convert("RGB")
    )

    h, w = original.shape[:2]

    cam = cv2.resize(
        cam,
        (w, h)
    )

    cam = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        cam,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return Image.fromarray(
        overlay
    )


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image):

    image_array = preprocess_for_model(
        image
    )

    processed = apply_model_preprocessing(
        image_array
    )

    prediction = model.predict(
        processed,
        verbose=0
    )[0]

    class_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[class_index]
    )

    return (
        image_array,
        processed,
        prediction,
        class_index,
        confidence
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.sidebar.header("Project Pipeline")

st.sidebar.markdown(
    """
    **1. Upload MRI**  
    ↓  
    **2. AES-128 Encryption**  
    ↓  
    **3. AES-128 Decryption**  
    ↓  
    **4. MobileNetV2**  
    ↓  
    **5. Grad-CAM**  
    ↓  
    **6. Grad-CAM++**
    """
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# MAIN PROCESS
# ============================================================

if uploaded_file is not None:

    original_bytes = uploaded_file.getvalue()

    try:

        original_image = Image.open(
            io.BytesIO(original_bytes)
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Unable to read image: {e}"
        )

        st.stop()


    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    st.subheader("1. Uploaded MRI")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            original_image,
            caption="Original MRI",
            width="stretch"
        )

    with col2:

        st.write(
            f"**Filename:** `{uploaded_file.name}`"
        )

        st.write(
            f"**Original size:** {len(original_bytes):,} bytes"
        )

        st.write(
            f"**Image dimensions:** "
            f"{original_image.width} × "
            f"{original_image.height}"
        )

        original_hash = calculate_sha256(
            original_bytes
        )

        st.write(
            f"**Original SHA-256:**"
        )

        st.code(
            original_hash
        )


    # --------------------------------------------------------
    # AES ENCRYPTION
    # --------------------------------------------------------

    st.subheader(
        "2. AES-128 Privacy Protection"
    )

    encryption_start = time.perf_counter()

    encrypted_bytes, iv = encrypt_aes_128(
        original_bytes
    )

    encryption_time = (
        time.perf_counter()
        - encryption_start
    ) * 1000


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Encryption Time",
            f"{encryption_time:.3f} ms"
        )

    with col2:

        st.metric(
            "Original Size",
            f"{len(original_bytes):,} B"
        )

    with col3:

        st.metric(
            "Encrypted Size",
            f"{len(encrypted_bytes):,} B"
        )


    st.success(
        "MRI image encrypted successfully using AES-128-CBC."
    )

    with st.expander(
        "View encryption information"
    ):

        st.write(
            "**Encryption:** AES-128-CBC"
        )

        st.write(
            "**Key size:** 128 bits"
        )

        st.write(
            "**IV size:** 128 bits"
        )

        st.code(
            iv.hex()
        )


    # --------------------------------------------------------
    # AES DECRYPTION
    # --------------------------------------------------------

    st.subheader(
        "3. Authorized Decryption"
    )

    decryption_start = time.perf_counter()

    decrypted_bytes = decrypt_aes_128(
        encrypted_bytes,
        iv
    )

    decryption_time = (
        time.perf_counter()
        - decryption_start
    ) * 1000


    decrypted_hash = calculate_sha256(
        decrypted_bytes
    )

    hash_match = (
        original_hash == decrypted_hash
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Decryption Time",
            f"{decryption_time:.3f} ms"
        )

    with col2:

        st.metric(
            "Recovered Size",
            f"{len(decrypted_bytes):,} B"
        )

    with col3:

        if hash_match:

            st.success(
                "SHA-256 MATCH"
            )

        else:

            st.error(
                "HASH MISMATCH"
            )


    if not hash_match:

        st.error(
            "Decrypted image does not match the uploaded image."
        )

        st.stop()

    else:

        st.success(
            "Decryption verified successfully. "
            "The recovered image exactly matches the original."
        )


    # --------------------------------------------------------
    # RECOVERED IMAGE
    # --------------------------------------------------------

    decrypted_image = Image.open(
        io.BytesIO(decrypted_bytes)
    ).convert("RGB")


    with st.expander(
        "View decrypted MRI"
    ):

        st.image(
            decrypted_image,
            caption="Authorized decrypted MRI",
            width="stretch"
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    st.subheader(
        "4. MobileNetV2 Tumor Classification"
    )

    prediction_start = time.perf_counter()

    (
        image_array,
        processed,
        probabilities,
        predicted_index,
        confidence
    ) = predict_image(
        decrypted_image
    )

    prediction_time = (
        time.perf_counter()
        - prediction_start
    ) * 1000


    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    display_class = DISPLAY_NAMES[
        predicted_class
    ]


    st.success(
        f"Prediction: {display_class}"
    )


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Class",
            display_class
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    st.write(
        f"**Prediction time:** "
        f"{prediction_time:.3f} ms"
    )


    # --------------------------------------------------------
    # PROBABILITY TABLE
    # --------------------------------------------------------

    st.write(
        "**Class probabilities**"
    )

    probability_data = {}

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        probability_data[
            DISPLAY_NAMES[class_name]
        ] = f"{probabilities[i] * 100:.2f}%"


    st.table(
        probability_data
    )


    # --------------------------------------------------------
    # EXPLAINABILITY
    # --------------------------------------------------------

    st.subheader(
        "5. Explainable AI"
    )

    st.write(
        "Grad-CAM and Grad-CAM++ visualize regions "
        "that contributed to the predicted class."
    )


    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    with st.spinner(
        "Generating Grad-CAM..."
    ):

        gradcam = generate_gradcam(
            processed,
            predicted_index
        )

        gradcam_image = create_heatmap_overlay(
            decrypted_image,
            gradcam
        )


    # --------------------------------------------------------
    # GRAD-CAM++
    # --------------------------------------------------------

    with st.spinner(
        "Generating Grad-CAM++..."
    ):

        gradcam_pp = generate_gradcam_plus_plus(
            processed,
            predicted_index
        )

        gradcam_pp_image = create_heatmap_overlay(
            decrypted_image,
            gradcam_pp
        )


    col1, col2 = st.columns(2)

    with col1:

        st.image(
            gradcam_image,
            caption=(
                f"Grad-CAM — {display_class}"
            ),
            width="stretch"
        )

    with col2:

        st.image(
            gradcam_pp_image,
            caption=(
                f"Grad-CAM++ — {display_class}"
            ),
            width="stretch"
        )


    # --------------------------------------------------------
    # RESULTS SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "6. Final Pipeline Summary"
    )


    summary = {
        "input_file": uploaded_file.name,
        "predicted_class": predicted_class,
        "display_class": display_class,
        "confidence": round(
            confidence,
            6
        ),
        "original_size_bytes": len(
            original_bytes
        ),
        "encrypted_size_bytes": len(
            encrypted_bytes
        ),
        "encryption_time_ms": round(
            encryption_time,
            3
        ),
        "decryption_time_ms": round(
            decryption_time,
            3
        ),
        "prediction_time_ms": round(
            prediction_time,
            3
        ),
        "sha256_verified": bool(
            hash_match
        ),
        "model": "MobileNetV2",
        "input_size": "224x224",
        "encryption": "AES-128-CBC",
        "explainability": [
            "Grad-CAM",
            "Grad-CAM++"
        ]
    }


    st.json(
        summary
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    timestamp = int(
        time.time()
    )

    result_file = (
        OUTPUT_DIR /
        f"prediction_{timestamp}.json"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.warning(
        """
        **Research demonstration only:** This system is not intended
        to replace radiologists or clinical diagnosis. Grad-CAM and
        Grad-CAM++ provide model-attribution visualizations and should
        not be interpreted as clinically validated tumor segmentation.
        """
    )


else:

    # ========================================================
    # LANDING PAGE
    # ========================================================

    st.markdown(
        """
        ### How to use

        1. Upload a brain MRI image using the uploader above.
        2. The image is encrypted using AES-128-CBC.
        3. The encrypted data is decrypted through the authorized pipeline.
        4. The recovered image is processed at 224 × 224 resolution.
        5. MobileNetV2 predicts one of four classes.
        6. Grad-CAM and Grad-CAM++ generate explanation heatmaps.

        **Supported classes**

        - Glioma
        - Meningioma
        - No Tumor
        - Pituitary
        """
    )

    st.info(
        "Upload an MRI image to start the complete privacy-preserving "
        "classification and explainability pipeline."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Privacy-Preserving Explainable Brain Tumor Detection "
    "Using Lightweight Deep Learning and Grad-CAM"
)
