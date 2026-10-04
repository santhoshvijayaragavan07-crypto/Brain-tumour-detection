from pathlib import Path

# Set this to the folder containing BOTH Train and Test.
DATASET_ROOT = Path(r"C:\Users\dell\Downloads\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset")
TRAIN_DIR = DATASET_ROOT / "Train"
TEST_DIR = DATASET_ROOT / "Test"

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 20
VALIDATION_FRACTION = 0.10

NOISE_MULTIPLIER = 1.0
L2_NORM_CLIP = 1.0
DELTA = 1e-5

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
