import sys
from pathlib import Path
import numpy as np, pandas as pd, tensorflow as tf
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from config import *
from src.data import make_official_splits, make_dataset
from src.model import build_lightweight_cnn

def main():
    tf.random.set_seed(SEED); np.random.seed(SEED)
    MODEL_DIR.mkdir(exist_ok=True); OUTPUT_DIR.mkdir(exist_ok=True)
    (tr_p,tr_y),(va_p,va_y),_,classes=make_official_splits(
        TRAIN_DIR,TEST_DIR,VALIDATION_FRACTION,SEED)
    model=build_lightweight_cnn(len(classes))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    h=model.fit(make_dataset(tr_p,tr_y,True,BATCH_SIZE),
                validation_data=make_dataset(va_p,va_y,False,BATCH_SIZE),
                epochs=EPOCHS)
    model.save(MODEL_DIR/"lightweight_brain_tumor_baseline.keras")
    pd.DataFrame(h.history).to_csv(OUTPUT_DIR/"baseline_history.csv",index=False)

if __name__=="__main__": main()
