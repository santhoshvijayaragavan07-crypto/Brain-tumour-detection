from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import IMG_SIZE, SEED, CLASS_ALIASES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def canonical_class(name):
    key = name.strip().lower().replace("-", "_")
    return CLASS_ALIASES.get(key, key)

def scan_split(split_dir):
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing dataset folder: {split_dir}")
    samples = []
    for folder in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = canonical_class(folder.name)
        for f in folder.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                samples.append((str(f), label))
    if not samples:
        raise RuntimeError(f"No images found under {split_dir}")
    return samples, sorted(set(y for _, y in samples))

def make_official_splits(train_dir, test_dir, validation_fraction=0.10, seed=SEED):
    train_samples, train_classes = scan_split(train_dir)
    test_samples, test_classes = scan_split(test_dir)
    if set(train_classes) != set(test_classes):
        raise RuntimeError(f"Train/Test class mismatch: {train_classes} vs {test_classes}")

    classes = sorted(set(train_classes))
    mapping = {c:i for i,c in enumerate(classes)}

    paths = np.array([p for p,_ in train_samples])
    y = np.array([mapping[v] for _,v in train_samples])
    test_paths = np.array([p for p,_ in test_samples])
    test_y = np.array([mapping[v] for _,v in test_samples])

    tr_p, va_p, tr_y, va_y = train_test_split(
        paths, y, test_size=validation_fraction,
        stratify=y, random_state=seed
    )
    return (tr_p,tr_y), (va_p,va_y), (test_paths,test_y), classes

def decode_resize(path, label, augment=False):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None,None,3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.08)
        image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label

def make_dataset(paths, labels, training=False, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((paths,labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p,y: decode_resize(p,y,training),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size, drop_remainder=training).prefetch(tf.data.AUTOTUNE)
