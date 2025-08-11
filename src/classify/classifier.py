import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, regularizers
import cv2

# ==== Config ====
# DEVNOTE: Path names are written for Windows. Replace \\ with / for Linux and MacOS.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "..", "..", "data", "classify")
SAVE_ROOT = os.path.join(BASE_DIR, "..", "..", "models")
FOLD = 1  # classification task folder number (1–10) or "full"
IMG_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 10
LEARN_RATE = 5e-4
AUTOTUNE = tf.data.AUTOTUNE

TRAIN_CSV = os.path.join(DATA_ROOT, f"classification-task\\fold-{FOLD}\\train.csv")
TEST_CSV = os.path.join(DATA_ROOT, f"classification-task\\fold-{FOLD}\\test.csv")
SYMBOL_CSV = os.path.join(DATA_ROOT, f"symbols.csv")
IMG_ROOT = os.path.join(DATA_ROOT, f"classification-task\\fold-{FOLD}\\")
NUM_CLASSES = 369  # unique symbol IDs in HASYv2

# ==== Model ====
def build_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape, kernel_regularizer=regularizers.L2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),

        # Block 2
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.L2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),

        # Block 3
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.L2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.L2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ==== Data Loader ====
def load_data(csv_path, symbols_csv_path):
    # Read the mapping from symbol_id to new sequential class index
    symbols_df = pd.read_csv(symbols_csv_path)
    unique_ids = sorted(symbols_df["symbol_id"].unique())
    symbol_id_to_class = {sid: idx for idx, sid in enumerate(unique_ids)}
    # print(symbol_id_to_class)

    print(f"[INFO] Created mapping for {len(symbol_id_to_class)} classes.")

    # Read the dataset CSV (train/test)
    df = pd.read_csv(csv_path)
    images, labels = [], []

    for _, row in df.iterrows():
        img_path = os.path.normpath(os.path.join(IMG_ROOT, row["path"]))
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        images.append(img)

        # Map original symbol_id to contiguous classifier ID
        labels.append(symbol_id_to_class[row["symbol_id"]])

    X = np.array(images)[..., np.newaxis]
    y = np.array(labels)
    return X, y

# ==== Dataset Prep ====
def prepare_dataset(X, y, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# ==== Train ====
def train():
    print(f"[INFO] Loading Fold {FOLD} training data...")
    X_train, y_train = load_data(TRAIN_CSV, SYMBOL_CSV)
    X_test, y_test = load_data(TEST_CSV, SYMBOL_CSV)

    train_ds = prepare_dataset(X_train, y_train, training=True)
    test_ds = prepare_dataset(X_test, y_test, training=False)

    model = build_model((IMG_SIZE, IMG_SIZE, 1), NUM_CLASSES)

    print("[INFO] Training model...")
    model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

    model.save(os.path.join(SAVE_ROOT, f"symbol_classifier_fold{FOLD}.h5"))
    print(f"[INFO] Model saved to symbol_classifier_fold{FOLD}.h5")

    print("[INFO] Final evaluation on test set:")
    loss, acc = model.evaluate(test_ds)
    print(f"[RESULT] Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    train()