import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split

# Config
DATA_DIR = "path/to/HASYv2"
LABELS_FILE = os.path.join(DATA_DIR, "symbols.csv")
IMG_DIR = os.path.join(DATA_DIR, "images")  # Adjust if needed
IMG_SIZE = 32  # HASY images are 32x32
BATCH_SIZE = 64
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE

# Load Labels
def load_labels(label_csv_path):
    df = pd.read_csv(label_csv_path)
    return df  # raw labels dataframe

# Load Images and Labels
def load_data(df):
    images = []
    labels = []
    
    for _, row in df.iterrows():
        filepath = os.path.join(IMG_DIR, row["path"])  # adjust column name if needed
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        images.append(img)
        labels.append(row["symbol_id"])  # adjust to your label column

    X = np.array(images)[..., np.newaxis]  # Add channel dimension
    y = np.array(labels)
    return X, y

# Prepare Dataset
def prepare_datasets(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return train_ds, test_ds

# Build Model (Placeholder)
def build_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        # TODO: Add Conv2D, MaxPooling2D, etc.
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Train Model
def train(train_ds, test_ds, input_shape, num_classes):
    model = build_model(input_shape, num_classes)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)

# Main
def main():
    labels_df = load_labels(LABELS_FILE)
    X, y = load_data(labels_df)
    train_ds, test_ds = prepare_datasets(X, y)
    input_shape = (IMG_SIZE, IMG_SIZE, 1)
    num_classes = len(np.unique(y))
    train(train_ds, test_ds, input_shape, num_classes)

if __name__ == "__main__":
    main()
