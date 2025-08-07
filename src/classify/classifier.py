import os
import tensorflow as tf

# Root dataset folder
DATA_ROOT = "Math-Symbol-Classifier/data/hasyv2"

# Fold 1 files (adjust as needed)
TRAIN_CSV = os.path.join(DATA_ROOT, "classification-task/fold-1/train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "classification-task/fold-1/test.csv")

# Image root folder (all image paths are relative to this)
IMG_ROOT = DATA_ROOT

def build_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

