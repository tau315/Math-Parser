import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt

def build_model(input_shape=(32, 32, 1), l2_reg=1e-4):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_reg))(c1)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_reg))(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_reg))(c2)
    p2 = layers.MaxPooling2D(2)(c2)

    # Bottleneck
    b = layers.Conv2D(128, 3, activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_reg))(p2)

    # Decoder
    u1 = layers.UpSampling2D(2)(b)
    concat1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat1)

    u2 = layers.UpSampling2D(2)(c3)
    concat2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(concat2)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)

    return models.Model(inputs, outputs)

def load_pair(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [32, 32])
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [32, 32])
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask

def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, mask

def get_dataset(image_paths, mask_paths, batch_size=16, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(256)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2. * intersection + smooth) / (total + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d = dice_loss(y_true, y_pred)
    return bce + d

def show_prediction(model, image):
    pred = model.predict(image[None])[0, ..., 0]
    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(image[..., 0], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.imshow(pred > 0.5, cmap='gray')
    plt.show()

