import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# ==== Config ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "..", "..", "data", "classify")
HASY_ROOT = os.path.join(DATA_ROOT, "hasyv2")  # folder with subfolders per symbol class
SAVE_ROOT = os.path.join(BASE_DIR, "..", "..", "models")

# ------------------------
# Model
# ------------------------
def build_unet_gray_binary(l2_reg=1e-4):
    """
    U-Net for grayscale input -> binary symbol mask
    """
    inputs = tf.keras.Input(shape=(None, None, 1))  # grayscale

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

    c3 = layers.Conv2D(128, 3, activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_reg))(p2)
    c3 = layers.Conv2D(128, 3, activation='relu', padding='same',
                       kernel_regularizer=regularizers.l2(l2_reg))(c3)
    p3 = layers.MaxPooling2D(2)(c3)

    # Bottleneck
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b)

    # Decoder
    u3 = layers.UpSampling2D(2)(b)
    x3 = layers.Concatenate()([u3, c3])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(x3)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)

    u2 = layers.UpSampling2D(2)(c6)
    x2 = layers.Concatenate()([u2, c2])
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(x2)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)

    u1 = layers.UpSampling2D(2)(c7)
    x1 = layers.Concatenate()([u1, c1])
    c8 = layers.Conv2D(32, 3, activation='relu', padding='same')(x1)
    c8 = layers.Conv2D(32, 3, activation='relu', padding='same')(c8)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c8)  # binary mask
    return models.Model(inputs, outputs, name="UNet_Gray_Binary")

# ------------------------
# Loss
# ------------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    total = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (total + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)

# ------------------------
# Padding / Unpadding
# ------------------------
def pad_to_multiple(img, multiple=8):
    h, w = tf.shape(img)[1], tf.shape(img)[2]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    padded = tf.pad(img, [[0,0],[0,pad_h],[0,pad_w],[0,0]])
    return padded, (pad_h, pad_w)

def unpad(img, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        img = img[:, :-pad_h, :, :]
    if pad_w > 0:
        img = img[:, :, :-pad_w, :]
    return img

# ------------------------
# Postprocessing
# ------------------------
def extract_bboxes_binary(mask, threshold=0.5):
    """
    mask: [H,W], values 0-1
    returns list of (x,y,w,h) bounding boxes
    """
    mask_np = (mask.numpy() > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(cnt) for cnt in contours]
    return bboxes

def visualize_bboxes_binary(image, bboxes):
    """
    Overlay bounding boxes on grayscale image
    """
    if isinstance(image, tf.Tensor):
        image = tf.squeeze(image).numpy()
        image = (image * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for x,y,w,h in bboxes:
        cv2.rectangle(img_rgb, (x,y), (x+w,y+h), (0,255,0), 2)
    return img_rgb

# ------------------------
# Load individual HASYv2 symbols from folder structure
# ------------------------
def load_hasyv2_symbols(img_size=32):
    """
    Reads all symbol images from HASYv2 folder structure
    Returns list of grayscale symbols normalized [0,1]
    """
    symbols = []
    for class_folder in os.listdir(HASY_ROOT):
        folder_path = os.path.join(HASY_ROOT, class_folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.endswith(".png"):
                img_path = os.path.join(folder_path, fname)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (img_size, img_size))
                img = img.astype(np.float32) / 255.0
                symbols.append(img)
    print(f"[INFO] Loaded {len(symbols)} HASYv2 symbols.")
    return symbols

# ------------------------
# Generate synthetic page
# ------------------------
def generate_page(symbols, canvas_height=128, canvas_width=512,
                  max_symbols_per_line=10, rotation_range=15, scale_range=(0.7,1.3)):
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    mask = np.zeros((canvas_height, canvas_width), dtype=np.float32)

    y_offset = 0
    while y_offset < canvas_height - 32:
        x_offset = 0
        symbols_in_line = np.random.randint(1, max_symbols_per_line + 1)
        for _ in range(symbols_in_line):
            sym = symbols[np.random.randint(len(symbols))]
            scale = np.random.uniform(scale_range[0], scale_range[1])
            new_size = max(1, int(32 * scale))
            sym_scaled = cv2.resize(sym, (new_size, new_size))
            M = cv2.getRotationMatrix2D((new_size//2,new_size//2), np.random.uniform(-rotation_range, rotation_range), 1.0)
            sym_rot = cv2.warpAffine(sym_scaled, M, (new_size, new_size), borderValue=0)

            y_pos = y_offset + np.random.randint(0, max(1, canvas_height - y_offset - new_size))
            x_pos = x_offset + np.random.randint(0, 5)
            if x_pos + new_size > canvas_width:
                break

            canvas[y_pos:y_pos+new_size, x_pos:x_pos+new_size] = np.maximum(
                canvas[y_pos:y_pos+new_size, x_pos:x_pos+new_size], sym_rot)
            mask[y_pos:y_pos+new_size, x_pos:x_pos+new_size] = np.maximum(
                mask[y_pos:y_pos+new_size, x_pos:x_pos+new_size], (sym_rot > 0).astype(np.float32))

            x_offset = x_pos + new_size
        y_offset += 32 + np.random.randint(0,5)

    return tf.convert_to_tensor(canvas[..., np.newaxis], dtype=tf.float32), \
           tf.convert_to_tensor(mask[..., np.newaxis], dtype=tf.float32)

# ------------------------
# Generate dataset from HASYv2
# ------------------------
def generate_dataset(symbols, num_pages=100, canvas_height=128, canvas_width=512):
    images, masks = [], []
    for _ in range(num_pages):
        img, msk = generate_page(symbols, canvas_height, canvas_width)
        images.append(img)
        masks.append(msk)
    images = tf.stack(images)
    masks = tf.stack(masks)
    return images, masks

# ------------------------
# Train UNet on synthetic pages
# ------------------------
def train_unet_on_hasyv2(num_pages=200, canvas_height=128, canvas_width=512, batch_size=8, epochs=5):
    symbols = load_hasyv2_symbols()
    images, masks = generate_dataset(symbols, num_pages, canvas_height, canvas_width)
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    unet = build_unet_gray_binary()
    unet.compile(optimizer='adam', loss=combined_loss)
    unet.fit(dataset, epochs=epochs)
    return unet

if __name__ == "__main__":
    # Train UNet on 200 synthetic pages, 128x512, batch size 8, for 5 epochs
    unet_model = train_unet_on_hasyv2(
        num_pages=200, 
        canvas_height=128, 
        canvas_width=512, 
        batch_size=8, 
        epochs=5
    )
    unet_model.save(os.path.join(SAVE_ROOT, "unet_hasyv2.h5"))
    pass