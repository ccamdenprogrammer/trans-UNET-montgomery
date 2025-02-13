from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
import glob

# Dataset paths
DATASET_PATH = "your file path here"
CXR_PATH = os.path.join(DATASET_PATH, "your file path here")
LEFT_MASK_PATH = os.path.join(DATASET_PATH, "your file path here")
RIGHT_MASK_PATH = os.path.join(DATASET_PATH, "syour file path here")

IMG_SIZE = (256, 256)  # Resize all images to 256x256
BATCH_SIZE = 8

# Improved data preprocessing
def load_images_and_masks(image_dir, mask_left_dir, mask_right_dir, image_size=(256, 256)):
    images = []
    masks = []

    for img_file in glob.glob(os.path.join(image_dir, '*.png')):
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

        # Check if corresponding masks exist
        left_mask_file = os.path.join(mask_left_dir, os.path.basename(img_file))
        right_mask_file = os.path.join(mask_right_dir, os.path.basename(img_file))

        if not os.path.exists(left_mask_file) or not os.path.exists(right_mask_file):
            print(f"Mask not found for {img_file}")
            continue

        left_mask = cv2.imread(left_mask_file, cv2.IMREAD_GRAYSCALE)
        right_mask = cv2.imread(right_mask_file, cv2.IMREAD_GRAYSCALE)

        # Combine the masks
        combined_mask = np.maximum(left_mask, right_mask)

        # Resize images and masks to the desired size
        img = cv2.resize(img, image_size)
        combined_mask = cv2.resize(combined_mask, image_size)

        # Normalize images and masks (0-1 range)
        img = img / 255.0
        combined_mask = combined_mask / 255.0

        # Append to lists
        images.append(img)
        masks.append(combined_mask)

    images = np.array(images)
    masks = np.array(masks)

    # Perform augmentation on the dataset
    images, masks = augment_data(images, masks)

    return images, masks

# Augmentation function for data
def augment_data(images, masks, image_size=(256, 256)):
    # Perform basic augmentation: flipping, rotation, and zoom
    aug_images = []
    aug_masks = []
    for img, mask in zip(images, masks):
        aug_images.append(cv2.resize(img, image_size))
        aug_masks.append(cv2.resize(mask, image_size))

        # Horizontal flip
        aug_images.append(cv2.resize(np.fliplr(img), image_size))
        aug_masks.append(cv2.resize(np.fliplr(mask), image_size))

        # Random rotation
        angle = np.random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((128, 128), angle, 1)
        aug_images.append(cv2.resize(cv2.warpAffine(img, M, (256, 256)), image_size))
        aug_masks.append(cv2.resize(cv2.warpAffine(mask, M, (256, 256)), image_size))

        # Random zoom
        zoom_factor = np.random.uniform(0.8, 1.2)
        aug_images.append(cv2.resize(cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor), image_size))
        aug_masks.append(cv2.resize(cv2.resize(mask, None, fx=zoom_factor, fy=zoom_factor), image_size))

    return np.array(aug_images), np.array(aug_masks)


# Load the images and masks
images, masks = load_images_and_masks(CXR_PATH, LEFT_MASK_PATH, RIGHT_MASK_PATH, image_size=IMG_SIZE)

# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Ensure the shapes are correct and the data is ready for training
X_train = X_train[..., np.newaxis]  # Add channel dimension for grayscale
X_val = X_val[..., np.newaxis]  # Add channel dimension for grayscale

# Helper function for convolution block
def conv_block(inputs, num_filters):
    """Convolutional block with 2 layers."""
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    return x

# Helper function for encoder block
def encoder_block(inputs, num_filters):
    """Encoder block with Conv2D and MaxPool."""
    x = conv_block(inputs, num_filters)
    p = tf.keras.layers.MaxPool2D((2, 2))(x)
    return x, p

# Helper function for decoder block
def decoder_block(inputs, skip_features, num_filters):
    """Decoder block with Conv2DTranspose and skip connection."""
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = tf.keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

# Building the TransUNet model
def build_transunet(input_shape):
    """Build TransUNet model."""
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder part
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Transformer-based bridge
    b1 = tf.keras.layers.Conv2D(1024, 3, padding="same")(p4)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.Activation("relu")(b1)

    b1 = tf.keras.layers.Conv2D(1024, 3, padding="same")(b1)
    b1 = tf.keras.layers.BatchNormalization()(b1)
    b1 = tf.keras.layers.Activation("relu")(b1)

    # Decoder part
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = tf.keras.Model(inputs, outputs, name="TransUNet")
    return model

# Compile the TransUNet model
model = build_transunet((256, 256, 1))
model.compile(optimizer=Adam(learning_rate=1e-5), loss=BinaryCrossentropy(), metrics=["accuracy"])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=22,
    batch_size=16,
    callbacks=[early_stopping]
)

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Visualize results
def plot_predictions(model, X, y_true, num_samples=3):
    """Visualize predictions."""
    preds = model.predict(X[:num_samples])
    preds = (preds > 0.5).astype(np.uint8)

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))

    for i in range(num_samples):
        axes[i, 0].imshow(X[i].squeeze(), cmap="gray")
        axes[i, 0].set_title("Original CXR")

        axes[i, 1].imshow(y_true[i].squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")

        axes[i, 2].imshow(preds[i].squeeze(), cmap="gray")
        axes[i, 2].set_title("Predicted Mask")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

# Show predictions on validation set
plot_predictions(model, X_val, y_val, num_samples=5)