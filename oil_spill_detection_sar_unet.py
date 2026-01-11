# ============================================================
# Oil Spill Detection and Impact Mapping in the Niger Delta
# Using Synthetic SAR Data and Deep Learning (U-Net)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D,
    concatenate, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------------
# Step 1: Synthetic SAR Image Generator
# ------------------------------------------------------------

def generate_sar_image(size=128):
    background = np.random.normal(0.6, 0.15, (size, size))

    spill = np.zeros((size, size))
    cx, cy = np.random.randint(30, size-30, 2)
    radius = np.random.randint(10, 25)

    y, x = np.ogrid[:size, :size]
    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
    spill[mask] = 1

    oil_texture = np.random.normal(0.25, 0.05, (size, size))
    oil_texture = gaussian_filter(oil_texture, sigma=2)

    sar_image = background * (1 - spill) + oil_texture * spill
    sar_image = np.clip(sar_image, 0, 1)

    return sar_image, spill

# ------------------------------------------------------------
# Step 2: Dataset Creation
# ------------------------------------------------------------

def create_dataset(samples=500, size=128):
    images, masks = [], []

    for _ in range(samples):
        img, msk = generate_sar_image(size)
        images.append(img)
        masks.append(msk)

    images = np.array(images)[..., np.newaxis]
    masks = np.array(masks)[..., np.newaxis]

    return images, masks

X, y = create_dataset(samples=600)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# Step 3: U-Net Model for Oil Spill Segmentation
# ------------------------------------------------------------

def unet_model(input_size=(128, 128, 1)):
    inputs = Input(input_size)

    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)

    u1 = UpSampling2D()(c3)
    m1 = concatenate([u1, c2])
    c4 = Conv2D(64, 3, activation='relu', padding='same')(m1)

    u2 = UpSampling2D()(c4)
    m2 = concatenate([u2, c1])
    c5 = Conv2D(32, 3, activation='relu', padding='same')(m2)

    outputs = Conv2D(1, 1, activation='sigmoid')(c5)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

model = unet_model()
model.summary()

# ------------------------------------------------------------
# Step 4: Model Training
# ------------------------------------------------------------

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=8
)

# ------------------------------------------------------------
# Step 5: Evaluation
# ------------------------------------------------------------

preds = model.predict(X_test)
preds_binary = (preds > 0.5).astype(int)

print(
    classification_report(
        y_test.flatten(),
        preds_binary.flatten(),
        target_names=["Water", "Oil Spill"]
    )
)

# ------------------------------------------------------------
# Step 6: Impact Mapping Visualization
# ------------------------------------------------------------

def visualize_results(index=0):
    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Synthetic SAR Image")
    plt.imshow(X_test[index].squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("Ground Truth Spill")
    plt.imshow(y_test[index].squeeze(), cmap='Reds')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Predicted Spill Impact")
    plt.imshow(preds_binary[index].squeeze(), cmap='Reds')
    plt.axis('off')

    plt.show()

visualize_results(index=3)
