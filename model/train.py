# -*- coding: utf-8 -*-
"""
CNN Model Training Script — MobileNetV2 Transfer Learning
-----------------------------------------------------------
Classes:
  0 — Normal  (undamaged asphalt/concrete road)
  1 — Pothole (deep surface cavities)
  2 — Crack   (surface fractures / linear damage)

Negative samples (anti-false-positive training):
  Labelled as class 0 so the model learns to NOT fire damage alerts
  on non-road objects that slip past the Road Validator CNN:
    - Skin-tone / face-like gradients
    - Indoor bright uniform surfaces (rooms)
    - Plain walls / whiteboards
    - Colourful random objects (books, furniture)
    - Vehicle interiors
    - Ceiling / sky textures
    - Mobile-screen vivid pixel grids
    - Book / paper with printed lines

Architecture: MobileNetV2 (ImageNet pretrained) + Custom Dense Head
  Frozen MobileNetV2 backbone + 512-unit Dense head + class_weight balancing

In-numpy augmentation:
  RandomFlip, RandomRotation, RandomContrast, RandomBrightness

Usage:
  python model/train.py

Output:
  model/road_damage_model.h5
  model/training_curves.png
  model/confusion_matrix.png
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_HEIGHT  = 128
IMG_WIDTH   = 128
NUM_CLASSES = 3
EPOCHS      = 40
BATCH_SIZE  = 16
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'road_damage_model.h5')
CLASS_NAMES = ['normal', 'pothole', 'crack']


# ── Synthetic Image Generators ─────────────────────────────────────────────────
def generate_road_surface(rng, h=IMG_HEIGHT, w=IMG_WIDTH):
    """Realistic grey asphalt texture with varied tone."""
    tone  = rng.integers(60, 160)
    base  = rng.integers(max(0, tone-20), min(255, tone+20),
                         size=(h, w, 3), dtype=np.uint8)
    noise = rng.normal(0, 14, (h, w, 3))
    # Occasionally add lane-marking (white stripe)
    if rng.random() < 0.25:
        col = rng.integers(10, w - 10)
        base[:, col:col+4] = 240
    return np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def generate_pothole(rng):
    """Road surface with 1–4 dark circular cavities of varied depth."""
    img = generate_road_surface(rng)
    n_holes = rng.integers(1, 5)
    for _ in range(n_holes):
        cx = rng.integers(15, IMG_WIDTH  - 15)
        cy = rng.integers(15, IMG_HEIGHT - 15)
        r  = rng.integers(10, 32)
        for row in range(max(0, cy - r), min(IMG_HEIGHT, cy + r)):
            for col in range(max(0, cx - r), min(IMG_WIDTH, cx + r)):
                if (row - cy)**2 + (col - cx)**2 <= r**2:
                    depth = rng.integers(8, 50, size=3, dtype=np.uint8)
                    img[row, col] = depth
        # Slight rim highlight
        for row in range(max(0, cy - r - 2), min(IMG_HEIGHT, cy + r + 2)):
            for col in range(max(0, cx - r - 2), min(IMG_WIDTH, cx + r + 2)):
                d = abs((row - cy)**2 + (col - cx)**2 - r**2)
                if d < 16:
                    img[row, col] = np.clip(
                        img[row, col].astype(int) + 35, 0, 255
                    ).astype(np.uint8)
    return img


def generate_crack(rng):
    """Road surface with 1–5 thin dark fracture lines (straight + jagged)."""
    img = generate_road_surface(rng)
    n_cracks = rng.integers(1, 6)
    for _ in range(n_cracks):
        x1 = rng.integers(0, IMG_WIDTH)
        y1 = rng.integers(0, IMG_HEIGHT)
        dx = rng.integers(-80, 80)
        dy = rng.integers(-80, 80)
        x2 = int(np.clip(x1 + dx, 0, IMG_WIDTH  - 1))
        y2 = int(np.clip(y1 + dy, 0, IMG_HEIGHT - 1))
        steps = max(abs(dx), abs(dy), 1)
        for t in range(steps):
            px = int(x1 + t * (x2 - x1) / steps) + rng.integers(-1, 2)
            py = int(y1 + t * (y2 - y1) / steps) + rng.integers(-1, 2)
            thickness = rng.integers(1, 4)
            for tw in range(-thickness, thickness + 1):
                for th in range(-thickness, thickness + 1):
                    nr, nc = py + th, px + tw
                    if 0 <= nr < IMG_HEIGHT and 0 <= nc < IMG_WIDTH:
                        img[nr, nc] = rng.integers(5, 45, size=3, dtype=np.uint8)
    return img


def generate_negative_sample(rng):
    """
    Generate 8 categories of non-road images labelled as class 0 (Normal).
    The Road Validator CNN is the primary gate; these train the damage model
    to be robust if any non-road image slips through validation.
    """
    kind = rng.integers(0, 8)

    if kind == 0:
        # Skin-tone gradient (face-like)
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        for i in range(IMG_HEIGHT):
            v = int(180 + 30 * np.sin(i / IMG_HEIGHT * np.pi))
            img[i, :] = [int(v * 0.55), int(v * 0.70), int(v * 0.90)]
        img = np.clip(img + rng.integers(-15, 15, img.shape), 0, 255).astype(np.uint8)

    elif kind == 1:
        # Indoor room: bright saturated colour
        hue  = int(rng.integers(0, 180))
        base = np.full((IMG_HEIGHT, IMG_WIDTH, 3), 200, dtype=np.uint8)
        base[:, :, 0] = hue
        base[:, :, 1] = int(rng.integers(100, 200))
        img = cv2.cvtColor(base, cv2.COLOR_HSV2BGR)
        img = np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)

    elif kind == 2:
        # Plain white/beige wall
        val = int(rng.integers(195, 255))
        img = np.full((IMG_HEIGHT, IMG_WIDTH, 3), val, dtype=np.uint8)
        img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)

    elif kind == 3:
        # Colourful random object (book, furniture)
        img = rng.integers(60, 220, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        for _ in range(int(rng.integers(2, 7))):
            x1 = int(rng.integers(0, IMG_WIDTH  - 20))
            y1 = int(rng.integers(0, IMG_HEIGHT - 20))
            x2 = x1 + int(rng.integers(10, 50))
            y2 = y1 + int(rng.integers(10, 50))
            img[y1:y2, x1:x2] = rng.integers(0, 255, 3).tolist()

    elif kind == 4:
        # Vehicle interior (dark grey + bright gauge circles)
        img = rng.integers(20, 60, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        for _ in range(int(rng.integers(1, 5))):
            cx = int(rng.integers(20, IMG_WIDTH  - 20))
            cy = int(rng.integers(20, IMG_HEIGHT - 20))
            r  = int(rng.integers(5, 18))
            for row in range(max(0, cy - r), min(IMG_HEIGHT, cy + r)):
                for col in range(max(0, cx - r), min(IMG_WIDTH, cx + r)):
                    if (row - cy)**2 + (col - cx)**2 <= r**2:
                        img[row, col] = [200, 200, 200]

    elif kind == 5:
        # Ceiling / sky: uniform bright blue or white
        img = np.full((IMG_HEIGHT, IMG_WIDTH, 3),
                      [int(rng.integers(150, 255)),
                       int(rng.integers(150, 255)),
                       int(rng.integers(100, 220))],
                      dtype=np.uint8)
        img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)

    elif kind == 6:
        # Mobile / computer screen — vivid pixel grid pattern
        img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        colors = [[255, 60, 60], [60, 255, 60], [60, 60, 255],
                  [255, 255, 60], [255, 60, 255], [60, 255, 255]]
        block = int(rng.integers(4, 16))
        for by in range(0, IMG_HEIGHT, block):
            for bx in range(0, IMG_WIDTH, block):
                c = colors[int(rng.integers(0, len(colors)))]
                img[by:by + block, bx:bx + block] = c

    else:
        # Book / paper with printed text lines
        img = np.full((IMG_HEIGHT, IMG_WIDTH, 3), 245, dtype=np.uint8)
        for row_y in range(10, IMG_HEIGHT, int(rng.integers(8, 18))):
            line_len = int(rng.integers(40, IMG_WIDTH - 10))
            start_x  = int(rng.integers(5, 15))
            img[row_y:row_y + 2, start_x:start_x + line_len] = [20, 20, 30]
        img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)

    return img.astype(np.uint8)


# ── Dataset Builder ────────────────────────────────────────────────────────────
def generate_synthetic_dataset(n_per_class=600, n_negatives=400):
    """
    Build dataset:
      n_per_class × normal road samples
      n_per_class × pothole samples
      n_per_class × crack samples
      n_negatives × non-road negative samples (labelled as class 0 / Normal)
    """
    rng = np.random.default_rng(42)
    X, y = [], []

    print("  Generating normal road samples...")
    for _ in range(n_per_class):
        X.append(generate_road_surface(rng)); y.append(0)

    print("  Generating pothole samples...")
    for _ in range(n_per_class):
        X.append(generate_pothole(rng)); y.append(1)

    print("  Generating crack samples...")
    for _ in range(n_per_class):
        X.append(generate_crack(rng)); y.append(2)

    print(f"  Adding {n_negatives} negative samples (8 non-road categories)...")
    for _ in range(n_negatives):
        X.append(generate_negative_sample(rng)); y.append(0)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)
    return X, y


# ── Numpy Data Augmentation ────────────────────────────────────────────────────
def augment_numpy(X, y, rng):
    """
    Apply numpy-based augmentation to training set.
    Avoids model-internal augmentation layers which conflict with
    EarlyStopping(restore_best_weights=True) deepcopy on TF 2.16 / Python 3.12.
    """
    aug_X, aug_y = [], []
    for img, label in zip(X, y):
        aug_X.append(img); aug_y.append(label)

        if rng.random() > 0.5:
            aug_X.append(np.fliplr(img)); aug_y.append(label)

        if rng.random() > 0.6:
            k = int(rng.integers(1, 4))
            aug_X.append(np.rot90(img, k=k)); aug_y.append(label)

        if rng.random() > 0.5:
            factor = 1.0 + rng.uniform(-0.15, 0.15)
            aug_X.append(np.clip(img * factor, 0, 1).astype(np.float32))
            aug_y.append(label)

    return np.array(aug_X, dtype=np.float32), np.array(aug_y, dtype=np.int32)


# ── MobileNetV2 CNN Architecture ──────────────────────────────────────────────
def build_cnn_model():
    """
    MobileNetV2 transfer learning for road damage classification.
    - Frozen MobileNetV2 backbone (ImageNet weights)
    - Custom trainable dense head with dropout
    - 3-class softmax output (normal, pothole, crack)
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze backbone

    model = models.Sequential(name='RoadDamageCNN_MobileNetV2')
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.50))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.40))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax', name='output'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Training ───────────────────────────────────────────────────────────────────
def train():
    print("=" * 65)
    print("  Road Damage CNN — MobileNetV2 Transfer Learning")
    print("=" * 65)

    print("\n[1/6] Generating synthetic dataset...")
    X, y = generate_synthetic_dataset(n_per_class=600, n_negatives=400)
    counts = np.bincount(y)
    print(f"      Total: {len(X)} samples | "
          f"Normal={counts[0]}, Pothole={counts[1]}, Crack={counts[2]}")

    print("[2/6] Train/validation split (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("[2b/6] Applying numpy augmentation to training set...")
    rng_aug = np.random.default_rng(999)
    X_train, y_train = augment_numpy(X_train, y_train, rng_aug)
    print(f"       Augmented training set: {len(X_train)} samples")

    # Class weights — compensate for extra normal samples (negatives → class 0)
    total = len(y_train)
    c0 = max(int(np.sum(y_train == 0)), 1)
    c1 = max(int(np.sum(y_train == 1)), 1)
    c2 = max(int(np.sum(y_train == 2)), 1)
    class_weight = {
        0: total / (3.0 * c0),
        1: total / (3.0 * c1),
        2: total / (3.0 * c2),
    }
    print(f"      Class weights: {class_weight}")

    print("[3/6] Building MobileNetV2 damage model...")
    model = build_cnn_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1)
    ]

    print("\n[4/6] Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[5/6] Saving training curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'],     label='Train Acc')
    axes[0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title('Model Accuracy (MobileNetV2)'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history['loss'],     label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss (MobileNetV2)'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), 'training_curves.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()

    print("[6/6] Saving confusion matrix...")
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    cm     = confusion_matrix(y_val, y_pred)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=CLASS_NAMES)
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax2, cmap='Blues', colorbar=False)
    ax2.set_title('Confusion Matrix — Road Damage CNN (MobileNetV2)')
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(__file__), 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=120)
    plt.close()

    print(f"\n[OK] Damage model saved  -> {MODEL_PATH}")
    print(f"[OK] Training curves     -> {plot_path}")
    print(f"[OK] Confusion matrix    -> {cm_path}")
    print("=" * 65)


if __name__ == '__main__':
    train()
