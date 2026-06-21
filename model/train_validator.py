# -*- coding: utf-8 -*-
"""
Road Validator CNN — MobileNetV2 Transfer Learning
----------------------------------------------------
Binary classifier: Road vs Non-Road

Classes:
  0 — Road     (asphalt, concrete, rural, highway, city roads with damage)
  1 — Non-Road (faces, indoor rooms, buildings, walls, furniture,
                 vehicles, phones, sky, vegetation, random objects,
                 mobile screens, books/paper)

If the validator predicts Non-Road with confidence >= 80%:
  -> Image is REJECTED immediately.
  -> No damage detection is performed.
  -> No data is stored.
  -> No alert is sent.

Architecture: MobileNetV2 (ImageNet pretrained) + Custom Dense Head
Output: model/road_validator.h5
        model/validator_curves.png
        model/validator_confusion.png
"""

import os
import numpy as np
import cv2
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

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_H          = 128
IMG_W          = 128
EPOCHS         = 35
BATCH_SIZE     = 16
VALIDATOR_PATH = os.path.join(os.path.dirname(__file__), 'road_validator.h5')


# ── Road Sample Generator ──────────────────────────────────────────────────────
def gen_road(rng):
    """Synthetic road surface: grey asphalt with varied tone and markings."""
    base_val = int(rng.integers(50, 155))
    img = rng.integers(
        max(30, base_val - 35),
        min(210, base_val + 35),
        (IMG_H, IMG_W, 3), dtype=np.uint8
    )
    noise = rng.normal(0, 16, (IMG_H, IMG_W, 3))
    img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Road lane markings
    if rng.random() > 0.45:
        line_x = int(rng.integers(20, IMG_W - 20))
        width  = int(rng.integers(2, 7))
        img[:, line_x:line_x + width] = int(rng.integers(160, 235))

    # Damage features (roads with potholes/cracks are STILL road)
    if rng.random() > 0.35:
        cx = int(rng.integers(20, IMG_W - 20))
        cy = int(rng.integers(20, IMG_H - 20))
        r  = int(rng.integers(5, 28))
        for row in range(max(0, cy - r), min(IMG_H, cy + r)):
            for col in range(max(0, cx - r), min(IMG_W, cx + r)):
                if (row - cy)**2 + (col - cx)**2 <= r**2:
                    img[row, col] = rng.integers(15, 55, 3, dtype=np.uint8)

    # Wet road (darker, bluish sheen)
    if rng.random() > 0.75:
        img = np.clip(img.astype(int) - 20, 0, 255).astype(np.uint8)
        img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + 10, 0, 255)

    return img


# ── Non-Road Sample Generators ─────────────────────────────────────────────────
def gen_face(rng):
    """Skin-tone gradient + eye circles simulating a human face."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    sr  = int(rng.integers(155, 225))
    sg  = int(rng.integers(105, 170))
    sb  = int(rng.integers(75, 135))
    for i in range(IMG_H):
        gradient = 1.0 - 0.3 * abs(i - IMG_H // 2) / (IMG_H // 2)
        img[i] = [int(sb * gradient), int(sg * gradient), int(sr * gradient)]
    img = np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8)
    for _ in range(2):
        cx = int(rng.integers(IMG_W // 4, 3 * IMG_W // 4))
        cy = int(rng.integers(IMG_H // 4, IMG_H // 2))
        r  = int(rng.integers(5, 14))
        for row in range(max(0, cy - r), min(IMG_H, cy + r)):
            for col in range(max(0, cx - r), min(IMG_W, cx + r)):
                if (row - cy)**2 + (col - cx)**2 <= r**2:
                    img[row, col] = [20, 20, 30]
    return img


def gen_indoor(rng):
    """Indoor room: walls, furniture, tiled floors."""
    kind = int(rng.integers(0, 5))
    if kind == 0:
        colour = rng.integers(130, 255, 3, dtype=np.uint8)
        img = np.full((IMG_H, IMG_W, 3), colour.tolist(), dtype=np.uint8)
        img = np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
    elif kind == 1:
        img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        for i in range(0, IMG_H, int(rng.integers(4, 12))):
            val = int(rng.integers(110, 185))
            img[i:i + 3] = [30, val // 2, val]
        img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    elif kind == 2:
        img = rng.integers(90, 225, (IMG_H, IMG_W, 3), dtype=np.uint8)
        tile = int(rng.integers(10, 28))
        for i in range(0, IMG_H, tile):
            for j in range(0, IMG_W, tile):
                if (i // tile + j // tile) % 2 == 0:
                    img[i:i + tile, j:j + tile] = np.clip(
                        img[i:i + tile, j:j + tile].astype(int) + 45, 0, 255)
    elif kind == 3:
        val = int(rng.integers(210, 255))
        img = np.full((IMG_H, IMG_W, 3), val, dtype=np.uint8)
        img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    else:
        # Curtain / drape pattern
        img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        base = rng.integers(80, 200, 3, dtype=np.uint8).tolist()
        for col in range(IMG_W):
            shade = int(30 * np.sin(col / 8.0))
            img[:, col] = [max(0, min(255, c + shade)) for c in base]
    return img.astype(np.uint8)


def gen_object(rng):
    """Random objects: phone, book, laptop, bottle."""
    img = rng.integers(30, 200, (IMG_H, IMG_W, 3), dtype=np.uint8)
    for _ in range(int(rng.integers(2, 7))):
        x1 = int(rng.integers(0, IMG_W - 20))
        y1 = int(rng.integers(0, IMG_H - 20))
        x2 = min(x1 + int(rng.integers(15, 55)), IMG_W)
        y2 = min(y1 + int(rng.integers(15, 55)), IMG_H)
        img[y1:y2, x1:x2] = rng.integers(0, 255, 3).tolist()
    return img


def gen_vehicle_interior(rng):
    """Car/truck interior: dark with bright dashboard gauges."""
    img = rng.integers(15, 60, (IMG_H, IMG_W, 3), dtype=np.uint8)
    for _ in range(int(rng.integers(2, 6))):
        cx = int(rng.integers(10, IMG_W - 10))
        cy = int(rng.integers(10, IMG_H - 10))
        r  = int(rng.integers(8, 22))
        for row in range(max(0, cy - r), min(IMG_H, cy + r)):
            for col in range(max(0, cx - r), min(IMG_W, cx + r)):
                if (row - cy)**2 + (col - cx)**2 <= r**2:
                    img[row, col] = rng.integers(130, 255, 3, dtype=np.uint8)
    return img


def gen_vegetation(rng):
    """Grass, trees, plants (green-dominant)."""
    hsv           = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    hsv[:, :, 0]  = int(rng.integers(32, 82))
    hsv[:, :, 1]  = int(rng.integers(95, 205))
    hsv[:, :, 2]  = int(rng.integers(75, 185))
    noise         = rng.integers(-18, 18, (IMG_H, IMG_W, 3))
    hsv           = np.clip(hsv.astype(int) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def gen_sky(rng):
    """Clear or cloudy sky."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    b   = int(rng.integers(175, 255))
    g   = int(rng.integers(135, 205))
    r   = int(rng.integers(75, 145))
    img[:] = [b, g, r]
    # Clouds
    if rng.random() > 0.5:
        for _ in range(int(rng.integers(1, 4))):
            cx = int(rng.integers(0, IMG_W))
            cy = int(rng.integers(0, IMG_H // 2))
            rr = int(rng.integers(10, 30))
            for row in range(max(0, cy - rr), min(IMG_H, cy + rr)):
                for col in range(max(0, cx - rr), min(IMG_W, cx + rr)):
                    if (row - cy)**2 + (col - cx)**2 <= rr**2:
                        img[row, col] = [245, 245, 245]
    return np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


def gen_building(rng):
    """Building facade: bricks, window grids."""
    img = rng.integers(95, 185, (IMG_H, IMG_W, 3), dtype=np.uint8)
    win_h = int(rng.integers(8, 22))
    win_w = int(rng.integers(8, 22))
    gap_h = int(rng.integers(4, 12))
    gap_w = int(rng.integers(4, 12))
    for row in range(0, IMG_H, win_h + gap_h):
        for col in range(0, IMG_W, win_w + gap_w):
            img[row:row + win_h, col:col + win_w] = rng.integers(25, 85, 3, dtype=np.uint8)
    return img


def gen_mobile_screen(rng):
    """Mobile / computer screen: vivid pixel-grid pattern."""
    img    = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    colors = [[255, 60, 60], [60, 255, 60], [60, 60, 255],
              [255, 255, 60], [255, 60, 255], [60, 255, 255],
              [255, 165, 0], [200, 200, 200]]
    block  = int(rng.integers(3, 18))
    for by in range(0, IMG_H, block):
        for bx in range(0, IMG_W, block):
            c = colors[int(rng.integers(0, len(colors)))]
            img[by:by + block, bx:bx + block] = c
    return img


def gen_book_paper(rng):
    """Book / paper with text lines."""
    img = np.full((IMG_H, IMG_W, 3), 248, dtype=np.uint8)
    for row_y in range(10, IMG_H, int(rng.integers(8, 20))):
        line_len = int(rng.integers(40, IMG_W - 10))
        start_x  = int(rng.integers(5, 18))
        img[row_y:row_y + 2, start_x:start_x + line_len] = [20, 20, 30]
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img


# ── Dataset Builder ────────────────────────────────────────────────────────────
def generate_validator_dataset(n_road=600, n_nonroad=500):
    """
    Returns X (float32 array), y (int32 labels):
      0 = Road
      1 = Non-Road
    """
    rng = np.random.default_rng(123)
    X, y = [], []

    print(f"  Generating {n_road} road samples...")
    for _ in range(n_road):
        X.append(gen_road(rng)); y.append(0)

    nonroad_gens = [
        gen_face, gen_indoor, gen_object, gen_vehicle_interior,
        gen_vegetation, gen_sky, gen_building,
        gen_mobile_screen, gen_book_paper
    ]
    per_gen   = n_nonroad // len(nonroad_gens)
    remainder = n_nonroad % len(nonroad_gens)

    print(f"  Generating {n_nonroad} non-road samples "
          f"({len(nonroad_gens)} categories)...")
    for i, gen_fn in enumerate(nonroad_gens):
        count = per_gen + (1 if i < remainder else 0)
        for _ in range(count):
            X.append(gen_fn(rng)); y.append(1)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)
    return X, y


# ── MobileNetV2 Validator Architecture ─────────────────────────────────────────
def build_validator_model():
    """
    MobileNetV2 transfer learning for Road vs Non-Road classification.
    - Frozen MobileNetV2 backbone (ImageNet weights)
    - Custom trainable dense head
    - Binary softmax output
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_H, IMG_W, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze backbone

    model = models.Sequential(name='RoadValidator_MobileNetV2')
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.45))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(2, activation='softmax', name='road_nonroad'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Main ───────────────────────────────────────────────────────────────────────
def train():
    print("=" * 65)
    print("  Road Validator CNN — MobileNetV2 Transfer Learning")
    print("=" * 65)

    print("\n[1/6] Generating dataset...")
    X, y = generate_validator_dataset(n_road=600, n_nonroad=500)
    counts = np.bincount(y)
    print(f"      Road={counts[0]}, Non-Road={counts[1]}, Total={len(X)}")

    print("[2/6] Train/val split (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("[2b/6] Numpy augmentation on training set...")
    rng_aug = np.random.default_rng(77)
    aug_X, aug_y = [], []
    for img, label in zip(X_train, y_train):
        aug_X.append(img); aug_y.append(label)
        if rng_aug.random() > 0.5:
            aug_X.append(np.fliplr(img)); aug_y.append(label)
        if rng_aug.random() > 0.6:
            k = int(rng_aug.integers(1, 4))
            aug_X.append(np.rot90(img, k=k)); aug_y.append(label)
        if rng_aug.random() > 0.5:
            factor = 1.0 + rng_aug.uniform(-0.15, 0.15)
            aug_X.append(np.clip(img * factor, 0, 1).astype(np.float32))
            aug_y.append(label)
    X_train = np.array(aug_X, dtype=np.float32)
    y_train = np.array(aug_y, dtype=np.int32)
    print(f"       Augmented training set: {len(X_train)} samples")

    # Class weights
    total = len(y_train)
    c0 = max(np.sum(y_train == 0), 1)
    c1 = max(np.sum(y_train == 1), 1)
    class_weight = {0: total / (2.0 * c0), 1: total / (2.0 * c1)}
    print(f"      Class weights: {class_weight}")

    print("[3/6] Building MobileNetV2 validator model...")
    model = build_validator_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(VALIDATOR_PATH, monitor='val_accuracy',
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
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(history.history['accuracy'],     label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Validator Accuracy (MobileNetV2)'); axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(history.history['loss'],     label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Validator Loss (MobileNetV2)'); axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    out_plot = os.path.join(os.path.dirname(__file__), 'validator_curves.png')
    plt.savefig(out_plot, dpi=120); plt.close()

    print("[6/6] Saving confusion matrix...")
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    cm     = confusion_matrix(y_val, y_pred)
    disp   = ConfusionMatrixDisplay(confusion_matrix=cm,
                                    display_labels=['Road', 'Non-Road'])
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax2, cmap='Greens', colorbar=False)
    ax2.set_title('Confusion Matrix — Road Validator (MobileNetV2)')
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(__file__), 'validator_confusion.png')
    plt.savefig(cm_path, dpi=120); plt.close()

    print(f"\n[OK] Validator model  -> {VALIDATOR_PATH}")
    print(f"[OK] Training curves  -> {out_plot}")
    print(f"[OK] Confusion matrix -> {cm_path}")
    print("=" * 65)


if __name__ == '__main__':
    train()
