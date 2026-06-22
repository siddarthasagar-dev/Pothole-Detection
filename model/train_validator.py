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
EPOCHS         = 4
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
                    img[row, col] = [0, 0, 0]
    return img


def gen_indoor(rng):
    """Indoor room patch: uniform background + rectangular shapes (furniture/books)."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    br  = int(rng.integers(180, 245))
    bg  = int(rng.integers(180, 245))
    bb  = int(rng.integers(180, 245))
    img[:] = [bb, bg, br]
    for _ in range(int(rng.integers(1, 5))):
        w = int(rng.integers(12, 45))
        h = int(rng.integers(12, 45))
        x = int(rng.integers(5, IMG_W - w - 5))
        y = int(rng.integers(5, IMG_H - h - 5))
        color = rng.integers(30, 180, 3, dtype=np.uint8)
        img[y:y + h, x:x + w] = color
    img = np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)
    return img


def gen_object(rng):
    """Vividly colored circles representing random household/outdoor objects."""
    img = rng.integers(220, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
    for _ in range(int(rng.integers(2, 6))):
        cx = int(rng.integers(20, IMG_W - 20))
        cy = int(rng.integers(20, IMG_H - 20))
        r  = int(rng.integers(10, 35))
        color = rng.integers(0, 255, 3, dtype=np.uint8)
        for row in range(max(0, cy - r), min(IMG_H, cy + r)):
            for col in range(max(0, cx - r), min(IMG_W, cx + r)):
                if (row - cy)**2 + (col - cx)**2 <= r**2:
                    img[row, col] = color
    img = np.clip(img + rng.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
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
def get_crops_from_image(img, max_crops_per_image=50, is_road=True):
    h, w = img.shape[:2]
    crops = []
    
    if is_road and h >= 256:
        y_start_min = int(h * 0.4)
    else:
        y_start_min = 0
        
    y_max = h - 128
    x_max = w - 128
    
    # Determine step size based on image size to get around 30-50 crops max per image
    if y_max > 0:
        y_step = max(32, y_max // 5)
    else:
        y_step = 32
        
    if x_max > 0:
        x_step = max(32, x_max // 8)
    else:
        x_step = 32
        
    y_range = range(y_start_min, y_max + 1, y_step) if y_max >= 0 else [0]
    x_range = range(0, x_max + 1, x_step) if x_max >= 0 else [0]
    
    for y_start in y_range:
        for x_start in x_range:
            if h < 128 or w < 128:
                crop = cv2.resize(img, (128, 128))
            else:
                crop = img[y_start:y_start+128, x_start:x_start+128]
            crops.append(crop)
            if len(crops) >= max_crops_per_image:
                return crops
    return crops


def generate_validator_dataset(n_road=1500, n_nonroad=1500):
    """
    Returns X (float32 array), y (int32 labels):
      0 = Road
      1 = Non-Road
    """
    rng = np.random.default_rng(123)
    X, y = [], []

    # 1. Load real road crops
    uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
    real_roads = [
        'f286f0f7d22546aa81b94b3ef8b89ceb.jpg',
        '593f4473a7a4480287f598a4b0398e76.jpg',
        'd84e06a5e3f44fb4b52552a0a1d285fa.jpg',
        'daa29be43b7747a8bfe82e6024da0375.jpeg',
        '44c16388cd9f4241a86d3c30214315b3.jpeg',
        '15b25bb04e3044b4b7ed291a550da5e7.jpeg',
        '4cdf6ae3d6634c278ca9ab1d3e3a9c02.jpeg',
        '21fa46c6ffde49ea993c4c3071ea82a5.jpeg',
        'c996a582d5be46de97ee35825f9ae819.jpeg',
        'e7ff9932d8b8437da5029d883dcff502.jpeg',
        '52624347e5564157866091660a1c438a.jpeg',
        '48e8321e209b467fa98fa46307e285d2.jpeg',
        'c07e6de2c687497d9c3e2d8a8158e984.jpeg',
        'eb5410d2ea9244d28da7f091b1cff28d.jpeg',
        '5b9b10295eae4e8197d64b97e1566f6e.jpeg',
        '6397c2e5a5c142baaabf7600c4087db6.jpeg',
        '2eeb8e76d578492ab2c54e57f961f36e.jpeg',
        '9365685c92874b789ec4b47c8ca6c30e.jpeg',
        'b72fcb269000489d89f6d7798dd0fdc8.jpg',
        'f62cdce087f548038f3edc2783942174.jpeg',
        'b26a188189294d6f90e91410b30f6084.jpeg',
        'c11fb53eb0a144e082fe3dd30f7fd325.jpg',
        'e932c4ced20c4ae883d370f88c7625a8.jpg',
        'cbc2bef49c31479ab7028c9645814899.jpeg',
        'ef9cb2ec89e141cdb78d5ac8633bd349.jpeg',
        '1a95f23c445348cbb7f0f5a65a85bd36.jpeg',
        '23bcf78d22864306aabdd3fe81943d94.jpeg',
        '3d473656eb174971bdf5fa35feb368a4.jpeg',
        'a24b1b4bbf404be79df127a0f38878d8.jpeg',
        '1d6ad1d2653447eda7bac44b3bafc17f.jpeg',
        'road_damage_new.jpg',
        'rural_road_new.jpg'
    ]
    
    real_non_roads = [
        '08504c8ff9a247db9b05a06a0ef0c112.png',
        '1246cfdbf30244158058311476782997.png',
        '125d80107165411792ccb9dc5f71c31b.png',
        '26a6a417e5204e2bb1b2dd737912f7fe.png',
        '39731cac18414de38ab03b90b7d837b6.png',
        '3abf487618eb493a811d81a2427a67a3.png',
        '4465861c6f5d41898e96ecc4bd1d3915.png',
        'd424d5b23f454d6b929c9d8b89e4f472.png',
        '5fa7c5e04a024b47ba543297e56b97d8.jpg',
        '27582a48968248dabe7266f70d856e4a.jpg',
        '85e3c41fadf74ff2a87f8da1b89c1be6.png',
        'krishna_silhouette_new.jpg'
    ]
    
    print("  Loading real road crops...")
    road_crops = 0
    for name in real_roads:
        path = os.path.join(uploads_dir, name)
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
            
        # Add resized whole image (crucial for matching predict.py validation input format)
        resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        X.append(resized)
        y.append(0)
        road_crops += 1
        
        crops = get_crops_from_image(img, max_crops_per_image=50, is_road=True)
        for crop in crops:
            X.append(crop)
            y.append(0)
            road_crops += 1
                
    print(f"    Loaded {road_crops} real road crops.")
    
    print("  Loading real non-road crops...")
    nonroad_crops = 0
    for name in real_non_roads:
        path = os.path.join(uploads_dir, name)
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
            
        # Add resized whole image
        resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        X.append(resized)
        y.append(1)
        nonroad_crops += 1
        
        crops = get_crops_from_image(img, max_crops_per_image=50, is_road=False)
        for crop in crops:
            X.append(crop)
            y.append(1)
            nonroad_crops += 1
                
    print(f"    Loaded {nonroad_crops} real non-road crops.")

    # 2. Add synthetic road samples to fill n_road
    remaining_road = max(0, n_road - road_crops)
    print(f"  Generating {remaining_road} synthetic road samples...")
    for _ in range(remaining_road):
        X.append(gen_road(rng))
        y.append(0)

    # 3. Add synthetic non-road samples to fill n_nonroad
    remaining_nonroad = max(0, n_nonroad - nonroad_crops)
    nonroad_gens = [
        gen_face, gen_indoor, gen_object, gen_vehicle_interior,
        gen_vegetation, gen_sky, gen_building,
        gen_mobile_screen, gen_book_paper
    ]
    per_gen = remaining_nonroad // len(nonroad_gens)
    remainder = remaining_nonroad % len(nonroad_gens)

    print(f"  Generating {remaining_nonroad} synthetic non-road samples...")
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
    X, y = generate_validator_dataset(n_road=1500, n_nonroad=1500)
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
