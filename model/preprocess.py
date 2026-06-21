"""
Edge Computing Image Preprocessing Pipeline
---------------------------------------------
Full edge-device preprocessing applied BEFORE sending data to the CNN.
Steps:
  1.  Load image (bytes or path)
  2.  Noise removal        — Gaussian blur + Non-local Means
  3.  Contrast Enhancement — CLAHE (Contrast Limited Adaptive Histogram Equalization)
  4.  Brightness Normalization — histogram stretching to [0, 255]
  5.  Sharpening           — Unsharp mask to enhance crack edges
  6.  Image Resizing       — to 128x128 (CNN input size)
  7.  Normalization        — pixel values to [0.0, 1.0]
  8.  Batch expansion      — (H,W,C) → (1,H,W,C)

Edge benefits:
  - Reduces noise before CNN inference → higher accuracy
  - Normalizes brightness/contrast variations (day/night/weather)
  - Sharpening improves visibility of fine cracks
  - Smaller normalized tensors → less cloud bandwidth
"""

import time
import cv2
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
IMG_HEIGHT = 128
IMG_WIDTH  = 128

# CLAHE parameters (Contrast Limited Adaptive Histogram Equalization)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)

# Non-local means denoising strength
NLM_H             = 6     # filter strength (lower = less denoising, more detail)
NLM_TEMPLATE_SIZE = 7
NLM_SEARCH_SIZE   = 21

# Sharpening strength (unsharp mask)
SHARPEN_AMOUNT = 0.6      # 0 = no sharpening, 1 = full sharpening


# ── Loaders ────────────────────────────────────────────────────────────────────
def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV BGR array."""
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def load_image_from_path(path: str) -> np.ndarray:
    """Load image from a filesystem path."""
    return cv2.imread(path)


# ── Step 1: Noise Removal ──────────────────────────────────────────────────────
def remove_noise(img: np.ndarray) -> np.ndarray:
    """
    Two-stage noise removal:
      a) Gaussian blur  — removes high-frequency sensor noise quickly.
      b) Non-local Means — preserves edges while smoothing textured noise.
    Applied on each BGR channel independently via fastNlMeansDenoisingColored.
    """
    blurred  = cv2.GaussianBlur(img, (3, 3), 0)
    denoised = cv2.fastNlMeansDenoisingColored(
        blurred, None,
        h=NLM_H, hColor=NLM_H,
        templateWindowSize=NLM_TEMPLATE_SIZE,
        searchWindowSize=NLM_SEARCH_SIZE
    )
    return denoised


# ── Step 2: Contrast Enhancement (CLAHE) ──────────────────────────────────────
def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE in the LAB colour space.
    CLAHE (Contrast Limited Adaptive Histogram Equalization):
      - Enhances local contrast in each image tile independently.
      - Prevents over-amplification of noise (clip limit = 2.0).
      - Applied only to the L (luminance) channel to preserve hue.
    This significantly improves visibility of cracks and potholes in
    dark, wet, or overcast road conditions.
    """
    lab    = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe  = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_eq   = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ── Step 3: Brightness Normalization ──────────────────────────────────────────
def normalize_brightness(img: np.ndarray) -> np.ndarray:
    """
    Stretch the histogram so the darkest pixel → 0 and brightest → 255.
    This compensates for:
      - Night-time / poorly lit road images (dark images become visible).
      - Overexposed midday images (washed-out areas become detailed).
    Applied per-channel so colour balance is maintained.
    """
    out = np.zeros_like(img, dtype=np.uint8)
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        lo = float(ch.min())
        hi = float(ch.max())
        if hi - lo < 1e-3:
            out[:, :, c] = ch.astype(np.uint8)
        else:
            stretched = ((ch - lo) / (hi - lo) * 255.0).clip(0, 255)
            out[:, :, c] = stretched.astype(np.uint8)
    return out


# ── Step 4: Sharpening (Unsharp Mask) ─────────────────────────────────────────
def sharpen_image(img: np.ndarray) -> np.ndarray:
    """
    Apply unsharp mask to enhance fine crack edges.

    Method:
      blurred = GaussianBlur(img, sigma=1.0)
      sharpened = img + amount * (img - blurred)

    Benefits:
      - Makes hairline cracks more prominent for CNN detection.
      - Does NOT affect smooth road surfaces significantly.
      - SHARPEN_AMOUNT=0.6 balances enhancement vs. noise amplification.
    """
    blurred   = cv2.GaussianBlur(img, (0, 0), 1.0)
    sharpened = cv2.addWeighted(img, 1.0 + SHARPEN_AMOUNT,
                                blurred, -SHARPEN_AMOUNT, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ── Step 5: Resize ─────────────────────────────────────────────────────────────
def resize_image(img: np.ndarray) -> np.ndarray:
    """Resize to CNN input dimensions (128×128) using INTER_AREA for downscaling."""
    return cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)


# ── Step 6: Pixel Normalization ────────────────────────────────────────────────
def normalize_pixels(img: np.ndarray) -> np.ndarray:
    """Scale pixel values from [0, 255] → [0.0, 1.0] (float32)."""
    return img.astype(np.float32) / 255.0


# ── Step 7: Segmentation (analysis only) ──────────────────────────────────────
def basic_segmentation(img: np.ndarray) -> np.ndarray:
    """
    Visual segmentation overlay using adaptive thresholding.
    Highlights candidate damage regions (cracks/potholes) for display.
    Not used in the CNN inference pipeline — for visualization only.
    """
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    mask_r  = np.zeros_like(thresh)
    mask_b  = thresh
    colored = cv2.merge([mask_r, mask_r, mask_b])
    return cv2.addWeighted(img, 0.8, colored, 0.2, 0)


# ── Full Edge Pipeline ─────────────────────────────────────────────────────────
def preprocess_for_model(file_bytes: bytes) -> np.ndarray:
    """
    Full edge-computing preprocessing pipeline (bytes input).
    Returns float32 batch tensor: shape (1, 128, 128, 3).

    Pipeline:
      load → denoise → CLAHE → brightness_norm → sharpen → resize → pixel_norm → batch
    """
    t0  = time.time()
    img = load_image_from_bytes(file_bytes)
    if img is None:
        raise ValueError("Could not decode image bytes.")

    img = remove_noise(img)
    img = enhance_contrast(img)
    img = normalize_brightness(img)
    img = sharpen_image(img)
    img = resize_image(img)
    img = normalize_pixels(img)

    elapsed = (time.time() - t0) * 1000
    print(f"[Edge] Preprocessing complete in {elapsed:.1f}ms")
    return np.expand_dims(img, axis=0)


def preprocess_from_path(path: str) -> np.ndarray:
    """
    Full edge-computing preprocessing pipeline (path input).
    Returns float32 batch tensor: shape (1, 128, 128, 3).
    """
    img = load_image_from_path(path)
    if img is None:
        raise ValueError(f"Could not load image from path: {path}")

    img = remove_noise(img)
    img = enhance_contrast(img)
    img = normalize_brightness(img)
    img = sharpen_image(img)
    img = resize_image(img)
    img = normalize_pixels(img)
    return np.expand_dims(img, axis=0)
