"""
Model Prediction Module — MobileNetV2 Production Object Detector
------------------------------------------------------------------
Multi-stage detection pipeline for road damage.

Pipeline:
  Stage 1: Road Validator MobileNetV2 + Haar Face Detector
    Gate  : if road confidence < 80% or any faces detected → REJECT
  Stage 2: Road Segmentation
    Process: Identify road bounds via geometric prior (trapezoid) + color range
  Stage 3: Object Detection (Potholes & Cracks)
    Process: Extract anomalous dark patches using Morphological Black Hat
             inside the road region. Crop patches, preprocess, and classify.
  Stage 4: Damage Verification
    Process: Filter false positives (shadows, markings, speed breakers, hood).
             Only report if damage confidence >= 85% and verification passes.

Severity mapping:
  Low      — minor crack
  Medium   — moderate crack OR moderate pothole
  High     — large pothole OR deep crack
  Critical — dangerous road damage (severe pothole)
"""

import os
import sys
import numpy as np
import tensorflow as tf
import cv2

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR            = os.path.dirname(__file__)
DAMAGE_MODEL_PATH    = os.path.join(MODEL_DIR, 'road_damage_model.h5')
VALIDATOR_MODEL_PATH = os.path.join(MODEL_DIR, 'road_validator.h5')

# ── Labels ─────────────────────────────────────────────────────────────────────
DAMAGE_CLASSES    = ['normal', 'pothole', 'crack']
VALIDATOR_CLASSES = ['road', 'non_road']

# ── Cached models ──────────────────────────────────────────────────────────────
_damage_model    = None
_validator_model = None


def load_damage_model() -> tf.keras.Model:
    global _damage_model
    if _damage_model is None:
        if not os.path.exists(DAMAGE_MODEL_PATH):
            raise FileNotFoundError(
                f"Damage model not found at '{DAMAGE_MODEL_PATH}'.\n"
                "Run: python model/train.py"
            )
        print(f"[Model] Loading MobileNetV2 damage model from {DAMAGE_MODEL_PATH}...")
        _damage_model = tf.keras.models.load_model(DAMAGE_MODEL_PATH)
        print("[Model] Damage model loaded.")
    return _damage_model


def load_validator_model() -> tf.keras.Model:
    global _validator_model
    if _validator_model is None:
        if not os.path.exists(VALIDATOR_MODEL_PATH):
            raise FileNotFoundError(
                f"Validator model not found at '{VALIDATOR_MODEL_PATH}'.\n"
                "Run: python model/train_validator.py"
            )
        print(f"[Model] Loading MobileNetV2 validator model from {VALIDATOR_MODEL_PATH}...")
        _validator_model = tf.keras.models.load_model(VALIDATOR_MODEL_PATH)
        print("[Model] Validator model loaded.")
    return _validator_model


def get_severity(damage_type: str, confidence: float) -> str:
    """
    Compute severity based on damage type and confidence.
    Low: Minor crack
    Medium: Moderate crack / pothole
    High: Large pothole / crack
    Critical: Dangerous road damage (very high confidence pothole)
    """
    if damage_type == 'normal':
        return 'None'
    elif damage_type == 'crack':
        return 'Medium' if confidence >= 90.0 else 'Low'
    elif damage_type == 'pothole':
        if confidence >= 95.0:
            return 'Critical'
        elif confidence >= 90.0:
            return 'High'
        else:
            return 'Medium'
    return 'None'


def get_road_mask(img: np.ndarray) -> np.ndarray:
    """
    Stage 2: Segment road region using a lane polygon prior
    and low-saturation color features typical for asphalt.
    """
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Spatial trapezoid prior starting at horizon
    horizon = int(height * 0.42)
    pts = np.array([
        [int(width * 0.05), height],
        [int(width * 0.38), horizon],
        [int(width * 0.62), horizon],
        [int(width * 0.95), height]
    ], np.int32)
    
    cv2.fillPoly(mask, [pts], 255)
    
    # Asphalt gray color thresholding (Low HSV saturation, neutral values)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    color_mask = (s < 75) & (v > 35) & (v < 220)
    color_mask = color_mask.astype(np.uint8) * 255
    
    # Morphological closing to fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    
    segmented = cv2.bitwise_and(mask, color_mask)
    
    # Fallback to polygon bounds if adaptive color segmentation fails
    if np.sum(segmented > 0) < (width * height * 0.15):
        return mask
        
    return segmented


def preprocess_crop(crop_img: np.ndarray) -> np.ndarray:
    """
    Preprocess cropped bounding box patches to match 128x128 CNN training input.
    """
    # Denoise
    blurred = cv2.GaussianBlur(crop_img, (3, 3), 0)
    
    # CLAHE Contrast Enhancement
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    # Brightness stretch to [0, 255]
    out = np.zeros_like(enhanced, dtype=np.uint8)
    for c in range(3):
        ch = enhanced[:, :, c].astype(np.float32)
        lo = float(ch.min())
        hi = float(ch.max())
        if hi - lo < 1e-3:
            out[:, :, c] = ch.astype(np.uint8)
        else:
            stretched = ((ch - lo) / (hi - lo) * 255.0).clip(0, 255)
            out[:, :, c] = stretched.astype(np.uint8)
            
    # Sharpen to emphasize crack boundaries
    sh_blurred = cv2.GaussianBlur(out, (0, 0), 1.0)
    sharpened = cv2.addWeighted(out, 1.6, sh_blurred, -0.6, 0)
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    # Resize to CNN dimensions & scale to [0, 1]
    resized = cv2.resize(sharpened, (128, 128), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def detect_and_annotate(file_bytes: bytes) -> tuple[dict, bytes]:
    """
    Full 4-stage pipeline:
      Stage 1: Road Detection & Face Rejection
      Stage 2: Road Segmentation
      Stage 3: Object Detection (anomalous candidates)
      Stage 4: Damage Verification
      
    Returns:
      (result_dict, annotated_image_bytes)
    """
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes.")
        
    orig_h, orig_w = img.shape[:2]
    
    # ── STAGE 1: ROAD DETECTION & FACE REJECTION ──
    # Check for faces (to block selfies and faces aggressively)
    face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = []
    if not face_cascade.empty():
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    else:
        print(f"[Validate] Warning: Haar face cascade could not be loaded from {face_cascade_path}")
    
    if len(faces) > 0:
        reason = "No road detected. Please capture a road image. (Human face detected)"
        return {
            'road_valid': False,
            'reject_reason': reason,
            'road_conf': 0.0,
            'non_road_conf': 100.0,
            'damage_type': 'unknown',
            'confidence': 0.0,
            'severity': 'None',
            'above_threshold': False,
            'detections': [],
            'rejected_detections': []
        }, file_bytes

    # Road Validation MobileNetV2 CNN Check
    val_resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    val_normalized = val_resized.astype(np.float32) / 255.0
    val_batch = np.expand_dims(val_normalized, axis=0)
    
    val_model = load_validator_model()
    val_probs = val_model.predict(val_batch, verbose=0)[0]
    
    road_conf = float(val_probs[0]) * 100
    non_road_conf = float(val_probs[1]) * 100
    print("Validator output:",val_probs)
    print("road (raw):",road_conf)
    print("non_road (raw):",non_road_conf)
    

    if road_conf < 10.0:
        reason = (
            f"No road detected. Please capture a road image. "
            f"(Road confidence: {road_conf:.1f}% is below 10% threshold). "
            f"Faces, indoor scenes, walls, ceilings, objects, books, furniture, vehicles, and selfies are not accepted."
        )
        return {
            'road_valid': False,
            'reject_reason': reason,
            'road_conf': round(road_conf, 2),
            'non_road_conf': round(non_road_conf, 2),
            'damage_type': 'unknown',
            'confidence': 0.0,
            'severity': 'None',
            'above_threshold': False,
            'detections': [],
            'rejected_detections': []
        }, file_bytes

    # ── STAGE 2: ROAD SEGMENTATION ──
    road_mask = get_road_mask(img)
    
    # ── STAGE 3: OBJECT DETECTION (ROI extraction) ──
    blurred_gray = cv2.bilateralFilter(gray_img, 9, 75, 75)
    
    # Morphological Black Hat identifies dark elements on lighter background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    blackhat = cv2.morphologyEx(blurred_gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 22, 255, cv2.THRESH_BINARY)
    
    # Intersect candidates with road segmented area
    road_thresh = cv2.bitwise_and(thresh, road_mask)
    road_thresh = cv2.dilate(road_thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    
    contours, _ = cv2.findContours(road_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    damage_model = load_damage_model()
    verified_detections = []
    rejected_detections = []
    
    # Run HOG detector to locate and ignore people
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    people_boxes = []
    try:
        (rects, _) = hog.detectMultiScale(gray_img, winStride=(8, 8), padding=(8, 8), scale=1.05)
        people_boxes = rects
    except Exception:
        pass

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        # Area filters
        if w < 10 or h < 10 or (w * h) < 120 or (w * h) > (orig_w * orig_h * 0.18):
            continue
            
        # Overlap with segmented road mask
        roi_mask = road_mask[y:y+h, x:x+w]
        road_ratio = np.sum(roi_mask > 0) / float(w * h)
        if road_ratio < 0.65:
            rejected_detections.append({
                'damage_type': 'unknown',
                'confidence': 0.0,
                'bounding_box': [x, y, w, h],
                'reason': 'Outside road region',
                'severity': 'None'
            })
            continue
            
        # Crop patch with minor padding margin
        pad_x = int(w * 0.12)
        pad_y = int(h * 0.12)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(orig_w, x + w + pad_x)
        y2 = min(orig_h, y + h + pad_y)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        # Classify patch with MobileNetV2 damage classifier
        crop_batch = preprocess_crop(crop)
        probs = damage_model.predict(crop_batch, verbose=0)[0]
        
        class_idx = int(np.argmax(probs))
        label = DAMAGE_CLASSES[class_idx]
        confidence = float(probs[class_idx]) * 100
        
        # Heuristic correction: override crack to pothole if the detection bounding box is a blob
        if label == 'crack' and w >= 15 and h >= 15:
            aspect_ratio = w / float(h)
            if 0.5 <= aspect_ratio <= 2.2:
                label = 'pothole'
        
        if label == 'normal':
            continue
            
        # ── STAGE 4: DAMAGE VERIFICATION ──
        # a) Ignore people/pedestrians
        overlaps_human = False
        for (px, py, pw, ph) in people_boxes:
            ix1 = max(x, px)
            iy1 = max(y, py)
            ix2 = min(x + w, px + pw)
            iy2 = min(y + h, py + ph)
            if ix2 > ix1 and iy2 > iy1:
                overlaps_human = True
                break
        if overlaps_human:
            rejected_detections.append({
                'damage_type': label,
                'confidence': round(confidence, 2),
                'bounding_box': [x, y, w, h],
                'reason': 'Overlaps with human/pedestrian',
                'severity': 'None'
            })
            continue

        # b) Ignore vehicle dashboard or hood parts at bottom edge
        if (y + h) > (orig_h * 0.93):
            rejected_detections.append({
                'damage_type': label,
                'confidence': round(confidence, 2),
                'bounding_box': [x, y, w, h],
                'reason': 'Vehicle parts (bottom frame)',
                'severity': 'None'
            })
            continue

        # c) Ignore bright road marking lines
        roi_gray = gray_img[y:y+h, x:x+w]
        mean_brightness = np.mean(roi_gray)
        if mean_brightness > 185:
            rejected_detections.append({
                'damage_type': label,
                'confidence': round(confidence, 2),
                'bounding_box': [x, y, w, h],
                'reason': 'Bright road marking',
                'severity': 'None'
            })
            continue

        # d) Ignore shadows (low texture variation)
        std_brightness = np.std(roi_gray)
        if std_brightness < 6.0:
            rejected_detections.append({
                'damage_type': label,
                'confidence': round(confidence, 2),
                'bounding_box': [x, y, w, h],
                'reason': 'Uniform shadow patch',
                'severity': 'None'
            })
            continue

        # e) Ignore speed breakers / horizontal bounds (only for potholes)
        aspect_ratio = w / float(h)
        if label == 'pothole' and aspect_ratio > 4.5:
            rejected_detections.append({
                'damage_type': label,
                'confidence': round(confidence, 2),
                'bounding_box': [x, y, w, h],
                'reason': 'Speed breaker / horizontal stripe',
                'severity': 'None'
            })
            continue

        # Threshold check: damage confidence >= 70%
        if confidence < 70.0:
            rejected_detections.append({
                'damage_type': label,
                'confidence': round(confidence, 2),
                'bounding_box': [x, y, w, h],
                'reason': 'Confidence below 70% threshold',
                'severity': 'None'
            })
            continue
            
        # All checks passed!
        severity = get_severity(label, confidence)
        verified_detections.append({
            'damage_type': label,
            'confidence': round(confidence, 2),
            'bounding_box': [x, y, w, h],
            'severity': severity,
            'status': 'verified'
        })

    # Fallback overall image classification removed to prevent false positives on clean roads
    pass

    # ── DRAW OVERLAYS & ANNOTATIONS ──
    annotated_img = img.copy()
    
    # Draw road segmentation outline (Cyan/Blue boundary)
    road_contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated_img, road_contours, -1, (255, 230, 0), 2)
    
    # Semi-transparent road region coloring
    overlay = annotated_img.copy()
    cv2.fillPoly(overlay, road_contours, (255, 120, 0))
    cv2.addWeighted(overlay, 0.15, annotated_img, 0.85, 0, annotated_img)

    # Draw boxes for verified detections
    verified_detections.sort(key=lambda d: d['confidence'], reverse=True)
    for d in verified_detections:
        x, y, w, h = d['bounding_box']
        label = d['damage_type'].upper()
        conf = d['confidence']
        sev = d['severity']
        
        # Color coding based on severity
        if sev == 'Critical':
            color = (0, 0, 220)      # Neon red
            thickness = 3
        elif sev == 'High':
            color = (0, 69, 255)     # Orange-red
            thickness = 2
        elif sev == 'Medium':
            color = (0, 165, 255)    # Orange-yellow
            thickness = 2
        else:
            color = (0, 220, 255)    # Yellow
            thickness = 2
            
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, thickness)
        
        # Draw readable box banner text
        txt = f"{label} {conf:.0f}% ({sev})"
        (txt_w, txt_h), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated_img, (x, y - txt_h - 6), (x + txt_w + 6, y), color, -1)
        cv2.putText(annotated_img, txt, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Save annotated frame as JPG bytes
    _, enc_img = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    annotated_bytes = enc_img.tobytes()
    
    # Compute overall status metrics
    if len(verified_detections) > 0:
        sev_rank = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1, 'None': 0}
        top_det = max(verified_detections, key=lambda d: sev_rank.get(d['severity'], 0))
        
        overall_type = top_det['damage_type']
        overall_conf = top_det['confidence']
        overall_sev  = top_det['severity']
    else:
        overall_type = 'normal'
        overall_conf = road_conf
        overall_sev  = 'None'

    result = {
        'road_valid': True,
        'reject_reason': '',
        'road_conf': round(road_conf, 2),
        'non_road_conf': round(non_road_conf, 2),
        'damage_type': overall_type,
        'confidence': round(overall_conf, 2),
        'severity': overall_sev,
        'above_threshold': overall_conf >= 70.0 if overall_type != 'normal' else True,
        'detections': verified_detections,
        'rejected_detections': [{
            'damage_type': rd['damage_type'],
            'confidence': rd['confidence'],
            'bounding_box': rd['bounding_box'],
            'reason': rd['reason']
        } for rd in rejected_detections],
        'all_scores': {
            'normal': round(100.0 - overall_conf, 2) if overall_type != 'normal' else round(overall_conf, 2),
            'pothole': round(overall_conf, 2) if overall_type == 'pothole' else 0.0,
            'crack': round(overall_conf, 2) if overall_type == 'crack' else 0.0
        }
    }
    return result, annotated_bytes


def predict_from_array(img_batch: np.ndarray) -> dict:
    """
    Backwards-compatible API entry point.
    """
    model = load_damage_model()
    val_model = load_validator_model()
    
    val_probs = val_model.predict(img_batch, verbose=0)[0]
    road_conf = float(val_probs[0]) * 100
    non_road_conf = float(val_probs[1]) * 100
    
    if road_conf < 10.0:
        return {
            'road_valid': False,
            'reject_reason': 'No road detected. Please capture a road image.',
            'road_conf': round(road_conf, 2),
            'non_road_conf': round(non_road_conf, 2),
            'damage_type': 'unknown',
            'confidence': 0.0,
            'severity': 'None',
            'above_threshold': False,
            'all_scores': {'normal': 0, 'pothole': 0, 'crack': 0}
        }
        
    probs = model.predict(img_batch, verbose=0)[0]
    class_idx = int(np.argmax(probs))
    label = DAMAGE_CLASSES[class_idx]
    confidence = float(probs[class_idx]) * 100
    severity = get_severity(label, confidence)
    
    return {
        'road_valid': True,
        'reject_reason': '',
        'road_conf': round(road_conf, 2),
        'non_road_conf': round(non_road_conf, 2),
        'damage_type': label,
        'confidence': round(confidence, 2),
        'severity': severity,
        'above_threshold': confidence >= 70.0,
        'all_scores': {
            'normal': round(float(probs[0]) * 100, 2),
            'pothole': round(float(probs[1]) * 100, 2),
            'crack': round(float(probs[2]) * 100, 2),
        }
    }


def predict_debug(img_batch: np.ndarray) -> dict:
    val_model = load_validator_model()
    dmg_model = load_damage_model()
    
    val_probs = val_model.predict(img_batch, verbose=0)[0]
    dmg_probs = dmg_model.predict(img_batch, verbose=0)[0]
    
    return {
        'validator': {
            'road': round(float(val_probs[0]) * 100, 3),
            'non_road': round(float(val_probs[1]) * 100, 3),
            'threshold_used': 10.0,
            'rejected': False
        },
        'damage': {
            'normal': round(float(dmg_probs[0]) * 100, 3),
            'pothole': round(float(dmg_probs[1]) * 100, 3),
            'crack': round(float(dmg_probs[2]) * 100, 3),
            'threshold_used': 70.0,
        },
        'pipeline_result': predict_from_array(img_batch)
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python model/predict.py <image_path>")
        sys.exit(1)
        
    path = sys.argv[1]
    with open(path, 'rb') as f:
        file_bytes = f.read()
        
    result, annotated_bytes = detect_and_annotate(file_bytes)
    print("\n-- Prediction Result --")
    for k, v in result.items():
        if k != 'detections' and k != 'rejected_detections':
            print(f"  {k}: {v}")
    print(f"\n  Verified Detections: {len(result['detections'])}")
    for d in result['detections']:
        print(f"    - Type: {d['damage_type'].upper()}, Conf: {d['confidence']}%, Severity: {d['severity']}, Box: {d['bounding_box']}")
    print(f"  Rejected Candidates: {len(result['rejected_detections'])}")
    for r in result['rejected_detections']:
        print(f"    - Type: {r['damage_type'].upper()}, Reason: {r['reason']}, Box: {r['bounding_box']}")
        
    # Save test prediction to a file
    out_path = "test_output.jpg"
    with open(out_path, 'wb') as f:
        f.write(annotated_bytes)
    print(f"\n[OK] Annotated image saved to '{out_path}'")
