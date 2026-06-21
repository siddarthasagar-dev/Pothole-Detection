"""
Flask REST API — Smart Road Damage Detection
----------------------------------------------
Endpoints:
  POST /detect             — Full pipeline: validate → preprocess → detect → store
  POST /predict/debug      — Full pipeline + raw probability vectors from both CNNs
  GET  /damages            — All damage records (supports ?type=pothole|crack|normal)
  GET  /damages/export     — Download all records as CSV
  GET  /stats              — Aggregate counts + severity breakdown
  GET  /record/<id>        — Single record
  DELETE /record/<id>      — Delete a record
  GET  /alerts/recent      — Last 10 authority alert JSON objects
  GET  /health             — API health check
  GET  /uploads/<file>     — Serve uploaded images

Two-stage pipeline:
  [Edge]  → preprocess_for_model (CLAHE, NLM denoise, brightness norm, sharpen)
  [Edge]  → validate_road MobileNetV2 (rejects non-road images >= 80% confidence)
  [Edge]  → predict_damage MobileNetV2 (only reports damage if >= 85% confidence)
  [Cloud] → SQLite insert (with duplicate merging by GPS distance)
  [Cloud] → Authority alert log + JSON file
"""

import base64
import csv
import io
import os
import sys
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS

# ── Path Setup ──────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(ROOT_DIR, 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
sys.path.insert(0, ROOT_DIR)

from model.preprocess import preprocess_for_model
from model.predict    import predict_from_array, predict_debug, detect_and_annotate
from backend.db       import (init_db, insert_record, get_all_records,
                               get_stats, get_record_by_id, delete_record,
                               get_severity_stats, get_records_by_damage,
                               get_recent_records)
from backend.alert    import send_driver_alert, send_authority_alert, get_recent_alerts

# ── App ─────────────────────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)
init_db()

@app.route('/')
def index_route():
    return app.send_static_file('index.html')

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}


def allowed_file(name: str) -> bool:
    return '.' in name and name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── POST /detect ────────────────────────────────────────────────────────────────
@app.route('/detect', methods=['POST'])
def detect():
    """
    Main detection endpoint.

    Form-data:
      image     (file)  — road image
      latitude  (float) — GPS latitude
      longitude (float) — GPS longitude
      address   (str)   — human-readable address (optional)

    Successful response (200):
      { success, record_id, road_valid, road_conf, non_road_conf,
        damage_type, confidence, severity, above_threshold,
        all_scores, location, driver_alert, is_duplicate,
        edge_processing, cloud_stored, timestamp }

    Rejected non-road response (422):
      { success: false, error_type: 'not_road_image', error, road_conf, non_road_conf }
    """
    # Validate request
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'success': False,
            'error': f'Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}'
        }), 400

    file_bytes = file.read()
    if len(file_bytes) == 0:
        return jsonify({'success': False, 'error': 'Empty file'}), 400

    # Extract GPS coords
    try:
        latitude  = float(request.form.get('latitude',  0.0))
        longitude = float(request.form.get('longitude', 0.0))
    except (ValueError, TypeError):
        latitude, longitude = 0.0, 0.0
    address = request.form.get('address', '')

    # Two-stage MobileNetV2 CNN inference with ROI object detection
    print("[Edge] Running MobileNetV2 multi-stage detection pipeline...")
    try:
        prediction, annotated_bytes = detect_and_annotate(file_bytes)
    except Exception as e:
        return jsonify({'success': False,
                        'error': f'Detection pipeline failed: {str(e)}'}), 422

    # Stage 1 result: Road Validation
    if not prediction.get('road_valid', True):
        print(f"[Validate] REJECTED — {prediction['reject_reason'][:60]}...")
        return jsonify({
            'success':       False,
            'error_type':    'not_road_image',
            'error':         prediction['reject_reason'],
            'road_conf':     prediction.get('road_conf', 0),
            'non_road_conf': prediction.get('non_road_conf', 100),
        }), 422

    damage_type     = prediction['damage_type']
    confidence      = prediction['confidence']
    severity        = prediction['severity']
    all_scores      = prediction['all_scores']
    above_threshold = prediction.get('above_threshold', True)
    detections      = prediction.get('detections', [])
    rejected_dets   = prediction.get('rejected_detections', [])

    # Store all valid road scans in the database so they appear in history
    gps_available = abs(latitude) > 0.00001 or abs(longitude) > 0.00001
    should_store  = True

    record_id    = None
    is_duplicate = False
    cloud_stored = False
    filename     = None

    if should_store:
        ext      = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save the annotated image with bounding boxes instead of the raw image
        with open(save_path, 'wb') as f:
            f.write(annotated_bytes)

        rel_path = os.path.join('uploads', filename)
        record_id, is_duplicate = insert_record(
            rel_path, damage_type, confidence, severity,
            latitude, longitude, address
        )
        cloud_stored = True
        print(f"[Cloud] Record #{record_id} stored "
              f"({'duplicate merged' if is_duplicate else 'new'}).")
    else:
        if not gps_available:
            print("[Gate] Not stored — GPS location is not available")
        elif damage_type == 'normal':
            print("[Gate] Not stored — road is clean / normal")
        else:
            print(f"[Gate] Not stored — confidence {confidence:.1f}% < 85% threshold")

    # Alerts removed as requested
    driver_alert = None

    return jsonify({
        'success':         True,
        'record_id':       record_id,
        'road_valid':      True,
        'road_conf':       prediction.get('road_conf', 100),
        'non_road_conf':   prediction.get('non_road_conf', 0),
        'damage_type':     damage_type,
        'confidence':      confidence,
        'severity':        severity,
        'above_threshold': above_threshold,
        'all_scores':      all_scores,
        'detections':      detections,
        'rejected_detections': rejected_dets,
        'is_duplicate':    is_duplicate,
        'location': {
            'latitude':  latitude,
            'longitude': longitude,
            'address':   address
        },
        'driver_alert':    driver_alert,
        'cloud_stored':    cloud_stored,
        'edge_processing': True,
        'timestamp':       datetime.utcnow().isoformat(),
        'annotated_image': base64.b64encode(annotated_bytes).decode('utf-8')
    })


# ── POST /predict/debug ──────────────────────────────────────────────────────
@app.route('/predict/debug', methods=['POST'])
def predict_debug_route():
    """
    Debug endpoint — returns full probability vectors from both MobileNetV2 CNNs.
    Accepts same form-data as /detect (image file required).
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file provided'}), 400

    file       = request.files['image']
    file_bytes = file.read()
    if len(file_bytes) == 0:
        return jsonify({'success': False, 'error': 'Empty file'}), 400

    try:
        img_batch = preprocess_for_model(file_bytes)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Preprocessing failed: {str(e)}'}), 422

    debug_result = predict_debug(img_batch)
    return jsonify({'success': True, 'debug': debug_result})


# ── GET /damages ─────────────────────────────────────────────────────────────
@app.route('/damages', methods=['GET'])
def damages():
    damage_type = request.args.get('type', '').lower()
    if damage_type in ('pothole', 'crack', 'normal'):
        records = get_records_by_damage(damage_type, limit=100)
    else:
        records = get_all_records(limit=200)
    return jsonify({'success': True, 'count': len(records), 'records': records})


# ── GET /damages/export ──────────────────────────────────────────────────────
@app.route('/damages/export', methods=['GET'])
def export_csv():
    """Download all damage records as a CSV file."""
    records = get_all_records(limit=10000)
    output  = io.StringIO()
    writer  = csv.writer(output)

    # Header
    writer.writerow([
        'ID', 'Damage Type', 'Confidence (%)', 'Severity',
        'Latitude', 'Longitude', 'Address',
        'Report Count', 'Is Duplicate', 'Timestamp', 'Image Path'
    ])

    for r in records:
        writer.writerow([
            r.get('id', ''),
            r.get('damage_type', ''),
            r.get('confidence', ''),
            r.get('severity', ''),
            r.get('latitude', ''),
            r.get('longitude', ''),
            r.get('address', ''),
            r.get('report_count', 1),
            'Yes' if r.get('is_duplicate') else 'No',
            r.get('timestamp', ''),
            r.get('image_path', ''),
        ])

    ts       = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = f"road_damage_report_{ts}.csv"
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


# ── GET /stats ───────────────────────────────────────────────────────────────
@app.route('/stats', methods=['GET'])
def stats():
    base_stats = get_stats()
    sev_stats  = get_severity_stats()
    return jsonify({
        'success': True,
        'stats': {**base_stats, 'severity': sev_stats}
    })


# ── GET /record/<id> ─────────────────────────────────────────────────────────
@app.route('/record/<int:rid>', methods=['GET'])
def record(rid: int):
    row = get_record_by_id(rid)
    if row:
        return jsonify({'success': True, 'record': row})
    return jsonify({'success': False, 'error': 'Record not found'}), 404


# ── DELETE /record/<id> ──────────────────────────────────────────────────────
@app.route('/record/<int:rid>', methods=['DELETE'])
def delete_record_route(rid: int):
    """Delete a damage record by ID."""
    deleted = delete_record(rid)
    if deleted:
        return jsonify({'success': True, 'message': f'Record #{rid} deleted.'})
    return jsonify({'success': False, 'error': 'Record not found'}), 404


# ── GET /alerts/recent ───────────────────────────────────────────────────────
@app.route('/alerts/recent', methods=['GET'])
def recent_alerts():
    """Return last N authority alert JSON payloads for dashboard feed."""
    n       = int(request.args.get('n', 10))
    alerts  = get_recent_alerts(n)
    # Also include recent DB records for alerts that may not have files
    db_recs = get_recent_records(n)
    return jsonify({
        'success':    True,
        'alerts':     alerts,
        'recent_damage': db_recs
    })


# ── Serve uploads ────────────────────────────────────────────────────────────
@app.route('/uploads/<filename>')
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


# ── Health ───────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':    'ok',
        'version':   'MobileNetV2',
        'timestamp': datetime.utcnow().isoformat()
    })


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("  Smart Road Damage Detection — Flask API (MobileNetV2)")
    print("  Listening on http://localhost:5001")
    print("=" * 65)
    app.run(host='0.0.0.0', port=5001, debug=True)
