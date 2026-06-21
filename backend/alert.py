"""
Alert System
--------------
Driver and authority alert generator with granular severity levels:
  Low    — minor crack (hairline fracture, 85–90% confidence)
  Medium — moderate damage (90–95% crack OR 85–90% pothole)
  High   — large pothole (>90%) or deep crack (>95%)
  Critical — dangerous road damage (≥95% pothole)

Authority alerts are only sent when:
  1. Road validation passed
  2. Damage confidence >= 85%
  3. GPS coordinates are available (non-zero)

Features:
  - Structured JSON alert files saved in alerts/ subfolder
  - get_recent_alerts(n) — read last N authority alerts for dashboard feed
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

# ── Alert Logger (Authority Channel) ──────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH    = os.path.join(ROOT_DIR, 'alerts.log')
ALERTS_DIR  = os.path.join(ROOT_DIR, 'alerts')
os.makedirs(ALERTS_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
authority_logger = logging.getLogger('authority')

# ── Minimum GPS quality for authority alerts ───────────────────────────────────
MIN_GPS_PRECISION = 0.001   # ~100 m


def build_driver_alert(damage_type: str, severity: str,
                       confidence: float,
                       latitude: float, longitude: float) -> dict:
    """
    Build the driver-facing alert payload.
    Severity drives the message and visual colour.
    """
    ALERTS = {
        'normal': {
            'title':   'Road Clear',
            'message': 'No road damage detected. Safe to continue.',
            'color':   '#10b981',
            'icon':    'check_circle',
            'sound':   False,
            'action':  'CLEAR'
        },
        'Low': {
            'title':   'Minor Crack Detected',
            'message': (
                'A minor crack has been detected on the road surface. '
                'Reduce speed slightly and proceed with care.'
            ),
            'color':   '#f59e0b',
            'icon':    'warning',
            'sound':   True,
            'action':  'CAUTION'
        },
        'Medium': {
            'title':   'Moderate Road Damage Detected',
            'message': (
                'Moderate road damage detected ahead. '
                'Slow down, stay alert, and avoid the damaged area.'
            ),
            'color':   '#f97316',
            'icon':    'error',
            'sound':   True,
            'action':  'SLOW_DOWN'
        },
        'High': {
            'title':   'LARGE POTHOLE DETECTED!',
            'message': (
                'Dangerous pothole detected directly ahead! '
                'Brake immediately, reduce speed, and steer around the hazard.'
            ),
            'color':   '#ef4444',
            'icon':    'dangerous',
            'sound':   True,
            'action':  'BRAKE'
        },
        'Critical': {
            'title':   '⚠️ CRITICAL ROAD DAMAGE!',
            'message': (
                'CRITICAL: Extremely dangerous road damage detected! '
                'Emergency braking required. Avoid the area completely if possible.'
            ),
            'color':   '#dc2626',
            'icon':    'dangerous',
            'sound':   True,
            'action':  'EMERGENCY_BRAKE'
        }
    }

    key   = 'normal' if damage_type == 'normal' else severity
    alert = dict(ALERTS.get(key, ALERTS['normal']))
    alert.update({
        'damage_type': damage_type,
        'severity':    severity,
        'confidence':  confidence,
        'latitude':    latitude,
        'longitude':   longitude,
        'timestamp':   datetime.utcnow().isoformat(),
    })
    return alert


def send_driver_alert(damage_type: str, severity: str,
                      confidence: float,
                      latitude: float, longitude: float) -> dict:
    """Compose driver alert and return payload (for frontend display)."""
    alert = build_driver_alert(damage_type, severity, confidence,
                               latitude, longitude)
    print(f"[DRIVER ALERT] {alert['title']} | "
          f"Severity: {severity} | Confidence: {confidence:.1f}% | "
          f"GPS: ({latitude:.5f}, {longitude:.5f})")
    return alert


def send_authority_alert(record_id: int, damage_type: str,
                         severity: str, confidence: float,
                         latitude: float, longitude: float,
                         image_path: str, is_duplicate: bool = False):
    """
    Send government authority alert.

    Conditions for sending:
      - damage_type != 'normal'
      - confidence >= 85%
      - GPS is available (not 0.0, 0.0)

    Writes a structured JSON file to alerts/ directory.
    """
    if damage_type == 'normal':
        return   # No authority alert for clear roads

    if confidence < 85.0:
        print(f"[AUTHORITY] Alert suppressed — confidence {confidence:.1f}% < 85%")
        return

    gps_ok = abs(latitude) > MIN_GPS_PRECISION or abs(longitude) > MIN_GPS_PRECISION
    if not gps_ok:
        print("[AUTHORITY] Alert suppressed — GPS coordinates unavailable")
        return

    action  = "UPDATED" if is_duplicate else "NEW REPORT"
    ts_str  = datetime.utcnow().isoformat()
    payload = {
        'action':      action,
        'record_id':   record_id,
        'damage_type': damage_type,
        'severity':    severity,
        'confidence':  confidence,
        'location': {
            'latitude':  latitude,
            'longitude': longitude
        },
        'image_path':  image_path,
        'reported_at': ts_str
    }

    # Log to authority channel (alerts.log)
    authority_logger.warning(json.dumps(payload))

    # Write structured JSON alert file
    safe_ts  = ts_str.replace(':', '-').replace('.', '-')
    json_path = os.path.join(ALERTS_DIR, f"alert_{record_id}_{safe_ts}.json")
    try:
        with open(json_path, 'w') as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        print(f"[AUTHORITY] Could not write JSON alert: {e}")

    tag = "[DUPLICATE UPDATE]" if is_duplicate else "[NEW ALERT]"
    print(f"[AUTHORITY] {tag} Type={damage_type.upper()} | "
          f"Severity={severity} | Conf={confidence:.1f}% | "
          f"GPS=({latitude:.5f}, {longitude:.5f}) | ID=#{record_id}")
    print(f"[CLOUD] Record #{record_id} stored/updated in cloud database.")


def get_recent_alerts(n: int = 10) -> list:
    """
    Read the last N structured JSON alert files from the alerts/ directory.
    Returns list of alert dicts sorted newest-first.
    Used by the frontend Alert Feed panel.
    """
    try:
        files = sorted(
            Path(ALERTS_DIR).glob('alert_*.json'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:n]
        alerts = []
        for f in files:
            try:
                with open(f) as fp:
                    alerts.append(json.load(fp))
            except Exception:
                pass
        return alerts
    except Exception:
        return []
