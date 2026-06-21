"""
Database Helper Module — SQLite ORM
-------------------------------------
SQLite helpers for road damage detection system.

Schema:
  id           INTEGER PRIMARY KEY
  image_path   TEXT    — relative path to saved image
  damage_type  TEXT    — 'normal', 'pothole', 'crack'
  confidence   REAL    — CNN confidence (0–100%)
  severity     TEXT    — 'None', 'Low', 'Medium', 'High', 'Critical'
  latitude     REAL    — GPS latitude
  longitude    REAL    — GPS longitude
  address      TEXT    — human-readable address
  report_count INTEGER — times this spot was reported
  is_duplicate INTEGER — 1 if merged with existing record
  timestamp    DATETIME

Duplicate detection:
  If a new detection is within 10 metres of an existing record
  AND both timestamps are within 24 hours:
    → Increment report_count on the existing record
    → Return existing record_id (do NOT create a new row)

Functions:
  - init_db()              — create / migrate tables
  - insert_record()        — insert or merge duplicate
  - find_duplicate()       — Haversine GPS duplicate check
  - get_all_records()      — paginated records
  - get_recent_records()   — last N damage records for alert feed
  - get_records_by_damage()— filter by damage type
  - get_stats()            — aggregate counts
  - get_severity_stats()   — counts by severity level
  - get_record_by_id()     — single record lookup
  - delete_record()        — remove a record
"""

import os
import math
import sqlite3
from datetime import datetime, timedelta

# ── DB Path ────────────────────────────────────────────────────────────────────
DB_DIR  = os.path.join(os.path.dirname(__file__), '..', 'database')
DB_PATH = os.path.join(DB_DIR, 'road_damage.db')

# Duplicate detection parameters
DUPLICATE_RADIUS_M = 10.0    # metres
DUPLICATE_HOURS    = 24      # hours


def get_connection() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create / migrate tables."""
    conn = get_connection()
    cur  = conn.cursor()

    # Create main table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS road_damage (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path   TEXT    NOT NULL,
            damage_type  TEXT    NOT NULL,
            confidence   REAL    DEFAULT 0.0,
            severity     TEXT    DEFAULT 'None',
            latitude     REAL    DEFAULT 0.0,
            longitude    REAL    DEFAULT 0.0,
            address      TEXT    DEFAULT '',
            report_count INTEGER DEFAULT 1,
            is_duplicate INTEGER DEFAULT 0,
            timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration: add new columns if upgrading from older schema
    existing = {row[1] for row in cur.execute("PRAGMA table_info(road_damage)")}
    for col, defn in [
        ('report_count', 'INTEGER DEFAULT 1'),
        ('is_duplicate', 'INTEGER DEFAULT 0'),
        ('address',      "TEXT DEFAULT ''"),
    ]:
        if col not in existing:
            cur.execute(f"ALTER TABLE road_damage ADD COLUMN {col} {defn}")
            print(f"[DB] Migrated: added column '{col}'")

    conn.commit()
    conn.close()
    print(f"[DB] Database ready at {DB_PATH}")


# ── Haversine distance ─────────────────────────────────────────────────────────
def _haversine_metres(lat1, lon1, lat2, lon2) -> float:
    """Return great-circle distance in metres between two GPS points."""
    R      = 6_371_000
    phi1   = math.radians(lat1)
    phi2   = math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def find_duplicate(damage_type: str, latitude: float, longitude: float) -> int | None:
    """
    Search for an existing record of the same damage type within
    DUPLICATE_RADIUS_M metres and within DUPLICATE_HOURS of now.

    Returns the existing record id, or None if no duplicate found.
    """
    if latitude == 0.0 and longitude == 0.0:
        return None   # No GPS — cannot check duplicates

    cutoff = (datetime.utcnow() - timedelta(hours=DUPLICATE_HOURS)).isoformat()
    conn   = get_connection()
    cur    = conn.cursor()
    cur.execute("""
        SELECT id, latitude, longitude
        FROM   road_damage
        WHERE  damage_type = ?
        AND    timestamp   >= ?
        AND    is_duplicate = 0
    """, (damage_type, cutoff))
    rows = cur.fetchall()
    conn.close()

    for row in rows:
        dist = _haversine_metres(latitude, longitude,
                                  row['latitude'], row['longitude'])
        if dist <= DUPLICATE_RADIUS_M:
            return row['id']
    return None


def increment_report_count(record_id: int):
    """Bump report_count and update timestamp on a duplicate."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        UPDATE road_damage
        SET    report_count = report_count + 1,
               timestamp    = ?
        WHERE  id = ?
    """, (datetime.utcnow().isoformat(), record_id))
    conn.commit()
    conn.close()


def insert_record(image_path: str, damage_type: str, confidence: float,
                  severity: str, latitude: float, longitude: float,
                  address: str = '') -> tuple[int, bool]:
    """
    Insert a new record OR merge with an existing duplicate.

    Returns (record_id, is_duplicate):
      is_duplicate=True  → existing record was updated (count incremented)
      is_duplicate=False → new record was created
    """
    # Duplicate merging disabled to ensure every scan gets its own image and unique history item
    # dup_id = find_duplicate(damage_type, latitude, longitude)
    # if dup_id is not None:
    #     increment_report_count(dup_id)
    #     print(f"[DB] Duplicate detected — updated record #{dup_id}")
    #     return dup_id, True

    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO road_damage
            (image_path, damage_type, confidence, severity,
             latitude, longitude, address, report_count, is_duplicate, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, ?)
    """, (image_path, damage_type, round(confidence, 2), severity,
          latitude, longitude, address, datetime.utcnow().isoformat()))
    conn.commit()
    record_id = cur.lastrowid
    conn.close()
    return record_id, False


def get_all_records(limit: int = 100) -> list:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT * FROM road_damage
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_recent_records(n: int = 10) -> list:
    """Return the last N records for the alert feed panel."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT id, damage_type, severity, confidence, latitude, longitude,
               address, timestamp
        FROM   road_damage
        WHERE  damage_type != 'normal'
        ORDER BY timestamp DESC
        LIMIT ?
    """, (n,))
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_records_by_damage(damage_type: str, limit: int = 100) -> list:
    """Filter records by damage type."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT * FROM road_damage
        WHERE  damage_type = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (damage_type, limit))
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def delete_record(record_id: int) -> bool:
    """Delete a record by ID. Returns True if a row was deleted."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("DELETE FROM road_damage WHERE id = ?", (record_id,))
    deleted = cur.rowcount > 0
    conn.commit()
    conn.close()
    if deleted:
        print(f"[DB] Record #{record_id} deleted.")
    return deleted


def get_stats() -> dict:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT damage_type, COUNT(*) as count
        FROM   road_damage
        GROUP BY damage_type
    """)
    rows = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM road_damage WHERE is_duplicate = 0")
    unique_count = cur.fetchone()[0]
    conn.close()

    stats = {'normal': 0, 'pothole': 0, 'crack': 0, 'total': 0,
             'unique_reports': unique_count}
    for row in rows:
        dt = row['damage_type']
        if dt in stats:
            stats[dt] = row['count']
        stats['total'] += row['count']
    return stats


def get_severity_stats() -> dict:
    """Return counts grouped by severity level. Used by analytics panel."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        SELECT severity, COUNT(*) as count
        FROM   road_damage
        WHERE  damage_type != 'normal'
        GROUP BY severity
    """)
    rows   = cur.fetchall()
    conn.close()
    result = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0, 'None': 0}
    for row in rows:
        sev = row['severity']
        if sev in result:
            result[sev] = row['count']
    return result


def get_record_by_id(record_id: int) -> dict | None:
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM road_damage WHERE id = ?", (record_id,))
    row  = cur.fetchone()
    conn.close()
    return dict(row) if row else None
