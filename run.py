"""
Smart Road Damage Detection & Warning System — Quick Launcher
--------------------------------------------------------------
Usage:
  python run.py                   # Start API (auto-train if models missing)
  python run.py --retrain         # Force re-train both CNNs, then start
  python run.py --retrain-damage  # Re-train damage model only
  python run.py --retrain-val     # Re-train validator model only

Requirements:
  Virtual environment: C:\\venv\\srd
  Install: pip install -r requirements.txt
"""

import subprocess
import sys
import os
import webbrowser
import time

ROOT         = os.path.dirname(os.path.abspath(__file__))

# Cross-platform python virtual environment path detection
if sys.platform == "win32":
    local_venv = os.path.join(ROOT, "venv", "Scripts", "python.exe")
    if os.path.exists(local_venv):
        VENV_PY = local_venv
    else:
        VENV_PY = r"C:\venv\srd\Scripts\python.exe"
else:
    local_venv = os.path.join(ROOT, "venv", "bin", "python")
    if os.path.exists(local_venv):
        VENV_PY = local_venv
    else:
        VENV_PY = "python3"

MODEL_DAMAGE = os.path.join(ROOT, "model", "road_damage_model.h5")
MODEL_VAL    = os.path.join(ROOT, "model", "road_validator.h5")

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   Smart Road Damage Detection & Warning System               ║
║   MobileNetV2 CNN + Edge Computing Architecture              ║
║   Road Validator + Damage Detector (Dual CNN Pipeline)       ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_cmd(cmd, **kw):
    return subprocess.run(cmd, **kw)


def check_model(path: str, name: str) -> bool:
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / 1_048_576
        print(f"  ✓ {name}: {path} ({size_mb:.2f} MB)")
        return True
    print(f"  ✗ {name}: NOT FOUND")
    return False


def train_model(script: str, label: str):
    print(f"\n[TRAIN] {label} (this may take 3–8 minutes on CPU)...")
    result = run_cmd([VENV_PY, script], cwd=ROOT)
    if result.returncode != 0:
        print(f"[ERROR] Training failed: {script}")
        sys.exit(1)
    print(f"[OK] {label} complete.")


def main():
    print(BANNER)
    args = sys.argv[1:]

    # Check venv
    if not os.path.exists(VENV_PY):
        print(f"[ERROR] Virtual environment not found at {VENV_PY}")
        print("  Run:  python -m venv C:\\venv\\srd")
        print("        C:\\venv\\srd\\Scripts\\pip install -r requirements.txt")
        sys.exit(1)
    print(f"[OK] Virtual environment found: {VENV_PY}\n")

    # Model status
    print("[Models]")
    damage_ok = check_model(MODEL_DAMAGE, "Road Damage CNN (MobileNetV2)")
    val_ok    = check_model(MODEL_VAL,    "Road Validator CNN (MobileNetV2)")
    print()

    # Determine what to train
    retrain_damage = '--retrain' in args or '--retrain-damage' in args or not damage_ok
    retrain_val    = '--retrain' in args or '--retrain-val' in args or not val_ok

    if retrain_val:
        train_model("model/train_validator.py", "Road Validator CNN (MobileNetV2)")
    if retrain_damage:
        train_model("model/train.py", "Road Damage CNN (MobileNetV2)")

    # Final model check
    print("\n[Models — Final Check]")
    check_model(MODEL_DAMAGE, "Road Damage CNN")
    check_model(MODEL_VAL,    "Road Validator CNN")
    print()

    # Start Flask API
    print("[INFO] Starting Flask API on http://localhost:5001 ...")
    proc = subprocess.Popen(
        [VENV_PY, "backend/app.py"],
        cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}
    )
    time.sleep(4)

    # Open dashboard
    dashboard_url = "http://localhost:5001/"
    print(f"[INFO] Opening dashboard: {dashboard_url}")
    webbrowser.open(dashboard_url)

    print("\n[INFO] System ready! Dashboard on http://localhost:5001")
    print("[INFO] Press Ctrl+C to stop the server.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n[INFO] Server stopped. Goodbye!")


if __name__ == '__main__':
    main()
