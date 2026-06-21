# Smart Road Damage Detection & Warning System

AI-powered road damage detection using **MobileNetV2 CNN** + **Edge Computing** with GPS tracking, real-time driver warnings, and government authority reporting.

## Features

- 🧠 **MobileNetV2 CNN** — Dual-model architecture (Road Validator + Damage Detector)
- ⚡ **Edge Computing** — OpenCV preprocessing (CLAHE, NLM denoise, brightness norm, sharpen)
- 📍 **GPS Tracking** — Browser geolocation with reverse geocoding
- 🗺️ **Leaflet Map** — Interactive GPS-pinned damage markers with severity colors
- 📊 **Dashboard** — Chart.js analytics with severity breakdown
- 📄 **Reports** — Authority report generation with GPS evidence
- 🔔 **Alerts** — Driver warnings + government authority notifications
- 💾 **SQLite** — Persistent storage with duplicate detection (10m radius)
- 🎨 **Dark Theme** — Professional glassmorphism UI with micro-animations

## Architecture

```
Image Capture → Preprocessing → Road Validation → Pothole/Crack Detection
→ GPS Capture → Database → Driver Warning → Authority Report → Dashboard
```

## Classes

| CNN | Class | Description |
|-----|-------|-------------|
| Validator | Road | Asphalt, concrete, highway, city roads |
| Validator | Non-Road | Faces, selfies, rooms, buildings, furniture, books, walls |
| Damage | Normal Road | Undamaged road surface |
| Damage | Pothole | Deep surface cavities |
| Damage | Crack | Surface fractures / linear damage |

## Quick Start

```bash
# 1. Create virtual environment
python -m venv C:\venv\srd
C:\venv\srd\Scripts\pip install -r requirements.txt

# 2. Launch (auto-trains models if missing)
python run.py
```

## Manual Setup

```bash
# Train models separately
C:\venv\srd\Scripts\python model/train_validator.py
C:\venv\srd\Scripts\python model/train.py

# Start API
C:\venv\srd\Scripts\python backend/app.py

# Open frontend
# Open frontend/index.html in browser
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/detect` | Full detection pipeline |
| POST | `/predict/debug` | Debug probabilities from both CNNs |
| GET | `/damages` | All damage records |
| GET | `/damages/export` | Download CSV |
| GET | `/stats` | Aggregate statistics |
| GET | `/record/<id>` | Single record |
| DELETE | `/record/<id>` | Delete record |
| GET | `/alerts/recent` | Authority alerts |
| GET | `/health` | API health check |

## Technology Stack

- **Backend**: Flask + SQLite3
- **Frontend**: HTML5 / CSS3 / JavaScript
- **AI/ML**: TensorFlow (MobileNetV2 transfer learning)
- **Computer Vision**: OpenCV
- **Maps**: Leaflet.js
- **Charts**: Chart.js
