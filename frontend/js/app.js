/**
 * Smart Road Damage Detection System
 * 
 * Features:
 * - Tab switching (Scanner, Dashboard, Map, History, Reports, Alerts)
 * - File upload with drag & drop
 * - Camera capture (front/rear)
 * - GPS acquisition (browser Geolocation API)
 * - Detection API call with animated pipeline steps
 * - Result rendering (gauge, scores, severity, bounding boxes)
 * - Chart.js dashboard analytics
 * - Leaflet.js GPS map with severity-coded markers
 * - History table with filtering and pagination
 * - Authority reports generation
 * - CSV export, record deletion
 * - Toast notification system
 */

// ── Configuration ──────────────────────────────────────────────────────────────
const API_BASE = window.location.origin;
const ROWS_PER_PAGE = 15;

// ── State ──────────────────────────────────────────────────────────────────────
let currentFile = null;
let gpsLatitude = 0.0;
let gpsLongitude = 0.0;
let gpsAddress = '';
let leafletMap = null;
let mapMarkers = [];
let damageChart = null;
let historyData = [];
let currentPage = 1;
let cameraStream = null;
let cameraFacing = 'environment';

// ══════════════════════════════════════════════════════════════════
//  INITIALIZATION
// ══════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  acquireGPS();
  loadStats();
  loadHistory();
  startClock();
  setupDragDrop();
  setupFileInput();

  // Auto-refresh
  setInterval(checkHealth, 30000);
  setInterval(() => {
    const active = document.querySelector('.tab-btn.active');
    if (active && active.id === 'tab-dashboard') loadStats();
  }, 15000);
});


// ── Clock ──────────────────────────────────────────────────────────────────────
function startClock() {
  const el = document.getElementById('nav-clock');
  const tick = () => {
    const now = new Date();
    el.textContent = now.toLocaleTimeString('en-US', { hour12: true });
  };
  tick();
  setInterval(tick, 1000);
}


// ── Health Check ───────────────────────────────────────────────────────────────
async function checkHealth() {
  const dot    = document.querySelector('.status-dot');
  const status = document.getElementById('api-status');
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.status === 'ok') {
      dot.classList.add('online');
      status.textContent = 'Online';
    }
  } catch {
    dot.classList.remove('online');
    status.textContent = 'Offline';
  }
}


// ══════════════════════════════════════════════════════════════════
//  TAB SWITCHING
// ══════════════════════════════════════════════════════════════════

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.remove('active');
    b.setAttribute('aria-selected', 'false');
  });
  document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));

  const btn = document.getElementById('tab-' + tab);
  const sec = document.getElementById('section-' + tab);
  if (btn) { btn.classList.add('active'); btn.setAttribute('aria-selected', 'true'); }
  if (sec) sec.classList.add('active');

  // Lazy-load tab content
  if (tab === 'dashboard') { loadStats(); loadAlertFeed(); }
  if (tab === 'map') { initMap(); loadMarkersFromHistory(); }
  if (tab === 'alerts') loadAlertFeedFull();
}


// ══════════════════════════════════════════════════════════════════
//  FILE UPLOAD & DRAG/DROP
// ══════════════════════════════════════════════════════════════════

function setupDragDrop() {
  const zone = document.getElementById('upload-zone');
  
  // Clicking the zone opens the file selector
  zone.addEventListener('click', () => {
    document.getElementById('file-input').click();
  });
  
  // Prevent default drag/drop behaviors on window to avoid browser navigation
  ['dragover', 'drop'].forEach(eventName => {
    window.addEventListener(eventName, e => e.preventDefault(), false);
  });

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
}

function setupFileInput() {
  document.getElementById('file-input').addEventListener('change', e => {
    if (e.target.files.length) handleFile(e.target.files[0]);
  });
}

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    showToast('Invalid File', 'Please select an image file.', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('File Too Large', 'Maximum file size is 10 MB.', 'error');
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('image-preview').src = e.target.result;
    document.getElementById('image-preview-container').style.display = 'block';
    document.getElementById('upload-zone').style.display = 'none';
    document.getElementById('btn-clear').style.display = 'inline-flex';
    document.getElementById('btn-detect').disabled = false;
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  currentFile = null;
  document.getElementById('image-preview').src = '';
  document.getElementById('image-preview-container').style.display = 'none';
  document.getElementById('upload-zone').style.display = 'block';
  document.getElementById('btn-clear').style.display = 'none';
  document.getElementById('btn-detect').disabled = true;
  document.getElementById('file-input').value = '';
  const banner = document.getElementById('validation-banner');
  if (banner) banner.style.display = 'none';
}

function clearUploadZone() {
  currentFile = null;
  document.getElementById('image-preview').src = '';
  document.getElementById('image-preview-container').style.display = 'none';
  document.getElementById('upload-zone').style.display = 'block';
  document.getElementById('btn-clear').style.display = 'none';
  document.getElementById('btn-detect').disabled = true;
  document.getElementById('file-input').value = '';
}


// ══════════════════════════════════════════════════════════════════
//  GPS ACQUISITION
// ══════════════════════════════════════════════════════════════════

function acquireGPS() {
  const display = document.getElementById('gps-display');
  const status  = document.getElementById('gps-status');

  if (!navigator.geolocation) {
    display.textContent = 'Geolocation not supported';
    status.textContent = 'Unavailable';
    status.className = 'gps-status error';
    return;
  }

  display.textContent = 'Acquiring…';
  status.textContent = 'Searching';
  status.className = 'gps-status';

  navigator.geolocation.getCurrentPosition(
    pos => {
      gpsLatitude  = pos.coords.latitude;
      gpsLongitude = pos.coords.longitude;
      display.textContent = `${gpsLatitude.toFixed(6)}, ${gpsLongitude.toFixed(6)}`;
      status.textContent = 'Acquired';
      status.className = 'gps-status acquired';
      // Reverse geocode (optional — uses free API)
      reverseGeocode(gpsLatitude, gpsLongitude);
    },
    err => {
      display.textContent = `Error: ${err.message}`;
      status.textContent = 'Failed';
      status.className = 'gps-status error';
      // Fallback: Default coords (Bangalore)
      gpsLatitude = 12.9716;
      gpsLongitude = 77.5946;
    },
    { enableHighAccuracy: true, timeout: 10000, maximumAge: 30000 }
  );
}

async function reverseGeocode(lat, lon) {
  try {
    const res = await fetch(`https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lon}&zoom=16`);
    const data = await res.json();
    if (data.display_name) {
      gpsAddress = data.display_name;
    }
  } catch { /* silent fail */ }
}


// ══════════════════════════════════════════════════════════════════
//  CAMERA
// ══════════════════════════════════════════════════════════════════

async function openCamera() {
  const modal = document.getElementById('camera-modal');
  const video = document.getElementById('camera-video');
  modal.classList.add('active');

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: cameraFacing, width: { ideal: 1280 }, height: { ideal: 720 } }
    });
    video.srcObject = cameraStream;
  } catch (err) {
    showToast('Camera Error', err.message, 'error');
    closeCamera();
  }
}

function closeCamera() {
  const modal = document.getElementById('camera-modal');
  const video = document.getElementById('camera-video');
  modal.classList.remove('active');
  if (cameraStream) {
    cameraStream.getTracks().forEach(t => t.stop());
    cameraStream = null;
  }
  video.srcObject = null;
}

function toggleCameraFacing() {
  cameraFacing = cameraFacing === 'environment' ? 'user' : 'environment';
  closeCamera();
  setTimeout(openCamera, 300);
}

function capturePhoto() {
  const video  = document.getElementById('camera-video');
  const canvas = document.getElementById('camera-canvas');
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  canvas.toBlob(blob => {
    const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
    handleFile(file);
    closeCamera();
  }, 'image/jpeg', 0.92);
}


// ══════════════════════════════════════════════════════════════════
//  DETECTION PIPELINE
// ══════════════════════════════════════════════════════════════════

async function runDetection() {
  if (!currentFile) return;

  const overlay = document.getElementById('loading-overlay');
  overlay.classList.add('active');
  
  // Hide validation banner on retry
  document.getElementById('validation-banner').style.display = 'none';

  const formData = new FormData();
  formData.append('image', currentFile);
  formData.append('latitude', gpsLatitude);
  formData.append('longitude', gpsLongitude);
  formData.append('address', gpsAddress);

  try {
    const res = await fetch(`${API_BASE}/detect`, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok || !data.success) {
      // Non-road rejection
      if (data.error_type === 'not_road_image') {
        overlay.classList.remove('active');
        showRejection(data);
        return;
      }
      throw new Error(data.error || 'Detection failed');
    }

    overlay.classList.remove('active');

    // On accepted & detected: Automatically clear the scanner image
    clearImage();

    // Update stats and history, then automatically open the popup for the new scan result
    loadStats();
    loadHistory().then(() => {
      if (data.record_id) {
        openHistoryPopup(data.record_id);
      }
    });

    showToast(
      data.damage_type === 'normal' ? 'Road Clear ✅' : `${data.damage_type.toUpperCase()} Detected!`,
      `Confidence: ${data.confidence.toFixed(1)}% | Severity: ${data.severity}`,
      data.damage_type === 'normal' ? 'success' : 'warning'
    );

  } catch (err) {
    overlay.classList.remove('active');
    showToast('Detection Error', err.message, 'error');
  }
}


// ══════════════════════════════════════════════════════════════════
//  RENDER RESULT
// ══════════════════════════════════════════════════════════════════

function showRejection(data) {
  const banner = document.getElementById('validation-banner');
  if (banner) {
    banner.style.display = 'flex';
    banner.className = 'validation-banner invalid';
    banner.innerHTML = `❌ <strong>REJECTED</strong> — ${data.error || 'Not a road image'}`;
  }
  showToast('Image Rejected', 'Not a road image. Please capture a road surface.', 'error');
}

function renderResult(data) {
  document.getElementById('result-placeholder').style.display = 'none';
  document.getElementById('result-card').style.display = 'block';
  document.getElementById('btn-debug').style.display = 'inline-flex';

  // Validation banner
  const banner = document.getElementById('validation-banner');
  banner.style.display = 'flex';
  banner.className = 'validation-banner valid';
  banner.innerHTML = `✅ Road validated`;

  // Validation chip
  const chip = document.getElementById('val-chip');
  chip.className = 'val-chip valid';
  chip.textContent = '🛡️ Road Validated';
  document.getElementById('val-scores').textContent =
    `Road: ${(data.road_conf || 0).toFixed(1)}% | Non-Road: ${(data.non_road_conf || 0).toFixed(1)}%`;

  // Result badge
  const badge = document.getElementById('result-badge');
  const type = data.damage_type;
  badge.textContent = type === 'normal' ? '✅ Normal Road' : type === 'pothole' ? '🕳️ Pothole Detected' : '⚡ Crack Detected';
  badge.className = 'result-badge ' + type;

  // Gauge
  const conf = data.confidence || 0;
  const arcLen = 172.79;
  const offset = arcLen - (arcLen * conf / 100);
  document.getElementById('gauge-fill').style.strokeDashoffset = offset;
  document.getElementById('gauge-label').textContent = conf.toFixed(1) + '%';

  // Confidence bar
  document.getElementById('conf-fill').style.width = conf + '%';
  document.getElementById('conf-pct').textContent = conf.toFixed(1) + '%';

  // Scores
  const scores = data.all_scores || {};
  document.getElementById('score-normal').textContent  = (scores.normal  || 0).toFixed(1) + '%';
  document.getElementById('score-pothole').textContent = (scores.pothole || 0).toFixed(1) + '%';
  document.getElementById('score-crack').textContent   = (scores.crack   || 0).toFixed(1) + '%';

  // Severity block
  const sevBlock = document.getElementById('severity-block');
  if (type !== 'normal' && data.severity && data.severity !== 'None') {
    sevBlock.style.display = 'flex';
    sevBlock.className = 'severity-block ' + data.severity.toLowerCase();
    document.getElementById('sev-icon').textContent = getSeverityIcon(data.severity);
    document.getElementById('sev-label').textContent = data.severity + ' Severity';
    document.getElementById('sev-msg').textContent = getSeverityMessage(data.severity, type);
  } else {
    sevBlock.style.display = 'none';
  }

  // Driver alert
  const alertBanner = document.getElementById('alert-banner');
  if (data.driver_alert) {
    alertBanner.style.display = 'block';
    alertBanner.style.borderColor = data.driver_alert.color || 'var(--accent-blue)';
    alertBanner.style.background = hexToRgba(data.driver_alert.color || '#3b82f6', 0.08);
    document.getElementById('alert-title').textContent = data.driver_alert.title || '';
    document.getElementById('alert-title').style.color = data.driver_alert.color || 'var(--text-primary)';
    document.getElementById('alert-msg').textContent = data.driver_alert.message || '';
  } else {
    alertBanner.style.display = 'none';
  }

  // Detections list
  renderDetections(data.detections || []);
  renderRejections(data.rejected_detections || []);

  // Meta row
  document.getElementById('meta-duplicate').style.display = data.is_duplicate ? 'inline-block' : 'none';
  document.getElementById('meta-cloud').textContent = data.cloud_stored ? '☁️ Stored' : '☁️ Not Stored';
  document.getElementById('meta-time').textContent = '🕐 ' + new Date().toLocaleTimeString();
}

function renderDetections(dets) {
  const wrap = document.getElementById('detections-list-wrap');
  const list = document.getElementById('detections-list');
  if (!dets.length) { wrap.style.display = 'none'; return; }
  wrap.style.display = 'block';
  list.innerHTML = dets.map(d => `
    <div style="display:flex;justify-content:space-between;align-items:center;padding:.3rem .5rem;background:rgba(16,185,129,.06);border-radius:4px;font-size:.72rem">
      <span><strong style="color:${d.damage_type === 'pothole' ? 'var(--accent-red)' : 'var(--accent-yellow)'}">${d.damage_type.toUpperCase()}</strong> ${d.confidence}%</span>
      <span class="sev-chip ${d.severity}">${d.severity}</span>
    </div>
  `).join('');
}

function renderRejections(rejs) {
  const wrap = document.getElementById('rejections-list-wrap');
  const list = document.getElementById('rejections-list');
  if (!rejs.length) { wrap.style.display = 'none'; return; }
  wrap.style.display = 'block';
  list.innerHTML = rejs.map(r => `
    <div style="display:flex;justify-content:space-between;align-items:center;padding:.3rem .5rem;background:rgba(239,68,68,.04);border-radius:4px;font-size:.72rem">
      <span style="color:var(--text-muted)">${(r.damage_type || 'unknown').toUpperCase()} ${r.confidence || 0}%</span>
      <span style="color:var(--accent-red);font-size:.65rem">${r.reason}</span>
    </div>
  `).join('');
}

function getSeverityIcon(sev) {
  return { Critical: '🚨', High: '🔴', Medium: '🟠', Low: '🟡' }[sev] || '⚠️';
}

function getSeverityMessage(sev, type) {
  const msgs = {
    Critical: 'CRITICAL: Emergency braking required! Extremely dangerous road damage.',
    High:     'Dangerous pothole detected! Brake immediately and steer around.',
    Medium:   'Moderate road damage. Slow down and stay alert.',
    Low:      'Minor crack detected. Proceed with care.'
  };
  return msgs[sev] || 'Road damage detected.';
}


// ══════════════════════════════════════════════════════════════════
//  DEBUG PANEL
// ══════════════════════════════════════════════════════════════════

function toggleDebug() {
  const panel = document.getElementById('debug-panel');
  if (panel.style.display === 'block') {
    panel.style.display = 'none';
    return;
  }
  panel.style.display = 'block';
  panel.textContent = 'Loading debug info…';

  if (!currentFile) return;
  const fd = new FormData();
  fd.append('image', currentFile);
  fetch(`${API_BASE}/predict/debug`, { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      panel.textContent = JSON.stringify(data.debug || data, null, 2);
    })
    .catch(e => { panel.textContent = 'Error: ' + e.message; });
}


// ══════════════════════════════════════════════════════════════════
//  DASHBOARD STATS & CHARTS
// ══════════════════════════════════════════════════════════════════

async function loadStats() {
  try {
    const res  = await fetch(`${API_BASE}/stats`);
    const data = await res.json();
    if (!data.success) return;

    const s = data.stats;
    // Scanner stats
    document.getElementById('stat-total').textContent   = s.total || 0;
    document.getElementById('stat-pothole').textContent = s.pothole || 0;
    document.getElementById('stat-crack').textContent   = s.crack || 0;
    document.getElementById('stat-normal').textContent  = s.normal || 0;

    // Dashboard stats
    const dt = document.getElementById('dash-total');
    if (dt) dt.textContent = s.total || 0;
    const du = document.getElementById('dash-unique');
    if (du) du.textContent = s.unique_reports || 0;
    const dp = document.getElementById('dash-pothole');
    if (dp) dp.textContent = s.pothole || 0;
    const dc = document.getElementById('dash-crack');
    if (dc) dc.textContent = s.crack || 0;

    // Severity
    if (s.severity) {
      const sc = document.getElementById('sev-critical');
      if (sc) sc.textContent = s.severity.Critical || 0;
      const sh = document.getElementById('sev-high');
      if (sh) sh.textContent = s.severity.High || 0;
      const sm = document.getElementById('sev-med');
      if (sm) sm.textContent = s.severity.Medium || 0;
      const sl = document.getElementById('sev-low');
      if (sl) sl.textContent = s.severity.Low || 0;
    }

    // Chart
    updateChart(s);
  } catch { /* silent */ }
}

function updateChart(stats) {
  const canvas = document.getElementById('damage-chart');
  if (!canvas) return;

  const chartData = {
    labels: ['Normal', 'Pothole', 'Crack'],
    datasets: [{
      data: [stats.normal || 0, stats.pothole || 0, stats.crack || 0],
      backgroundColor: ['rgba(16,185,129,0.75)', 'rgba(239,68,68,0.75)', 'rgba(245,158,11,0.75)'],
      borderColor: ['#10b981', '#ef4444', '#f59e0b'],
      borderWidth: 2,
      hoverOffset: 8,
    }]
  };

  if (damageChart) {
    damageChart.data = chartData;
    damageChart.update();
  } else {
    damageChart = new Chart(canvas, {
      type: 'doughnut',
      data: chartData,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '60%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 }, padding: 16, usePointStyle: true }
          }
        }
      }
    });
  }
}


// ══════════════════════════════════════════════════════════════════
//  ALERT FEED
// ══════════════════════════════════════════════════════════════════

async function loadAlertFeed() {
  try {
    const res  = await fetch(`${API_BASE}/alerts/recent?n=10`);
    const data = await res.json();
    const container = document.getElementById('alert-feed');
    const items = data.recent_damage || [];
    if (!items.length) { container.innerHTML = '<div class="feed-empty">No alerts yet</div>'; return; }
    container.innerHTML = items.map(a => buildFeedCard(a)).join('');
  } catch { /* silent */ }
}

async function loadAlertFeedFull() {
  try {
    const res  = await fetch(`${API_BASE}/alerts/recent?n=50`);
    const data = await res.json();
    const container = document.getElementById('alert-feed-full');
    const items = [...(data.alerts || []), ...(data.recent_damage || [])];
    if (!items.length) { container.innerHTML = '<div class="feed-empty">No authority alerts yet.</div>'; return; }
    // Deduplicate by record_id / id
    const seen = new Set();
    const unique = items.filter(i => {
      const id = i.record_id || i.id;
      if (seen.has(id)) return false;
      seen.add(id);
      return true;
    });
    container.innerHTML = unique.map(a => buildFeedCard(a)).join('');
  } catch { /* silent */ }
}

function buildFeedCard(a) {
  const type = a.damage_type || 'unknown';
  const sev  = a.severity || 'None';
  const conf = a.confidence ? a.confidence.toFixed(1) + '%' : '—';
  const time = a.reported_at || a.timestamp || '';
  const lat  = a.location ? a.location.latitude : (a.latitude || 0);
  const lon  = a.location ? a.location.longitude : (a.longitude || 0);
  const fmtTime = time ? safeFormatDateTime(time) : '—';

  return `
    <div class="feed-card ${type}">
      <div class="feed-card-header">
        <div class="feed-card-title" style="color:${type === 'pothole' ? 'var(--accent-red)' : type === 'crack' ? 'var(--accent-yellow)' : 'var(--accent-green)'}">
          ${type === 'pothole' ? '🕳️' : type === 'crack' ? '⚡' : '✅'} ${type.toUpperCase()} — ${sev}
        </div>
        <div class="feed-card-time">${fmtTime}</div>
      </div>
      <div class="feed-card-body">
        Confidence: ${conf} · GPS: (${Number(lat).toFixed(5)}, ${Number(lon).toFixed(5)})
        ${a.action ? ` · ${a.action}` : ''}
      </div>
    </div>`;
}


// ══════════════════════════════════════════════════════════════════
//  LEAFLET MAP
// ══════════════════════════════════════════════════════════════════

function initMap() {
  if (leafletMap) { leafletMap.invalidateSize(); return; }
  const lat = gpsLatitude || 12.9716;
  const lon = gpsLongitude || 77.5946;
  leafletMap = L.map('map').setView([lat, lon], 13);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19
  }).addTo(leafletMap);

  setTimeout(() => leafletMap.invalidateSize(), 300);
}

async function loadMarkersFromHistory() {
  if (!leafletMap) initMap();
  clearMapMarkers();

  try {
    const res  = await fetch(`${API_BASE}/damages`);
    const data = await res.json();
    if (!data.success || !data.records) return;

    data.records.forEach(r => {
      if (!r.latitude || !r.longitude) return;
      if (Math.abs(r.latitude) < 0.001 && Math.abs(r.longitude) < 0.001) return;

      const color = getMarkerColor(r.damage_type, r.severity);
      const icon = L.divIcon({
        className: 'custom-marker',
        html: `<div style="width:16px;height:16px;background:${color};border-radius:50%;border:2px solid #fff;box-shadow:0 0 8px ${color}"></div>`,
        iconSize: [16, 16]
      });

      const marker = L.marker([r.latitude, r.longitude], { icon })
        .addTo(leafletMap)
        .bindPopup(`
          <div style="font-family:Inter,sans-serif;font-size:13px">
            <strong style="color:${color}">${(r.damage_type || '').toUpperCase()}</strong><br/>
            Confidence: ${(r.confidence || 0).toFixed(1)}%<br/>
            Severity: ${r.severity || 'None'}<br/>
            Reports: ${r.report_count || 1}<br/>
            <small>${safeFormatDateTime(r.timestamp)}</small>
          </div>
        `);
      mapMarkers.push(marker);
    });

    if (mapMarkers.length) {
      const group = L.featureGroup(mapMarkers);
      leafletMap.fitBounds(group.getBounds().pad(0.15));
    }
  } catch { /* silent */ }
}

function clearMapMarkers() {
  mapMarkers.forEach(m => leafletMap.removeLayer(m));
  mapMarkers = [];
}

function getMarkerColor(type, severity) {
  if (type === 'normal') return '#10b981';
  if (severity === 'Critical') return '#dc2626';
  if (severity === 'High') return '#ef4444';
  if (severity === 'Medium') return '#f97316';
  if (severity === 'Low') return '#f59e0b';
  return '#3b82f6';
}


// ══════════════════════════════════════════════════════════════════
//  RECENT SCAN HISTORY
// ══════════════════════════════════════════════════════════════════

function cleanTimestampString(tsString) {
  if (!tsString) return '';
  let cleanString = tsString.replace(' ', 'T');
  
  const dotIndex = cleanString.indexOf('.');
  if (dotIndex !== -1) {
    let beforeDot = cleanString.substring(0, dotIndex);
    let afterDot = cleanString.substring(dotIndex + 1);
    let nonDigitIndex = afterDot.search(/\D/);
    let digits = '';
    let suffix = '';
    if (nonDigitIndex !== -1) {
      digits = afterDot.substring(0, nonDigitIndex);
      suffix = afterDot.substring(nonDigitIndex);
    } else {
      digits = afterDot;
    }
    digits = digits.substring(0, 3);
    cleanString = beforeDot + '.' + digits + suffix;
  }
  
  const tIndex = cleanString.indexOf('T');
  if (tIndex !== -1) {
    const timePart = cleanString.substring(tIndex + 1);
    if (!timePart.includes('Z') && !timePart.includes('+') && !timePart.includes('-')) {
      cleanString += 'Z';
    }
  } else {
    if (!cleanString.includes('Z') && !cleanString.includes('+') && !cleanString.includes('-')) {
      cleanString += 'Z';
    }
  }
  return cleanString;
}

function safeFormatTime(tsString) {
  if (!tsString) return '—';
  try {
    const cleanString = cleanTimestampString(tsString);
    const date = new Date(cleanString);
    if (isNaN(date.getTime())) {
      const match = tsString.match(/\d{2}:\d{2}/);
      return match ? match[0] : '—';
    }
    return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
  } catch (e) {
    console.error('Error parsing time:', e);
    return '—';
  }
}

function safeFormatDateTime(tsString) {
  if (!tsString) return '—';
  try {
    const cleanString = cleanTimestampString(tsString);
    const date = new Date(cleanString);
    if (isNaN(date.getTime())) {
      return tsString;
    }
    return date.toLocaleString();
  } catch (e) {
    console.error('Error parsing datetime:', e);
    return tsString || '—';
  }
}

async function loadHistory() {
  try {
    const res  = await fetch(`${API_BASE}/damages`);
    const data = await res.json();
    historyData = data.records || [];
    renderRecentHistory();
    return historyData;
  } catch (err) {
    console.error('Failed to load history:', err);
    return [];
  }
}

function renderRecentHistory() {
  const list = document.getElementById('recent-history-list');
  if (!list) return;

  if (!historyData.length) {
    list.innerHTML = `<div style="text-align:center;color:var(--text-muted);font-size:0.75rem;padding:1.5rem">No scans recorded yet</div>`;
    return;
  }

  list.innerHTML = historyData.map(r => {
    const type = r.damage_type || 'unknown';
    const conf = (r.confidence || 0).toFixed(1);
    const sev  = r.severity || 'None';
    const ts   = safeFormatTime(r.timestamp);
    const icon = type === 'normal' ? '✅' : type === 'pothole' ? '🕳️' : '⚡';
    
    return `
      <div class="history-item-card" onclick="openHistoryPopup(${r.id})" style="display:flex;align-items:center;justify-content:space-between;padding:0.6rem 0.75rem;background:rgba(255,255,255,0.02);border:1px solid var(--border);border-radius:var(--radius-sm);cursor:pointer;transition:background 0.2s,border-color 0.2s;">
        <div style="display:flex;align-items:center;gap:0.5rem;">
          <span style="font-size:1rem">${icon}</span>
          <div>
            <div style="font-size:0.75rem;font-weight:700;color:var(--text-primary);text-transform:capitalize;">${type} (${conf}%)</div>
            <div style="font-size:0.65rem;color:var(--text-muted);">${ts}</div>
          </div>
        </div>
        <span class="sev-chip ${sev}" style="font-size:0.65rem;padding:0.15rem 0.35rem;border-radius:3px;">${sev}</span>
      </div>`;
  }).join('');
}

function openHistoryPopup(recordId) {
  const record = historyData.find(r => r.id === recordId);
  if (!record) {
    showToast('Error', 'Record not found locally.', 'error');
    return;
  }
  
  document.getElementById('history-detail-id').textContent = `#${record.id}`;
  document.getElementById('history-detail-type').textContent = record.damage_type;
  document.getElementById('history-detail-conf').textContent = `${(record.confidence || 0).toFixed(1)}%`;
  document.getElementById('history-detail-sev').textContent = record.severity || 'None';
  document.getElementById('history-detail-gps').textContent = `${(record.latitude || 0).toFixed(6)}, ${(record.longitude || 0).toFixed(6)}`;
  document.getElementById('history-detail-address').textContent = record.address || 'No address recorded';
  document.getElementById('history-detail-time').textContent = safeFormatDateTime(record.timestamp);
  
  const imgUrl = record.image_path ? `${API_BASE}/${record.image_path.replace(/\\/g, '/')}` : '';
  document.getElementById('history-modal-image').src = imgUrl;
  
  // Set delete action
  const deleteBtn = document.getElementById('history-detail-delete-btn');
  deleteBtn.onclick = () => {
    closeHistoryModal();
    deleteRecord(record.id);
  };
  
  document.getElementById('history-detail-modal').classList.add('active');
}

function closeHistoryModal() {
  document.getElementById('history-detail-modal').classList.remove('active');
}

async function deleteRecord(id) {
  if (!confirm(`Delete record #${id}?`)) return;
  try {
    const res = await fetch(`${API_BASE}/record/${id}`, { method: 'DELETE' });
    const data = await res.json();
    if (data.success) {
      showToast('Deleted', `Record #${id} removed.`, 'success');
      loadHistory();
      loadStats();
    } else {
      showToast('Error', data.error || 'Delete failed', 'error');
    }
  } catch (e) {
    console.error('Delete failed:', e);
    showToast('Error', e.message, 'error');
  }
}


// ══════════════════════════════════════════════════════════════════
//  REPORTS
// ══════════════════════════════════════════════════════════════════

async function generateReport() {
  const preview = document.getElementById('report-preview');
  const summary = document.getElementById('report-summary');

  try {
    const [statsRes, damagesRes] = await Promise.all([
      fetch(`${API_BASE}/stats`),
      fetch(`${API_BASE}/damages`)
    ]);
    const statsData   = await statsRes.json();
    const damagesData = await damagesRes.json();

    if (!statsData.success || !damagesData.success) throw new Error('Failed to load data');

    const s       = statsData.stats;
    const records = damagesData.records || [];
    const now     = new Date().toLocaleString();
    const sevData = s.severity || {};

    // Summary card
    summary.innerHTML = `
      <div style="padding:.65rem .85rem;background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.25);border-radius:var(--radius-sm);font-size:.82rem;color:var(--accent-green)">
        ✅ Report generated with <strong>${records.length}</strong> records at ${now}
      </div>`;

    // Report preview
    preview.innerHTML = `
      <div style="border-bottom:1px solid var(--border);padding-bottom:.75rem;margin-bottom:.75rem">
        <div style="font-size:1.1rem;font-weight:800;margin-bottom:.3rem">📋 Road Damage Report</div>
        <div style="font-size:.72rem;color:var(--text-muted)">Generated: ${now}</div>
        <div style="font-size:.72rem;color:var(--text-muted)">System: Smart Road Damage Detection</div>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-bottom:1rem">
        <div style="padding:.5rem;background:rgba(59,130,246,.06);border-radius:var(--radius-sm);text-align:center">
          <div style="font-size:1.3rem;font-weight:800;color:var(--accent-blue)">${s.total || 0}</div>
          <div style="font-size:.68rem;color:var(--text-muted)">Total Reports</div>
        </div>
        <div style="padding:.5rem;background:rgba(239,68,68,.06);border-radius:var(--radius-sm);text-align:center">
          <div style="font-size:1.3rem;font-weight:800;color:var(--accent-red)">${s.pothole || 0}</div>
          <div style="font-size:.68rem;color:var(--text-muted)">Potholes</div>
        </div>
        <div style="padding:.5rem;background:rgba(245,158,11,.06);border-radius:var(--radius-sm);text-align:center">
          <div style="font-size:1.3rem;font-weight:800;color:var(--accent-yellow)">${s.crack || 0}</div>
          <div style="font-size:.68rem;color:var(--text-muted)">Cracks</div>
        </div>
        <div style="padding:.5rem;background:rgba(16,185,129,.06);border-radius:var(--radius-sm);text-align:center">
          <div style="font-size:1.3rem;font-weight:800;color:var(--accent-green)">${s.unique_reports || 0}</div>
          <div style="font-size:.68rem;color:var(--text-muted)">Unique Locations</div>
        </div>
      </div>

      <div style="font-size:.82rem;font-weight:700;margin-bottom:.4rem">Severity Distribution</div>
      <div style="display:flex;flex-direction:column;gap:.3rem;margin-bottom:1rem">
        ${['Critical', 'High', 'Medium', 'Low'].map(s => `
          <div style="display:flex;justify-content:space-between;padding:.3rem .5rem;background:rgba(30,41,59,.4);border-radius:4px;font-size:.75rem">
            <span>${s}</span>
            <span style="font-family:'JetBrains Mono',monospace;font-weight:700">${sevData[s] || 0}</span>
          </div>
        `).join('')}
      </div>

      <div style="font-size:.82rem;font-weight:700;margin-bottom:.4rem">Recent Damage Locations (Top 10)</div>
      <div style="display:flex;flex-direction:column;gap:.3rem">
        ${records.slice(0, 10).map(r => `
          <div style="display:flex;justify-content:space-between;padding:.3rem .5rem;background:rgba(30,41,59,.4);border-radius:4px;font-size:.72rem">
            <span style="color:${r.damage_type === 'pothole' ? 'var(--accent-red)' : 'var(--accent-yellow)'}">
              ${(r.damage_type || '').toUpperCase()} #${r.id}
            </span>
            <span style="font-family:'JetBrains Mono',monospace;color:var(--text-muted)">
              ${(r.latitude || 0).toFixed(5)}, ${(r.longitude || 0).toFixed(5)}
            </span>
          </div>
        `).join('')}
      </div>
    `;

    showToast('Report Generated', `${records.length} records compiled`, 'success');
  } catch (e) {
    showToast('Error', e.message, 'error');
  }
}


// ══════════════════════════════════════════════════════════════════
//  CSV EXPORT
// ══════════════════════════════════════════════════════════════════

function exportCSV() {
  window.open(`${API_BASE}/damages/export`, '_blank');
}


// ══════════════════════════════════════════════════════════════════
//  TOAST NOTIFICATIONS
// ══════════════════════════════════════════════════════════════════

function showToast(title, msg, type = 'info') {
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  toast.className = 'toast ' + type;
  toast.innerHTML = `<div class="toast-title">${title}</div><div class="toast-msg">${msg}</div>`;
  container.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}


// ══════════════════════════════════════════════════════════════════
//  UTILITY
// ══════════════════════════════════════════════════════════════════

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
