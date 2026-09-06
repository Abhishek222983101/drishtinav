/* DrishtiNav - Frontend Logic */

let map = null;
let trajectoryLayer = null;
let outageLayer = null;

/* ---- Run Simulation ---- */
async function runSimulation() {
    const btn = document.getElementById('runBtn');
    const status = document.getElementById('status');
    const resultsDiv = document.getElementById('results');

    btn.disabled = true;
    btn.textContent = 'RUNNING...';
    status.innerHTML = '<span class="status-dot running"></span> Processing IMU data...';
    resultsDiv.style.display = 'none';

    try {
        const resp = await fetch('/api/run');
        if (!resp.ok) throw new Error('Server error');
        const data = await resp.json();

        status.innerHTML = '<span class="status-dot done"></span> Simulation complete';

        renderMetrics(data.metrics_summary);
        renderLeafletMap(data);
        renderDriftChart(data);
        renderSpeedChart(data);

        resultsDiv.style.display = 'block';
        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
        status.innerHTML = '<span class="status-dot" style="background:#FF4136"></span> Error: ' + err.message;
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.textContent = 'RUN SIMULATION';
    }
}

/* ---- Metrics ---- */
function renderMetrics(m) {
    setMetric('dr-drift', m.dr_rmse.toFixed(1) + 'm', m.dr_drift_pct.toFixed(1) + '%');
    setMetric('ekf-drift', m.ekf_rmse.toFixed(1) + 'm', m.ekf_drift_pct.toFixed(1) + '%');
    setMetric('map-drift', m.map_rmse.toFixed(1) + 'm', m.map_drift_pct.toFixed(1) + '%');
    document.getElementById('speed-rmse').textContent = m.speed_rmse.toFixed(2);

    setBadge('dr-badge', m.dr_pass);
    setBadge('ekf-badge', m.ekf_pass);
    setBadge('map-badge', m.map_pass);
}

function setMetric(id, value, drift) {
    document.getElementById(id).textContent = value;
    const card = document.getElementById(id).closest('.metric-card');
    const sub = card.querySelector('.metric-sub');
    if (drift) sub.textContent = 'Drift: ' + drift;
}

function setBadge(id, pass) {
    const el = document.getElementById(id);
    el.textContent = pass ? 'PASS' : 'FAIL';
    el.className = 'metric-badge ' + (pass ? 'pass' : 'fail');
}

/* ---- Leaflet Map ---- */
function renderLeafletMap(data) {
    if (map) { map.remove(); map = null; }

    map = L.map('trajectory-map', { zoomControl: true }).setView([0, 0], 13);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: 'OpenStreetMap',
        maxZoom: 19
    }).addTo(map);

    // Convert relative ENU to lat/lon around Delhi center
    const refLat = 28.6139;
    const refLon = 77.2090;
    const toLL = (x, y) => [refLat + y / 111320, refLon + x / (111320 * Math.cos(refLat * Math.PI / 180))];

    // Ground Truth
    const gtPts = data.ground_truth.x.map((x, i) => toLL(x, data.ground_truth.y[i]));
    L.polyline(gtPts, { color: '#1A1A1A', weight: 4, opacity: 0.9 }).addTo(map).bindPopup('Ground Truth');

    // Dead Reckoning
    const drPts = data.dead_reckoning.x.map((x, i) => toLL(x, data.dead_reckoning.y[i]));
    L.polyline(drPts, { color: '#FF6B35', weight: 3, opacity: 0.85, dashArray: '8,6' }).addTo(map).bindPopup('Dead Reckoning');

    // EKF Fusion
    const ekfPts = data.ekf_fusion.x.map((x, i) => toLL(x, data.ekf_fusion.y[i]));
    L.polyline(ekfPts, { color: '#004E89', weight: 3, opacity: 0.85 }).addTo(map).bindPopup('EKF Fusion');

    // Map Matched
    const mapPts = data.map_matched.x.map((x, i) => toLL(x, data.map_matched.y[i]));
    L.polyline(mapPts, { color: '#2ECC40', weight: 3, opacity: 0.85 }).addTo(map).bindPopup('Map-Matched');

    // GPS Outage zone (red polygon)
    const tunnel = data.tunnel;
    const gtLen = data.ground_truth.x.length;
    if (tunnel.start_idx < gtLen && tunnel.end_idx < gtLen) {
        const tStart = Math.min(tunnel.start_idx, gtLen - 1);
        const tEnd = Math.min(tunnel.end_idx, gtLen - 1);
        const outagePts = [];
        for (let i = tStart; i <= tEnd; i++) {
            if (i < data.ground_truth.x.length) {
                outagePts.push(toLL(data.ground_truth.x[i], data.ground_truth.y[i]));
            }
        }
        if (outagePts.length > 1) {
            L.polyline(outagePts, { color: '#FF4136', weight: 12, opacity: 0.25 }).addTo(map)
                .bindPopup('GPS Outage Zone (Tunnel)');
        }
    }

    // Fit bounds
    if (gtPts.length > 0) {
        map.fitBounds(L.latLngBounds(gtPts), { padding: [30, 30] });
    }
}

/* ---- Drift Chart (Canvas) ---- */
function renderDriftChart(data) {
    const canvas = document.getElementById('driftCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const drErr = data.dead_reckoning.metrics.error_over_time;
    const ekfErr = data.ekf_fusion.metrics.error_over_time;
    const mapErr = data.map_matched.metrics.error_over_time;
    const tunnel = data.tunnel;

    const maxErr = Math.max(
        Math.max(...drErr.slice(0, 10).concat([1])),
        Math.max(...ekfErr.slice(0, 10).concat([1])),
        Math.max(...mapErr.slice(0, 10).concat([1]))
    ) * 1.1;

    const n = Math.min(drErr.length, ekfErr.length, mapErr.length);
    const padL = 50, padR = 10, padT = 10, padB = 30;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    // Background
    ctx.fillStyle = '#FFF8F0';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padT + plotH * (1 - i / 4);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(W - padR, y);
        ctx.stroke();
        ctx.fillStyle = '#999';
        ctx.font = '10px JetBrains Mono';
        ctx.fillText((maxErr * i / 4).toFixed(1) + 'm', 5, y + 3);
    }

    // Tunnel shading
    const tunnelStartX = padL + (tunnel.start_idx / n) * plotW;
    const tunnelEndX = padL + (tunnel.end_idx / n) * plotW;
    ctx.fillStyle = 'rgba(255, 65, 54, 0.12)';
    ctx.fillRect(tunnelStartX, padT, tunnelEndX - tunnelStartX, plotH);
    ctx.fillStyle = '#FF4136';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('GPS OUTAGE', tunnelStartX + 5, padT + 15);

    // Draw lines
    function drawLine(arr, color, lineWidth) {
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
            const x = padL + (i / n) * plotW;
            const y = padT + plotH * (1 - arr[i] / maxErr);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    drawLine(drErr, '#FF6B35', 2.5);
    drawLine(ekfErr, '#004E89', 2);
    drawLine(mapErr, '#2ECC40', 2);

    // 10% threshold line (approx based on total distance)
    const totalDist = data.metrics_summary.total_distance;
    const threshold10 = totalDist * 0.10;
    if (threshold10 < maxErr) {
        const y10 = padT + plotH * (1 - threshold10 / maxErr);
        ctx.setLineDash([5, 5]);
        ctx.strokeStyle = '#FFD23F';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(padL, y10);
        ctx.lineTo(W - padR, y10);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#FFD23F';
        ctx.fillText('10% limit', W - padR - 55, y10 - 4);
    }

    // X-axis label
    ctx.fillStyle = '#1A1A1A';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText('Time (samples)', W / 2, H - 5);
    ctx.textAlign = 'left';
}

/* ---- Speed Chart (Canvas) ---- */
function renderSpeedChart(data) {
    const canvas = document.getElementById('speedCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const aiSpeed = data.speed.ai_predicted.map(v => v * 3.6);
    const gtSpeed = data.speed.gps_ground_truth.map(v => v * 3.6);
    const tunnel = data.tunnel;
    const n = Math.min(aiSpeed.length, gtSpeed.length);

    const maxSpd = Math.max(Math.max(...aiSpeed), Math.max(...gtSpeed.filter(v => !isNaN(v)))) * 1.15;

    const padL = 50, padR = 10, padT = 10, padB = 30;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    ctx.fillStyle = '#FFF8F0';
    ctx.fillRect(0, 0, W, H);

    // Grid
    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padT + plotH * (1 - i / 4);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(W - padR, y);
        ctx.stroke();
        ctx.fillStyle = '#999';
        ctx.font = '10px JetBrains Mono';
        ctx.fillText((maxSpd * i / 4).toFixed(0) + ' km/h', 2, y + 3);
    }

    // Tunnel shading
    const tunnelStartX = padL + (tunnel.start_idx / n) * plotW;
    const tunnelEndX = padL + (tunnel.end_idx / n) * plotW;
    ctx.fillStyle = 'rgba(255, 65, 54, 0.12)';
    ctx.fillRect(tunnelStartX, padT, tunnelEndX - tunnelStartX, plotH);

    function drawLine(arr, color, lineWidth) {
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
            const x = padL + (i / n) * plotW;
            const val = isNaN(arr[i]) ? 0 : arr[i];
            const y = padT + plotH * (1 - val / maxSpd);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    drawLine(gtSpeed, '#1A1A1A', 2);
    drawLine(aiSpeed, '#FF6B35', 2);

    ctx.fillStyle = '#1A1A1A';
    ctx.font = '10px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText('Time (samples)', W / 2, H - 5);
    ctx.textAlign = 'left';
}

/* ---- Init ---- */
document.addEventListener('DOMContentLoaded', () => {
    console.log('DrishtiNav loaded');
});
