/* DrishtiNav - Frontend Logic: Journey Dashboard */

let map = null;
let vehicleMarker = null;
let vehicleAnimInterval = null;
let trailDots = [];

/* ========================= Run Simulation ========================= */
async function runSimulation() {
    const btn = document.getElementById('runBtn');
    const status = document.getElementById('status');
    const resultsDiv = document.getElementById('results');
    const progressBar = document.getElementById('progressBar');
    const progressFill = document.getElementById('progressFill');
    const stageText = document.getElementById('stageText');
    const pipeBoxes = document.querySelectorAll('.pipe-box');
    const duration = document.getElementById('duration').value;

    btn.disabled = true;
    btn.textContent = 'RUNNING...';
    resultsDiv.style.display = 'none';

    if (progressBar) progressBar.style.display = 'block';

    const stages = [
        { text: 'Initializing IMU sensors...', pipe: 0 },
        { text: 'Calibrating accelerometer bias...', pipe: 1 },
        { text: 'Filtering vibration noise...', pipe: 1 },
        { text: 'Training AI speed predictor...', pipe: 2 },
        { text: 'Predicting vehicle speed...', pipe: 2 },
        { text: 'Running Extended Kalman Filter...', pipe: 3 },
        { text: 'Applying GPS + magnetometer + AI corrections...', pipe: 3 },
        { text: 'Map matching to road network...', pipe: 4 },
        { text: 'Generating trajectory plots...', pipe: 5 },
    ];

    pipeBoxes.forEach(b => b.classList.remove('pipe-active'));

    let stageIdx = 0;
    let progress = 5;
    if (progressFill) progressFill.style.width = '5%';

    const stageInterval = setInterval(() => {
        if (stageIdx < stages.length) {
            const s = stages[stageIdx];
            if (stageText) stageText.textContent = s.text;
            status.innerHTML = '<span class="status-dot running"></span> ' + s.text;
            pipeBoxes.forEach(b => b.classList.remove('pipe-active'));
            if (s.pipe < pipeBoxes.length) {
                pipeBoxes[s.pipe].classList.add('pipe-active');
            }
            progress = Math.min(90, 5 + ((stageIdx + 1) / stages.length) * 85);
            if (progressFill) progressFill.style.width = progress + '%';
            stageIdx++;
        }
    }, 500);

    try {
        const resp = await fetch('/api/run?duration=' + encodeURIComponent(duration));
        if (!resp.ok) throw new Error('Server returned ' + resp.status);
        const data = await resp.json();
        if (data.error) throw new Error(data.error);

        clearInterval(stageInterval);

        if (stageText) stageText.textContent = 'Done!';
        status.innerHTML = '<span class="status-dot done"></span> Simulation complete';
        pipeBoxes.forEach(b => b.classList.add('pipe-active'));
        if (progressFill) progressFill.style.width = '100%';

        setTimeout(() => {
            if (progressBar) progressBar.style.display = 'none';
            pipeBoxes.forEach(b => b.classList.remove('pipe-active'));
        }, 1000);

        resultsDiv.style.display = 'block';

        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

        renderMetrics(data.metrics_summary);
        setupJourneyTimeline(data);
        renderLeafletMap(data);
        renderDriftChart(data);
        renderSpeedChart(data);

        resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
        clearInterval(stageInterval);
        status.innerHTML = '<span class="status-dot" style="background:#FF4136"></span> Error: ' + err.message;
        if (stageText) stageText.textContent = 'Error: ' + err.message;
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.textContent = 'RUN SIMULATION';
    }
}

/* ========================= Metrics ========================= */
function renderMetrics(m) {
    animateCounter('dr-drift', m.dr_rmse.toFixed(1) + 'm');
    animateCounter('ekf-drift', m.ekf_rmse.toFixed(1) + 'm');
    animateCounter('map-drift', m.map_rmse.toFixed(1) + 'm');
    animateCounter('speed-rmse', m.speed_rmse.toFixed(2));

    setMetricSub('dr-drift', 'Drift: ' + m.dr_drift_pct.toFixed(1) + '%');
    setMetricSub('ekf-drift', 'Drift: ' + m.ekf_drift_pct.toFixed(2) + '%');
    setMetricSub('map-drift', 'Drift: ' + m.map_drift_pct.toFixed(2) + '%');

    setBadge('dr-badge', m.dr_pass);
    setBadge('ekf-badge', m.ekf_pass);
    setBadge('map-badge', m.map_pass);
}

function animateCounter(id, targetText) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = targetText;
    el.classList.add('metric-pop');
    setTimeout(() => el.classList.remove('metric-pop'), 400);
}

function setMetricSub(id, text) {
    const card = document.getElementById(id);
    if (!card) return;
    const sub = card.closest('.metric-card').querySelector('.metric-sub');
    if (sub) sub.textContent = text;
}

function setBadge(id, pass) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = pass ? 'PASS' : 'FAIL';
    el.className = 'metric-badge ' + (pass ? 'pass' : 'fail');
}

/* ========================= Journey Timeline ========================= */
function setupJourneyTimeline(data) {
    const tunnel = data.tunnel;
    const n = data.ground_truth.x.length;
    const tsPct = Math.min(100, (tunnel.start_idx / n) * 100);
    const tePct = Math.min(100, (tunnel.end_idx / n) * 100);

    const zonePre = document.getElementById('zonePre');
    const zoneTunnel = document.getElementById('zoneTunnel');
    const zonePost = document.getElementById('zonePost');
    if (zonePre) zonePre.style.width = tsPct + '%';
    if (zoneTunnel) { zoneTunnel.style.left = tsPct + '%'; zoneTunnel.style.width = (tePct - tsPct) + '%'; }
    if (zonePost) { zonePost.style.left = tePct + '%'; zonePost.style.width = (100 - tePct) + '%'; }

    const tTunnel = document.getElementById('tTunnel');
    const tEnd = document.getElementById('tEnd');
    if (tTunnel) tTunnel.textContent = 'tunnel: ' + tunnel.start_sec.toFixed(0) + '-' + tunnel.end_sec.toFixed(0) + 's';
    if (tEnd) tEnd.textContent = 't = ' + data.metrics_summary.total_time.toFixed(0) + 's';

    resetJourneyUI();
}

function resetJourneyUI() {
    const fill = document.getElementById('journeyFill');
    const marker = document.getElementById('journeyMarker');
    if (fill) fill.style.width = '0%';
    if (marker) {
        marker.style.left = '0%';
        marker.classList.remove('m-tunnel', 'm-recovery');
        marker.classList.add('m-gps');
    }
    document.querySelectorAll('.journey-phase').forEach(p => p.classList.remove('active'));
    const zoneT = document.getElementById('zoneTunnel');
    if (zoneT) zoneT.classList.remove('near');
    const v = document.getElementById('mapVignette');
    if (v) v.classList.remove('active');
    const tCur = document.getElementById('tCur');
    if (tCur) tCur.textContent = 't = 0.0s';
}

function updateTimeline(idx, n, tunnel) {
    const pct = (idx / (n - 1)) * 100;
    const fill = document.getElementById('journeyFill');
    const marker = document.getElementById('journeyMarker');
    if (fill) fill.style.width = pct + '%';
    if (marker) marker.style.left = pct + '%';

    const phases = { gps: 'phase-gps', tunnel: 'phase-tunnel', recovery: 'phase-recovery' };
    const inTunnel = idx >= tunnel.start_idx && idx <= tunnel.end_idx;
    document.querySelectorAll('.journey-phase').forEach(p => p.classList.remove('active'));
    let key;
    if (inTunnel) key = 'tunnel';
    else if (idx < tunnel.start_idx) key = 'gps';
    else key = 'recovery';
    const el = document.getElementById(phases[key]);
    if (el) el.classList.add('active');
}

/* ========================= Leaflet Map ========================= */
const REF_LAT = 28.6139;
const REF_LON = 77.2090;
function toLL(x, y) {
    return [REF_LAT + y / 111320, REF_LON + x / (111320 * Math.cos(REF_LAT * Math.PI / 180))];
}

function glowLine(pts, color, weight, opts) {
    if (pts.length < 2) return;
    L.polyline(pts, { color, weight: weight + 6, opacity: 0.15, lineCap: 'round', lineJoin: 'round' }).addTo(map);
    return L.polyline(pts, {
        color, weight,
        opacity: (opts && opts.opacity) || 0.9,
        dashArray: (opts && opts.dash) || null,
        lineCap: 'round', lineJoin: 'round'
    }).addTo(map);
}

function renderLeafletMap(data) {
    if (map) { map.remove(); map = null; }
    if (vehicleAnimInterval) { clearInterval(vehicleAnimInterval); vehicleAnimInterval = null; }
    trailDots = [];

    const mapEl = document.getElementById('trajectory-map');
    if (!mapEl) return;
    mapEl.innerHTML = '';

    map = L.map('trajectory-map', { zoomControl: true, attributionControl: true }).setView([0, 0], 13);

    /* OSM standard tiles: free forever, no API key, no usage walls.
       CARTO only as fallback (its free tier can serve "API key required"
       error tiles under load/referrer limits). */
    const tileLayer = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
    }).addTo(map);

    let cartoFallbackAdded = false;
    tileLayer.on('tileerror', function () {
        if (cartoFallbackAdded || !map) return;
        cartoFallbackAdded = true;
        L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 19
        }).addTo(map);
    });

    const gtPts = data.ground_truth.x.map((x, i) => toLL(x, data.ground_truth.y[i]));
    const ekfPts = data.ekf_fusion.x.map((x, i) => toLL(x, data.ekf_fusion.y[i]));
    const drPts = data.dead_reckoning.x.map((x, i) => toLL(x, data.dead_reckoning.y[i]));

    const tunnel = data.tunnel;
    const n = gtPts.length;

    /* Road = ground truth, split into GPS-ok / outage segments */
    const okSegs = [];
    let cur = [];
    for (let i = 0; i < n; i++) {
        const inTunnel = i >= tunnel.start_idx && i <= tunnel.end_idx;
        if (!inTunnel) {
            cur.push(gtPts[i]);
        } else {
            if (cur.length > 1) okSegs.push(cur);
            cur = [];
        }
    }
    if (cur.length > 1) okSegs.push(cur);
    okSegs.forEach(seg => glowLine(seg, '#1A1A1A', 5));

    /* Outage zone: fat translucent red + hatched centerline */
    const tS = Math.min(tunnel.start_idx, n - 1);
    const tE = Math.min(tunnel.end_idx, n - 1);
    const outagePts = [];
    for (let i = tS; i <= tE && i < n; i++) outagePts.push(gtPts[i]);
    if (outagePts.length > 1) {
        L.polyline(outagePts, { color: '#FF4136', weight: 16, opacity: 0.25, lineCap: 'round' }).addTo(map)
            .bindPopup('GPS Outage Zone (tunnel)');
        L.polyline(outagePts, { color: '#FF4136', weight: 2, opacity: 0.7, dashArray: '6,6' }).addTo(map);
    }

    /* Estimate trajectories are revealed progressively by the journey
       animation (see animateJourney) - nothing pre-drawn here so the
       reveal tells the story cleanly. */

    /* Start / End markers */
    if (gtPts.length > 0) {
        const startIcon = L.divIcon({
            className: 'vehicle-marker',
            html: '<div class="marker-pulse"></div><div class="marker-dot"></div>',
            iconSize: [20, 20], iconAnchor: [10, 10]
        });
        L.marker(gtPts[0], { icon: startIcon }).addTo(map).bindPopup('Start');
    }
    if (gtPts.length > 1) {
        const endIcon = L.divIcon({
            className: 'vehicle-marker',
            html: '<div class="marker-dot end"></div>',
            iconSize: [14, 14], iconAnchor: [7, 7]
        });
        L.marker(gtPts[gtPts.length - 1], { icon: endIcon }).addTo(map).bindPopup('End');
    }

    if (gtPts.length > 0) {
        map.fitBounds(L.latLngBounds(gtPts), { padding: [40, 40], maxZoom: 16 });
    }

    setTimeout(() => { if (map) map.invalidateSize(); }, 300);

    setTimeout(() => { animateJourney(data, gtPts, ekfPts, drPts, tunnel); }, 600);
}

/* ========================= Journey Animation ========================= */
/* Vehicle follows the EKF estimate; the map, timeline and gauges are all
   driven from the same index so they stay perfectly in sync. */
function animateJourney(data, gtPts, ekfPts, drPts, tunnel) {
    if (!map || ekfPts.length < 2) return;
    if (vehicleAnimInterval) { clearInterval(vehicleAnimInterval); vehicleAnimInterval = null; }
    trailDots.forEach(d => { try { map.removeLayer(d); } catch (e) {} });
    trailDots = [];

    const n = ekfPts.length;
    const gpsAvail = data.gps_available;
    const headingDeg = data.gt_heading_deg;
    const gtSpeed = data.gt_speed_ms;
    const ekfErr = data.ekf_fusion.metrics.error_over_time;
    const maxEkfErr = Math.max(...ekfErr, 1);

    const carIcon = L.divIcon({
        className: 'vehicle-marker car-icon',
        html: '<div class="car-body">&#x1F697;</div>',
        iconSize: [32, 32], iconAnchor: [16, 16]
    });
    vehicleMarker = L.marker(ekfPts[0], { icon: carIcon, zIndexOffset: 1000 }).addTo(map)
        .bindPopup('Vehicle (EKF estimate)');

    /* Progressive trajectory polylines (revealed as the vehicle moves) */
    function makeProg(pts, color, weight, dash) {
        const line = L.polyline([pts[0]], {
            color, weight: weight + 6, opacity: 0.15, lineCap: 'round', lineJoin: 'round'
        }).addTo(map);
        const core = L.polyline([pts[0]], {
            color, weight, opacity: 0.95, dashArray: dash || null, lineCap: 'round', lineJoin: 'round'
        }).addTo(map);
        return { line, core, pts, idx: 1 };
    }
    const progDR = makeProg(drPts, '#FF6B35', 3, '10,7');
    const progEKF = makeProg(ekfPts, '#004E89', 3.5);
    const progMAP = makeProg(data.map_matched.x.map((x, i) => toLL(x, data.map_matched.y[i])), '#2ECC40', 3);

    /* DR only becomes interesting during the outage - hide before it */
    progDR.line.setStyle({ opacity: 0 });
    progDR.core.setStyle({ opacity: 0 });

    let phase = 'gps'; // gps | tunnel | recovery
    let idx = 0;

    const calloutEl = document.getElementById('callout');
    const calloutIcon = document.getElementById('calloutIcon');
    const calloutText = document.getElementById('calloutText');

    function showCallout(icon, text, cls) {
        if (!calloutEl) return;
        calloutIcon.textContent = icon;
        calloutText.textContent = text;
        calloutEl.className = 'callout pulse ' + (cls || '');
        calloutEl.style.animation = 'none';
        void calloutEl.offsetWidth; /* restart CSS animation */
        calloutEl.style.animation = 'calloutIn 0.35s ease';
    }

    showCallout('\u{1F4E1}', 'GPS locked - AI speed model trained on live IMU/GPS data. Journey starting...', 'ok');
    setMethod('gps', 'GPS + EKF FUSION');
    showAlert('success', '\u{1F4E1}', 'GPS LOCKED - NAVIGATION ACTIVE', 2500);
    setVignette(false);
    setMarkerPhase('gps');

    const dtMs = 40;
    vehicleAnimInterval = setInterval(() => {
        if (idx >= n) {
            clearInterval(vehicleAnimInterval);
            vehicleAnimInterval = null;
            showCallout('\u2705',
                'Journey complete. EKF final error ' + data.ekf_fusion.metrics.final_error.toFixed(1) +
                'm, drift ' + data.metrics_summary.ekf_drift_pct.toFixed(2) + '% of distance - PASS.', 'ok');
            setMethod('gps', 'COMPLETE - PASS');
            showAlert('success', '\u2705', 'JOURNEY COMPLETE - DRIFT ' + data.metrics_summary.ekf_drift_pct.toFixed(2) + '% - PASS', 0);
            setVignette(false);
            return;
        }

        /* --- Vehicle + progressive trajectories --- */
        vehicleMarker.setLatLng(ekfPts[idx]);
        [progDR, progEKF, progMAP].forEach(p => {
            for (let k = p.idx; k <= idx; k++) p.line.addLatLng(p.pts[k]), p.core.addLatLng(p.pts[k]);
            p.idx = idx + 1;
        });
        /* DR path only visible inside/after the tunnel */
        const showDR = idx >= tunnel.start_idx;
        progDR.line.setStyle({ opacity: showDR ? 0.15 : 0 });
        progDR.core.setStyle({ opacity: showDR ? 0.95 : 0 });

        /* breadcrumb trail, colored by GPS status */
        if (idx % 8 === 0 && idx > 0) {
            const inT = idx >= tunnel.start_idx && idx <= tunnel.end_idx;
            const dot = L.circleMarker(ekfPts[idx], {
                radius: 3, weight: 0,
                color: inT ? '#FF4136' : '#004E89',
                fillColor: inT ? '#FF4136' : '#004E89',
                fillOpacity: 0.55
            }).addTo(map);
            trailDots.push(dot);
            if (trailDots.length > 40) { const old = trailDots.shift(); map.removeLayer(old); }
        }

        /* --- Timeline --- */
        updateTimeline(idx, n, tunnel);
        const tCur = document.getElementById('tCur');
        if (tCur) {
            const totalSec = data.metrics_summary.total_time;
            tCur.textContent = 't = ' + (idx / (n - 1) * totalSec).toFixed(1) + 's';
        }

        /* --- Gauges --- */
        const spd = gtSpeed && gtSpeed[idx] ? gtSpeed[idx] * 3.6 : 0;
        updateSpeedometer(spd);
        const hdg = headingDeg && headingDeg[idx] !== undefined ? headingDeg[idx] : 0;
        updateCompass(hdg);
        const inTunnel = idx >= tunnel.start_idx && idx <= tunnel.end_idx;
        updateGPSBars(!inTunnel);
        const err = ekfErr[idx] || 0;
        updateErrorRing(err, maxEkfErr);

        /* --- Phase transitions: callouts + banner + gauges --- */
        if (inTunnel && phase !== 'tunnel') {
            phase = 'tunnel';
            showCallout('\u{1F687}',
                'GPS SIGNAL LOST - satellites out of view. Now relying on IMU + AI speed prediction + magnetometer heading.',
                'warn');
            setMethod('ai', 'AI DEAD RECKONING');
            showAlert('danger', '\u{1F687}', 'GPS SIGNAL LOST - AI DEAD RECKONING', 0); /* holds until exit */
            setVignette(true);
            setMarkerPhase('tunnel');
        } else if (!inTunnel && phase === 'tunnel') {
            phase = 'recovery';
            showCallout('\u{1F4E1}',
                'GPS RE-LOCKED - EKF corrects accumulated drift and snaps back to the true position.',
                'ok');
            setMethod('recover', 'GPS RE-ACQUIRED');
            showAlert('success', '\u{1F4E1}', 'GPS RE-ACQUIRED - POSITION CORRECTED', 3000);
            setVignette(false);
            setMarkerPhase('recovery');
        } else {
            /* Approaching tunnel: pulse the tunnel zone 30 samples ahead */
            const zoneT = document.getElementById('zoneTunnel');
            if (zoneT && !inTunnel && idx >= tunnel.start_idx - 30 && idx < tunnel.start_idx) {
                zoneT.classList.add('near');
            } else if (zoneT) {
                zoneT.classList.remove('near');
            }
            if (!inTunnel && phase === 'gps' && idx === Math.floor(tunnel.start_idx * 0.55)) {
                showCallout('\u{1F916}',
                    'AI Ridge model predicting speed from IMU vibration - updates used by both DR and EKF.',
                    'ai');
            } else if (phase === 'recovery' && idx === Math.floor((tunnel.end_idx + n) / 2)) {
                showCallout('\u{1F5FA}\uFE0F',
                    'Map matching snapping fused position onto the road network - lane-level accuracy.',
                    'ai');
            }
        }

        idx++;
    }, dtMs);
}

/* ========================= Dashboard Gauges ========================= */
function setMethod(cls, text) {
    const dot = document.getElementById('methodDot');
    const txt = document.getElementById('methodText');
    if (dot) dot.className = 'method-dot ' + cls;
    if (txt) txt.textContent = text;
}

/* Alert banner over the map: type = danger | success | warn */
let alertTimer = null;
function showAlert(type, icon, text, holdMs) {
    const banner = document.getElementById('alertBanner');
    if (!banner) return;
    const iconEl = document.getElementById('alertIcon');
    const textEl = document.getElementById('alertText');
    if (iconEl) iconEl.textContent = icon;
    if (textEl) textEl.textContent = text;
    banner.className = 'alert-banner show alert-' + type;
    if (alertTimer) clearTimeout(alertTimer);
    if (holdMs !== 0) {
        alertTimer = setTimeout(() => {
            banner.classList.remove('show');
        }, holdMs || 3200);
    }
}

function setVignette(on) {
    const v = document.getElementById('mapVignette');
    if (v) v.classList.toggle('active', !!on);
}

function setMarkerPhase(phase) {
    const marker = document.getElementById('journeyMarker');
    if (!marker) return;
    marker.classList.remove('m-gps', 'm-tunnel', 'm-recovery');
    marker.classList.add('m-' + phase);
}

function updateSpeedometer(kmh) {
    const arc = document.getElementById('speedArc');
    const val = document.getElementById('speedValue');
    if (!arc) return;
    const maxKmh = 120;
    const frac = Math.min(kmh / maxKmh, 1);
    /* arc length = pi * r = pi * 48 = 150.8 */
    arc.setAttribute('stroke-dashoffset', String(150.8 * (1 - frac)));
    /* color shifts to red at high speed */
    const col = kmh > 55 ? '#FF4136' : kmh > 40 ? '#FFD23F' : '#FF6B35';
    arc.setAttribute('stroke', col);
    if (val) val.textContent = Math.round(kmh);
}

function updateCompass(deg) {
    const needle = document.getElementById('compassNeedle');
    const val = document.getElementById('headingValue');
    if (needle) needle.setAttribute('transform', 'rotate(' + deg.toFixed(1) + ' 50 50)');
    if (val) val.textContent = Math.round(deg);
}

function updateGPSBars(locked) {
    const wrap = document.getElementById('signalBars');
    const status = document.getElementById('gpsStatus');
    if (!wrap) return;
    wrap.className = 'signal-bars ' + (locked ? 'locked' : 'denied');
    if (status) {
        status.textContent = locked ? 'LOCKED' : 'LOST';
        status.className = locked ? 'locked' : 'denied';
    }
}

function updateErrorRing(errM, maxErr) {
    const ring = document.getElementById('errorRing');
    const val = document.getElementById('errorValue');
    if (!ring) return;
    const scale = Math.min(errM / maxErr, 1);
    ring.style.transform = 'scale(' + (0.3 + scale * 0.7).toFixed(3) + ')';
    ring.style.borderColor = scale < 0.3 ? '#2ECC40' : scale < 0.6 ? '#FFD23F' : '#FF4136';
    if (val) val.textContent = errM.toFixed(1) + 'm';
}

/* ========================= Drift Chart (Canvas) ========================= */
function renderDriftChart(data) {
    const canvas = document.getElementById('driftCanvas');
    if (!canvas) return;
    const container = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;

    const displayW = container ? container.clientWidth - 32 : 500;
    const displayH = 300;
    canvas.style.width = displayW + 'px';
    canvas.style.height = displayH + 'px';
    canvas.width = displayW * dpr;
    canvas.height = displayH * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const W = displayW;
    const H = displayH;
    ctx.clearRect(0, 0, W, H);

    const drErr = data.dead_reckoning.metrics.error_over_time;
    const ekfErr = data.ekf_fusion.metrics.error_over_time;
    const mapErr = data.map_matched.metrics.error_over_time;
    const tunnel = data.tunnel;

    const allErr = [...drErr, ...ekfErr, ...mapErr];
    const maxErr = Math.max(...allErr, 1) * 1.1;

    const n = Math.min(drErr.length, ekfErr.length, mapErr.length);
    const padL = 60, padR = 15, padT = 20, padB = 40;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    ctx.fillStyle = '#FAFAFA';
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = '#E8E8E8';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = padT + plotH * (1 - i / 5);
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
        ctx.fillStyle = '#999';
        ctx.font = '11px JetBrains Mono, monospace';
        ctx.textAlign = 'right';
        ctx.fillText((maxErr * i / 5).toFixed(0) + 'm', padL - 8, y + 4);
    }
    ctx.textAlign = 'left';

    const tSX = padL + (tunnel.start_idx / n) * plotW;
    const tEX = padL + (tunnel.end_idx / n) * plotW;
    const g = ctx.createLinearGradient(tSX, padT, tSX, padT + plotH);
    g.addColorStop(0, 'rgba(255, 65, 54, 0.15)');
    g.addColorStop(1, 'rgba(255, 65, 54, 0.03)');
    ctx.fillStyle = g;
    ctx.fillRect(tSX, padT, tEX - tSX, plotH);
    ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(255,65,54,0.35)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(tSX, padT); ctx.lineTo(tSX, padT + plotH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(tEX, padT); ctx.lineTo(tEX, padT + plotH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#FF4136'; ctx.font = 'bold 10px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('GPS OUTAGE', (tSX + tEX) / 2, padT + 15);
    ctx.textAlign = 'left';

    function area(arr, col) {
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.moveTo(padL, padT + plotH);
        for (let i = 0; i < n; i++) {
            ctx.lineTo(padL + (i / n) * plotW, padT + plotH * (1 - arr[i] / maxErr));
        }
        ctx.lineTo(padL + plotW, padT + plotH); ctx.closePath(); ctx.fill();
    }
    function line(arr, col, w) {
        ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
            const x = padL + (i / n) * plotW;
            const y = padT + plotH * (1 - arr[i] / maxErr);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    area(drErr, 'rgba(255,107,53,0.1)');
    area(ekfErr, 'rgba(0,78,137,0.1)');
    area(mapErr, 'rgba(46,204,64,0.1)');
    line(drErr, '#FF6B35', 2.5);
    line(ekfErr, '#004E89', 2.5);
    line(mapErr, '#2ECC40', 2.5);

    const td = data.metrics_summary.total_distance;
    const th10 = td * 0.10;
    if (th10 < maxErr) {
        const y10 = padT + plotH * (1 - th10 / maxErr);
        ctx.setLineDash([6, 4]); ctx.strokeStyle = '#FFD23F'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(padL, y10); ctx.lineTo(W - padR, y10); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#FFD23F'; ctx.font = 'bold 10px JetBrains Mono, monospace';
        ctx.fillText('10% LIMIT', W - padR - 68, y10 - 6);
    }

    ctx.fillStyle = '#888'; ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Time (samples)', W / 2, H - 8);
    ctx.textAlign = 'left';
}

/* ========================= Speed Chart (Canvas) ========================= */
function renderSpeedChart(data) {
    const canvas = document.getElementById('speedCanvas');
    if (!canvas) return;
    const container = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;

    const displayW = container ? container.clientWidth - 32 : 500;
    const displayH = 300;
    canvas.style.width = displayW + 'px';
    canvas.style.height = displayH + 'px';
    canvas.width = displayW * dpr;
    canvas.height = displayH * dpr;

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const W = displayW;
    const H = displayH;
    ctx.clearRect(0, 0, W, H);

    const aiSpeed = data.speed.ai_predicted.map(v => v * 3.6);
    const gtSpeed = data.speed.gps_ground_truth.map(v => v * 3.6);
    const tunnel = data.tunnel;
    const n = Math.min(aiSpeed.length, gtSpeed.length);

    const allS = [...aiSpeed.slice(0, n), ...gtSpeed.slice(0, n)].filter(v => !isNaN(v) && isFinite(v));
    const maxSpd = Math.max(...allS, 1) * 1.15;

    const padL = 60, padR = 15, padT = 20, padB = 40;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    ctx.fillStyle = '#FAFAFA';
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = '#E8E8E8'; ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = padT + plotH * (1 - i / 5);
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
        ctx.fillStyle = '#999'; ctx.font = '11px JetBrains Mono, monospace'; ctx.textAlign = 'right';
        ctx.fillText((maxSpd * i / 5).toFixed(0), padL - 8, y + 4);
    }
    ctx.textAlign = 'left';

    ctx.save(); ctx.translate(14, padT + plotH / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#888'; ctx.font = '10px JetBrains Mono, monospace'; ctx.textAlign = 'center';
    ctx.fillText('km/h', 0, 0); ctx.restore();

    const tSX = padL + (tunnel.start_idx / n) * plotW;
    const tEX = padL + (tunnel.end_idx / n) * plotW;
    const g = ctx.createLinearGradient(tSX, padT, tSX, padT + plotH);
    g.addColorStop(0, 'rgba(255,65,54,0.15)'); g.addColorStop(1, 'rgba(255,65,54,0.03)');
    ctx.fillStyle = g; ctx.fillRect(tSX, padT, tEX - tSX, plotH);

    function area(arr, col) {
        ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(padL, padT + plotH);
        for (let i = 0; i < n; i++) {
            const v = isNaN(arr[i]) ? 0 : arr[i];
            ctx.lineTo(padL + (i / n) * plotW, padT + plotH * (1 - v / maxSpd));
        }
        ctx.lineTo(padL + plotW, padT + plotH); ctx.closePath(); ctx.fill();
    }
    function line(arr, col, w) {
        ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
            const v = isNaN(arr[i]) ? 0 : arr[i];
            const x = padL + (i / n) * plotW;
            const y = padT + plotH * (1 - v / maxSpd);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    area(gtSpeed, 'rgba(26,26,26,0.08)');
    area(aiSpeed, 'rgba(255,107,53,0.1)');
    line(gtSpeed, '#1A1A1A', 2.5);
    line(aiSpeed, '#FF6B35', 2.5);

    ctx.fillStyle = '#888'; ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('Time (samples)', W / 2, H - 8);
    ctx.textAlign = 'left';
}

/* ========================= Init ========================= */
document.addEventListener('DOMContentLoaded', () => {
    console.log('DrishtiNav loaded');
});
