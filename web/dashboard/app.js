/* DrishtiNav evaluation dashboard */
// categorical slots validated for colour-vision deficiency (dataviz validator: CVD dE >= 9.2 all pairs);
// the reference path is neutral ink and GNSS-denied stretches are a neutral band, not a series colour
const COLORS = { truth: '#1A1A1A', ins: '#2a78d6', ai: '#eb6834', full: '#1baf7a', gnss: '#9aa0a6', outage: '#52514e' };
const $ = id => document.getElementById(id);
const CFG = window.DRISHTI || { data: '/data', engine: '' };     // written by /config.js
const ORDER = ['ins', 'ai', 'full'];                   // ablation order (the server's JSON keys are sorted)
const cfgs = () => ORDER.filter(k => data.configs[k]).map(k => [k, data.configs[k]]);
let scenarios = [], data = null, map = null, layers = [], car = null, playing = false, cursor = 0, raf = null;

async function init() {
  scenarios = await (await fetch(`${CFG.data}/scenarios.json`)).json();
  const sel = $('scenario');
  for (const s of scenarios) {
    const o = document.createElement('option');
    o.value = s.id; o.textContent = s.title;
    sel.appendChild(o);
  }
  sel.addEventListener('change', updateHint);
  updateHint();
  $('runBtn').addEventListener('click', () => run(false));
  $('liveBtn').addEventListener('click', () => run(true));
  $('playBtn').addEventListener('click', togglePlay);
  $('scrub').addEventListener('input', e => { cursor = +e.target.value / 1000 * (data.t.length - 1); render(); });
  window.addEventListener('resize', () => data && drawCharts());
  loadBenchmark();
}

function updateHint() {
  const s = scenarios.find(x => x.id === $('scenario').value);
  $('outageGroup').style.display = s && s.outage === 'injected' ? '' : 'none';
  $('scenarioHint').textContent = !s ? '' : s.kind === 'synthetic'
    ? 'Physics simulation on real OpenStreetMap roads: GNSS is lost for the whole 1.3 km Pragati Maidan tunnel (and degraded by multipath at the portals). The IMU includes mount misalignment, holder wobble, engine/road vibration and potholes.'
    : 'Real IO-VNBD smartphone recording (held-out test drive, never used for training). GNSS outages are injected repeatedly after a 120 s warm-up; the reference is the VBOX / CAN log.';
}

function setStatus(cls, text) { $('status').innerHTML = `<span class="dot ${cls}"></span> ${text}`; }

async function run(live) {
  const sc = scenarios.find(x => x.id === $('scenario').value);
  const outage = sc.outage === 'tunnel' ? 'tunnel' : $('outage').value;
  $('runBtn').disabled = $('liveBtn').disabled = true;
  let url, t0 = performance.now();
  if (live) {
    const seed = Math.max(1, parseInt($('seed').value || '11', 10));
    url = `${CFG.engine}/api/run?scenario=${encodeURIComponent(sc.id)}&outage=${outage === 'tunnel' ? 0 : outage}&seed=${seed}&configs=full`;
    setStatus('run', `Computing on the live Python engine (seed ${seed})… a sleeping free-tier server needs ~1 min to wake`);
  } else {
    url = `${CFG.data}/runs/${sc.runs[outage]}.json`;
    setStatus('run', 'Loading engine results…');
  }
  try {
    const r = await fetch(url);
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || r.status);
    data = j;
    $('results').hidden = false;
    renderKpis(); drawMap(); renderTable(); renderDiag();
    cursor = 0; render();
    const where = live ? `live engine, seed ${j.seed}, computed in ${j.compute_s} s (${((performance.now() - t0) / 1000).toFixed(0)} s incl. network)` : 'precomputed by the same engine';
    setStatus('ok', `Done — ${j.distance_m.toFixed(0)} m, ${j.duration_s.toFixed(0)} s @ ${j.rate_hz} Hz · ${where}`);
    $('results').scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    setStatus('err', (live ? 'Live engine error: ' : 'Error: ') + e.message);
  } finally { $('runBtn').disabled = $('liveBtn').disabled = false; }
}

function fmt(v, d = 1) { return v == null || !isFinite(v) ? '—' : (+v).toFixed(d); }

function renderKpis() {
  const k = $('kpis'); k.innerHTML = '';
  for (const [key, c] of cfgs()) {
    const s = c.summary || {};
    const drift = s.drift_pct_median, pass = drift != null && drift < 10;
    const div = document.createElement('div');
    div.className = 'kpi';
    div.innerHTML = `<div class="lbl"><span class="sw" style="background:${COLORS[key]}"></span>${c.label}</div>
      <div class="num">${fmt(drift, 2)} %</div>
      <div class="sub">median drift · end err ${fmt(s.end_err_median_m)} m · max ${fmt(s.max_err_mean_m)} m</div>
      <div class="sub">pass-rate &lt;10 %: ${s.pass_rate_10pct != null ? (100 * s.pass_rate_10pct).toFixed(0) + ' %' : '—'} · ${fmt(c.us_per_step, 0)} µs/step</div>
      <span class="pass ${pass ? 'ok' : 'bad'}">${pass ? 'PASS (<10 %)' : 'ABOVE 10 %'}</span>`;
    k.appendChild(div);
  }
  const full = data.configs.full;
  const cap = document.createElement('div');
  cap.className = 'kpi';
  cap.innerHTML = `<div class="lbl">GNSS-denied distance</div><div class="num">${fmt(full.summary.mean_distance_m, 0)} m</div>
    <div class="sub">${data.outages.length} outage(s) · ${data.profile} profile</div>
    <div class="sub">engine capacity ≈ ${fmt(1e6 / full.us_per_step, 0)} steps/s (need ${data.rate_hz})</div>`;
  k.appendChild(cap);
}

function poly(lat, lon) { const p = []; for (let i = 0; i < lat.length; i++) if (lat[i] != null) p.push([lat[i], lon[i]]); return p; }

function drawMap() {
  if (!map) {
    map = L.map('map', { zoomControl: true, preferCanvas: true });
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap contributors' }).addTo(map);
  }
  layers.forEach(l => map.removeLayer(l)); layers = [];
  const add = l => { l.addTo(map); layers.push(l); return l; };
  const tr = poly(data.truth.lat, data.truth.lon);
  add(L.polyline(tr, { color: COLORS.truth, weight: 5, opacity: .85 }));
  // outage stretches on the reference path
  let seg = [];
  for (let i = 0; i < data.denied.length; i++) {
    if (data.denied[i]) seg.push([data.truth.lat[i], data.truth.lon[i]]);
    else if (seg.length) { add(L.polyline(seg, { color: COLORS.outage, weight: 13, opacity: .22 })); seg = []; }
  }
  if (seg.length) add(L.polyline(seg, { color: COLORS.outage, weight: 13, opacity: .22 }));
  for (const g of data.gnss) add(L.circleMarker(g, { radius: 2, color: COLORS.gnss, weight: 1, fillOpacity: .6 }));
  const dash = { ins: '6 6', ai: null, full: null };
  for (const [key, c] of cfgs()) {
    const lat = key === 'full' ? c.matched_lat : c.lat, lon = key === 'full' ? c.matched_lon : c.lon;
    add(L.polyline(poly(lat, lon), { color: COLORS[key], weight: key === 'full' ? 3.5 : 2.5, dashArray: dash[key], opacity: .95 }));
  }
  car = add(L.marker(tr[0], { icon: L.divIcon({ className: 'car-icon', html: '🚗', iconSize: [24, 24] }) }));
  map.fitBounds(L.latLngBounds(tr).pad(0.05));
  $('mapTitle').textContent = data.title;
  $('legend').innerHTML = `<span><i style="background:${COLORS.truth}"></i>Reference</span>` +
    `<span><i style="background:${COLORS.outage};opacity:.4;height:8px"></i>GNSS denied</span>` +
    `<span><i style="background:${COLORS.gnss}"></i>GNSS fixes</span>` +
    cfgs().map(([k, c]) => `<span><i style="background:${COLORS[k]}"></i>${c.label}</span>`).join('');
}

function drawCharts() {
  const bands = data.outages.map(o => [o.start_s, o.end_s]);
  const tc = data.t[Math.round(cursor)];
  Charts.lineChart($('errChart'), {
    x: data.t, bands, yLabel: 'error (m)', xLabel: 'time (s)', cursor: tc,
    series: cfgs().map(([k, c]) => ({ y: c.err, color: COLORS[k], label: c.label, dash: k === 'ins' ? [5, 4] : null })),
  });
  const full = data.configs.full, ai = data.configs.ai || full;
  Charts.lineChart($('spdChart'), {
    x: data.t, bands, yLabel: 'speed (km/h)', xLabel: 'time (s)', cursor: tc,
    series: [
      { y: data.truth.speed.map(v => v == null ? null : v * 3.6), color: COLORS.truth, label: 'reference', width: 2.5 },
      { y: ai.ai_speed.map(v => v == null ? null : v * 3.6), color: COLORS.ai, label: 'SpeedNet', width: 1.2 },
      { y: full.speed.map(v => v == null ? null : v * 3.6), color: COLORS.full, label: 'fused', width: 2 },
    ],
  });
  Charts.modeStrip($('modeStrip'), data.t, full.mode, tc);
}

function render() {
  if (!data) return;
  const i = Math.max(0, Math.min(data.t.length - 1, Math.round(cursor)));
  const f = data.configs.full;
  const ll = [f.matched_lat[i], f.matched_lon[i]];
  if (car && ll[0] != null) car.setLatLng(ll);
  $('scrub').value = Math.round(i / (data.t.length - 1) * 1000);
  $('clock').textContent = `t = ${data.t[i].toFixed(0)} s`;
  const m = f.mode[i];
  $('tMode').textContent = m; $('tMode').className = 'mode-pill mode-' + m;
  $('tSpeed').textContent = fmt(f.speed[i] * 3.6, 0) + ' km/h';
  $('tHead').textContent = fmt(f.heading[i], 0) + '°';
  $('tErr').textContent = fmt(f.err[i]) + ' m';
  $('tAi').textContent = f.ai_speed[i] == null ? '—' : fmt(f.ai_speed[i] * 3.6, 0) + ' km/h';
  $('tSig').textContent = fmt(f.sigma[i]) + ' m';
  $('tConf').textContent = fmt(f.match_conf[i], 2);
  $('tStat').textContent = f.stationary[i] ? 'yes (ZUPT)' : 'no';
  drawCharts();
}

function togglePlay() {
  playing = !playing;
  $('playBtn').textContent = playing ? '❚❚ Pause' : '▶ Play';
  if (!playing) { cancelAnimationFrame(raf); return; }
  let last = performance.now();
  const dtData = (data.t[data.t.length - 1] - data.t[0]) / (data.t.length - 1);
  const loop = now => {
    const speedup = 12;                                        // 12x real time
    cursor += (now - last) / 1000 * speedup / dtData; last = now;
    if (cursor >= data.t.length - 1) { cursor = data.t.length - 1; togglePlay(); }
    render();
    if (playing) raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
}

function renderTable() {
  const keys = cfgs().map(([k]) => k);
  const rows = data.configs.full.outages;
  let h = '<tr><th>#</th><th>start (s)</th><th>length (s)</th><th>distance (m)</th><th>mean speed</th>' +
    keys.map(k => `<th>${data.configs[k].label}<br>end err · drift</th>`).join('') + '</tr>';
  rows.forEach((r, j) => {
    h += `<tr><td>${j + 1}</td><td>${r.start_s.toFixed(0)}</td><td>${r.length_s.toFixed(0)}</td><td>${r.distance_m.toFixed(0)}</td><td>${r.mean_speed_kmh.toFixed(0)} km/h</td>`;
    for (const k of keys) {
      const o = data.configs[k].outages[j];
      const drift = o.distance_m > 50 ? o.drift_pct : null;
      h += `<td class="${drift == null ? '' : drift < 10 ? 'good' : 'badc'}">${o.end_err_m.toFixed(1)} m · ${drift == null ? 'stopped' : drift.toFixed(1) + ' %'}</td>`;
    }
    h += '</tr>';
  });
  $('outTable').innerHTML = h;
}

function renderDiag() {
  const d = data.configs.full.diagnostics, a = d.alignment;
  $('diag').innerHTML = `roll ${fmt(a.roll_deg)}° · pitch ${fmt(a.pitch_deg)}° · yaw-mount ${fmt(a.yaw_mis_deg)}°<br>` +
    `forward axis ${a.forward_converged ? 'converged ✓' : 'learning…'} · gyro ${a.gyro_frame} frame, scale ${fmt(a.gyro_scale, 3)}<br>` +
    `gyro bias ${fmt(d.gyro_bias * 180 / Math.PI * 3600, 0)} °/h · potholes/bumps flagged: ${d.bumps}<br>` +
    `mode transitions: ${d.transitions.length} · EKF updates ${d.ekf_updates.updates}, gated-out ${d.ekf_updates.rejected}` +
    ` · mount re-seats detected: ${a.mount_events || 0}`;
}

async function loadBenchmark() {
  try {
    const r = await fetch(`${CFG.data}/benchmark.json`);
    const j = await r.json();
    if (j.error) { $('benchMeta').textContent = j.error; return; }
    $('benchMeta').innerHTML = `Held-out drives <b>${j.meta.test_drives.join(', ')}</b> · ${j.meta.km} km · ${j.meta.gnss}. ${j.meta.protocol}.`;
    const labels = { ins_nhc: 'INS + NHC + ZUPT', ai_speed: '+ AI speed (SpeedNet)', full: '+ HMM map matching' };
    let h = '<tr><th>Outage</th><th>Configuration</th><th># outages</th><th>mean distance</th><th>end err median</th><th>drift median</th><th>drift aggregate</th><th>pass &lt;10 %</th></tr>';
    const bOrder = ['ins_nhc', 'ai_speed', 'full'];
    for (const L of Object.keys(j.summary).sort((a, b) => a - b)) {
      for (const c of bOrder.filter(k => j.summary[L][k])) {
        const s = j.summary[L][c];
        const ok = s.drift_pct_median < 10;
        h += `<tr><td>${L} s</td><td>${labels[c] || c}</td><td>${s.n_outages}</td><td>${s.mean_distance_m.toFixed(0)} m</td>
          <td>${s.end_err_median_m.toFixed(1)} m</td><td class="${ok ? 'good' : 'badc'}">${s.drift_pct_median.toFixed(1)} %</td>
          <td>${s.drift_pct_aggregate.toFixed(1)} %</td><td>${(100 * s.pass_rate_10pct).toFixed(0)} %</td></tr>`;
      }
    }
    $('benchTable').innerHTML = h;
    if (j.synthetic) {
      const names = { ins: 'INS + NHC + ZUPT', ai: '+ AI speed (SpeedNet)', full: '+ HMM map matching' };
      let t = '<tr><th>IMU</th><th>Configuration</th><th>drift mean</th><th>drift median</th><th>drift p90</th><th>worst</th><th>exit error (mean)</th><th>seeds &lt;10 %</th></tr>';
      for (const [prof, cfgsSyn] of Object.entries(j.synthetic)) {
        for (const k of ['ins', 'ai', 'full'].filter(k => cfgsSyn[k])) {
          const s = cfgsSyn[k];
          t += `<tr><td>${prof === 'fog' ? 'FOG @200 Hz' : 'smartphone @10 Hz'}</td><td>${names[k]}</td>
            <td class="${s.drift_pct_mean < 10 ? 'good' : 'badc'}">${s.drift_pct_mean.toFixed(1)} %</td><td>${s.drift_pct_median.toFixed(1)} %</td>
            <td>${s.drift_pct_p90.toFixed(1)} %</td><td>${s.drift_pct_max.toFixed(1)} %</td><td>${s.exit_err_mean_m.toFixed(0)} m</td>
            <td>${(100 * s.pass_rate_10pct).toFixed(0)} %</td></tr>`;
        }
      }
      $('synTable').innerHTML = t;
    }
    if (j.speednet) {
      const m = j.speednet;
      $('speedMetrics').innerHTML = `${m.params.toLocaleString()} parameters · ${m.window_s} s window @10 Hz<br>` +
        ['train', 'val', 'test'].map(k => `${k.padEnd(5)}: RMSE ${m[k].rmse_ms.toFixed(2)} m/s · MAE ${m[k].mae_ms.toFixed(2)} · σ-calibration ${m[k].calibration_std_of_z.toFixed(2)} · stationary acc ${(100 * m[k].stationary_accuracy).toFixed(1)} %`).join('<br>') +
        `<br><span class="muted">${m.dataset}</span>`;
    }
  } catch (e) { $('benchMeta').textContent = 'Benchmark unavailable: ' + e.message; }
}

init();
