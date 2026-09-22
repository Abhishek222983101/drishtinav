/* DrishtiNav phone app: live sensors or replay -> on-device engine -> smooth map. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const E = window.DrishtiEngine;
  const CFG = window.DRISHTI || { data: '/data', engine: '' };    // written by /config.js
  const RATE = 10;                         // engine / SpeedNet rate (Hz)
  let map, marker, pathLine, truthLine, gnssLayer, speednet = null;
  let engine = null, mode = 'idle', timer = null, replay = null, recorder = null;
  let lastOut = null, prevOut = null, lastOutWall = 0, drStart = null, drDist = 0;
  let pathPts = [];

  /* ------------------------------------------------------------ map */
  function initMap() {
    map = L.map('map', { zoomControl: false, attributionControl: true }).setView([28.6139, 77.209], 13);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OpenStreetMap' }).addTo(map);
    const icon = L.divIcon({ className: 'car', iconSize: [34, 34], iconAnchor: [17, 17],
      html: '<svg viewBox="0 0 34 34"><circle cx="17" cy="17" r="15" fill="#004E89" stroke="#fff" stroke-width="3"/><path d="M17 6 L25 25 L17 21 L9 25 Z" fill="#FFD23F" stroke="#1A1A1A" stroke-width="1.5"/></svg>' });
    marker = L.marker([0, 0], { icon, zIndexOffset: 1000 });
    pathLine = L.polyline([], { color: '#1baf7a', weight: 5, opacity: .95 }).addTo(map);
    truthLine = L.polyline([], { color: '#1A1A1A', weight: 3, opacity: .45, dashArray: '4 6' }).addTo(map);
    gnssLayer = L.layerGroup().addTo(map);
  }

  /* smooth icon: interpolate between the last two engine outputs at display rate */
  function animate() {
    if (lastOut && isFinite(lastOut.matchedLat)) {
      let lat = lastOut.matchedLat, lon = lastOut.matchedLon, hd = lastOut.headingDeg;
      if (prevOut && isFinite(prevOut.matchedLat)) {
        const period = 1000 / RATE / (replay ? replay.speed : 1);
        const a = Math.min(1, (performance.now() - lastOutWall) / period);
        lat = prevOut.matchedLat + (lastOut.matchedLat - prevOut.matchedLat) * a;
        lon = prevOut.matchedLon + (lastOut.matchedLon - prevOut.matchedLon) * a;
        let dh = ((lastOut.headingDeg - prevOut.headingDeg + 540) % 360) - 180;
        hd = prevOut.headingDeg + dh * a;
      }
      if (!map.hasLayer(marker)) { marker.addTo(map); map.setView([lat, lon], 17); }
      marker.setLatLng([lat, lon]);
      const svg = marker.getElement() && marker.getElement().querySelector('svg');
      if (svg) svg.style.transform = `rotate(${hd}deg)`;
      if (followMap) {
        // keep the car centred in the part of the map that the bottom sheet does not cover
        const sheet = $('sheet'), size = map.getSize();
        const covered = sheet.classList.contains('collapsed') ? 92 : sheet.offsetHeight;
        const p = map.latLngToContainerPoint([lat, lon]);
        const dx = p.x - size.x / 2, dy = p.y - (size.y - covered + 60) / 2;
        if (Math.abs(dx) > 1 || Math.abs(dy) > 1) map.panBy([dx, dy], { animate: false });
      }
    }
    requestAnimationFrame(animate);
  }
  let followMap = true;

  /* ------------------------------------------------------- engine I/O */
  function newEngine(frame, roads) {
    engine = new E.NavEngine({ speednet, roads, frame });
    lastOut = prevOut = null; drStart = null; pathPts = []; pathLine.setLatLngs([]); gnssLayer.clearLayers();
  }

  function onOutput(o, truth) {
    prevOut = lastOut; lastOut = o; lastOutWall = performance.now();
    if (isFinite(o.matchedLat)) {
      pathPts.push([o.matchedLat, o.matchedLon]);
      if (pathPts.length % 5 === 0) pathLine.setLatLngs(pathPts);
    }
    $('modePill').textContent = o.mode; $('modePill').className = 'pill mode-' + o.mode;
    $('hSpeed').textContent = (o.speed * 3.6).toFixed(0);
    $('hHead').textContent = isFinite(o.headingDeg) ? o.headingDeg.toFixed(0) : '—';
    $('hSig').textContent = isFinite(o.posSigma) ? o.posSigma.toFixed(1) : '—';
    $('dAi').textContent = isFinite(o.aiSpeed) ? `${(o.aiSpeed * 3.6).toFixed(0)} km/h ± ${(o.aiSigma * 3.6).toFixed(0)}` : 'warming up (20 s)';
    $('dStat').textContent = o.stationary ? 'yes — ZUPT active' : (isFinite(o.pStat) ? `no (p=${o.pStat.toFixed(2)})` : 'no');
    $('dMap').textContent = o.matchConf ? `${(100 * o.matchConf).toFixed(0)} %${o.inTunnel ? ' · in tunnel' : ''}` : 'no roads loaded';
    const a = engine.align.state;
    $('dMount').textContent = `${a.roll_deg.toFixed(0)}° / ${a.pitch_deg.toFixed(0)}° / ${a.yaw_mis_deg.toFixed(0)}°${a.forward_converged ? '' : ' (learning)'}${a.mount_events ? ` · re-seated ×${a.mount_events}` : ''}`;
    $('dRate').textContent = `${RATE} Hz · ${o.latencyMs.toFixed(2)} ms`;
    if (truth) {
      const [te, tn] = engine.frame.toEN(truth[0], truth[1]);
      $('dErr').textContent = Math.hypot(o.e - te, o.n - tn).toFixed(1) + ' m';
    }
    // outage banner with dead-reckoned distance
    const b = $('banner');
    if (o.mode === 'DR') {
      if (drStart == null) { drStart = o.t; drDist = 0; }
      drDist += o.speed / RATE;
      b.hidden = false; b.className = 'banner dr';
      b.textContent = `GNSS lost — AI dead reckoning · ${(o.t - drStart).toFixed(0)} s · ${drDist.toFixed(0)} m${o.inTunnel ? ' · tunnel' : ''}`;
    } else if (o.mode === 'RECOVERY') {
      b.hidden = false; b.className = 'banner'; b.textContent = 'GNSS re-acquired — blending back smoothly';
    } else if (drStart != null && o.mode === 'GNSS') {
      drStart = null; setTimeout(() => { if (lastOut && lastOut.mode === 'GNSS') $('banner').hidden = true; }, 2500);
    }
  }

  /* ------------------------------------------------------------ live */
  const acc = { a: [0, 0, 0], g: [0, 0, 0], n: 0 };
  let latestFix = null, fixConsumed = true, t0Live = 0, wakeLock = null;

  async function startLive() {
    stopAll();
    try {
      if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        const p = await DeviceMotionEvent.requestPermission();
        if (p !== 'granted') throw new Error('motion permission denied');
      }
    } catch (e) { alertBanner('Motion sensors unavailable: ' + e.message); return; }
    if (!window.isSecureContext) alertBanner('Sensors need HTTPS (see README: python -m drishtinav serve --https)');
    mode = 'live'; $('btnLive').classList.add('active'); $('btnLive').textContent = '■ Stop live';
    newEngine(null, null); roadsRequested = false;
    window.addEventListener('devicemotion', onMotion);
    watchId = navigator.geolocation.watchPosition(onFix, e => alertBanner('GNSS: ' + e.message),
      { enableHighAccuracy: true, maximumAge: 0, timeout: 10000 });
    try { wakeLock = await navigator.wakeLock.request('screen'); } catch (e) { /* optional */ }
    t0Live = performance.now();
    timer = setInterval(liveTick, 1000 / RATE);
  }
  let watchId = null;

  function onMotion(ev) {
    const a = ev.accelerationIncludingGravity, r = ev.rotationRate;
    if (!a || a.x == null) return;
    acc.a[0] += a.x; acc.a[1] += a.y; acc.a[2] += a.z;
    if (r && r.alpha != null) { acc.g[0] += r.beta * Math.PI / 180; acc.g[1] += r.gamma * Math.PI / 180; acc.g[2] += r.alpha * Math.PI / 180; }
    acc.n++;
  }

  /* Offline road database for map matching: a bundled extract if we are inside
   * one, otherwise download ~2.5 km around us from Overpass once and cache it. */
  let roadsRequested = false;
  async function ensureRoads(lat, lon) {
    if (roadsRequested || !engine || engine.roadsData) return;
    roadsRequested = true;
    try {
      const idx = await (await fetch(`${CFG.data}/osm/index.json`)).json();
      const hit = idx.find(r => r.bbox[0] < lat && lat < r.bbox[2] && r.bbox[1] < lon && lon < r.bbox[3]);
      let data = hit ? await (await fetch(`${CFG.data}/osm/${hit.file}`)).json() : loadCachedRoads();
      const inside = d => d && d.bbox[0] < lat && lat < d.bbox[2] && d.bbox[1] < lon && lon < d.bbox[3];
      if (!inside(data)) data = await fetchOverpass(lat, lon);
      if (data && engine) { engine.roadsData = data; if (engine.frame) engine.buildRoads(); }
    } catch (e) { alertBanner('Map matching off (no road data): ' + e.message); }
  }

  async function fetchOverpass(lat, lon) {
    const d = 0.025, bb = [lat - d, lon - d, lat + d, lon + d];
    const hw = 'motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link';
    const q = `[out:json][timeout:60];way["highway"~"^(${hw})$"](${bb.join(',')});out body;>;out skel qt;`;
    const osm = await (await fetch('https://overpass-api.de/api/interpreter', { method: 'POST', body: 'data=' + encodeURIComponent(q),
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' } })).json();
    const nodes = new Map(), idx = new Map(), outNodes = [], ways = [];
    for (const el of osm.elements) if (el.type === 'node') nodes.set(el.id, [el.lat, el.lon]);
    for (const el of osm.elements) {
      if (el.type !== 'way') continue;
      const t = el.tags || {}, refs = [];
      for (const id of el.nodes) { if (!nodes.has(id)) continue; if (!idx.has(id)) { idx.set(id, outNodes.length); outNodes.push(nodes.get(id)); } refs.push(idx.get(id)); }
      if (refs.length < 2) continue;
      const ow = t.oneway;
      ways.push({ id: el.id, nodes: refs, hw: t.highway || '', name: t.name || '', tunnel: t.tunnel === 'yes' ? 1 : 0,
        oneway: (ow === 'yes' || ow === '1' || (t.highway || '').startsWith('motorway') || t.junction === 'roundabout') ? 1 : ow === '-1' ? -1 : 0 });
    }
    const data = { name: 'local', bbox: bb, nodes: outNodes, ways };
    try { localStorage.setItem('drishtinav.roads', JSON.stringify(data)); } catch (e) { /* quota: keep in memory */ }
    return data;
  }

  function onFix(p) {
    const c = p.coords;
    ensureRoads(c.latitude, c.longitude);
    latestFix = { lat: c.latitude, lon: c.longitude, speed: c.speed != null ? c.speed : null,
      course: c.heading != null && isFinite(c.heading) ? c.heading : null, accuracy: c.accuracy || 10 };
    fixConsumed = false;
    $('hGnss').textContent = c.accuracy ? c.accuracy.toFixed(0) : '—';
    $('hGnssSub').textContent = $('chkOutage').checked ? 'blocked (sim)' : 'm accuracy';
  }

  function liveTick() {
    if (!acc.n) return;
    const a = acc.a.map(v => v / acc.n), g = acc.g.map(v => v / acc.n);
    acc.a = [0, 0, 0]; acc.g = [0, 0, 0]; acc.n = 0;
    const t = (performance.now() - t0Live) / 1000;
    let fix = null;
    if (!fixConsumed && latestFix && !$('chkOutage').checked) fix = latestFix;
    fixConsumed = true;
    const o = engine.step(t, a, g, fix, $('chkOutage').checked);
    if (recorder) recorder.push([t.toFixed(3), ...a.map(v => v.toFixed(4)), ...g.map(v => v.toFixed(5)),
      ...(fix ? [fix.lat, fix.lon, fix.speed ?? '', fix.course ?? '', fix.accuracy] : ['', '', '', '', ''])].join(','));
    onOutput(o, null);
  }

  /* ---------------------------------------------------------- replay */
  async function startReplay() {
    stopAll();
    const key = $('replaySel').value;
    alertBanner('Loading drive…');
    const drv = await (await fetch(`${CFG.data}/drives/${key}.json`)).json();
    const roads = drv.roads ? await (await fetch(`${CFG.data}/osm/${drv.roads}`)).json() : null;
    $('banner').hidden = true;
    const frame = new E.LocalFrame(drv.truth[0][0], drv.truth[0][1]);
    newEngine(frame, roads);
    truthLine.setLatLngs(drv.truth.filter((_, i) => i % 5 === 0));
    map.fitBounds(truthLine.getBounds(), { padding: [30, 30] });
    followMap = false; setTimeout(() => { followMap = true; }, 2500);
    replay = { drv, i: 0, speed: +$('replaySpeed').value };
    mode = 'replay'; $('btnReplay').classList.add('active'); $('btnReplay').textContent = '■ Stop replay';
    timer = setInterval(replayTick, 1000 / RATE / replay.speed);
  }

  function replayTick() {
    const r = replay, d = r.drv;
    if (r.i >= d.t.length) { stopAll(); alertBanner('Replay finished'); return; }
    const i = r.i++;
    const g = d.gnss[i];
    const blocked = $('chkOutage').checked;
    const fix = g && !blocked ? { lat: g[0], lon: g[1], speed: g[2], course: g[3], accuracy: g[4] } : null;
    if (fix && i % 20 === 0) L.circleMarker([fix.lat, fix.lon], { radius: 2, color: '#777', weight: 1 }).addTo(gnssLayer);
    $('hGnss').textContent = g ? g[4].toFixed(0) : (d.denied[i] || blocked ? '✕' : '…');
    $('hGnssSub').textContent = d.denied[i] ? 'tunnel / outage' : (blocked ? 'blocked (sim)' : 'm accuracy');
    onOutput(engine.step(d.t[i], d.accel[i], d.gyro[i], fix, blocked), d.truth[i]);
  }

  /* ------------------------------------------------------------ misc */
  function stopAll() {
    clearInterval(timer); timer = null;
    if (mode === 'live') {
      window.removeEventListener('devicemotion', onMotion);
      if (watchId != null) navigator.geolocation.clearWatch(watchId);
      if (wakeLock) wakeLock.release().catch(() => {});
    }
    mode = 'idle'; replay = null;
    for (const [id, txt] of [['btnLive', '▶ Live (phone sensors)'], ['btnReplay', '⟲ Replay demo']]) { $(id).classList.remove('active'); $(id).textContent = txt; }
  }

  function alertBanner(msg) { const b = $('banner'); b.hidden = false; b.className = 'banner'; b.textContent = msg; }

  function toggleRecord() {
    if (!recorder) {
      recorder = ['t,ax,ay,az,gx,gy,gz,lat,lon,speed,course,hacc'];
      $('btnRecord').classList.add('active'); $('btnRecord').textContent = '■ Save';
      if (mode !== 'live') alertBanner('Recording starts with Live mode');
      return;
    }
    const blob = new Blob([recorder.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = `drishtinav_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.csv`;
    a.click();
    recorder = null; $('btnRecord').classList.remove('active'); $('btnRecord').textContent = '● Record';
  }

  function loadCachedRoads() {
    try { const s = localStorage.getItem('drishtinav.roads'); return s ? JSON.parse(s) : null; } catch (e) { return null; }
  }

  async function loadScenarios() {
    try {
      const list = await (await fetch(`${CFG.data}/scenarios.json`)).json();
      for (const s of list.filter(s => s.drive)) {
        const o = document.createElement('option'); o.value = s.drive; o.textContent = s.title; $('replaySel').appendChild(o);
      }
    } catch (e) { $('replaySel').innerHTML = '<option>offline — replay unavailable</option>'; }
  }

  async function init() {
    initMap();
    try { speednet = await (await fetch('/models/speednet.json')).json(); } catch (e) { alertBanner('SpeedNet model not loaded: AI speed disabled'); }
    loadScenarios();
    $('btnLive').onclick = () => (mode === 'live' ? stopAll() : startLive());
    $('btnReplay').onclick = () => (mode === 'replay' ? stopAll() : startReplay());
    $('btnRecord').onclick = toggleRecord;
    $('grab').onclick = () => $('sheet').classList.toggle('collapsed');
    map.on('dragstart', () => { followMap = false; setTimeout(() => { followMap = true; }, 8000); });
    if ('serviceWorker' in navigator) navigator.serviceWorker.register('/app/sw.js', { scope: '/app/' }).catch(() => {});
    requestAnimationFrame(animate);
  }
  init();
})();
