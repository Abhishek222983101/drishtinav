/* DrishtiNav — story layer: sticky nav, before/after live demo, technical-pipeline toggle.
 * Kept fully independent from app.js (the interactive engine dashboard) so a failure here
 * can never break the "Run the engine" flow — everything below is wrapped defensively. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const CFG = window.DRISHTI || { data: '/data', engine: '' };

  /* --------------------------------------------------------- view router */
  // The site is an app-shell: exactly one [data-view] section is visible at
  // a time, switched by nav clicks or the URL hash (#checklist, #demo, …).
  // Nothing scroll-based — this is real navigation, not a fade-in-on-scroll bar.
  const VIEWS = ['home', 'checklist', 'how', 'demo', 'app'];

  function showView(name, opts = {}) {
    if (!VIEWS.includes(name)) name = 'home';
    document.querySelectorAll('[data-view]').forEach(el => {
      el.classList.toggle('view-active', el.dataset.view === name);
    });
    document.querySelectorAll('.topnav-links a').forEach(a => {
      const target = (a.getAttribute('href') || '').replace('#', '') || 'home';
      a.classList.toggle('active', target === name);
    });
    if (!opts.silent) window.scrollTo(0, 0);
    document.dispatchEvent(new CustomEvent('view:shown', { detail: { name } }));
  }

  function initNav() {
    const nav = $('topnav');
    if (!nav) return;

    document.body.addEventListener('click', e => {
      const a = e.target.closest('a[href^="#"]');
      if (!a) return;
      const name = a.getAttribute('href').replace('#', '') || 'home';
      if (!VIEWS.includes(name)) return; // e.g. plain external anchors, if any
      e.preventDefault();
      if (location.hash !== '#' + name) location.hash = name;
      else showView(name);
      const links = document.querySelector('.topnav-links');
      if (links) links.classList.remove('open');
    });

    window.addEventListener('hashchange', () => {
      showView((location.hash || '#home').slice(1));
    });

    const toggle = $('navToggle');
    const links = document.querySelector('.topnav-links');
    if (toggle && links) {
      toggle.addEventListener('click', () => links.classList.toggle('open'));
    }

    showView((location.hash || '#home').slice(1), { silent: true });
  }

  /* ------------------------------------------------- technical pipeline toggle */
  function initTechToggle() {
    const btn = $('techToggle'), box = $('pipeline');
    if (!btn || !box) return;
    btn.addEventListener('click', () => {
      const show = box.hidden;
      box.hidden = !show;
      btn.textContent = show ? '🔧 Hide the technical pipeline' : '🔧 Show the technical pipeline';
      if (show) box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
  }

  /* --------------------------------------------------- before/after live demo */
  const COLORS_BA = { road: '#c9c6bf', deny: 'rgba(255,65,54,0.14)', bad: '#2a78d6', good: '#1baf7a', dot: '#1A1A1A' };

  function project(lat, lon, box, w, h, pad) {
    const sx = (w - 2 * pad) / Math.max(box.lon1 - box.lon0, 1e-9);
    const sy = (h - 2 * pad) / Math.max(box.lat1 - box.lat0, 1e-9);
    const s = Math.min(sx, sy);
    const ox = pad + (w - 2 * pad - s * (box.lon1 - box.lon0)) / 2;
    const oy = pad + (h - 2 * pad - s * (box.lat1 - box.lat0)) / 2;
    return [ox + (lon - box.lon0) * s, h - (oy + (lat - box.lat0) * s)]; // flip Y (screen space)
  }

  function setupCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || 340, h = w * 0.62;
    cv.width = w * dpr; cv.height = h * dpr;
    cv.style.height = h + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function drawFrame(cv, box, truthPts, pathPts, deniedMask, frac, color, opts) {
    const { ctx, w, h } = setupCanvas(cv);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, h);
    const P = (lat, lon) => project(lat, lon, box, w, h, 14);

    // GPS-denied shaded band, drawn as a thick translucent road segment
    ctx.lineWidth = 16; ctx.lineCap = 'round'; ctx.strokeStyle = COLORS_BA.deny;
    ctx.beginPath();
    let penDown = false;
    for (let i = 0; i < truthPts.length; i++) {
      if (deniedMask[i]) {
        const [x, y] = P(truthPts[i][0], truthPts[i][1]);
        if (!penDown) { ctx.moveTo(x, y); penDown = true; } else ctx.lineTo(x, y);
      } else penDown = false;
    }
    ctx.stroke();

    // reference road
    ctx.lineWidth = 3; ctx.strokeStyle = COLORS_BA.road; ctx.beginPath();
    truthPts.forEach(([lat, lon], i) => { const [x, y] = P(lat, lon); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();

    // travelled path up to `frac` of the way through — a fading trail, not a static line
    const n = Math.max(1, Math.round(pathPts.length * frac));
    const TRAIL = 26; // most-recent points drawn opaque -> transparent
    for (let i = Math.max(1, n - TRAIL); i < n; i++) {
      const a = pathPts[i - 1], b = pathPts[i];
      if (a[0] == null || b[0] == null) continue;
      const age = (n - i) / TRAIL; // 0 = newest
      ctx.globalAlpha = i < n - TRAIL + 4 ? Math.max(0, 1 - age) * 0.5 : 1;
      ctx.lineWidth = 3.5; ctx.strokeStyle = color; ctx.lineCap = 'round';
      const [x1, y1] = P(a[0], a[1]), [x2, y2] = P(b[0], b[1]);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }
    // the rest of the driven path, drawn faint once it's outside the trail window
    if (n - TRAIL > 1) {
      ctx.globalAlpha = 0.22; ctx.lineWidth = 2.5; ctx.strokeStyle = color; ctx.beginPath();
      let started = false;
      for (let i = 0; i < n - TRAIL; i++) {
        const p = pathPts[i];
        if (p[0] == null) continue;
        const [x, y] = P(p[0], p[1]);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;

    // the moving marker — oriented in the direction of travel, with a live-lock
    // pulse ring when GPS is available and a broken-signal glyph when it isn't
    let head = null;
    for (let i = n - 1; i >= 1; i--) {
      if (pathPts[i][0] != null && pathPts[i - 1][0] != null) { head = i; break; }
    }
    if (head != null) {
      const [x, y] = P(pathPts[head][0], pathPts[head][1]);
      const [px, py] = P(pathPts[head - 1][0], pathPts[head - 1][1]);
      const heading = Math.atan2(y - py, x - px);
      const denied = !!(opts && opts.denied);
      const entryFade = Math.min(1, frac / 0.03); // avoid a hard snap at loop restart
      ctx.globalAlpha = entryFade;

      if (!denied) {
        const pulse = 9 + 4 * (0.5 + 0.5 * Math.sin(performance.now() / 260));
        ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.globalAlpha = entryFade * 0.35;
        ctx.beginPath(); ctx.arc(x, y, pulse, 0, Math.PI * 2); ctx.stroke();
        ctx.globalAlpha = entryFade;
      }

      // a solid, convex arrowhead — the shape real map apps use for a heading
      // marker; the previous concave dart shape read as a smudge at this size
      ctx.save();
      ctx.translate(x, y); ctx.rotate(heading);
      ctx.beginPath(); ctx.ellipse(-1, 0, 8, 8, 0, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0,0,0,.14)'; ctx.fill();
      ctx.beginPath();
      ctx.moveTo(12, 0); ctx.lineTo(-7, 7); ctx.lineTo(-3, 0); ctx.lineTo(-7, -7); ctx.closePath();
      ctx.fillStyle = denied ? COLORS_BA.dot : color;
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
      ctx.stroke();
      ctx.fill();
      ctx.restore();

      if (denied) {
        ctx.globalAlpha = entryFade * (0.55 + 0.45 * Math.sin(performance.now() / 180));
        ctx.font = '600 12px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('📡', x + 14, y - 12);
      }
      ctx.globalAlpha = 1;
    }
  }

  async function initBeforeAfter() {
    const bad = $('baCanvasBad'), good = $('baCanvasGood'), caption = $('baCaption');
    if (!bad || !good) return;
    let j;
    try {
      const r = await fetch(`${CFG.data}/runs/sim__delhi_pragati_tunnel__smartphone__tunnel.json`);
      j = await r.json();
      if (j.error) throw new Error(j.error);
    } catch (e) {
      console.error('before/after: fetch failed', e);
      if (caption) caption.textContent = 'Live trajectory data unavailable right now — see the interactive demo below instead.';
      return;
    }

    try {
      setupBeforeAfterAnimation(j, bad, good, caption);
    } catch (e) {
      console.error('before/after: setup failed', e);
      if (caption) caption.textContent = 'Live trajectory data unavailable right now — see the interactive demo below instead.';
    }
  }

  function setupBeforeAfterAnimation(j, bad, good, caption) {
    const truth = j.truth.lat.map((la, i) => [la, j.truth.lon[i]]);
    const box = { lat0: Math.min(...j.truth.lat), lat1: Math.max(...j.truth.lat), lon0: Math.min(...j.truth.lon), lon1: Math.max(...j.truth.lon) };
    const badCfg = j.configs.ins || j.configs.full;
    const goodCfg = j.configs.full;
    const badPts = badCfg.lat.map((la, i) => [la, badCfg.lon[i]]);
    const goodPts = goodCfg.matched_lat.map((la, i) => [la ?? goodCfg.lat[i], goodCfg.matched_lon[i] ?? goodCfg.lon[i]]);
    const denied = j.denied;
    const n = truth.length;
    const deniedStart = denied.indexOf(1);
    const deniedEnd = n - 1 - [...denied].reverse().indexOf(1);

    function renderAt(frac) {
      const idxNow = Math.min(n - 1, Math.round(frac * n));
      const nowDenied = !!denied[idxNow];
      drawFrame(bad, box, truth, badPts, denied, frac, COLORS_BA.bad, { denied: nowDenied });
      drawFrame(good, box, truth, goodPts, denied, frac, COLORS_BA.good, { denied: nowDenied });
      if (caption) {
        const idx = Math.min(n - 1, Math.round(frac * n));
        const inOutage = idx >= deniedStart && idx <= deniedEnd && deniedStart >= 0;
        const badErr = badCfg.err[idx], goodErr = goodCfg.err[idx];
        caption.innerHTML = inOutage
          ? `<b>📡 GPS lost right now (t=${j.t[idx].toFixed(0)}s)</b> — basic physics is off by <b class="ba-num-bad">${(badErr ?? 0).toFixed(0)} m</b>, DrishtiNav is off by <b class="ba-num-good">${(goodErr ?? 0).toFixed(0)} m</b>`
          : `t=${j.t[idx].toFixed(0)}s — GPS available, both approaches tracking closely (basic: ${(badErr ?? 0).toFixed(1)} m, DrishtiNav: ${(goodErr ?? 0).toFixed(1)} m)`;
      }
    }

    // Paint one real frame immediately, regardless of visibility/timing — the
    // panel must never show blank or placeholder content, even for a fraction
    // of a second or if IntersectionObserver / RAF is ever throttled.
    renderAt(0);

    let playing = true;
    const observer = ('IntersectionObserver' in window) ? new IntersectionObserver(es => {
      playing = es.some(e => e.isIntersecting);
    }, { threshold: 0.1 }) : null;
    if (observer) observer.observe($('beforeAfter'));

    // The canvas lives inside the "home" app-shell view; when it's hidden
    // (display:none) clientWidth reads 0, so re-measure the moment it's shown.
    document.addEventListener('view:shown', e => {
      if (e.detail.name === 'home') requestAnimationFrame(() => renderAt(lastFrac));
    });

    const CYCLE_S = 9;
    let t0 = performance.now();
    let lastFrac = 0;
    function loop(now) {
      if (playing) {
        let frac = ((now - t0) / 1000 / CYCLE_S) % 1;
        if (frac < 0) frac = 0;
        lastFrac = frac;
        renderAt(frac);
      }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }

  function init() {
    try { initNav(); } catch (e) { console.error('nav init failed', e); }
    try { initTechToggle(); } catch (e) { console.error('tech toggle init failed', e); }
    try { initBeforeAfter(); } catch (e) { console.error('before/after init failed', e); }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
