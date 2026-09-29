/* DrishtiNav — story layer: sticky nav, before/after live demo, technical-pipeline toggle.
 * Kept fully independent from app.js (the interactive engine dashboard) so a failure here
 * can never break the "Run the engine" flow — everything below is wrapped defensively. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const CFG = window.DRISHTI || { data: '/data', engine: '' };

  /* ---------------------------------------------------------- sticky nav */
  function initNav() {
    const nav = $('topnav');
    const hero = document.querySelector('.hero');
    if (!nav || !hero) return;
    const onScroll = () => {
      nav.classList.toggle('visible', window.scrollY > hero.offsetHeight * 0.6);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    const toggle = $('navToggle');
    const links = document.querySelector('.topnav-links');
    if (toggle && links) {
      toggle.addEventListener('click', () => links.classList.toggle('open'));
      links.querySelectorAll('a').forEach(a => a.addEventListener('click', () => links.classList.remove('open')));
    }

    // scroll-spy: highlight the nav link for the section currently in view
    const sections = [...document.querySelectorAll('section[id]')].filter(s => !s.hidden);
    const navLinks = [...document.querySelectorAll('.topnav-links a')];
    if (sections.length && navLinks.length && 'IntersectionObserver' in window) {
      const spy = new IntersectionObserver(entries => {
        for (const e of entries) {
          if (!e.isIntersecting) continue;
          const link = navLinks.find(a => a.getAttribute('href') === '#' + e.target.id);
          if (link) { navLinks.forEach(a => a.classList.remove('active')); link.classList.add('active'); }
        }
      }, { rootMargin: '-40% 0px -50% 0px' });
      sections.forEach(s => spy.observe(s));
    }
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

  function drawFrame(cv, box, truthPts, pathPts, deniedMask, frac, color) {
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

    // travelled path up to `frac` of the way through
    const n = Math.max(1, Math.round(pathPts.length * frac));
    ctx.lineWidth = 3.5; ctx.strokeStyle = color; ctx.beginPath();
    let started = false;
    for (let i = 0; i < n; i++) {
      const p = pathPts[i];
      if (p[0] == null) continue;
      const [x, y] = P(p[0], p[1]);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // the moving dot
    if (n > 0 && pathPts[n - 1][0] != null) {
      const [x, y] = P(pathPts[n - 1][0], pathPts[n - 1][1]);
      ctx.fillStyle = color; ctx.strokeStyle = '#1A1A1A'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.fill(); ctx.stroke();
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
      drawFrame(bad, box, truth, badPts, denied, frac, COLORS_BA.bad);
      drawFrame(good, box, truth, goodPts, denied, frac, COLORS_BA.good);
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

    const CYCLE_S = 9;
    let t0 = performance.now();
    function loop(now) {
      if (playing) {
        let frac = ((now - t0) / 1000 / CYCLE_S) % 1;
        if (frac < 0) frac = 0;
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
