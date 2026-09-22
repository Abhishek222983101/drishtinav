/* Minimal dependency-free canvas line charts (works offline in the field). */
(function (global) {
  function setup(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || canvas.parentElement.clientWidth;
    const h = canvas.height && canvas.dataset.h ? +canvas.dataset.h : (canvas.getAttribute('height') | 0) || 240;
    canvas.dataset.h = h;
    canvas.width = w * dpr; canvas.height = h * dpr; canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function niceMax(v) {
    if (!isFinite(v) || v <= 0) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p;
    return 10 * p;
  }

  /**
   * lineChart(canvas, {x, series:[{y, color, width, dash, label}], bands:[[x0,x1]], yLabel, xLabel, yMax, cursor})
   */
  function lineChart(canvas, o) {
    const { ctx, w, h } = setup(canvas);
    const L = 48, R = 12, T = 12, B = 30;
    const pw = w - L - R, ph = h - T - B;
    const xs = o.x, x0 = xs[0], x1 = xs[xs.length - 1];
    let ymax = o.yMax;
    if (ymax == null) {
      ymax = 0;
      for (const s of o.series) for (const v of s.y) if (v != null && isFinite(v) && v > ymax) ymax = v;
      ymax = niceMax(ymax * 1.05);
    }
    const X = v => L + (v - x0) / (x1 - x0 || 1) * pw;
    const Y = v => T + ph - Math.min(v, ymax) / ymax * ph;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, h);
    for (const [a, b] of o.bands || []) {
      ctx.fillStyle = 'rgba(82,81,78,0.12)';
      ctx.fillRect(X(a), T, Math.max(1, X(b) - X(a)), ph);
    }
    ctx.strokeStyle = '#e4e4e4'; ctx.lineWidth = 1; ctx.fillStyle = '#555';
    ctx.font = '11px JetBrains Mono, monospace'; ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const v = ymax * i / 4, y = Y(v);
      ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(L + pw, y); ctx.stroke();
      ctx.fillText((v >= 100 ? v.toFixed(0) : v.toFixed(1)), L - 6, y + 4);
    }
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
      const v = x0 + (x1 - x0) * i / 5;
      ctx.fillText(v.toFixed(0), X(v), T + ph + 16);
    }
    if (o.xLabel) ctx.fillText(o.xLabel, L + pw / 2, h - 2);
    if (o.yLabel) { ctx.save(); ctx.translate(11, T + ph / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(o.yLabel, 0, 0); ctx.restore(); }
    for (const s of o.series) {
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.setLineDash(s.dash || []);
      ctx.beginPath(); let pen = false;
      for (let i = 0; i < xs.length; i++) {
        const v = s.y[i];
        if (v == null || !isFinite(v)) { pen = false; continue; }
        const px = X(xs[i]), py = Y(v);
        if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py);
      }
      ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.strokeStyle = '#1A1A1A'; ctx.lineWidth = 2; ctx.strokeRect(L, T, pw, ph);
    if (o.cursor != null) {
      ctx.strokeStyle = '#004E89'; ctx.lineWidth = 1.5; ctx.beginPath();
      ctx.moveTo(X(o.cursor), T); ctx.lineTo(X(o.cursor), T + ph); ctx.stroke();
    }
    let lx = L + 8;
    ctx.textAlign = 'left';
    for (const s of o.series) {
      if (!s.label) continue;
      ctx.fillStyle = s.color; ctx.fillRect(lx, T + 6, 14, 3);
      ctx.fillStyle = '#1A1A1A'; ctx.fillText(s.label, lx + 18, T + 11);
      lx += ctx.measureText(s.label).width + 34;
    }
  }

  function modeStrip(canvas, t, modes, cursor) {
    const { ctx, w, h } = setup(canvas);
    const colors = { GNSS: '#2ECC40', DR: '#FF4136', RECOVERY: '#FFD23F', DEGRADED: '#FF851B', INIT: '#ccc' };
    const t0 = t[0], t1 = t[t.length - 1];
    for (let i = 0; i < t.length - 1; i++) {
      ctx.fillStyle = colors[modes[i]] || '#ccc';
      const a = (t[i] - t0) / (t1 - t0) * w, b = (t[i + 1] - t0) / (t1 - t0) * w;
      ctx.fillRect(a, 0, Math.max(1, b - a + 0.5), h);
    }
    if (cursor != null) {
      ctx.fillStyle = '#1A1A1A';
      ctx.fillRect((cursor - t0) / (t1 - t0) * w - 1, 0, 3, h);
    }
  }

  global.Charts = { lineChart, modeStrip };
})(window);
