// Canvas plots. Colours come from the CSS variables each time a plot is drawn, so the theme toggle just works.
// Line charts support a hover read-out (frequency / level).
const Plots = (() => {
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const isLight = () => document.documentElement.getAttribute("data-theme") === "light";
  const col = () => ({ text: css("--text"), muted: css("--muted"), faint: css("--faint"), line: css("--line"), line2: css("--line2"),
    accent: css("--accent-text") || css("--accent"), bg: css("--bg"), panel: css("--panel"), ok: css("--ok") });
  function rgba(hex, a) {
    const h = hex.replace("#", ""); const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }
  const FONT = "11.5px 'IBM Plex Mono', ui-monospace, Consolas, monospace";

  function fmtHz(v) {
    if (v === null || v === undefined || isNaN(v)) return "-";
    const a = Math.abs(v);
    if (a >= 1e6) return (v / 1e6).toFixed(3) + " MHz";
    if (a >= 1e3) return (v / 1e3).toFixed(2) + " kHz";
    return v.toFixed(1) + " Hz";
  }
  const fmtDb = (v) => (v === null || v === undefined ? "-" : v.toFixed(1) + " dB");

  function prep(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || canvas.parentElement.clientWidth || 600;
    const H = parseInt(canvas.dataset.h || "240", 10);
    canvas.style.height = H + "px";
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H); ctx.font = FONT;
    return { ctx, W, H, C: col() };
  }
  function empty(canvas, msg) { const { ctx, W, H, C } = prep(canvas); ctx.fillStyle = C.faint; ctx.textAlign = "center"; ctx.fillText(msg, W / 2, H / 2); }

  // ------------------------------------------------------------------ generic line chart with hover
  function lineChart(canvas, xs, ys, o = {}) {
    const state = { xs, ys, o };
    canvas._lc = state;
    const render = (hoverIdx) => {
      const { ctx, W, H, C } = prep(canvas);
      const L = 48, R = 12, T = 12, B = 30;
      const xmin = xs[0], xmax = xs[xs.length - 1];
      let ymax = Math.max(...ys), ymin = Math.min(...ys);
      if (o.yMin !== undefined) ymin = o.yMin;
      if (o.yMax !== undefined) ymax = o.yMax; else { ymax += 3; ymin -= 3; }
      const X = (v) => L + ((v - xmin) / (xmax - xmin)) * (W - L - R);
      const Y = (v) => T + ((ymax - v) / (ymax - ymin)) * (H - T - B);
      Object.assign(state, { L, R, W, xmin, xmax, X, Y });
      ctx.strokeStyle = C.line; ctx.fillStyle = C.muted; ctx.lineWidth = 1;
      const nx = W < 460 ? 3 : 5;
      for (let i = 0; i <= 4; i++) { const v = ymin + ((ymax - ymin) * i) / 4; ctx.beginPath(); ctx.moveTo(L, Y(v) + .5); ctx.lineTo(W - R, Y(v) + .5); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(v.toFixed(0), L - 7, Y(v) + 4); }
      for (let i = 0; i <= nx; i++) { const v = xmin + ((xmax - xmin) * i) / nx; ctx.textAlign = "center"; ctx.fillText(fmtHz(v).replace(" ", ""), X(v), H - B + 17); ctx.beginPath(); ctx.moveTo(X(v) + .5, H - B); ctx.lineTo(X(v) + .5, H - B + 4); ctx.stroke(); }
      if (o.band) { ctx.fillStyle = rgba(C.accent, 0.13); ctx.fillRect(X(o.band[0]), T, X(o.band[1]) - X(o.band[0]), H - T - B); }
      ctx.strokeStyle = C.text; ctx.lineWidth = 1.3; ctx.beginPath();
      xs.forEach((v, i) => (i ? ctx.lineTo(X(v), Y(ys[i])) : ctx.moveTo(X(v), Y(ys[i])))); ctx.stroke();
      if (o.hline !== undefined) { ctx.strokeStyle = C.faint; ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(L, Y(o.hline)); ctx.lineTo(W - R, Y(o.hline)); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = C.faint; ctx.textAlign = "left"; ctx.fillText("noise floor", L + 6, Y(o.hline) - 6); }
      if (o.marker !== undefined) { ctx.strokeStyle = C.accent; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(o.marker), T); ctx.lineTo(X(o.marker), H - B); ctx.stroke(); ctx.fillStyle = C.accent; ctx.textAlign = "left"; ctx.fillText(o.markerLabel || "", X(o.marker) + 6, T + 13); }
      ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.fillText(o.ylabel || "", L + 6, T + 13);
      if (hoverIdx !== undefined && hoverIdx >= 0) {
        const hx = X(xs[hoverIdx]), hy = Y(ys[hoverIdx]);
        ctx.strokeStyle = C.accent; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(hx + .5, T); ctx.lineTo(hx + .5, H - B); ctx.stroke();
        ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(hx, hy, 3.5, 0, 6.3); ctx.fill();
        const label = `${fmtHz(xs[hoverIdx])}  ${ys[hoverIdx].toFixed(1)} ${o.unit || "dB"}`;
        const tw = ctx.measureText(label).width + 14, tx = Math.min(Math.max(hx - tw / 2, L), W - R - tw);
        ctx.fillStyle = C.panel; ctx.strokeStyle = C.line2; ctx.fillRect(tx, T + 22, tw, 24); ctx.strokeRect(tx + .5, T + 22.5, tw, 24);
        ctx.fillStyle = C.text; ctx.textAlign = "left"; ctx.fillText(label, tx + 7, T + 38);
      }
    };
    state.render = render; render();
    if (!canvas._hoverBound) {
      canvas._hoverBound = true;
      canvas.addEventListener("mousemove", (e) => {
        const s = canvas._lc; if (!s || !s.X) return;
        const r = canvas.getBoundingClientRect(), px = e.clientX - r.left;
        const f = s.xmin + ((px - s.L) / (s.W - s.L - s.R)) * (s.xmax - s.xmin);
        let lo = 0, hi = s.xs.length - 1;
        while (hi - lo > 1) { const m = (lo + hi) >> 1; (s.xs[m] < f ? (lo = m) : (hi = m)); }
        s.render(Math.abs(s.xs[lo] - f) < Math.abs(s.xs[hi] - f) ? lo : hi);
      });
      canvas.addEventListener("mouseleave", () => canvas._lc && canvas._lc.render());
    }
  }

  const psd = (canvas, d) => lineChart(canvas, d.psd.freq, d.psd.db, { band: d.params.band_low_hz !== undefined ? [d.params.band_low_hz, d.params.band_high_hz] : null, hline: d.params.noise_floor_db, ylabel: "dB/Hz" });
  const rateSpectrum = (canvas, spec, rate) => lineChart(canvas, spec.freq, spec.db, { yMin: -60, yMax: 3, marker: rate, markerLabel: fmtHz(rate), ylabel: "relative dB", unit: "dB" });

  // ------------------------------------------------------------------ waterfall
  function colormap(t) {
    t = Math.max(0, Math.min(1, t));
    const S = isLight() ? [[250, 249, 246], [247, 214, 190], [226, 120, 60], [172, 52, 8], [60, 18, 4]]
                        : [[10, 10, 12], [48, 28, 16], [170, 76, 18], [255, 138, 61], [255, 240, 224]];
    const s = t * (S.length - 1), i = Math.min(Math.floor(s), S.length - 2), u = s - i;
    return S[i].map((a, k) => Math.round(a + (S[i + 1][k] - a) * u));
  }
  function waterfall(canvas, w) {
    const { ctx, W, H, C } = prep(canvas);
    const L = 48, R = 12, T = 6, B = 30;
    const off = document.createElement("canvas"); off.width = w.cols; off.height = w.rows;
    const og = off.getContext("2d"), img = og.createImageData(w.cols, w.rows), flat = w.db.flat();
    const sorted = Float32Array.from(flat).sort(), lo = sorted[Math.floor(sorted.length * 0.02)], hi = sorted[Math.floor(sorted.length * 0.995)];
    flat.forEach((v, i) => { const [r, g, b] = colormap((v - lo) / (hi - lo + 1e-9)); img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255; });
    og.putImageData(img, 0, 0); ctx.imageSmoothingEnabled = false; ctx.drawImage(off, L, T, W - L - R, H - T - B);
    ctx.strokeStyle = C.line; ctx.strokeRect(L + .5, T + .5, W - L - R, H - T - B);
    ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.fillText("0 ms", L - 7, T + 12); ctx.fillText((w.t_end * 1000).toFixed(0) + " ms", L - 7, H - B);
    const fmin = w.freq[0], fmax = w.freq[w.freq.length - 1], nx = W < 460 ? 3 : 5; ctx.textAlign = "center";
    for (let i = 0; i <= nx; i++) ctx.fillText(fmtHz(fmin + ((fmax - fmin) * i) / nx).replace(" ", ""), L + ((W - L - R) * i) / nx, H - 10);
  }

  // ------------------------------------------------------------------ constellation / eye / histogram
  const IDEAL = {
    BPSK: [[1, 0], [-1, 0]],
    QPSK: [0, 1, 2, 3].map((k) => [Math.cos(Math.PI / 4 + (k * Math.PI) / 2), Math.sin(Math.PI / 4 + (k * Math.PI) / 2)]),
    "8PSK": [0, 1, 2, 3, 4, 5, 6, 7].map((k) => [Math.cos((k * Math.PI) / 4), Math.sin((k * Math.PI) / 4)]),
    "16QAM": [-3, -1, 1, 3].flatMap((a) => [-3, -1, 1, 3].map((b) => [a / Math.sqrt(10), b / Math.sqrt(10)])),
  };
  function constellation(canvas, pts, mod) {
    const { ctx, W, H, C } = prep(canvas);
    const S = Math.min(W, H), cx = W / 2, cy = H / 2, scale = (S / 2 - 22) / 1.6;
    ctx.strokeStyle = C.line; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(8, cy + .5); ctx.lineTo(W - 8, cy + .5); ctx.moveTo(cx + .5, 8); ctx.lineTo(cx + .5, H - 8); ctx.stroke();
    ctx.fillStyle = rgba(C.accent, 0.55); pts.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(cx + x * scale, cy - y * scale, 2, 0, 6.3); ctx.fill(); });
    const ideal = IDEAL[mod];
    if (ideal) {      // phase is only known up to a fixed rotation: draw ideal points at the best-matching rotation
      const M = mod === "BPSK" ? 2 : mod === "8PSK" ? 8 : 4; let bestRot = 0, bestD = Infinity;
      for (let r = 0; r < M; r++) {
        const a = (2 * Math.PI * r) / M, ca = Math.cos(a), sa = Math.sin(a); let d = 0;
        for (let i = 0; i < pts.length; i += 7) { let m = Infinity; for (const [ix, iy] of ideal) { const rx = ix * ca - iy * sa, ry = ix * sa + iy * ca; m = Math.min(m, (pts[i][0] - rx) ** 2 + (pts[i][1] - ry) ** 2); } d += m; }
        if (d < bestD) { bestD = d; bestRot = a; }
      }
      const ca = Math.cos(bestRot), sa = Math.sin(bestRot); ctx.strokeStyle = C.text; ctx.lineWidth = 1.4;
      ideal.forEach(([ix, iy]) => { ctx.beginPath(); ctx.arc(cx + (ix * ca - iy * sa) * scale, cy - (ix * sa + iy * ca) * scale, 7, 0, 6.3); ctx.stroke(); });
    }
    ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.fillText("I", W - 18, cy - 7); ctx.fillText("Q", cx + 7, 18);
    ctx.textAlign = "right"; ctx.fillText("circles = ideal points", W - 10, H - 10);
  }
  function eye(canvas, traces) {
    const { ctx, W, H, C } = prep(canvas); const n = traces[0].length, L = 8, R = 8, T = 10, B = 24;
    const X = (i) => L + (i / (n - 1)) * (W - L - R), Y = (v) => T + ((1 - v) / 2) * (H - T - B);
    ctx.strokeStyle = C.line; ctx.beginPath(); ctx.moveTo(L, Y(0) + .5); ctx.lineTo(W - R, Y(0) + .5); ctx.moveTo(X((n - 1) / 2) + .5, T); ctx.lineTo(X((n - 1) / 2) + .5, H - B); ctx.stroke();
    ctx.strokeStyle = rgba(C.text, 0.2); ctx.lineWidth = 1;
    traces.forEach((tr) => { ctx.beginPath(); tr.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.stroke(); });
    ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.fillText("-1 symbol", X(0) + 30, H - 7); ctx.fillText("sampling instant", X((n - 1) / 2), H - 7); ctx.fillText("+1 symbol", X(n - 1) - 30, H - 7);
  }
  function histogram(canvas, h, tones) {
    const { ctx, W, H, C } = prep(canvas); const L = 10, R = 10, T = 14, B = 28, e = h.edges, c = h.counts, cmax = Math.max(...c), xmin = e[0], xmax = e[e.length - 1];
    const X = (v) => L + ((v - xmin) / (xmax - xmin)) * (W - L - R);
    ctx.fillStyle = rgba(C.accent, 0.8); c.forEach((v, i) => { const x0 = X(e[i]), x1 = X(e[i + 1]), hh = (v / cmax) * (H - T - B); ctx.fillRect(x0 + 1, H - B - hh, Math.max(x1 - x0 - 2, 1), hh); });
    ctx.strokeStyle = C.text; ctx.setLineDash([5, 4]); tones.forEach((t) => { ctx.beginPath(); ctx.moveTo(X(t) + .5, T); ctx.lineTo(X(t) + .5, H - B); ctx.stroke(); });
    ctx.setLineDash([]); ctx.fillStyle = C.muted; ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) { const v = xmin + ((xmax - xmin) * i) / 4; ctx.fillText(fmtHz(v).replace(" ", ""), X(v), H - 9); }
  }

  // ------------------------------------------------------------------ frame structure
  function frameColumns(canvas, f) {
    const { ctx, W, H, C } = prep(canvas); const L = 34, R = 10, T = 26, B = 24, w = W - L - R, h = H - T - B, n = f.frame_len;
    const X = (c) => L + (c / n) * w, Y = (v) => T + (1 - v) * h;
    const fill = (a, b, al) => { ctx.fillStyle = rgba(C.accent, al); ctx.fillRect(X(a), T, Math.max(X(b) - X(a), 2), h); };
    fill(0, f.sync_len, 0.3); if (f.header_bits) fill(f.sync_len, f.sync_len + f.header_bits, 0.13);
    ctx.fillStyle = C.muted; ctx.textAlign = "left";
    ctx.fillText(`sync ${f.sync_len} | header ${f.header_bits} | payload ${f.payload_bits} bits`, L, T - 9);
    ctx.textAlign = "right"; ctx.fillText("frame " + n + " bits", W - R, T - 9);
    ctx.strokeStyle = C.line; [0, 0.5, 1].forEach((v) => { ctx.beginPath(); ctx.moveTo(L, Y(v) + .5); ctx.lineTo(W - R, Y(v) + .5); ctx.stroke(); ctx.fillStyle = C.muted; ctx.textAlign = "right"; ctx.fillText(v.toFixed(1), L - 6, Y(v) + 4); });
    const draw = (arr, color) => { ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.beginPath(); arr.forEach((v, i) => { const x = X(i * f.columns.step), y = Y(Math.min(Math.max(v, 0), 1)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); };
    draw(f.columns.pred.map((v) => Math.max(0, (v - 0.5) * 2)), C.accent); draw(f.columns.bias, C.text);
    ctx.fillStyle = C.muted; ctx.textAlign = "center"; ctx.fillText("bit position within frame", L + w / 2, H - 6);
  }

  // ------------------------------------------------------------------ accuracy by SNR
  function accuracyCurve(canvas, rows, label) {
    const { ctx, W, H, C } = prep(canvas); const L = 44, R = 12, T = 12, B = 30;
    const xs = rows.map((r) => r.snr), xmin = Math.min(...xs), xmax = Math.max(...xs);
    const X = (v) => L + ((v - xmin) / (xmax - xmin || 1)) * (W - L - R), Y = (v) => T + (1 - v) * (H - T - B);
    ctx.strokeStyle = C.line; ctx.fillStyle = C.muted; ctx.lineWidth = 1;
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => { ctx.beginPath(); ctx.moveTo(L, Y(v) + .5); ctx.lineTo(W - R, Y(v) + .5); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText((v * 100) + "%", L - 6, Y(v) + 4); });
    const step = Math.ceil(rows.length / 8); ctx.textAlign = "center";
    rows.forEach((r, i) => { if (i % step === 0) ctx.fillText(r.snr, X(r.snr), H - 12); });
    ctx.fillText("SNR (dB)", L + (W - L - R) / 2, H - 1);
    ctx.strokeStyle = C.accent; ctx.lineWidth = 2; ctx.beginPath(); rows.forEach((r, i) => (i ? ctx.lineTo(X(r.snr), Y(r.acc)) : ctx.moveTo(X(r.snr), Y(r.acc)))); ctx.stroke();
    ctx.fillStyle = C.accent; rows.forEach((r) => { ctx.beginPath(); ctx.arc(X(r.snr), Y(r.acc), 3, 0, 6.3); ctx.fill(); });
    ctx.fillStyle = C.muted; ctx.textAlign = "left"; ctx.fillText(label || "", L + 8, Y(0.08));
  }

  return { fmtHz, fmtDb, empty, psd, rateSpectrum, waterfall, constellation, eye, histogram, frameColumns, accuracyCurve };
})();
