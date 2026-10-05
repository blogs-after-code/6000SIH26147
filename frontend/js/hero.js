// Landing-page illustration: a scrolling waterfall of a noisy narrow-band signal and a constellation that tightens
// and loosens as the "SNR" changes. Purely decorative; pauses when hidden and respects reduced motion.
(function () {
  const c = document.getElementById("heroCanvas");
  if (!c) return;
  const ctx = c.getContext("2d");
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const pts = Array.from({ length: 360 }, () => ({ s: Math.floor(Math.random() * 4), a: gauss(), b: gauss() }));
  const rows = [];
  let W = 0, H = 0, frame = 0, last = 0;
  function size() {
    const d = window.devicePixelRatio || 1;
    W = c.clientWidth; H = c.clientHeight;
    c.width = W * d; c.height = H * d; ctx.setTransform(d, 0, 0, d, 0, 0);
  }
  size(); addEventListener("resize", size);
  const lerp = (a, b, t) => a + (b - a) * t;
  function rgb(hex) { const n = parseInt(hex.replace("#", ""), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }

  function spectrumRow(snr) {
    const n = 96, row = new Array(n);
    for (let i = 0; i < n; i++) {
      const x = (i - n / 2) / (n * 0.09);
      const sig = Math.abs(x) < 1.6 ? 1 - 0.18 * x * x : 0;
      row[i] = Math.max(0, 0.18 + 0.12 * Math.abs(gauss()) + sig * (0.35 + snr * 0.5));
    }
    return row;
  }
  function draw(t) {
    if (document.hidden) { requestAnimationFrame(draw); return; }
    if (t - last < 40 && !reduce) { requestAnimationFrame(draw); return; }
    last = t; frame++;
    const accent = css("--accent-text") || css("--accent"), line = css("--line"), bg = css("--bg"), muted = css("--muted");
    const snr = reduce ? 0.6 : 0.5 + 0.5 * Math.sin(t / 2600);
    rows.unshift(spectrumRow(snr)); if (rows.length > 56) rows.pop();
    ctx.clearRect(0, 0, W, H);
    // waterfall (left)
    const wx = 14, wy = 14, ww = W * 0.52 - 20, wh = H - 28, [r, g, b] = rgb(accent), [br, bgc, bb] = rgb(bg);
    ctx.fillStyle = bg; ctx.fillRect(wx, wy, ww, wh);
    const rh = wh / 56;
    rows.forEach((row, ri) => {
      const cw = ww / row.length;
      row.forEach((v, ci) => {
        const k = Math.min(0.85, v * 0.9);
        ctx.fillStyle = `rgb(${lerp(br, r, k) | 0},${lerp(bgc, g, k) | 0},${lerp(bb, b, k) | 0})`;
        ctx.fillRect(wx + ci * cw, wy + ri * rh, cw + 0.6, rh + 0.6);
      });
    });
    ctx.strokeStyle = line; ctx.strokeRect(wx + .5, wy + .5, ww, wh);
    ctx.fillStyle = muted; ctx.font = "11px 'JetBrains Mono', monospace"; ctx.fillText("waterfall", wx + 8, wy + wh - 8);
    // constellation (right)
    const cx = W * 0.52 + (W * 0.48) / 2, cy = H / 2, R = Math.min(W * 0.48, H) * 0.3, sigma = 0.07 + (1 - snr) * 0.34;
    ctx.strokeStyle = line; ctx.beginPath(); ctx.moveTo(cx - R * 1.5, cy + .5); ctx.lineTo(cx + R * 1.5, cy + .5); ctx.moveTo(cx + .5, cy - R * 1.5); ctx.lineTo(cx + .5, cy + R * 1.5); ctx.stroke();
    const rot = reduce ? 0 : Math.sin(t / 3100) * 0.18 * (1 - snr);
    const cs = Math.cos(rot), sn = Math.sin(rot);
    pts.forEach((p) => {
      const bx = (p.s & 1 ? 1 : -1) * 0.7071 + p.a * sigma * 0.7, by = (p.s & 2 ? 1 : -1) * 0.7071 + p.b * sigma * 0.7;
      ctx.fillStyle = accent; ctx.globalAlpha = 0.7;
      ctx.beginPath(); ctx.arc(cx + (bx * cs - by * sn) * R * 1.3, cy - (bx * sn + by * cs) * R * 1.3, 2, 0, 6.3); ctx.fill();
    });
    ctx.globalAlpha = 1; ctx.fillStyle = muted;
    ctx.fillText("SNR " + Math.round(8 + snr * 18) + " dB", cx - R * 1.5, cy - R * 1.65);
    if (!reduce) requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();
