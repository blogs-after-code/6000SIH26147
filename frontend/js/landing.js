// Scroll reveal
const io = new IntersectionObserver((entries) => {
  entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
}, { threshold: 0.12 });
document.querySelectorAll(".reveal").forEach((el, i) => {
  el.style.transitionDelay = (i % 4) * 70 + "ms";
  io.observe(el);
});

// Live accuracy figure from the server's held-out evaluation
fetch("/api/model-report").then((r) => (r.ok ? r.json() : Promise.reject()))
  .then((d) => { document.getElementById("statAcc").textContent = Math.round(d.accuracy * 100) + "%"; })
  .catch(() => { document.getElementById("statAcc").textContent = "n/a"; });

// Hero illustration: QPSK symbols, noise shrinks and grows so the constellation "resolves"
(function hero() {
  const c = document.getElementById("heroCanvas");
  const ctx = c.getContext("2d");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const N = 520;
  const pts = Array.from({ length: N }, () => ({
    s: Math.floor(Math.random() * 4),
    gx: gauss(), gy: gauss(),
  }));
  function gauss() { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function size() {
    const dpr = window.devicePixelRatio || 1;
    c.width = c.clientWidth * dpr; c.height = c.clientHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  size(); window.addEventListener("resize", size);

  function draw(t) {
    const W = c.clientWidth, H = c.clientHeight;
    ctx.clearRect(0, 0, W, H);
    const cx = W / 2, cy = H * 0.42, R = Math.min(W, H) * 0.3;
    // axes
    ctx.strokeStyle = "rgba(147,162,196,.18)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(10, cy); ctx.lineTo(W - 10, cy); ctx.moveTo(cx, 10); ctx.lineTo(cx, H * 0.84); ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, cy, R * 1.41, 0, Math.PI * 2); ctx.stroke();
    const phase = reduce ? 0.9 : 0.5 + 0.5 * Math.cos(t * 0.0007);       // 1 = noisy, 0 = clean
    const sigma = 0.06 + 0.42 * phase;
    const rot = reduce ? 0 : Math.sin(t * 0.0004) * 0.25 * phase;        // carrier wobble when noisy
    const cs = Math.cos(rot), sn = Math.sin(rot);
    for (const p of pts) {
      const bx = (p.s & 1 ? 1 : -1) * 0.7071, by = (p.s & 2 ? 1 : -1) * 0.7071;
      let x = bx + p.gx * sigma * 0.7, y = by + p.gy * sigma * 0.7;
      const xr = x * cs - y * sn, yr = x * sn + y * cs;
      ctx.fillStyle = p.s % 2 ? "rgba(76,194,255,.75)" : "rgba(139,123,255,.75)";
      ctx.beginPath(); ctx.arc(cx + xr * R * 1.3, cy - yr * R * 1.3, 2.2, 0, Math.PI * 2); ctx.fill();
    }
    // scrolling waveform strip
    const wy = H * 0.93, amp = H * 0.045;
    ctx.strokeStyle = "rgba(76,194,255,.8)"; ctx.lineWidth = 1.6; ctx.beginPath();
    for (let x = 0; x < W; x += 3) {
      const tt = x * 0.045 + t * 0.004;
      const clean = Math.sin(tt) * (Math.sin(tt * 0.23) > 0 ? 1 : -1);
      const noise = Math.sin(tt * 7.3) * 0.6 + Math.sin(tt * 13.1) * 0.4;
      const y = wy - amp * (clean * (1 - 0.5 * phase) + noise * phase * 0.9);
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = "rgba(147,162,196,.8)"; ctx.font = "12px system-ui, sans-serif";
    ctx.fillText("SNR " + Math.round(26 - 20 * phase) + " dB", W - 78, cy - R * 1.5);
    if (!reduce) requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
})();
