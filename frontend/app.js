const $ = (id) => document.getElementById(id);
const statusEl = $("status");

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.className = "status" + (isError ? " error" : "");
}

async function postAndRender(promise) {
  setStatus("Analyzing...");
  try {
    const res = await promise;
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    render(data);
    setStatus("Done.");
  } catch (e) {
    setStatus(e.message, true);
  }
}

$("uploadForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append("file", $("file").files[0]);
  fd.append("fmt", $("fmt").value);
  if ($("fs").value) fd.append("fs", $("fs").value);
  postAndRender(fetch("/api/analyze", { method: "POST", body: fd }));
});

$("demoBtn").addEventListener("click", () => {
  const body = { mod: $("dMod").value, snr_db: +$("dSnr").value, cfo_hz: +$("dCfo").value };
  postAndRender(fetch("/api/demo", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }));
});

function fmtHz(v) {
  if (v === null || v === undefined) return "-";
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(3) + " MHz";
  if (a >= 1e3) return (v / 1e3).toFixed(2) + " kHz";
  return v.toFixed(1) + " Hz";
}
const fmtDb = (v) => (v === null || v === undefined ? "-" : v.toFixed(1) + " dB");

function render(d) {
  const p = d.params, gt = d.ground_truth;
  const rows = [
    ["Sampling rate", fmtHz(d.fs), d.source.kind === "raw" ? "user supplied" : ""],
    ["Samples / duration", `${d.n_samples} / ${(d.duration_s * 1000).toFixed(1)} ms`, ""],
    ["Centre frequency offset", fmtHz(p.center_freq_hz), gt ? "truth " + fmtHz(gt.cfo_hz) : ""],
    ["Occupied bandwidth (99%)", fmtHz(p.bandwidth_hz), gt && gt.mod.indexOf("FSK") < 0 ? "truth ~" + fmtHz(gt.symbol_rate * (1 + gt.beta)) : ""],
    ["SNR (full band)", fmtDb(p.snr_db_fullband), gt ? "truth " + gt.snr_db_fullband + " dB" : ""],
    ["SNR (in-band)", fmtDb(p.snr_db_inband), ""],
    ["Noise floor", fmtDb(p.noise_floor_db), "PSD level"],
    ["DC offset", p.dc_offset.toFixed(4), ""],
  ];
  $("params").querySelector("tbody").innerHTML = rows
    .map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join("");
  drawPSD(d);
  drawWaterfall(d);
}

function drawPSD(d) {
  const c = $("psd"), g = c.getContext("2d"), W = c.width, H = c.height;
  const L = 60, R = 10, T = 10, B = 30;
  g.clearRect(0, 0, W, H);
  const f = d.psd.freq, y = d.psd.db;
  const fmin = f[0], fmax = f[f.length - 1];
  const ymax = Math.max(...y) + 3, ymin = Math.min(...y) - 3;
  const X = (v) => L + ((v - fmin) / (fmax - fmin)) * (W - L - R);
  const Y = (v) => T + ((ymax - v) / (ymax - ymin)) * (H - T - B);

  g.strokeStyle = "#26314d"; g.fillStyle = "#8fa0bf"; g.font = "12px sans-serif"; g.lineWidth = 1;
  for (let i = 0; i <= 5; i++) {
    const v = ymin + ((ymax - ymin) * i) / 5;
    g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(W - R, Y(v)); g.stroke();
    g.fillText(v.toFixed(0), 8, Y(v) + 4);
  }
  for (let i = 0; i <= 6; i++) {
    const v = fmin + ((fmax - fmin) * i) / 6;
    g.fillText(fmtHz(v), X(v) - 22, H - 8);
  }
  const p = d.params;
  if (p.band_low_hz !== undefined) {           // shade the detected band
    g.fillStyle = "rgba(62,166,255,0.12)";
    g.fillRect(X(p.band_low_hz), T, X(p.band_high_hz) - X(p.band_low_hz), H - T - B);
  }
  g.strokeStyle = "#3ea6ff"; g.lineWidth = 1.5; g.beginPath();
  f.forEach((v, i) => (i ? g.lineTo(X(v), Y(y[i])) : g.moveTo(X(v), Y(y[i]))));
  g.stroke();
  g.strokeStyle = "#ffb347"; g.setLineDash([6, 4]); g.beginPath();   // noise floor
  g.moveTo(L, Y(p.noise_floor_db)); g.lineTo(W - R, Y(p.noise_floor_db)); g.stroke(); g.setLineDash([]);
  g.fillStyle = "#8fa0bf"; g.fillText("PSD (dB/Hz)", L + 6, T + 12);
}

function colormap(t) {            // 0..1 -> RGB (dark blue -> cyan -> yellow -> red)
  t = Math.max(0, Math.min(1, t));
  const stops = [[5, 10, 50], [20, 80, 160], [40, 200, 200], [250, 230, 60], [240, 70, 40]];
  const s = t * (stops.length - 1), i = Math.min(Math.floor(s), stops.length - 2), u = s - i;
  return stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * u));
}

function drawWaterfall(d) {
  const w = d.waterfall, c = $("wf"), g = c.getContext("2d");
  const off = document.createElement("canvas");
  off.width = w.cols; off.height = w.rows;
  const og = off.getContext("2d"), img = og.createImageData(w.cols, w.rows);
  const flat = w.db.flat();
  const sorted = [...flat].sort((a, b) => a - b);
  const lo = sorted[Math.floor(sorted.length * 0.02)], hi = sorted[Math.floor(sorted.length * 0.995)];
  flat.forEach((v, i) => {
    const [r, gg, b] = colormap((v - lo) / (hi - lo));
    img.data[i * 4] = r; img.data[i * 4 + 1] = gg; img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
  });
  og.putImageData(img, 0, 0);
  const L = 60, B = 26, pw = c.width - L - 10, ph = c.height - B - 6;
  g.clearRect(0, 0, c.width, c.height);
  g.imageSmoothingEnabled = false;
  g.drawImage(off, L, 6, pw, ph);
  g.fillStyle = "#8fa0bf"; g.font = "12px sans-serif";
  g.fillText("0 s", 12, 16); g.fillText((w.t_end * 1000).toFixed(0) + " ms", 4, ph);
  const fmin = w.freq[0], fmax = w.freq[w.freq.length - 1];
  for (let i = 0; i <= 6; i++) g.fillText(fmtHz(fmin + ((fmax - fmin) * i) / 6), L + (pw * i) / 6 - 22, c.height - 8);
}
