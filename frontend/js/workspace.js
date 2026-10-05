const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- pipeline step tracker
const STEPS = [
  "Ingest file", "Spectrum analysis", "Symbol rate & timing", "Carrier & modulation", "Demodulation",
  "De-interleaving", "FEC decoding", "Bitstream correlation",
];
const LOCKED_FROM = 5;
const stepEls = [];
STEPS.forEach((name, i) => {
  const li = document.createElement("li");
  li.innerHTML = `<span class="dot"></span><span>${name}</span>` + (i >= LOCKED_FROM ? `<span class="phase">Phase 3</span>` : "");
  if (i >= LOCKED_FROM) li.classList.add("locked");
  $("stepList").appendChild(li);
  stepEls.push(li);
});
function setSteps(done, active, error) {
  stepEls.forEach((li, i) => {
    if (i >= LOCKED_FROM) return;
    li.classList.remove("done", "active", "error");
    if (i < done) li.classList.add("done");
    else if (i === active) li.classList.add(error ? "error" : "active");
  });
}
setSteps(0, -1);

// ---------------------------------------------------------------- state
let session = null, lastFile = null, lastSpec = null, lastDemod = null;

function status(msg, kind) {
  const s = $("status");
  s.className = "status" + (msg ? " show " + (kind || "busy") : "");
  s.innerHTML = kind === "err" ? msg : msg ? `<span class="spinner"></span>${msg}` : "";
}
const esc = (t) => String(t).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

// ---------------------------------------------------------------- drag & drop
const dz = $("dropzone"), input = $("fileInput");
dz.addEventListener("click", () => input.click());
dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0]); });
input.addEventListener("change", () => { if (input.files.length) pickFile(input.files[0]); });

const EXT_FMT = { ".cf32": "cf32", ".cfile": "cf32", ".cs16": "ci16", ".cs8": "ci8", ".cu8": "cu8" };
function pickFile(f) {
  lastFile = f;
  const ext = (f.name.match(/\.[^.]+$/) || [""])[0].toLowerCase();
  $("fileName").textContent = f.name;
  $("fileSize").textContent = (f.size / 1024 / 1024 >= 1 ? (f.size / 1048576).toFixed(2) + " MB" : (f.size / 1024).toFixed(1) + " KB");
  $("fileInfo").classList.add("show");
  const raw = ext !== ".wav";
  $("rawFields").classList.toggle("show", raw);
  if (EXT_FMT[ext]) $("fmt").value = EXT_FMT[ext];
  status("");
}

$("startBtn").addEventListener("click", () => {
  if (!lastFile) return;
  const raw = $("rawFields").classList.contains("show");
  if (raw && !$("fs").value) { status("Enter the sampling rate: raw IQ files have no header to read it from.", "err"); return; }
  run(() => API.upload(lastFile, $("fmt").value, raw ? $("fs").value : null));
});
$("demoBtn").addEventListener("click", () => run(() => API.demo({
  mod: $("dMod").value, snr_db: +$("dSnr").value, cfo_hz: +$("dCfo").value, beta: +$("dBeta").value,
})));
$("rerunBtn").addEventListener("click", () => rerun());

// ---------------------------------------------------------------- orchestration
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function run(loader) {
  $("results").classList.remove("show");
  try {
    setSteps(0, 0); status("Loading and normalizing the file...");
    const d = await loader();
    session = d.session;
    setSteps(1, 1); status("Analyzing the spectrum...");
    await sleep(250);
    renderSpectrum(d);
    $("results").classList.add("show");
    await demodulate({});
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) {
    setSteps(stepEls.findIndex((l) => l.classList.contains("active")), stepEls.findIndex((l) => l.classList.contains("active")), true);
    status(esc(e.message), "err");
  }
}

async function demodulate(over) {
  setSteps(2, 2); status("Estimating symbol rate and recovering timing...");
  const p = Promise.resolve(API.demod(session, over));
  await sleep(300); setSteps(3, 3); status("Recovering carrier and classifying the modulation...");
  const r = await p;
  await sleep(200); setSteps(4, 4); status("Demodulating...");
  await sleep(200);
  renderDemod(r);
  setSteps(5, -1);
  status("");
}

async function rerun() {
  const over = { beta: +$("oBeta").value || 0.35 };
  if ($("oMod").value) over.mod = $("oMod").value;
  if ($("oRate").value) over.symbol_rate = +$("oRate").value;
  try { await demodulate(over); } catch (e) { status(esc(e.message), "err"); }
}

// ---------------------------------------------------------------- rendering
function tile(k, v, s, cls) { return `<div class="tile ${cls || ""}"><div class="k">${k}</div><div class="v">${v}</div>${s ? `<div class="s">${s}</div>` : ""}</div>`; }

function renderSpectrum(d) {
  lastSpec = d;
  const p = d.params, gt = d.ground_truth;
  $("summary").innerHTML =
    tile("File", esc(d.source.filename || "-"), d.source.kind === "raw" ? "raw IQ: " + d.source.fmt : d.source.kind) +
    tile("Sampling rate", Plots.fmtHz(d.fs), d.source.kind === "raw" ? "supplied by you" : "from header") +
    tile("Duration", (d.duration_s * 1000).toFixed(1) + " ms", d.n_samples.toLocaleString() + " samples") +
    tile("Centre offset", Plots.fmtHz(p.center_freq_hz), gt ? "truth " + Plots.fmtHz(gt.cfo_hz) : "") +
    tile("Bandwidth (99%)", Plots.fmtHz(p.bandwidth_hz), "") +
    tile("SNR (in-band)", Plots.fmtDb(p.snr_db_inband), gt ? "truth " + gt.snr_db_fullband + " dB full-band" : "");
  Plots.psd($("cPsd"), d);
  Plots.waterfall($("cWf"), d.waterfall);
}

function renderDemod(r) {
  lastDemod = r;
  // warnings
  $("warns").innerHTML = (r.warnings || []).map((w) => `<div class="warn">&#9888; ${esc(w)}</div>`).join("");

  const isFsk = r.family === "fsk";
  $("cardConst").style.display = isFsk ? "none" : "";
  $("cardEye").style.display = isFsk ? "none" : "";
  $("cardHist").style.display = isFsk ? "" : "none";

  // modulation card
  $("modName").textContent = r.modulation;
  $("modSrc").textContent = "via " + r.modulation_source;
  if (r.class_probs && Object.keys(r.class_probs).length > 1) {
    $("probs").innerHTML = Object.entries(r.class_probs).sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `<div class="prob"><span>${k}</span><div class="bar"><i style="width:${(v * 100).toFixed(0)}%"></i></div><span>${(v * 100).toFixed(0)}%</span></div>`).join("");
  } else if (isFsk) {
    const e = r.family_evidence || {};
    $("probs").innerHTML = `<p class="cap">Constant-envelope signal with separable tones: classified as FSK. Tone count comes from clustering the measured frequencies.</p>`;
  } else {
    $("probs").innerHTML = `<p class="cap">Modulation set manually.</p>`;
  }
  const mk = [];
  if (r.features && Object.keys(r.features).length) Object.entries(r.features).forEach(([k, v]) => mk.push([k, v]));
  if (isFsk) { mk.push(["tone spacing", Plots.fmtHz(r.fsk.tone_spacing_hz)], ["modulation index h", r.fsk.mod_index_h]); }
  $("modKv").innerHTML = mk.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("");

  // symbol rate card
  Plots.rateSpectrum($("cRate"), r.plots.rate_spectrum, r.symbol_rate);
  const rk = [["Symbol rate", Plots.fmtHz(r.symbol_rate)], ["Samples / symbol", r.sps.toFixed(2)], ["Line strength", r.symbol_rate_pmr + "x"]];
  if (r.carrier) rk.push(["Residual carrier offset", Plots.fmtHz(r.carrier.residual_cfo_hz)]);
  $("rateKv").innerHTML = rk.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("");

  // quality
  const q = r.quality || {};
  let qh = tile("Symbols", (q.n_symbols || 0).toLocaleString());
  if (q.esn0_db !== undefined) {
    qh += tile("Symbol SNR", q.esn0_db + " dB", "from EVM", q.esn0_db >= 12 ? "good" : q.esn0_db < 8 ? "bad" : "");
    qh += tile("EVM", (q.evm * 100).toFixed(1) + " %", "lower is better");
    qh += tile("Timing lock", (q.timing_lock * 100).toFixed(0) + " %", "", q.timing_lock > 0.8 ? "good" : "");
  }
  $("quality").innerHTML = qh;
  const g = r.ground_truth_check;
  $("truthBox").innerHTML = g ? `<p class="cap" style="margin-top:14px"><b>Ground-truth check:</b> ${g.ber === null ? esc(g.note) :
    `bit error rate <b style="color:${g.ber < 0.01 ? "var(--good)" : "var(--warn)"}">${(g.ber * 100).toFixed(3)} %</b> against the generator's true bits.`}</p>` : "";

  // plots
  if (isFsk) {
    Plots.histogram($("cHist"), r.fsk.hist, r.fsk.tones_hz);
  } else {
    Plots.constellation($("cConst"), r.plots.constellation, r.modulation);
    Plots.eye($("cEye"), r.plots.eye);
  }

  // bits
  $("bitsSub").textContent = `${r.bits.count.toLocaleString()} bits recovered. First 512 shown.`;
  $("bitsBox").textContent = r.bits.preview.replace(/(.{8})/g, "$1 ");
  $("bitsNote").textContent = r.bits.note;
  $("dlTxt").href = `/api/bits/${session}?fmt=txt`;
  $("dlBin").href = `/api/bits/${session}?fmt=bin`;
}

// ---------------------------------------------------------------- classifier reliability panel
API.report().then((m) => {
  $("modelBars").innerHTML = `<div class="prob"><span><b>Overall</b></span><div class="bar"><i style="width:${m.accuracy * 100}%"></i></div><span>${(m.accuracy * 100).toFixed(1)}%</span></div>` +
    m.accuracy_by_esn0_db.filter((b) => b.accuracy !== null).map((b) =>
      `<div class="prob"><span>${b.esn0_range[0]}-${b.esn0_range[1]} dB</span><div class="bar"><i style="width:${b.accuracy * 100}%"></i></div><span>${(b.accuracy * 100).toFixed(0)}%</span></div>`).join("");
  $("modelNote").textContent = `Symbol SNR bands (Es/N0). ${m.n_test} held-out synthetic signals. ${m.note}`;
}).catch(() => { $("modelBars").innerHTML = `<p class="cap">No evaluation report found. Run scripts/train_classifier.py.</p>`; });

// Re-draw the plots when the window is resized
let rt;
window.addEventListener("resize", () => {
  clearTimeout(rt);
  rt = setTimeout(() => { if (lastSpec) renderSpectrum(lastSpec); if (lastDemod) renderDemod(lastDemod); }, 200);
});
