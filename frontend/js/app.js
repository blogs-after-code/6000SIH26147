// SanketSetu front end: routing, theme, upload and analysis flow, rendering of results.
const $ = (id) => document.getElementById(id);
const esc = (t) => String(t === null || t === undefined ? "" : t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fmtHz = Plots.fmtHz, fmtDb = Plots.fmtDb;

function toast(msg) {
  const t = document.createElement("div"); t.className = "toast"; t.textContent = msg; document.body.appendChild(t);
  setTimeout(() => t.remove(), 1800);
}
async function copyText(text, what) {
  try { await navigator.clipboard.writeText(text); toast((what || "Text") + " copied"); }
  catch (e) { toast("Copy is blocked by the browser"); }
}

// ------------------------------------------------------------------ theme
(function initTheme() {
  const saved = localStorage.getItem("sanketsetu-theme");
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  $("themeBtn").onclick = () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("sanketsetu-theme", next); } catch (e) { /* private mode */ }
    redrawAll();
  };
})();

// ------------------------------------------------------------------ router
function route() {
  const h = (location.hash || "#/").replace("#/", "");
  const inApp = h === "analyze";
  $("view-landing").classList.toggle("hidden", inApp);
  $("view-app").classList.toggle("hidden", !inApp);
  document.querySelectorAll("[data-nav]").forEach((a) => a.classList.toggle("active", a.dataset.nav === (inApp ? "analyze" : h || "home")));
  if (inApp) { window.scrollTo(0, 0); redrawAll(); }
  else if (["coverage", "report", "how", "limits", "problem", "evidence"].includes(h)) setTimeout(() => { const el = $(h); if (el) el.scrollIntoView({ behavior: "smooth" }); }, 30);
  else window.scrollTo(0, 0);
}
addEventListener("hashchange", route);

// ------------------------------------------------------------------ backend status + landing metrics
let MODELS = null;
(async function status() {
  const el = $("apiStatus");
  let h = null;
  // A free hosting plan sleeps when idle and needs up to about a minute to wake: keep trying before declaring it offline.
  for (let i = 0; i < 12 && !h; i++) {
    try { h = await API.health(); }
    catch (e) {
      el.innerHTML = `<span class="dot"></span><span>${i === 0 ? "connecting to server" : "server waking up, please wait"}</span>`;
      await new Promise((r) => setTimeout(r, 5000));
    }
  }
  if (h) {
    el.innerHTML = `<span class="dot ok"></span><span>server ready</span>`;
    el.title = `Feature classifier: ${h.classifier_ready ? "loaded" : "missing"}. CNN (public data): ${h.cnn_ready ? "loaded" : "not installed"}.`;
  } else { el.innerHTML = `<span class="dot bad"></span><span>server offline</span>`; }
  try {
    MODELS = await API.models();
    if (MODELS.rf) $("mRf").textContent = Math.round(MODELS.rf.accuracy * 100) + "%";
    if (MODELS.cnn_available && MODELS.cnn) {
      $("mCnn").textContent = Math.round((MODELS.cnn.test_accuracy_ge10dB ?? MODELS.cnn.test_accuracy) * 100) + "%";
      $("mCnnNote").textContent = `${MODELS.cnn.dataset}, test frames at 10 dB SNR and above. Simulated data.`;
    } else { $("mCnn").textContent = "not trained"; $("mCnnNote").textContent = "Training notebook for Kaggle is included; see the Models tab for the steps."; }
  } catch (e) { /* landing still works */ }
})();

// ------------------------------------------------------------------ state
const S = { session: null, spec: null, demod: null, decode: null, label: "", file: null, tab: "overview", busy: false, times: {} };

// ------------------------------------------------------------------ pipeline rail
const STEPS = [["Ingest .IQ / .wav", ""], ["Signal parameters", ""], ["Symbol timing", ""], ["Demodulate", ""], ["De-interleave", ""], ["FEC decode", ""], ["Bit-stream correlation", ""]];
function renderRail(states, subs) {
  $("railSteps").innerHTML = STEPS.map(([n], i) => {
    const st = (states && states[i]) || "later";
    const ic = st === "done" ? "&#10003;" : st === "fail" ? "!" : st === "skip" ? "&ndash;" : "";
    const sub = (subs && subs[i]) || "", tm = (S.times && S.times[i]) ? S.times[i] : "";
    return `<li class="${st}"><span class="ic">${ic}</span><span>${n}${sub ? `<span class="sub">${esc(sub)}</span>` : ""}</span><span class="tm">${tm}</span></li>`;
  }).join("");
}
renderRail([]);

// ------------------------------------------------------------------ alerts and progress
let alerts = [];
function setAlerts(list) { alerts = list; drawAlerts(); }
function drawAlerts() {
  $("alerts").innerHTML = alerts.map((a, i) => `<div class="alert ${a.level}"><span class="ai">${a.level === "err" ? "ERROR" : a.level === "warn" ? "CHECK" : "NOTE"}</span><div class="grow">${a.title ? `<b>${esc(a.title)}</b> ` : ""}${esc(a.text)}</div><button type="button" data-dismiss="${i}" aria-label="Dismiss">&times;</button></div>`).join("");
}
$("alerts").addEventListener("click", (e) => { const b = e.target.closest("[data-dismiss]"); if (b) { alerts.splice(+b.dataset.dismiss, 1); drawAlerts(); } });
let ticker = null, t0 = 0;
function progress(text) {
  const p = $("progress");
  if (!text) { p.classList.add("hidden"); clearInterval(ticker); return; }
  $("progressText").textContent = text; p.classList.remove("hidden");
  if (!ticker) { t0 = Date.now(); }
  clearInterval(ticker); ticker = setInterval(() => { $("progressTime").textContent = Math.round((Date.now() - t0) / 1000) + "s"; }, 500);
}
function endProgress() { $("progress").classList.add("hidden"); clearInterval(ticker); ticker = null; }

// ------------------------------------------------------------------ tabs
function selectTab(name) {
  S.tab = name;
  document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t.dataset.tab === name)));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("on", p.id === "p-" + name));
  drawTab(name);
}
$("tabs").addEventListener("click", (e) => { const t = e.target.closest(".tab"); if (t) selectTab(t.dataset.tab); });
$("tabs").addEventListener("keydown", (e) => {
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  const tabs = [...document.querySelectorAll(".tab")], i = tabs.findIndex((t) => t.dataset.tab === S.tab);
  const n = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length]; selectTab(n.dataset.tab); n.focus();
});

// ------------------------------------------------------------------ input handling
const drop = $("drop"), fileInput = $("fileInput");
drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) pick(e.dataTransfer.files[0]); });
fileInput.addEventListener("change", () => { if (fileInput.files.length) pick(fileInput.files[0]); });
const EXT_FMT = { ".cf32": "cf32", ".cfile": "cf32", ".cs16": "ci16", ".cs8": "ci8", ".cu8": "cu8" };
function pick(f) {
  S.file = f;
  const ext = (f.name.match(/\.[^.]+$/) || [""])[0].toLowerCase(), kb = f.size / 1024, raw = ext !== ".wav";
  $("fileName").textContent = f.name; $("fileSize").textContent = kb >= 1024 ? (kb / 1024).toFixed(2) + " MB" : kb.toFixed(1) + " KB";
  $("rawFields").classList.toggle("hidden", !raw);
  if (EXT_FMT[ext]) $("fmt").value = EXT_FMT[ext];
  $("picked").classList.remove("hidden");
  if (raw) $("fs").focus();
}
$("startBtn").onclick = () => {
  if (!S.file) return;
  const raw = !$("rawFields").classList.contains("hidden");
  if (raw && !$("fs").value) { setAlerts([{ level: "err", title: "Sampling rate needed.", text: "Raw IQ files have no header, so the sampling rate cannot be read from the file. Enter it and try again." }]); $("fs").focus(); return; }
  run(() => API.upload(S.file, $("fmt").value, raw ? $("fs").value : null), S.file.name);
};

const PRESETS = {
  plain: { n_frames: 0, fec: "none", conv: "none", interleaver: "none" },
  frames: { n_frames: 40, fec: "none", conv: "none", interleaver: "none" },
  conv: { n_frames: 60, fec: "none", conv: "conv_k7", interleaver: "none" },
  rsconv: { n_frames: 14, fec: "rs_204_188", conv: "conv_k7", interleaver: "none" },
  ldpc: { n_frames: 16, fec: "none", conv: "none", ldpc: "ldpc_512_256", interleaver: "none" },
  ldpcrs: { n_frames: 14, fec: "rs_204_188", conv: "none", ldpc: "ldpc_512_256", interleaver: "none" },
  il: { n_frames: 80, fec: "none", conv: "conv_k7", interleaver: "block", ia: 12, ib: 20 },
};
function applyPreset(name) {
  const p = PRESETS[name]; if (!p) return;
  $("dFrames").value = p.n_frames; $("dFec").value = p.fec; $("dConv").value = p.conv; $("dLdpc").value = p.ldpc || "none"; $("dIl").value = p.interleaver;
  if (p.ia) { $("dIa").value = p.ia; $("dIb").value = p.ib; }
}
$("preset").onchange = () => applyPreset($("preset").value);
["dFrames", "dFec", "dConv", "dLdpc", "dIl", "dIa", "dIb"].forEach((id) => $(id).addEventListener("change", () => { $("preset").value = "custom"; }));
applyPreset("conv");
$("demoBtn").onclick = () => {
  const body = { mod: $("dMod").value, snr_db: +$("dSnr").value, cfo_hz: +$("dCfo").value, beta: +$("dBeta").value, gray: $("dGray").value === "1",
    n_frames: Math.max(0, Math.min(80, +$("dFrames").value || 0)), fec: $("dFec").value, conv: $("dConv").value, ldpc: $("dLdpc").value, interleaver: $("dIl").value, ia: +$("dIa").value, ib: +$("dIb").value };
  if ((body.fec !== "none" || body.conv !== "none" || body.ldpc !== "none" || body.interleaver !== "none") && body.n_frames === 0) body.n_frames = 16;
  $("xIl").value = body.interleaver; $("xIa").value = body.ia; $("xIb").value = body.ib;     // the operator-supplied interleaver
  run(() => API.demo(body), `synthetic ${body.mod}${body.conv !== "none" ? " + conv" : ""}${body.fec !== "none" ? " + RS" : ""}${body.ldpc !== "none" ? " + LDPC" : ""}`);
};
$("newBtn").onclick = () => { $("viewResults").classList.add("hidden"); $("viewInput").classList.remove("hidden"); $("railFileCard").style.display = "none"; setAlerts([]); endProgress(); renderRail([]); S.demod = S.decode = S.spec = null; $("pageSub").textContent = "Drop a capture to begin. Every estimate is shown with its evidence and can be overridden."; };
$("rerunBtn").onclick = async () => {
  const over = { beta: +$("oBeta").value || 0.35 };
  if ($("oMod").value) over.mod = $("oMod").value;
  if ($("oRate").value) over.symbol_rate = +$("oRate").value;
  try { await runDemod(over); await runDecode(); selectTab("overview"); } catch (e) { fail(e); }
};
$("redecodeBtn").onclick = async () => { try { await runDecode(); } catch (e) { fail(e); } };

// ------------------------------------------------------------------ analysis flow
function fail(e) {
  const states = [...document.querySelectorAll("#railSteps li")].map((li) => (li.classList.contains("done") ? "done" : li.classList.contains("active") ? "fail" : "later"));
  renderRail(states); endProgress(); S.busy = false;
  setAlerts([{ level: "err", title: "Analysis stopped.", text: e.message }]);
}
async function run(loader, label) {
  if (S.busy) return; S.busy = true; S.times = {}; S.label = label;
  setAlerts([]); S.demod = S.decode = null;
  try {
    renderRail(["active"]); progress("Loading and normalising the file...");
    const t = Date.now(); const d = await loader(); S.session = d.session; S.spec = d;
    S.times[1] = ((Date.now() - t) / 1000).toFixed(1) + "s";
    $("viewInput").classList.add("hidden"); $("viewResults").classList.remove("hidden");
    $("railFileCard").style.display = ""; $("railFile").textContent = label; $("pageSub").textContent = label;
    renderRail(["done", "done"]); renderAll(); selectTab("overview");
    await runDemod({}); await runDecode();
    endProgress(); S.busy = false; renderRail(railStates()); window.scrollTo(0, 0);
  } catch (e) { fail(e); }
}
async function runDemod(over) {
  renderRail(["done", "done", "active"]); progress("Estimating symbol rate and recovering timing...");
  const t = Date.now(); const p = API.demod(S.session, over);
  await sleep(250); renderRail(["done", "done", "done", "active"]); progress("Recovering carrier, classifying the modulation, demodulating...");
  S.demod = await p; S.decode = null; S.times[3] = ((Date.now() - t) / 1000).toFixed(1) + "s";
  renderAll();
}
async function runDecode() {
  renderRail(["done", "done", "done", "done", "active"]); progress("Decoding: bit mapping, de-interleaving, error correction, framing (up to ~30 s)...");
  const il = $("xIl").value, body = il === "none" ? {} : { interleaver: { kind: il, a: +$("xIa").value, b: +$("xIb").value, seed: +$("xSeed").value || 1 } };
  const t = Date.now(); S.decode = await API.decode(S.session, body); S.times[6] = ((Date.now() - t) / 1000).toFixed(1) + "s";
  endProgress(); renderRail(railStates()); renderAll();
}
function railStates() {
  const d = S.decode, il = d && d.interleaver || {}, cv = d && d.conv || {}, rs = d && d.rs || {}, fr = d && d.frame || {}, ld = d && d.ldpc || {};
  return ["done", "done", "done", "done",
    il.kind && il.kind !== "none" ? (il.found ? "done" : "fail") : "skip",
    cv.found || rs.found || ld.found ? "done" : "skip", fr.found ? "done" : "fail"];
}

// ------------------------------------------------------------------ rendering
function stat(k, v, s, cls) { return `<div class="stat ${cls || ""}"><div class="k">${k}</div><div class="v" title="${esc(v)}">${v}</div>${s ? `<div class="s" title="${esc(s)}">${esc(s)}</div>` : ""}</div>`; }
const skel = (k) => `<div class="stat skel"><div class="k">${k}</div><div class="v muted">...</div></div>`;
function kv(rows) { return `<table class="kv">${rows.map(([a, b, n]) => `<tr><td>${a}</td><td>${b}${n ? `<span class="note">${n}</span>` : ""}</td></tr>`).join("")}</table>`; }
function find(state, name, val, note) { const sym = state === "ok" ? "&#10003;" : state === "warn" ? "!" : "&ndash;"; return `<div class="find ${state}"><span class="st">${sym}</span><span class="nm">${name}</span><span class="vl">${val}${note ? `<small>${note}</small>` : ""}</span></div>`; }
const topProb = (r) => (r && r.class_probs && Object.keys(r.class_probs).length > 1 ? Object.entries(r.class_probs).sort((a, b) => b[1] - a[1])[0] : null);

function renderAll() { renderSummary(); renderTexts(); drawTab(S.tab); }

function renderSummary() {
  const d0 = S.spec, r = S.demod, d = S.decode; if (!d0) return;
  const p = d0.params;
  let html = "";
  if (r) {
    const tp = topProb(r), conf = tp ? Math.round(tp[1] * 100) : null;
    html += stat("Modulation", esc(r.modulation), conf !== null ? `${conf}% confidence` : r.modulation_source === "override" ? "set manually" : "tone clustering", "hero-stat");
    html += stat("Symbol rate", fmtHz(r.symbol_rate), `${r.sps.toFixed(2)} samples per symbol`);
  } else { html += skel("Modulation") + skel("Symbol rate"); }
  html += stat("SNR (in band)", fmtDb(p.snr_db_inband), r && r.quality && r.quality.esn0_db !== undefined ? `symbol SNR ${r.quality.esn0_db} dB` : `centre ${fmtHz(p.center_freq_hz)}`);
  if (d) {
    const cv = d.conv || {}, rs = d.rs || {}, fr = d.frame || {}, m = d.message || {}, ld = d.ldpc || {};
    const fec = [cv.found ? `Conv 1/${cv.rate.split("/")[1]} K${cv.K}` : "", ld.found ? `LDPC (${ld.n},${ld.k})` : "", rs.found ? `RS (${rs.n},${rs.k})` : ""].filter(Boolean).join(" + ");
    html += stat("Error correction", fec || "none found", ld.found ? `${ld.converged}/${ld.n_blocks} LDPC blocks fixed, channel BER ${(ld.channel_ber * 100).toFixed(2)}%` : cv.found ? `channel BER ${(cv.channel_ber * 100).toFixed(2)}%` : rs.found ? `${rs.corrected_symbols} symbols corrected` : "stream looks uncoded", fec ? "" : "na");
    html += stat("Frame", fr.found ? fr.frame_len + " bits" : "not found", fr.found ? `${fr.sync_name || "sync " + fr.sync_hex}` : "no repeating sync word", fr.found ? "" : "na");
    html += stat("Message", m.found ? "recovered" : "not found", m.found ? `${m.bytes} bytes, ${Math.round(m.printable * 100)}% printable` : "needs frames with a sync word", m.found ? "good" : "na");
  } else if (r) { html += skel("Error correction") + skel("Frame") + skel("Message"); }
  else html += skel("Error correction") + skel("Frame") + skel("Message");
  $("summary").innerHTML = html;
}

function renderTexts() {
  const d0 = S.spec, r = S.demod, d = S.decode; if (!d0) return;
  const p = d0.params, gt = d0.ground_truth;
  // ---- spectrum table
  $("specTable").innerHTML = kv([
    ["Sampling rate", fmtHz(d0.fs), d0.source.kind === "raw" ? "supplied by you" : d0.source.kind === "wav" ? "from the WAV header" : "synthetic"],
    ["Duration", (d0.duration_s * 1000).toFixed(1) + " ms", d0.n_samples.toLocaleString() + " samples"],
    ["Centre offset", fmtHz(p.center_freq_hz), gt ? "truth " + fmtHz(gt.cfo_hz) : ""],
    ["Bandwidth (99%)", fmtHz(p.bandwidth_hz)],
    ["SNR, in band", fmtDb(p.snr_db_inband)], ["SNR, full band", fmtDb(p.snr_db_fullband), gt ? "truth " + gt.snr_db_fullband + " dB" : ""],
    ["Noise floor", fmtDb(p.noise_floor_db) + "/Hz"], ["DC offset", p.dc_offset.toFixed(4)]]);
  if (!r) return;
  const isFsk = r.family === "fsk";
  // ---- demod
  $("constTitle").textContent = isFsk ? "Tone histogram" : "Constellation";
  $("constSub").textContent = isFsk ? "Measured frequency of each symbol. Each peak is one tone." : "Recovered symbols. Tight clusters on the rings mean a clean demodulation.";
  $("constCap").textContent = isFsk ? "The number of well-separated peaks gives the order of the FSK." : "The pattern is correct up to a fixed rotation; the decoding stage resolves it.";
  $("miniTitle").textContent = $("constTitle").textContent; $("miniSub").textContent = isFsk ? "Measured tone per symbol" : "Symbols after timing and carrier recovery";
  $("eyeCard").classList.toggle("hidden", isFsk);
  $("bitsSub").textContent = `${r.bits.count.toLocaleString()} bits recovered. First 512 shown.`;
  $("bitsBox").textContent = r.bits.preview.replace(/(.{8})/g, "$1 ");
  $("bitsNote").textContent = r.bits.note;
  $("dlTxt").href = API.url(`/api/bits/${S.session}?fmt=txt`); $("dlBin").href = API.url(`/api/bits/${S.session}?fmt=bin`); $("dlReport").href = API.url(`/api/report/${S.session}`); $("dlReportTop").href = API.url(`/api/report/${S.session}`);
  const q = r.quality || {}, g = r.ground_truth_check;
  $("qualTable").innerHTML = kv([
    ["Symbols", (q.n_symbols || 0).toLocaleString()],
    q.esn0_db !== undefined ? ["Symbol SNR", q.esn0_db + " dB", "from the error vector magnitude"] : null,
    q.evm !== undefined ? ["EVM", (q.evm * 100).toFixed(1) + " %", "lower is better"] : null,
    q.timing_lock !== undefined ? ["Timing lock", (q.timing_lock * 100).toFixed(0) + " %"] : null,
    r.carrier ? ["Residual carrier offset", fmtHz(r.carrier.residual_cfo_hz)] : null,
    ["Symbol-rate line", r.symbol_rate_pmr + "x", "strength over the spectral floor"],
    isFsk ? ["Tone spacing", fmtHz(r.fsk.tone_spacing_hz), "modulation index h = " + r.fsk.mod_index_h] : null,
    g ? ["Ground truth", g.ber === null ? esc(g.note) : `bit error rate ${(g.ber * 100).toFixed(3)} %`, "demo signals only"] : null].filter(Boolean));
  // ---- overview found list
  const tp = topProb(r), conf = tp ? Math.round(tp[1] * 100) : null;
  const cv = d && d.conv || {}, rs = d && d.rs || {}, fr = d && d.frame || {}, il = d && d.interleaver || {}, ld = d && d.ldpc || {};
  let rows = find("ok", "Signal", `${fmtHz(p.center_freq_hz)} offset, ${fmtHz(p.bandwidth_hz)} wide`, `SNR ${fmtDb(p.snr_db_inband)} in band`);
  rows += find(conf !== null && conf < 60 ? "warn" : "ok", "Modulation", esc(r.modulation), conf !== null ? `${conf}% by the feature classifier` : r.modulation_source === "override" ? "set manually" : "tone clustering");
  rows += find("ok", "Symbol rate", fmtHz(r.symbol_rate), `${r.sps.toFixed(2)} samples per symbol`);
  if (d) {
    rows += find("ok", "Bit mapping", esc(d.bit_mapping), "phase ambiguity and labelling resolved by structure");
    rows += il.kind && il.kind !== "none" ? find(il.found ? "ok" : "warn", "Interleaver", `${esc(il.kind)} ${il.a} x ${il.b}`, il.found ? `start offset ${il.offset} bits` : "no structure appeared: wrong type or size?") : find("no", "Interleaver", "none supplied", "type and size cannot be found blindly");
    rows += cv.found ? find("ok", "Convolutional code", `rate ${esc(cv.rate)}, K=${cv.K}`, `generators (octal) ${cv.generators_octal.join(", ")}`) : find("no", "Convolutional code", "not detected", "");
    rows += rs.found ? find("ok", "Reed-Solomon", `RS(${rs.n},${rs.k}), corrects ${rs.t}`, `${rs.corrected_symbols} symbols fixed, ${rs.uncorrectable} words lost`) : find("no", "Reed-Solomon", "not detected", "");
    rows += fr.found ? find("ok", "Frame", `${fr.frame_len} bits, ${fr.n_frames} frames`, `sync ${fr.sync_hex}${fr.sync_name ? " (" + esc(fr.sync_name) + ")" : ""}, header ${fr.header_bits} bits, payload ${fr.payload_bits} bits`) : find("warn", "Frame", "no repeating structure", "needs a sync word and about a dozen frames");
    rows += ld.found ? find(ld.failed ? "warn" : "ok", "LDPC code", `${esc(ld.name)}: (${ld.n}, ${ld.k}), rate ${esc(ld.rate)}`, `${ld.converged} of ${ld.n_blocks} blocks decoded, ${ld.corrected_bits} bits corrected, start offset ${ld.offset}${ld.source === "uploaded" ? ", your matrix" : ""}`) : find("no", "LDPC code", "not detected", "checked the built-in codes and any matrix you uploaded");
  } else rows += find("no", "Decoding", "running...", "");
  $("findList").innerHTML = rows;
  // ---- decoding tab + message
  if (d) {
    const m = d.message || {};
    const msgHtml = m.found ? esc(m.text) : "No message recovered. Without a repeating sync word the stream cannot be framed.";
    ["ovMsg", "msgFull"].forEach((id) => { $(id).textContent = m.found ? m.text : "No message recovered. Without a repeating sync word the stream cannot be framed."; $(id).classList.toggle("empty", !m.found); });
    const meta = m.found ? `${m.bytes} bytes from ${m.n_frames} frames, ${Math.round(m.printable * 100)}% printable text` : "nothing recovered";
    $("ovMsgMeta").textContent = meta; $("msgMeta").textContent = meta;
    const btns = m.found ? `<button class="btn secondary small" data-copy-msg type="button">Copy</button> <a class="btn secondary small" href="${API.url(`/api/payload/${S.session}`)}">.bin</a>` : "";
    $("ovMsgBtns").innerHTML = btns; $("msgBtns").innerHTML = btns;
    const gc = d.ground_truth_check; $("gtBox").innerHTML = gc ? `<b>Ground truth:</b> ${(gc.payload_byte_accuracy * 100).toFixed(1)}% of payload bytes match what was transmitted. Decoded in ${d.elapsed_s}s.` : (m.found ? `Decoded in ${d.elapsed_s}s.` : "");
    const fm = d.frames || [];
    $("frameTable").innerHTML = fm.length ? `<table class="table"><tr><th>#</th><th>Header</th><th>Payload</th></tr>${fm.map((f, i) => `<tr><td>${i}</td><td class="mono">${esc(f.header_hex)}</td><td class="mono">${esc(f.payload_ascii.slice(0, 56))}</td></tr>`).join("")}</table>` : `<p class="muted">No frames recovered.</p>`;
    $("hypTable").innerHTML = `<table class="table"><tr><th>Hypothesis</th><th><span class="tip" tabindex="0" data-tip="How strongly a convolutional code shows up in the bits (0 to 1). Random bits score about 0.05.">Code</span></th><th><span class="tip" tabindex="0" data-tip="Strength of the repeating frame structure. Above about 10 is real.">Frame z</span></th><th><span class="tip" tabindex="0" data-tip="How clearly an LDPC code from the library shows up (parity checks satisfied). Above 7 is real.">LDPC z</span></th></tr>` +
      d.candidates.map((c) => `<tr class="${c.chosen ? "sel" : ""}"><td>${esc(c.label)}${c.chosen ? " &#9664;" : ""}</td><td class="mono">${c.conv === null ? "-" : c.conv}</td><td class="mono">${c.frame_z === null ? "-" : c.frame_z}</td><td class="mono">${c.ldpc_z === undefined || c.ldpc_z === null ? "-" : c.ldpc_z}</td></tr>`).join("") + `</table><p class="cap">Hypotheses that differ only by a constant XOR score identically; the final polarity is chosen from a known sync word or from how text-like the payload is.</p>`;
  } else {
    ["ovMsg", "msgFull"].forEach((id) => { $(id).textContent = "Decoding has not finished yet."; $(id).classList.add("empty"); });
    $("ovMsgBtns").innerHTML = ""; $("msgBtns").innerHTML = ""; $("frameTable").innerHTML = ""; $("hypTable").innerHTML = "";
  }
  try { renderLdpc(d); } catch (e) { console.warn("LDPC panel:", e); }
  renderAlerts(); renderModels();
}
function renderLdpc(d) {
  const ld = d && d.ldpc || {};
  $("ldpcResult").innerHTML = !d ? "" : ld.found
    ? `<b>Found:</b> ${esc(ld.name)} (${ld.n}, ${ld.k}), rate ${esc(ld.rate)}. Before decoding ${(ld.checks_violated_before * 100).toFixed(0)}% of parity checks were violated (random bits: about 50%). ${ld.converged} of ${ld.n_blocks} blocks decoded in ${ld.mean_iterations} iterations on average, ${ld.corrected_bits} bit errors corrected.`
    : "<b>No LDPC code found.</b> The stream does not match any built-in code or uploaded matrix.";
  if (typeof API.ldpcCodes !== "function") return;     // stale cached api.js: skip the optional list instead of breaking the analysis
  if (typeof API.ldpcCodes !== "function") return;            // stale cached api.js: skip the list, never block the analysis
  API.ldpcCodes().then((c) => { $("ldpcList").innerHTML = "Built-in library: " + c.map((x) => `<a href="${API.url(`/api/ldpc-codes/${x.name}.alist`)}" download>${esc(x.name)}</a>`).join(", ") + ". Download one to see the accepted format."; }).catch(() => {});
}
$("ldpcBtn").onclick = () => $("ldpcFile").click();
$("ldpcFile").onchange = async () => {
  const f = $("ldpcFile").files[0]; if (!f || !S.session) return;
  try { const r = await API.ldpcUpload(S.session, f); toast(`Matrix read: (${r.code.n}, ${r.code.k}). Re-running the decoder...`); $("ldpcFile").value = ""; await runDecode(); }
  catch (e) { toast(e.message); setAlerts([{ level: "warn", title: "Matrix not used.", text: e.message }]); }
};
$("ldpcClear").onclick = async () => { if (!S.session) return; try { await API.ldpcClear(S.session); toast("Uploaded matrices removed."); await runDecode(); } catch (e) { toast(e.message); } };
document.addEventListener("click", (e) => { if (e.target.closest("[data-copy-msg]") && S.decode && S.decode.message) copyText(S.decode.message.text, "Message"); });
$("copyBits").onclick = () => S.demod && copyText(S.demod.bits.preview, "Bits");

function renderAlerts() {
  const list = [];
  const r = S.demod, d = S.decode;
  (r && r.warnings || []).forEach((w) => list.push({ level: /CNN/.test(w) || /disagree/.test(w) ? "warn" : "warn", text: w }));
  (d && d.warnings || []).forEach((w) => list.push({ level: "info", text: w }));
  setAlerts(list);
}

// ------------------------------------------------------------------ models tab
function bars(rows) { return rows.map(([n, v]) => `<div class="bar-row"><span>${esc(n)}</span><div class="tr"><i style="width:${(v * 100).toFixed(0)}%"></i></div><span class="pc">${(v * 100).toFixed(0)}%</span></div>`).join(""); }
function renderModels() {
  const r = S.demod, m = MODELS, deep = r && r.deep;
  let html = "";
  // this capture
  const rfRows = r && r.class_probs && Object.keys(r.class_probs).length > 1 ? Object.entries(r.class_probs).sort((a, b) => b[1] - a[1]) : null;
  let cnnBlock;
  if (!r) cnnBlock = `<p class="muted">Run an analysis first.</p>`;
  else if (deep && deep.available && deep.top) cnnBlock = bars(deep.top.map((t) => [t.label, t.prob])) + `<p class="cap">${deep.n_frames} frames of ${deep.frame_len} samples averaged. Trained on ${esc(deep.dataset || "public data")}.</p>` + (deep.note ? `<p class="cap"><b>Caveat:</b> ${esc(deep.note)}</p>` : "");
  else if (deep && deep.error) cnnBlock = `<p class="muted">The CNN could not score this capture: ${esc(deep.error)}</p>`;
  else cnnBlock = `<p class="muted">Not installed. ${esc((deep && deep.reason) || (m && m.cnn_reason) || "")}</p>`;
  const agree = deep && deep.available && deep.top && deep.warnings_trusted === false ? `<span class="chip">CNN informational only</span>` : deep && deep.available && deep.top ? (deep.agrees === true ? `<span class="chip ok">agrees with ${esc(r.modulation)}</span>` : deep.agrees === false ? `<span class="chip warn">disagrees with ${esc(r.modulation)}</span>` : `<span class="chip warn">outside supported set</span>`) : "";
  html += `<div class="card"><div class="card-h"><div><h3>Two independent opinions on this capture</h3><p class="sub">Different training data and methods. Agreement raises confidence; disagreement is a reason to look closer.</p></div>${agree}</div><div class="card-b"><div class="cols-eq">
    <div><p class="section-label">Feature-based classifier</p>${rfRows ? bars(rfRows) : `<p class="muted">${r ? "Modulation set manually, or FSK (decided by tone clustering)." : "Run an analysis first."}</p>`}</div>
    <div><p class="section-label">CNN trained on public data</p>${cnnBlock}</div></div></div></div>`;
  // RF report
  if (m && m.rf) {
    html += `<div class="card"><div class="card-h"><div><h3>Feature-based classifier</h3><p class="sub">Random forest on spectral-line and amplitude features. Hold-out set of ${m.rf.n_test} signals from this project's own generator.</p></div><span class="chip">${Math.round(m.rf.accuracy * 100)}% overall</span></div><div class="card-b"><canvas class="plot" id="cRf" data-h="200"></canvas>
      <p class="cap"><b>Caveat:</b> trained and tested on the same simulator, so this is an upper bound. Real captures will score lower.</p></div></div>`;
  }
  // CNN report or how-to
  if (m && m.cnn_available && m.cnn) {
    const c = m.cnn;
    html += `<div class="card"><div class="card-h"><div><h3>CNN trained on ${esc(c.dataset)}</h3><p class="sub">${esc(c.arch)}. ${c.n_train.toLocaleString()} training frames, ${c.n_test.toLocaleString()} test frames never seen in training.</p></div><span class="chip ok">${Math.round(c.test_accuracy_ge10dB * 100)}% at 10 dB and above</span></div><div class="card-b"><canvas class="plot" id="cCnn" data-h="220"></canvas>
      <p class="cap">Overall test accuracy ${(c.test_accuracy * 100).toFixed(1)}%. ${esc(c.note || "")}</p>
      ${c.top_confusions_ge10dB && c.top_confusions_ge10dB.length ? `<p class="section-label" style="margin-top:14px">Most common confusions at 10 dB and above</p><table class="table"><tr><th>True</th><th>Predicted as</th><th>Count</th></tr>${c.top_confusions_ge10dB.slice(0, 5).map((x) => `<tr><td>${esc(x[1])}</td><td>${esc(x[2])}</td><td class="mono">${x[0]}</td></tr>`).join("")}</table>` : ""}</div></div>`;
  } else {
    html += `<div class="card"><div class="card-h"><div><h3>Train the CNN on public data (Kaggle)</h3><p class="sub">Not installed yet${m && m.cnn_reason ? ": " + esc(m.cnn_reason) : ""}. A free Kaggle GPU trains it in well under an hour.</p></div><span class="chip warn">not installed</span></div><div class="card-b">
      <ol class="steps-howto">
        <li>Open <b>kaggle/train_modulation_cnn.ipynb</b> from the project folder in a new Kaggle notebook (File, Import).</li>
        <li><b>Add Input</b>: search <code>radioml2018</code> (20 GB, best) or <code>radioml2016</code> (600 MB, quick start). Set the accelerator to <b>GPU</b> and Internet to <b>On</b>.</li>
        <li>Set <code>SMOKE = True</code> in the first code cell and run all cells once to check that everything works (about 3 minutes). Then set it to <code>False</code> and run again.</li>
        <li>Download <code>sanketsetu_model.zip</code> from the notebook's Output tab and unzip it.</li>
        <li>Copy <code>${esc(m ? m.cnn_file : "radioml_cnn.onnx")}</code> and <code>radioml_cnn.json</code> into <code>backend/models/</code>, run <code>pip install onnxruntime</code>, and restart the server.</li>
      </ol></div></div>`;
  }
  $("modelsBody").innerHTML = html;
  const tb = document.querySelector('.tab[data-tab="models"]'); tb.querySelector(".badge")?.remove();
  if (deep && deep.available && deep.warnings_trusted && deep.agrees === false) tb.insertAdjacentHTML("beforeend", `<span class="badge warn">!</span>`);
  if (S.tab === "models") drawTab("models");
}

// ------------------------------------------------------------------ drawing (only the visible tab)
function drawTab(name) {
  const d0 = S.spec, r = S.demod, d = S.decode; if (!d0) return;
  if (name === "overview") {
    Plots.psd($("cPsdMini"), d0);
    if (r) { if (r.family === "fsk") Plots.histogram($("cMini"), r.fsk.hist, r.fsk.tones_hz); else Plots.constellation($("cMini"), r.plots.constellation, r.modulation); } else Plots.empty($("cMini"), "analysing...");
  } else if (name === "spectrum") { Plots.psd($("cPsd"), d0); Plots.waterfall($("cWf"), d0.waterfall); }
  else if (name === "demod") {
    if (!r) { ["cConst", "cEye", "cRate"].forEach((i) => Plots.empty($(i), "analysing...")); return; }
    const fsk = r.family === "fsk"; $("cConst").classList.toggle("hidden", fsk); $("cHist").classList.toggle("hidden", !fsk);
    if (fsk) Plots.histogram($("cHist"), r.fsk.hist, r.fsk.tones_hz); else { Plots.constellation($("cConst"), r.plots.constellation, r.modulation); Plots.eye($("cEye"), r.plots.eye); }
    Plots.rateSpectrum($("cRate"), r.plots.rate_spectrum, r.symbol_rate);
  } else if (name === "decode") {
    if (d && d.frame && d.frame.found && d.frame.columns) Plots.frameColumns($("cFrame"), d.frame); else Plots.empty($("cFrame"), d ? "no repeating frame structure found" : "decoding...");
  } else if (name === "models" && MODELS) {
    if (MODELS.rf && $("cRf")) Plots.accuracyCurve($("cRf"), MODELS.rf.accuracy_by_esn0_db.filter((b) => b.accuracy !== null).map((b) => ({ snr: (b.esn0_range[0] + b.esn0_range[1]) / 2, acc: b.accuracy })), "accuracy by symbol SNR (Es/N0)");
    if (MODELS.cnn_available && $("cCnn")) Plots.accuracyCurve($("cCnn"), MODELS.cnn.accuracy_by_snr.map((b) => ({ snr: b.snr, acc: b.acc })), "test accuracy by SNR");
  }
}
function redrawAll() { if (!$("view-app").classList.contains("hidden") && S.spec) drawTab(S.tab); }
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(redrawAll, 150); });

route();
