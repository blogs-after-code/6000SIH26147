// Thin wrapper around the backend REST API. Every call rejects with a readable Error.
const API = (() => {
  const BASE = String(window.SANKETSETU_API_BASE || "").replace(/\/+$/, "");
  const url = (p) => BASE + p;                       // "/api/x" -> "https://backend.example/api/x" (or unchanged when same-origin)
  async function call(path, opts) {
    let res;
    try { res = await fetch(url(path), opts); }
    catch (e) { throw new Error(BASE ? "Cannot reach the server. If the demo has been idle it may be waking up: wait about a minute and try again." : "Cannot reach the server. Is it running? (uvicorn backend.main:app)"); }
    let data = {};
    try { data = await res.json(); } catch (e) { /* non-JSON body */ }
    if (!res.ok) {
      const d = data.detail;
      throw new Error(typeof d === "string" ? d : Array.isArray(d) ? d.map((x) => x.msg).join("; ") : `Request failed (${res.status})`);
    }
    return data;
  }
  const json = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  return {
    url,
    health: () => call("/api/health"),
    models: () => call("/api/models"),
    upload(file, fmt, fs) {
      const fd = new FormData();
      fd.append("file", file); fd.append("fmt", fmt);
      if (fs) fd.append("fs", fs);
      return call("/api/upload", { method: "POST", body: fd });
    },
    demo: (p) => call("/api/demo", json(p)),
    demod: (sid, p) => call(`/api/demod/${sid}`, json(p)),
    decode: (sid, p) => call(`/api/decode/${sid}`, json(p)),
    ldpcCodes: () => call("/api/ldpc-codes"),
    ldpcUpload(sid, file) { const fd = new FormData(); fd.append("file", file); return call(`/api/ldpc/${sid}`, { method: "POST", body: fd }); },
    ldpcClear: (sid) => call(`/api/ldpc/${sid}`, { method: "DELETE" }),
  };
})();
