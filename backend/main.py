"""FastAPI entry point. Run:  uvicorn backend.main:app --reload"""
import json
import os
import uuid
from collections import OrderedDict
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.core.generator import MOD_ORDER, gen_signal
from backend.core.chain import build_chain, RS_PRESETS, CONV_PRESETS
from backend.core.fec import ldpc as ldpc_mod
from backend.core import decode as decoder
from backend.core.loader import load_file
from backend.core.pipeline import demodulate, get_classifier
from backend.core.dl_classifier import get_deep, ONNX_NAME
from backend.core.spectral import analyze

MAX_BYTES = 200 * 1024 * 1024
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT, "frontend")
REPORT_PATH = os.path.join(ROOT, "backend", "models", "eval_report.json")

app = FastAPI(title="SanketSetu - automated .IQ and .wav analysis")
# The front end may be hosted elsewhere (Vercel). Allow any *.vercel.app page plus any origins listed in CORS_ORIGINS
# (comma separated, e.g. a custom domain). Same-origin use (local run, or the Render URL itself) needs no CORS.
_extra = [o.strip().rstrip("/") for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_extra, allow_origin_regex=r"https://([a-z0-9-]+\.)*vercel\.app",
                   allow_methods=["*"], allow_headers=["*"], expose_headers=["Content-Disposition"])

# Single-user demo store: session id -> loaded signal and stage results (last 8 kept).
STORE: "OrderedDict[str, dict]" = OrderedDict()


def _put(x, fs, source, spec, truth=None):
    sid = uuid.uuid4().hex[:10]
    STORE[sid] = dict(x=x, fs=fs, source=source, spec=spec, truth=truth, bits=None, demod=None)
    while len(STORE) > 8:
        STORE.popitem(last=False)
    return sid


def _get(sid):
    if sid not in STORE:
        raise HTTPException(404, "Unknown or expired session - upload the file again")
    return STORE[sid]


class DemoRequest(BaseModel):
    mod: str = "QPSK"
    fs: float = 100e3
    symbol_rate: float = 12.5e3
    snr_db: float = 15.0          # full-band SNR
    cfo_hz: float = 0.0
    beta: float = 0.35
    n_symbols: int = 4000
    gray: bool = False            # Gray-labelled constellation (the receiver does not know this)
    # optional transmit chain: message -> frames -> RS -> convolutional -> interleaver
    fec: str = "none"             # none | rs_255_223 | rs_204_188 | rs_ccsds | rs_64_48
    conv: str = "none"            # none | conv_k3 | conv_k7 | conv_k7_r13 | conv_k9
    ldpc: str = "none"            # none | ldpc_256_128 | ldpc_512_256 | ldpc_1024_512
    interleaver: str = "none"     # none | block | diagonal | convolutional | prandom
    ia: int = 12
    ib: int = 20
    n_frames: int = 0             # 0 = no framing (random bits, as in Phase 2)


class InterleaverSpec(BaseModel):
    kind: str = "none"
    a: int = 8
    b: int = 8
    seed: int = 1


class DecodeRequest(BaseModel):
    interleaver: Optional[InterleaverSpec] = None


class DemodRequest(BaseModel):
    mod: Optional[str] = None            # override classifier: BPSK/QPSK/8PSK/16QAM/2FSK/4FSK
    symbol_rate: Optional[float] = None  # override estimator (Hz)
    beta: float = 0.35                   # RRC roll-off assumed for the matched filter


@app.get("/api/health")
def health():
    clf, deep = get_classifier(), get_deep()
    return {"status": "ok", "modulations": list(MOD_ORDER), "classifier_ready": clf.available,
            "cnn_ready": deep.available, "cnn_reason": deep.reason}


@app.get("/api/model-report")
def model_report():
    if not os.path.exists(REPORT_PATH):
        raise HTTPException(404, "No evaluation report - run scripts/train_classifier.py")
    return json.load(open(REPORT_PATH))


@app.get("/api/models")
def models():
    """Everything the UI needs to describe the classifiers: RF report (synthetic hold-out) and the CNN (public data)."""
    rf = json.load(open(REPORT_PATH)) if os.path.exists(REPORT_PATH) else None
    deep = get_deep()
    return {"rf": rf, "cnn_available": deep.available, "cnn_reason": deep.reason, "cnn": deep.meta,
            "cnn_file": ONNX_NAME, "cnn_transfer": getattr(deep, "transfer", None),
            "cnn_size_bytes": os.path.getsize(os.path.join(os.path.dirname(REPORT_PATH), ONNX_NAME)) if os.path.exists(os.path.join(os.path.dirname(REPORT_PATH), ONNX_NAME)) else None}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), fmt: str = Form("cf32"), fs: Optional[float] = Form(None)):
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (limit 200 MB)")
    try:
        x, fs_used, info = load_file(file.filename or "", data, fmt=fmt, fs=fs)
        result = analyze(x, fs_used)
    except ValueError as e:
        raise HTTPException(400, str(e))
    result["source"] = dict(filename=file.filename, size_bytes=len(data), **info)
    result["session"] = _put(x, fs_used, result["source"], result["params"])
    return result


def _clean(o):
    """Make numpy values JSON friendly."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.hex()
    return o


@app.post("/api/demo")
def demo(req: DemoRequest):
    chain = None
    try:
        k = int(np.log2(MOD_ORDER[req.mod.upper()]))
        bits = None
        if req.n_frames > 0:
            if req.fec != "none" and req.fec not in RS_PRESETS:
                raise ValueError("unknown fec preset")
            if req.conv != "none" and req.conv not in CONV_PRESETS:
                raise ValueError("unknown conv preset")
            nf = req.n_frames
            while True:                                      # keep the capture a sensible size
                bits, chain = build_chain(req.fec, req.conv, req.interleaver, req.ia, req.ib, nf, ldpc=req.ldpc)
                if len(bits) <= 70000 or nf <= 8:
                    break
                nf -= 2
            chain["n_frames"] = nf
        x, truth = gen_signal(req.mod, fs=req.fs, symbol_rate=req.symbol_rate, n_symbols=req.n_symbols,
                              snr_db=req.snr_db, cfo_hz=req.cfo_hz, beta=req.beta, seed=7, bits=bits,
                              gray=req.gray)
        result = analyze(x, req.fs)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))
    tbits = truth.pop("bits")
    result["source"] = dict(filename=f"synthetic {req.mod}", kind="synthetic", notes=["Generated on the server"])
    gt = dict(truth)
    if chain:
        gt["chain"] = {k_: v for k_, v in chain.items() if k_ not in ("message",)}
    result["ground_truth"] = _clean(gt)
    result["session"] = _put(x, req.fs, result["source"], result["params"], truth=dict(truth, bits=tbits))
    STORE[result["session"]]["chain"] = chain
    return result


@app.post("/api/demod/{sid}")
def run_demod(sid: str, req: DemodRequest):
    s = _get(sid)
    if req.mod is not None and req.mod.upper() not in MOD_ORDER:
        raise HTTPException(400, f"Unsupported modulation {req.mod}")
    try:
        res, bits, internal = demodulate(s["x"], s["fs"], s["spec"], mod=req.mod.upper() if req.mod else None,
                                         symbol_rate=req.symbol_rate, beta=req.beta)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    s["bits"], s["demod"], s["internal"] = bits, res, internal
    if s.get("truth") is not None:
        from backend.core.evaluate import ber_aligned
        t = s["truth"]
        k = internal["k"]
        if t["mod"] == res["modulation"]:
            best = (1.0, 0, 0)
            rots = internal["M"]
            for r in range(rots):
                if internal["kind"] == "linear":
                    from backend.core.demod import demap_linear
                    b, _ = demap_linear(internal["syms"] * np.exp(2j * np.pi * r / rots), internal["mod"], gray=t.get("gray", False))
                else:
                    b = bits
                ber, lag = ber_aligned(t["bits"], b, k)
                if ber < best[0]:
                    best = (ber, lag, r)
            res["ground_truth_check"] = dict(ber=round(float(best[0]), 5), symbol_lag=best[1],
                                             phase_rotation_index=best[2],
                                             note="BER vs the generator's true bits (demo signals only)")
        else:
            res["ground_truth_check"] = dict(ber=None, note=f"Detected {res['modulation']} but truth is {t['mod']}")
    return res


@app.post("/api/decode/{sid}")
def run_decode(sid: str, req: DecodeRequest):
    s = _get(sid)
    if not s.get("internal"):
        raise HTTPException(400, "Run demodulation first")
    il = None
    if req.interleaver and req.interleaver.kind != "none":
        il = req.interleaver.model_dump()
        if il["kind"] not in ("block", "diagonal", "convolutional", "prandom"):
            raise HTTPException(400, "Unsupported interleaver type")
        if not (2 <= il["a"] <= 4096 and 1 <= il["b"] <= 4096):
            raise HTTPException(400, "Interleaver size out of range")
    chain = s.get("chain")
    try:
        res = decoder.decode(s["internal"], interleaver=il, truth=chain, ldpc_codes=s.get("ldpc_codes"))
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    s["decoded"] = dict(payload=res.get("_payload"), bits=res.get("_bits"))
    s["decode_res"] = _clean(res)
    return s["decode_res"]


@app.post("/api/ldpc/{sid}")
async def upload_ldpc(sid: str, file: UploadFile = File(...)):
    """Register a parity-check matrix (alist, dense 0/1 text, or 'row col' pairs) for this session. The next decode
    run looks for it, together with the built-in codes."""
    s = _get(sid)
    data = await file.read()
    if len(data) > 8 * 1024 * 1024:
        raise HTTPException(413, "Matrix file too large (limit 8 MB)")
    try:
        code = ldpc_mod.parse_matrix(data.decode("utf-8", "replace"), name=(file.filename or "uploaded"))
    except (ValueError, IndexError) as e:
        raise HTTPException(400, f"Could not read the parity-check matrix: {e}")
    if code.n > 20000 or code.m > 20000:
        raise HTTPException(400, "Matrix too large (limit 20000 x 20000)")
    s["ldpc_codes"] = [c for c in s.get("ldpc_codes", []) if c.name != code.name] + [code]
    return dict(ok=True, code=code.info(), registered=[c.name for c in s["ldpc_codes"]])


@app.get("/api/ldpc-codes")
def ldpc_codes():
    return [c.info() for c in ldpc_mod.builtin_library()]


@app.get("/api/ldpc-codes/{name}.alist")
def ldpc_alist(name: str):
    if name not in ldpc_mod.BUILTIN:
        raise HTTPException(404, "Unknown built-in code")
    return Response(ldpc_mod.to_alist(ldpc_mod.get_builtin(name)), media_type="text/plain",
                    headers={"Content-Disposition": f"attachment; filename={name}.alist"})


@app.delete("/api/ldpc/{sid}")
def clear_ldpc(sid: str):
    _get(sid)["ldpc_codes"] = []
    return dict(ok=True)


@app.get("/api/report/{sid}")
def download_report(sid: str):
    """PDF report of everything run so far in this session."""
    s = _get(sid)
    from backend.core.report import build_report, TOOL_NAME
    try:
        pdf = build_report(s, analyze)
    except Exception as e:                                   # report problems must not look like a server crash
        raise HTTPException(500, f"Could not build the report: {e}")
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename={TOOL_NAME.lower()}_report_{sid}.pdf"})


@app.get("/api/payload/{sid}")
def download_payload(sid: str):
    s = _get(sid)
    d = s.get("decoded")
    if not d or d.get("payload") is None:
        raise HTTPException(400, "No decoded payload - run decoding first")
    return Response(d["payload"], media_type="application/octet-stream",
                    headers={"Content-Disposition": "attachment; filename=payload.bin"})


@app.get("/api/bits/{sid}")
def download_bits(sid: str, fmt: str = "txt"):
    s = _get(sid)
    if s["bits"] is None:
        raise HTTPException(400, "Run demodulation first")
    b = s["bits"]
    if fmt == "bin":
        pad = (-len(b)) % 8
        packed = np.packbits(np.concatenate([b, np.zeros(pad, np.uint8)]))
        return Response(packed.tobytes(), media_type="application/octet-stream",
                        headers={"Content-Disposition": "attachment; filename=bitstream.bin"})
    return Response("".join(map(str, b.tolist())), media_type="text/plain",
                    headers={"Content-Disposition": "attachment; filename=bitstream.txt"})


@app.middleware("http")
async def no_stale_frontend(request, call_next):
    """Always revalidate the front-end files, so an updated project never runs against a cached older script."""
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
