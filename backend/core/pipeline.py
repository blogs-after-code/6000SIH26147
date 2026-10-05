"""Phase 2 orchestrator: spectral params + IQ  ->  modulation, symbol rate, constellation, bits."""
import numpy as np

from . import demod as D
from .classifier import ModClassifier, features
from .dl_classifier import get_deep
from .generator import MOD_ORDER
from scipy.cluster.vq import kmeans2

_clf = None


def get_classifier():
    global _clf
    if _clf is None:
        _clf = ModClassifier()
    return _clf


def _dprime(s, K):
    c0 = np.quantile(s, (np.arange(K) + 0.5) / K)
    cent, _ = kmeans2(s.reshape(-1, 1), c0.reshape(-1, 1), minit="matrix", iter=30)
    cent = np.sort(cent[:, 0])
    sig = np.sqrt(np.mean((s - cent[np.abs(s[:, None] - cent[None, :]).argmin(axis=1)]) ** 2)) + 1e-12
    gaps = np.diff(cent)
    return float(gaps.min() / sig)


def detect_family(xb, fs, bw):
    """FSK (constant envelope, separable tones) vs linear. Returns (family, evidence)."""
    cv = D.envelope_cv(xb)
    fr, _, _ = D.estimate_symbol_rate_fsk(xb, fs, bw)
    s, _ = D.demod_fsk(xb, fs, fr)
    d2, d4 = _dprime(s, 2), _dprime(s, 4)
    is_fsk = (cv < 0.22) or (d2 > 4.2) or (d4 > 3.9)
    return ("fsk" if is_fsk else "linear"), dict(envelope_cv=round(cv, 3), tone_d2=round(d2, 2), tone_d4=round(d4, 2))


def _downsample_spec(f, S, fmax, n=300):
    m = f <= fmax
    f, S = f[m], S[m]
    step = max(len(f) // n, 1)
    L = (len(f) // step) * step
    Sm = S[:L].reshape(-1, step).max(axis=1)
    fm = f[:L].reshape(-1, step).mean(axis=1)
    db = 20 * np.log10(Sm / (Sm.max() + 1e-30) + 1e-6)
    return dict(freq=np.round(fm, 1).tolist(), db=np.round(db, 1).tolist())


def demodulate(x, fs, spec, mod=None, symbol_rate=None, beta=0.35, family=None):
    """spec = params dict from spectral.analyze. Returns (result dict, bits array, internal dict)."""
    warnings = []
    fc, bw = spec.get("center_freq_hz"), spec.get("bandwidth_hz")
    if fc is None or bw is None:
        raise ValueError("No signal detected in the spectrum - cannot demodulate")
    xb = D.channelize(x, fs, fc, bw)

    if family is None:
        if mod in D.LINEAR:
            family, evid = "linear", {}
        elif mod in ("2FSK", "4FSK"):
            family, evid = "fsk", {}
        else:
            family, evid = detect_family(xb, fs, bw)
    else:
        evid = {}
    res = dict(family=family, family_evidence=evid, center_freq_hz=fc, bandwidth_hz=bw)

    # ------------------------------------------------------------ FSK
    if family == "fsk":
        Rs, pmr, spec_rate = D.estimate_symbol_rate_fsk(xb, fs, bw)
        if symbol_rate:
            Rs = float(symbol_rate)
        s, _ = D.demod_fsk(xb, fs, Rs)
        idx, cent, M = D.fsk_levels(s)
        if mod in ("2FSK", "4FSK"):
            M = MOD_ORDER[mod]
            idx, cent, M = _force_levels(s, M)
        k = int(np.log2(M))
        bits = ((idx[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).reshape(-1)
        hist, edges = np.histogram(s, bins=60)
        spacing = float(np.mean(np.diff(cent))) if len(cent) > 1 else 0.0
        res.update(
            modulation=f"{M}FSK", modulation_source="override" if mod else "tone clustering",
            symbol_rate=float(Rs), symbol_rate_pmr=round(float(pmr), 1), sps=float(fs / Rs),
            fsk=dict(tones_hz=np.round(cent, 1).tolist(), tone_spacing_hz=round(spacing, 1),
                     mod_index_h=round(spacing / Rs, 3),
                     hist=dict(counts=hist.tolist(), edges=np.round(edges, 1).tolist())),
            quality=dict(n_symbols=int(len(s))),
            bits=_bits_info(bits, ambiguity=1),
            plots=dict(rate_spectrum=_downsample_spec(*spec_rate, fmax=2.0 * bw)),
        )
        if M == 4 and spacing > 0 and spacing / Rs < 0.5:
            warnings.append("Tone spacing is small relative to symbol rate; 4FSK decisions may be unreliable")
        res["deep"] = _deep_opinion(xb, fs, Rs, f"{M}FSK", warnings)
        res["warnings"] = warnings
        return res, bits, dict(kind="fsk", syms=s, M=M, k=k, idx=idx)

    # ------------------------------------------------------------ linear
    Rs, pmr, spec_rate = D.estimate_symbol_rate_linear(xb, fs, bw)
    if symbol_rate:
        Rs = float(symbol_rate)
    y, tinfo = D.recover_symbols(xb, fs, Rs, beta)
    if mod in D.LINEAR:
        label, probs, feats, src = mod, {mod: 1.0}, {}, "override"
    else:
        label, probs, feats = get_classifier().predict(y)
        src = "random forest on cumulant / line features"
    y2, cinfo = D.carrier_recover(y, label)
    y2 = D.agc(y2)
    bits, evm = D.demap_linear(y2, label)
    # Verification by constellation fit. The classifier's features assume near-random data; structured
    # traffic (text, constant headers) biases symbol statistics and can fool it. A hypothesis that fits
    # badly (large EVM) is re-examined: the LOWEST-order modulation that fits about as well as the best wins.
    if mod not in D.LINEAR and -20 * np.log10(evm + 1e-9) < 10:
        fits = {}
        for H in D.LINEAR:
            try:
                yH, cH = D.carrier_recover(y, H)
                bH, eH = D.demap_linear(D.agc(yH), H)
                fits[H] = (eH, yH, cH, bH)
            except Exception:
                continue
        if fits:
            emin = min(v[0] for v in fits.values())
            ok = [h for h, v in fits.items() if v[0] <= max(1.5 * emin, emin + 0.03)]
            if label not in ok:
                order = ["BPSK", "QPSK", "8PSK", "16QAM"]
                new_label = [h for h in order if h in ok][0]
                warnings.append(f"Classifier said {label} but the constellation fits {new_label} much better "
                                "(structured data can bias the classifier features); using the better fit")
                label = new_label
                src = "constellation fit (overruled classifier)"
                eH, yH, cH, bH = fits[label][0], fits[label][1], fits[label][2], fits[label][3]
                y2, cinfo = D.carrier_recover(y, label)
                y2 = D.agc(y2)
                bits, evm = D.demap_linear(y2, label)
    Mrot = D.PSK_M[label]
    Rs_ref = tinfo["symbol_rate_refined"]
    cfo_hz = cinfo["cfo_cycles_per_symbol"] * Rs_ref
    esn0 = float(-20 * np.log10(evm + 1e-9))
    if esn0 < 8:
        warnings.append("Low symbol SNR - expect bit errors; classification confidence is reduced")
    if tinfo["timing_lock"] < 0.5:
        warnings.append("Weak timing lock - symbol rate estimate may be wrong; try overriding it")
    top = sorted(probs.items(), key=lambda kv: -kv[1])
    if len(top) > 1 and top[0][1] < 0.6:
        warnings.append("Classifier is not confident; consider selecting the modulation manually")

    deep = _deep_opinion(xb, fs, Rs_ref, label, warnings)
    npts = min(len(y2), 3000)
    sel = np.linspace(0, len(y2) - 1, npts).astype(int)
    pts = np.stack([y2[sel].real, y2[sel].imag], axis=1)
    res.update(
        modulation=label, modulation_source=src,
        class_probs={k_: round(v, 4) for k_, v in probs.items()}, features=feats,
        symbol_rate=float(Rs_ref), symbol_rate_coarse=float(Rs), symbol_rate_pmr=round(float(pmr), 1),
        sps=float(fs / Rs_ref), beta_assumed=beta,
        carrier=dict(residual_cfo_hz=round(float(cfo_hz), 2), mth_power=cinfo["M"],
                     line_pmr=round(cinfo["mth_power_pmr"], 1)),
        quality=dict(evm=round(evm, 4), esn0_db=round(esn0, 1),
                     timing_lock=round(tinfo["timing_lock"], 3), n_symbols=int(len(y2))),
        bits=_bits_info(bits, ambiguity=Mrot),
        plots=dict(constellation=np.round(pts, 3).tolist(),
                   eye=np.round(tinfo["eye"], 3).tolist(),
                   rate_spectrum=_downsample_spec(*spec_rate, fmax=2.0 * bw)),
        warnings=warnings, deep=deep,
    )
    return res, bits, dict(kind="linear", syms=y2, mod=label, M=Mrot, k=int(np.log2(MOD_ORDER[label])))


def _deep_opinion(xb, fs, Rs, chosen, warnings):
    """Second opinion from the CNN trained on public data (if its model files are installed)."""
    clf = get_deep()
    if not clf.available:
        return dict(available=False, reason=clf.reason)
    try:
        out = clf.predict(xb, fs, Rs)
    except Exception as e:
        return dict(available=True, error=str(e))
    top = out["top"][0]
    out["agrees"] = clf.agrees(top["label"], chosen)
    if not out["warnings_trusted"]:                      # informational only: labels unconfirmed or poor transfer
        out["note"] = ("Informational only: this CNN was trained on simulated RadioML data and has not been shown to "
                       "agree with ground truth on this tool's own signals" +
                       ("" if out["labels_verified"] else "; its class-name order is also not yet confirmed") + ".")
        return out
    if top["prob"] >= 0.6 and out["agrees"] is False:
        warnings.append(f"The CNN trained on {out['dataset']} suggests {top['label']} ({top['prob'] * 100:.0f}%), "
                        f"which disagrees with {chosen}. Treat the modulation as uncertain and compare the constellation.")
    if top["prob"] >= 0.6 and out["agrees"] is None:
        warnings.append(f"The CNN suggests {top['label']} ({top['prob'] * 100:.0f}%), a modulation this app cannot "
                        "demodulate yet. The result below may be meaningless.")
    return out


def _force_levels(s, M):
    c0 = np.quantile(s, (np.arange(M) + 0.5) / M)
    cent, lab = kmeans2(s.reshape(-1, 1), c0.reshape(-1, 1), minit="matrix", iter=30)
    order = np.argsort(cent[:, 0])
    return np.argsort(order)[lab].astype(int), cent[order, 0], M


def _bits_info(bits, ambiguity):
    return dict(count=int(len(bits)), preview="".join(map(str, bits[:512].tolist())),
                phase_ambiguity=int(ambiguity),
                note=("Bits are correct up to a %d-fold phase ambiguity and an unknown start offset; "
                      "the blind decoding stage below resolves both." % ambiguity) if ambiguity > 1 else
                     "Bit polarity is fixed by tone order; the start offset is resolved by the decoding stage below.")


def extract_features_from_iq(x, fs, spec):
    """Used by training: same path as inference, stops after timing recovery."""
    xb = D.channelize(x, fs, spec["center_freq_hz"], spec["bandwidth_hz"])
    Rs, _, _ = D.estimate_symbol_rate_linear(xb, fs, spec["bandwidth_hz"])
    y, _ = D.recover_symbols(xb, fs, Rs, 0.35)
    return features(y)
