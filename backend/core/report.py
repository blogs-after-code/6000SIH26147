"""
PDF analysis report for one session (parameters, plots, modulation, decoding, message, warnings).

Everything printed comes from the session's own measurements; nothing is filled in by hand. The honest-limits footer is
part of the report on purpose, so a printed page cannot be mistaken for a validated real-world result.
"""
import datetime
import io
import os

import numpy as np

TOOL_NAME = "SanketSetu"
TOOL_VERSION = "0.4.0"

FOOTER = ("Method limits: modulation, symbol-rate and code identification are estimates. Accuracy figures for this tool "
          "were measured on its own simulator, not on real captures. Interleaver type and size are operator-supplied; "
          "raw IQ sampling rate comes from the operator. Treat unconfirmed results as hypotheses.")


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    buf.seek(0)
    return buf


def _fig_psd(an, params):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, db = np.array(an["psd"]["freq"]) / 1e3, np.array(an["psd"]["db"])
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    ax.plot(f, db, lw=.8, color="#1f4e79")
    if params.get("band_low_hz") is not None:
        ax.axvspan(params["band_low_hz"] / 1e3, params["band_high_hz"] / 1e3, color="#f2a900", alpha=.25, label="detected band")
        ax.legend(fontsize=7, loc="upper right")
    ax.axhline(params["noise_floor_db"], color="grey", ls=":", lw=.8)
    ax.set(xlabel="frequency (kHz)", ylabel="PSD (dB)", title="Power spectrum")
    ax.grid(alpha=.3)
    return _png(fig)


def _fig_waterfall(an):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = an["waterfall"]
    z = np.array(w["db"]).reshape(w["rows"], w["cols"])
    f = np.array(w["freq"]) / 1e3
    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    ax.imshow(z, aspect="auto", origin="lower", cmap="magma", extent=[f[0], f[-1], 0, w["t_end"]])
    ax.set(xlabel="frequency (kHz)", ylabel="time (s)", title="Waterfall")
    return _png(fig)


def _fig_const(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if res.get("fsk"):
        h = res["fsk"]["hist"]
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        ax.stairs(h["counts"], h["edges"], color="#1f4e79", fill=True, alpha=.6)
        ax.set(xlabel="measured frequency (Hz)", ylabel="symbols", title="Tone histogram")
        return _png(fig)
    p = np.array(res["plots"]["constellation"])
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.scatter(p[:, 0], p[:, 1], s=2, alpha=.35, color="#1f4e79")
    ax.set(xlabel="I", ylabel="Q", title="Constellation", aspect="equal")
    ax.grid(alpha=.3)
    return _png(fig)


def _printable(text, n=500):
    t = "".join(ch if (ch.isprintable() or ch in "\n ") else "." for ch in str(text))
    return (t[:n] + " ...") if len(t) > n else t


def build_report(s: dict, analyzer) -> bytes:
    """s: session dict from backend.main.STORE. analyzer: backend.core.spectral.analyze."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=18, spaceAfter=2, alignment=0)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1f4e79"))
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9, leading=12)
    SM = ParagraphStyle("SM", parent=B, fontSize=7.5, leading=10, textColor=colors.HexColor("#555555"))
    MONO = ParagraphStyle("MONO", parent=B, fontName="Courier", fontSize=8, leading=10)

    def table(rows, widths=(55 * mm, 115 * mm)):
        t = Table([[Paragraph(str(a), B), Paragraph(str(b), B)] for a, b in rows], colWidths=widths)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .3, colors.HexColor("#cccccc")),
                               ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f3f3")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    spec, x, fs = s["spec"], s["x"], s["fs"]
    an = analyzer(x, fs)
    src = s.get("source") or {}
    story = [Paragraph(f"{TOOL_NAME} analysis report", H1),
             Paragraph(f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} | {TOOL_NAME} {TOOL_VERSION}", SM),
             Spacer(1, 4)]

    story += [Paragraph("1. Capture", H2), table([
        ("File", src.get("filename", "-")), ("Kind", src.get("kind", "-") + (f" ({src['fmt']})" if src.get("fmt") else "")),
        ("Sampling rate", f"{fs:,.0f} Hz"), ("Samples", f"{len(x):,} ({len(x) / fs:.3f} s)"),
        ("Notes", "; ".join(src.get("notes", [])) or "-")])]

    f = lambda k, u="", d=1: "-" if spec.get(k) is None else f"{spec[k]:,.{d}f}{u}"
    story += [Paragraph("2. Signal parameters (measured)", H2), table([
        ("Centre frequency", f("center_freq_hz", " Hz")), ("Occupied bandwidth (99 %)", f("bandwidth_hz", " Hz")),
        ("SNR, full band", f("snr_db_fullband", " dB")), ("SNR, in band", f("snr_db_inband", " dB")),
        ("Noise floor", f("noise_floor_db", " dB")), ("Power", f("power_dbfs", " dBFS")), ("DC offset", f("dc_offset", "", 3))]),
        Spacer(1, 4), Image(_fig_psd(an, spec), width=165 * mm, height=60 * mm),
        Image(_fig_waterfall(an), width=165 * mm, height=55 * mm)]

    r = s.get("demod")
    if r:
        q = r.get("quality") or {}
        rows = [("Modulation", f"{r['modulation']} (from: {r.get('modulation_source', '-')})"),
                ("Symbol rate", f"{r['symbol_rate']:,.1f} Hz ({r.get('sps', 0):.2f} samples/symbol)")]
        if q.get("esn0_db") is not None:
            rows += [("Symbol SNR (from EVM)", f"{q['esn0_db']} dB"), ("EVM", f"{q['evm'] * 100:.1f} %"),
                     ("Timing lock", f"{q['timing_lock'] * 100:.0f} %")]
        if r.get("carrier"):
            rows.append(("Residual carrier offset", f"{r['carrier']['residual_cfo_hz']} Hz"))
        if r.get("fsk"):
            rows += [("Tones", ", ".join(f"{t:.0f}" for t in r["fsk"]["tones_hz"]) + " Hz"),
                     ("Modulation index h", r["fsk"]["mod_index_h"])]
        rows.append(("Bits recovered", f"{r['bits']['count']:,} (correct up to the phase ambiguity noted below)"))
        if r.get("class_probs") and len(r["class_probs"]) > 1:
            rows.append(("Feature classifier", ", ".join(f"{k} {v * 100:.0f} %" for k, v in sorted(r["class_probs"].items(), key=lambda kv: -kv[1]))))
        d = r.get("deep") or {}
        if d.get("available") and d.get("top"):
            rows.append(("CNN (RadioML)", ", ".join(f"{t['label']} {t['prob'] * 100:.0f} %" for t in d["top"]) +
                         (f". {d['note']}" if d.get("note") else "")))
        gt = r.get("ground_truth_check")
        if gt:
            rows.append(("Ground truth (demo only)", f"BER {gt['ber']}" if gt.get("ber") is not None else gt.get("note", "")))
        story += [Paragraph("3. Modulation and demodulation", H2), table(rows), Spacer(1, 4), Image(_fig_const(r), width=80 * mm, height=80 * mm)]

    dec = s.get("decode_res")
    if dec:
        rows = []
        c, rs, fr, m = dec.get("conv"), dec.get("rs"), dec.get("frame"), dec.get("message")
        ld = dec.get("ldpc")
        if c:
            rows.append(("Convolutional code", f"rate {c.get('rate')}, K={c.get('K')}, generators {', '.join(c.get('generators_octal', []))} (octal), "
                         f"estimated channel BER {c.get('channel_ber')}" if c.get("found") else "not found"))
        if ld:
            rows.append(("LDPC", f"{ld.get('name')} ({ld.get('n')}, {ld.get('k')}), rate {ld.get('rate')}, "
                         f"{ld.get('converged')} of {ld.get('n_blocks')} blocks decoded, {ld.get('corrected_bits')} bit errors corrected, "
                         f"start offset {ld.get('offset')}, matrix {ld.get('source')}" if ld.get("found") else "not found"))
        if rs:
            rows.append(("Reed-Solomon", f"RS({rs.get('n')},{rs.get('k')}), t={rs.get('t')}, {rs.get('codewords')} codewords, "
                         f"{rs.get('corrected_symbols')} symbols corrected, {rs.get('uncorrectable')} uncorrectable" if rs.get("found") else "not found"))
        if fr:
            rows.append(("Frame structure", f"{fr.get('frame_len')} bits per frame, {fr.get('n_frames')} frames, sync {fr.get('sync_name', '')} "
                         f"0x{fr.get('sync_hex')}, header {fr.get('header_bits')} bits, payload {fr.get('payload_bits')} bits" if fr.get("found") else "not found"))
        gt = dec.get("ground_truth_check")
        if gt:
            rows.append(("Ground truth (demo only)", f"payload byte accuracy {gt.get('payload_byte_accuracy')}"))
        story += [Paragraph("4. De-interleaving, FEC decoding and framing", H2), table(rows or [("Result", "nothing decoded")])]
        if m and m.get("found"):
            story += [Paragraph(f"Recovered message ({m.get('bytes')} bytes; first 500 characters, non-printable bytes shown as '.')", B),
                      Paragraph(_printable(m.get("text", "")).replace("&", "&amp;").replace("<", "&lt;"), MONO)]
        for w in dec.get("warnings", []) or []:
            story.append(Paragraph("&bull; " + str(w).replace("&", "&amp;").replace("<", "&lt;"), B))

    warns = list((r or {}).get("warnings", []))
    story.append(Paragraph("5. Warnings from the analysis", H2))
    story += [Paragraph("&bull; " + w.replace("&", "&amp;").replace("<", "&lt;"), B) for w in warns] or [Paragraph("None raised.", B)]
    story += [Spacer(1, 10), Paragraph(FOOTER, SM)]

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                      title=f"{TOOL_NAME} report", author=TOOL_NAME).build(story)
    return buf.getvalue()
