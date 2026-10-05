"""Generate sample files + ground-truth JSON into ./samples
Run from project root:  python scripts/make_samples.py"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.core.generator import gen_signal, save_cf32, save_wav_iq

os.makedirs("samples", exist_ok=True)
CASES = [("qpsk_12db_cfo5k", "QPSK", 12, 5e3), ("bpsk_8db_cfo-10k", "BPSK", 8, -10e3),
         ("16qam_20db_cfo0", "16QAM", 20, 0), ("2fsk_15db_cfo3k", "2FSK", 15, 3e3)]
for name, mod, snr, cfo in CASES:
    x, t = gen_signal(mod, snr_db=snr, cfo_hz=cfo, seed=11)
    save_cf32(f"samples/{name}.cf32", x)
    save_wav_iq(f"samples/{name}.wav", x, t["fs"])
    t.pop("bits")
    json.dump(t, open(f"samples/{name}.truth.json", "w"), indent=2)
    print("wrote", name)


# ---- coded captures (frames -> Reed-Solomon -> convolutional code), for testing the Phase 3 decoder ----
from backend.core.chain import build_chain
CODED = [("coded_qpsk_rs204_conv_k7", "QPSK", 12, 4e3, dict(fec="rs_204_188", conv="conv_k7", n_frames=14)),
         ("coded_bpsk_conv_k7", "BPSK", 6, 1e3, dict(conv="conv_k7", n_frames=60)),
         ("coded_2fsk_plain_frames", "2FSK", 12, 2e3, dict(n_frames=40))]
for name, mod, snr, cfo, chain in CODED:
    bits, truth_c = build_chain(**chain)
    x, t = gen_signal(mod, snr_db=snr, cfo_hz=cfo, seed=11, bits=bits)
    save_cf32(f"samples/{name}.cf32", x)
    t.pop("bits")
    t["chain"] = {k: v for k, v in truth_c.items() if k != "message"}
    t["message_text"] = truth_c["message"].decode("ascii", "replace")
    json.dump(t, open(f"samples/{name}.truth.json", "w"), indent=2, default=str)
    print("wrote", name)
