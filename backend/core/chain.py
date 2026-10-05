"""
Transmit-side chain used to make test signals with a KNOWN answer:
   message -> frames (sync + header + payload) -> [Reed-Solomon] -> [LDPC] -> [convolutional code] -> [interleaver] -> bits
The receiver (decode.py) has to undo all of this blindly.
"""
import numpy as np

from .fec import conv as C
from .fec import ldpc as LD
from .fec import interleave as I
from .fec import rs as R

ASM = 0x1ACFFC1D
WORDS = ("ALPHA BRAVO CHARLIE DELTA ECHO FOXTROT GOLF HOTEL INDIA JULIET KILO LIMA MIKE NOVEMBER OSCAR PAPA "
         "QUEBEC ROMEO SIERRA TANGO UNIFORM VICTOR WHISKEY XRAY YANKEE ZULU telemetry status nominal packet "
         "uplink downlink carrier beacon sensor 0123456789").split()

RS_PRESETS = {                               # name: (n, k, primitive poly, first root)
    "rs_255_223": (255, 223, 0x11D, 0),
    "rs_204_188": (204, 188, 0x11D, 0),
    "rs_ccsds":   (255, 223, 0x187, 112),
    "rs_64_48":   (64, 48, 0x12B, 1),
}
CONV_PRESETS = {                             # name: (textbook octal generators, K)
    "conv_k3":  (("7", "5"), 3),
    "conv_k7":  (("171", "133"), 7),
    "conv_k7_r13": (("171", "133", "165"), 7),
    "conv_k9":  (("561", "753"), 9),
}


def make_message(n_bytes, seed=5):
    rng = np.random.default_rng(seed)
    out = b""
    while len(out) < n_bytes:
        out += (" ".join(rng.choice(WORDS, 40)) + " ").encode()
    return out[:n_bytes]


def build_frames(payload_len, n_frames, message):
    """Return list of bytes objects: ASM(4) + header(4: counter16, length, flags) + payload."""
    frames = []
    for i in range(n_frames):
        hdr = bytes([(i >> 8) & 255, i & 255, payload_len & 255, 1])
        pl = message[i * payload_len:(i + 1) * payload_len]
        pl = pl + bytes(payload_len - len(pl))
        frames.append(ASM.to_bytes(4, "big") + hdr + pl)
    return frames


def _bits(b):
    return np.unpackbits(np.frombuffer(b, np.uint8))


def build_chain(fec="none", conv="none", interleaver="none", ia=8, ib=8, n_frames=16, payload_len=24,
                message=None, seed=1, ldpc="none"):
    """fec: none or an RS preset; conv: none or a conv preset; interleaver: none/block/diagonal/convolutional/prandom.
    Returns (bits, truth)."""
    if fec != "none":
        n, k, prim, fcr = RS_PRESETS[fec]
        payload_len = k - 8
    message = message or make_message(n_frames * payload_len + 16)
    frames = build_frames(payload_len, n_frames, message)
    truth = dict(fec=fec, conv=conv, interleaver=interleaver, ia=ia, ib=ib, n_frames=n_frames,
                 payload_len=payload_len, message=message[: n_frames * payload_len],
                 frame_bytes=len(frames[0]))
    if fec != "none":
        words = [bytes(R.rs_encode(list(f), n, k, prim, fcr)) for f in frames]
        truth.update(rs=dict(n=n, k=k, prim=prim, fcr=fcr))
        stream = b"".join(words)
    else:
        stream = b"".join(frames)
    bits = _bits(stream)
    if ldpc != "none":
        if ldpc not in LD.BUILTIN:
            raise ValueError("unknown ldpc preset")
        code = LD.get_builtin(ldpc)
        bits = code.encode(bits)                                  # zero-padded to a whole number of blocks
        truth.update(ldpc_params=dict(name=ldpc, n=code.n, k=code.k, blocks=int(len(bits) // code.n)))
    truth["ldpc"] = ldpc
    if conv != "none":
        octs, K = CONV_PRESETS[conv]
        gens = [C.from_octal(o, K) for o in octs]
        bits = C.conv_encode(bits, gens, K)
        truth.update(conv_params=dict(octal=list(octs), K=K, rate=f"1/{len(octs)}"))
    if interleaver != "none":
        bits = I.interleave(bits, interleaver, ia, ib, seed)
    truth["n_bits"] = int(len(bits))
    return bits.astype(np.uint8), truth
