"""Scoring helpers: compare received bits with ground truth despite the
unavoidable start offset and phase ambiguity of blind demodulation."""
import numpy as np


def ber_aligned(tx_bits, rx_bits, k, max_lag_syms=64):
    """Best BER over symbol lags in [-max_lag, max_lag]. Returns (ber, lag_symbols)."""
    tx = np.asarray(tx_bits, np.uint8)
    rx = np.asarray(rx_bits, np.uint8)
    best = (1.0, 0)
    for lag in range(-max_lag_syms, max_lag_syms + 1):
        s = lag * k
        a = tx[max(0, s): ]
        b = rx[max(0, -s): ]
        n = min(len(a), len(b))
        if n < 200:
            continue
        ber = float(np.mean(a[:n] != b[:n]))
        if ber < best[0]:
            best = (ber, lag)
    return best
