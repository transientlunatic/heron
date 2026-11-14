"""Reproduce SimInspiralTDFromFD's duration/start-frequency/FFT-length bounds vs q (M=60, f_min=20 Hz, dt=1/4096).

Predicts the retained waveform length (5.060 s at q=0.10, 1.871 s at q=0.90),
f_start (~17.4-17.7 Hz) and where the power-of-two FFT length changes (q~0.154).

LAL archaeology (manager task #399). Requires lalsuite. Conditioning code is
functionally identical between lalsuite 7.22 (our conda pin) and 7.26.16.
"""

import lal
import lalsimulation as ls
import numpy as np

M = 60.0
fmin = 20.0
dT = 1 / 4096.0
MS = lal.MSUN_SI


def bounds(q):
    m1 = M / (1 + q) * MS
    m2 = M * q / (1 + q) * MS
    tch = ls.SimInspiralChirpTimeBound(fmin, m1, m2, 0, 0)
    s = ls.SimInspiralFinalBlackHoleSpinBound(0, 0)
    tm = ls.SimInspiralMergeTimeBound(m1, m2) + ls.SimInspiralRingdownTimeBound(
        m1 + m2, s
    )
    tex = 3 / fmin
    fst = ls.SimInspiralChirpStartFrequencyBound(1.1 * tch, m1, m2)
    tch2 = ls.SimInspiralChirpTimeBound(fst, m1, m2, 0, 0)
    n = round((tch2 + tm + 2 * tex) / dT)
    e = int(np.frexp(n)[1])
    N = 2**e
    keep = round((tch2 + tm) / dT)
    return fst, tm, N, keep


qs = np.round(np.arange(0.10, 0.991, 0.01), 3)
prev = None
for q in qs:
    fst, tm, N, keep = bounds(q)
    flag = "" if prev is None or N == prev else "  <-- FFT length changes"
    if q in (0.1, 0.2, 0.25, 0.28, 0.29, 0.3, 0.4, 0.5, 0.7, 0.9) or flag:
        print(
            f"q={q:.2f} fstart={fst:6.2f}Hz tmerge={tm*1e3:5.1f}ms Nfft={N} keep={keep*dT:6.3f}s{flag}"
        )
    prev = N
# fine scan for the exact boundary
f = lambda q: bounds(q)[2]
lo, hi = 0.10, 0.99
qq = np.linspace(lo, hi, 20001)
Ns = np.array([f(x) for x in qq])
for i in np.where(np.diff(Ns) != 0)[0]:
    print("boundary at q≈%.5f: %d -> %d" % (qq[i], Ns[i], Ns[i + 1]))
