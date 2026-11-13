"""Fine scan of LAL's peak-sample epoch convention (epoch = -dt*argmax(hp^2+hc^2)) across q~0.27-0.31.

Usage: python scripts/lal_peak_alignment.py IMRPhenomD|IMRPhenomXAS
Shows the true (log-parabola-interpolated) peak offset from the t=0 sample and
the one-sample jump at q~0.2915 (D) / q~0.2837 (XAS).

LAL archaeology (manager task #399). Requires lalsuite. Conditioning code is
functionally identical between lalsuite 7.22 (our conda pin) and 7.26.16.
"""

import sys

import lal
import lalsimulation as ls
import numpy as np

M = 60.0
fmin = 20.0
dT = 1 / 4096.0
MS = lal.MSUN_SI
apx = ls.GetApproximantFromString(sys.argv[1])


def gen(q):
    m1 = M / (1 + q) * MS
    m2 = M * q / (1 + q) * MS
    hp, hc = ls.SimInspiralChooseTDWaveform(
        m1,
        m2,
        0,
        0,
        0,
        0,
        0,
        0,
        100e6 * lal.PC_SI,
        0,
        0,
        0,
        0,
        0,
        dT,
        fmin,
        fmin,
        lal.CreateDict(),
        apx,
    )
    a2 = hp.data.data**2 + hc.data.data**2
    i = int(np.argmax(a2))
    y0, y1, y2 = np.log(a2[i - 1 : i + 2])
    d = (
        0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    )  # sub-sample offset of the true peak (log-parabola)
    t0 = float(hp.epoch)  # = -dT*maxind
    return t0, i, d * dT, np.sqrt(a2[i]), len(a2)


qs = np.round(np.arange(0.270, 0.3101, 0.001), 4)
rows = [gen(q) for q in qs]
print(" q      epoch(s)      len   peak_idx  subsample_off(us)  peakAmp")
for q, (t0, i, off, A, n) in zip(qs, rows):
    print(f"{q:.3f} {t0:+.6f} {n:6d} {i:6d} {off*1e6:+8.1f} {A:.4e}")
