"""Locate every one-sample t=0 jump over q=0.10-0.99 for IMRPhenomD and IMRPhenomXAS (M=60).

LAL archaeology (manager task #399). Requires lalsuite. Conditioning code is
functionally identical between lalsuite 7.22 (our conda pin) and 7.26.16.
"""

import lal
import lalsimulation as ls
import numpy as np

fmin = 20.0
dT = 1 / 4096.0
MS = lal.MSUN_SI


def gen(apx, M, q, fmin=20.0):
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
        ls.GetApproximantFromString(apx),
    )
    a2 = hp.data.data**2 + hc.data.data**2
    i = int(np.argmax(a2))
    y0, y1, y2 = np.log(a2[i - 1 : i + 2])
    return 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * dT


for apx in ("IMRPhenomD", "IMRPhenomXAS"):
    qs = np.round(np.arange(0.10, 0.991, 0.0025), 4)
    off = np.array([gen(apx, 60, q) for q in qs]) * 1e6
    j = np.where(np.abs(np.diff(off)) > 150)[0]
    print(
        f"{apx}: true-peak offset from t=0 sample spans {off.min():+.0f}..{off.max():+.0f} us; jumps at q≈",
        [f"{(qs[k]+qs[k+1])/2:.4f}" for k in j],
    )
    print(
        "   offsets at q=0.1,0.3,0.5,0.7,0.9:",
        [f"{off[np.argmin(abs(qs-x))]:+.0f}" for x in (0.1, 0.3, 0.5, 0.7, 0.9)],
    )
