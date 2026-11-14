"""Mass-scaling test: h(M=70) vs rescaled h(M=60) (scale-equivalent f_min), aLIGO PSD.

Compares mismatch on raw LAL time axes, after placing each interpolated peak at
t=0, and with integer-sample time maximisation. Shows the ~1e-3 non-scaling
residual is the sub-sample peak-alignment convention.

LAL archaeology (manager task #399). Requires lalsuite. Conditioning code is
functionally identical between lalsuite 7.22 (our conda pin) and 7.26.16.
"""

import lal
import lalsimulation as ls
import numpy as np
from scipy.interpolate import CubicSpline

dT = 1 / 4096.0
MS = lal.MSUN_SI


def gen(apx, M, q, fmin):
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
    t = np.arange(len(hp.data.data)) * dT + float(hp.epoch)
    h = hp.data.data - 1j * hc.data.data
    a2 = abs(h) ** 2
    i = int(np.argmax(a2))
    y0, y1, y2 = np.log(a2[i - 1 : i + 2])
    off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * dT
    return t, h, off


def psd(f):
    out = np.full_like(f, np.inf)
    m = (f >= 25) & (f <= 1000)
    out[m] = np.array([ls.SimNoisePSDaLIGOZeroDetHighPower(x) for x in f[m]])
    return out


def overlap(h1, h2, maxtime):
    n = len(h1)
    F1 = np.fft.fft(h1)
    F2 = np.fft.fft(h2)
    f = np.abs(np.fft.fftfreq(n, dT))
    w = 1 / psd(f)
    z = (
        np.fft.ifft(np.conj(F1) * F2 * w) * n
        if maxtime
        else np.array([np.sum(np.conj(F1) * F2 * w)])
    )
    return np.max(np.abs(z)) / np.sqrt(
        np.sum(abs(F1) ** 2 * w).real * np.sum(abs(F2) ** 2 * w).real
    )


def test(apx, q, M=70.0, Mref=60.0, fmin=20.0):
    lam = M / Mref
    t1, h1, o1 = gen(apx, M, q, fmin)
    t0, h0, o0 = gen(apx, Mref, q, fmin * lam)  # scale-equivalent start frequency
    T = np.arange(-1.2, 0.06, dT)

    def onto(t, h, shift=0.0, s=1.0, a=1.0):
        sp_r = CubicSpline(t * s, h.real * a, extrapolate=False)
        sp_i = CubicSpline(t * s, h.imag * a, extrapolate=False)
        return np.nan_to_num(sp_r(T - shift) + 1j * sp_i(T - shift))

    win = np.where(T < -1.0, 0.5 - 0.5 * np.cos(np.pi * (T + 1.2) / 0.2), 1.0)
    A = onto(t1, h1) * win
    B = onto(t0, h0, s=lam, a=lam) * win
    A2 = onto(t1, h1, shift=-o1) * win
    B2 = onto(t0, h0, shift=-o0 * lam, s=lam, a=lam) * win  # put each true peak at t=0
    return (
        1 - overlap(A, B, False),
        1 - overlap(A2, B2, False),
        1 - overlap(A, B, True),
        o1 * 1e6,
        o0 * lam * 1e6,
    )


for apx in ("IMRPhenomD", "IMRPhenomXAS"):
    for q in (0.3, 0.5, 0.8):
        r = test(apx, q)
        print(
            f"{apx:12s} q={q}: mismatch raw-axes {r[0]:.2e} | true-peak-aligned {r[1]:.2e} | time-maximised {r[2]:.2e}   (peak offsets M70 {r[3]:+.0f}us, rescaled-M60 {r[4]:+.0f}us)"
        )
