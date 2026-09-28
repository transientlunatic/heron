"""Fetch and condition real GW150914 strain data for heron.inference.

Downloads open H1/L1 strain around the GW150914 trigger from GWOSC (via
gwpy), conditions it (high-pass + Tukey edge taper — see
``heron.inference.strain`` for why both steps and in that order), and saves
the result as HDF5 files readable straight back into
``heron.inference.network.NetworkLikelihood``. This is the "real strain
segment ingestion + windowing" gap flagged in the GW150914 / heron recovery
task — a prerequisite for, not the same as, the full GW150914 recovery run
(that also needs the total-mass rescaling check, still unvalidated).

Usage::

    python scripts/fetch_gw150914_strain.py
    python scripts/fetch_gw150914_strain.py --sanity-check
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

GW150914_GPS = 1126259462.4


def fetch_and_save(
    detectors: list[str],
    trigger_time: float,
    duration: float,
    post_trigger_duration: float,
    sample_rate: float,
    f_low: float,
    roll_off: float,
    outdir: str,
) -> dict[str, str]:
    from heron.inference.strain import fetch_gwosc_strain

    os.makedirs(outdir, exist_ok=True)
    written = {}
    for det in detectors:
        print(f"Fetching {det} strain around GPS {trigger_time}...")
        strain, times = fetch_gwosc_strain(
            det, trigger_time, duration=duration,
            post_trigger_duration=post_trigger_duration,
            sample_rate=sample_rate, f_low=f_low, roll_off=roll_off,
        )
        path = os.path.join(outdir, f"{det}.hdf5")
        with h5py.File(path, "w") as f:
            f.create_dataset("strain", data=strain)
            f.create_dataset("times", data=times)
            f.attrs["trigger_time"] = trigger_time
            f.attrs["duration"] = duration
            f.attrs["post_trigger_duration"] = post_trigger_duration
            f.attrs["sample_rate"] = sample_rate
            f.attrs["f_low"] = f_low
            f.attrs["roll_off"] = roll_off
        written[det] = path
        print(
            f"  {det}: {len(times)} samples, "
            f"[{times[0]:.3f}, {times[-1]:.3f}] GPS -> {path}"
        )
    return written


def load_strain(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a strain HDF5 file written by :func:`fetch_and_save`."""
    with h5py.File(path, "r") as f:
        return np.asarray(f["strain"]), np.asarray(f["times"])


def sanity_check(strain_paths: dict[str, str], psd_dir: str, checkpoint: str) -> None:
    """Wire the ingested strain into NetworkLikelihood and confirm it runs.

    Deliberately NOT a GW150914 recovery: the checkpoint is trained at
    total_mass=60 and this evaluates it near GW150914's approximate total
    mass (~65) via the surrogate's total-mass rescaling, which the "GW150914
    / heron recovery" task explicitly flags as implemented-but-unvalidated.
    Sky location/psi below are the codebase's generic placeholder values
    (same as scripts/network_injection_ns.py's build_truth), not GW150914's
    real sky position. This only checks that real strain + real PSDs +
    the surrogate produce finite, sane, intrinsic-parameter-sensitive
    likelihoods end-to-end -- a wiring check, not a scientific result.
    """
    from heron.inference.detectors import Detector, load_psd_ascii
    from heron.inference.network import NetworkLikelihood
    from heron.models.gp.demod import DemodGPSurrogate

    print("\n--- sanity check: wiring real strain into NetworkLikelihood ---")
    surrogate = DemodGPSurrogate.load(checkpoint, device="cpu")

    data, times = {}, None
    detectors = []
    for det, path in strain_paths.items():
        strain, t = load_strain(path)
        data[det] = strain
        times = t if times is None else times
        psd_path = os.path.join(psd_dir, f"{det}.dat")
        detectors.append(Detector.from_name(det, psd_fn=load_psd_ascii(psd_path)))

    like = NetworkLikelihood(
        data=data, times=times, detectors=detectors, surrogate=surrogate,
        use_waveform_uncertainty=False,  # matched-filter: cheapest sane check
    )

    base = {
        "tc": GW150914_GPS, "ra": 1.95, "dec": -1.27, "psi": 0.82,
        "inclination": 0.5, "coalescence_phase": 1.1,
        "luminosity_distance": 440.0, "total_mass": 65.0,
    }
    print("Evaluating log-likelihood across a coarse mass_ratio scan "
          "(total_mass=65, matched-filter/no-K):")
    for q in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
        logl = like({**base, "mass_ratio": q})
        print(f"  q={q:.2f}: logL = {logl:.2f}")
    print("(finite, varying values above confirm the ingestion -> "
          "likelihood pipeline runs end-to-end; this is not a recovery.)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detectors", default="H1,L1")
    parser.add_argument("--trigger-time", type=float, default=GW150914_GPS)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--post-trigger-duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=float, default=4096.0)
    parser.add_argument("--f-low", type=float, default=20.0)
    parser.add_argument("--roll-off", type=float, default=0.4)
    parser.add_argument("--outdir", default=os.path.join("data", "strain", "GW150914"))
    parser.add_argument("--psd-dir", default=os.path.join("data", "psds", "GW150914"))
    parser.add_argument(
        "--checkpoint",
        default=os.path.join("checkpoints", "phenomd_nonspinning_dense30_demod.pt"),
    )
    parser.add_argument("--sanity-check", action="store_true")
    args = parser.parse_args()

    written = fetch_and_save(
        detectors=args.detectors.split(","),
        trigger_time=args.trigger_time,
        duration=args.duration,
        post_trigger_duration=args.post_trigger_duration,
        sample_rate=args.sample_rate,
        f_low=args.f_low,
        roll_off=args.roll_off,
        outdir=args.outdir,
    )

    if args.sanity_check:
        sanity_check(written, args.psd_dir, args.checkpoint)


if __name__ == "__main__":
    main()
