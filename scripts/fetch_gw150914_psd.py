"""Fetch the GWTC-2.1 PE data release for GW150914 and extract its PSDs.

The PSD stored in a GWTC catalog PE metafile is the on-source noise estimate
(BayesWave) that the original LVK analysis actually conditioned on -- not a
generic design curve -- so this is the right PSD to use for both the real
GW150914 recovery and, to keep the two apples-to-apples, the simulated-data
validation campaign (see the "Load externally-generated (BayesWave) PSDs"
task).

Uses ``pesummary.gw.fetch`` to locate/download the Zenodo-hosted metafile and
``asimov-gwdata``'s ``datafind.metafiles.Metafile`` to extract the per-IFO PSD
(matching the workflow ``gwdata --settings <yaml>`` with ``data: [psds]``
would run) into plain two-column ASCII files readable by
``heron.inference.detectors.load_psd_ascii``.

Usage::

    python scripts/fetch_gw150914_psd.py
    python scripts/fetch_gw150914_psd.py --event GW150914 --analysis C01:IMRPhenomXPHM
"""
from __future__ import annotations

import argparse
import os
import tempfile

from datafind.metafiles import Metafile


def fetch_and_extract(
    event: str,
    catalog: str,
    analysis: str,
    outdir: str,
) -> dict[str, str]:
    from pesummary.gw.fetch import fetch_open_samples

    os.makedirs(outdir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Downloading {event} ({catalog}) PE data release...")
        metafile_path = fetch_open_samples(
            event, catalog=catalog, read_file=False, delete_on_exit=False,
            outdir=tmpdir,
        )
        print(f"Extracting PSDs from analysis '{analysis}'...")
        written = {}
        with Metafile(str(metafile_path)) as metafile:
            for ifo, psd in metafile.psd(analysis).items():
                path = os.path.join(outdir, f"{ifo}.dat")
                psd.to_ascii(path)
                written[ifo] = path
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="GW150914")
    parser.add_argument("--catalog", default="GWTC-2.1-confident")
    parser.add_argument("--analysis", default="C01:IMRPhenomXPHM")
    parser.add_argument(
        "--outdir", default=os.path.join("data", "psds", "GW150914"),
    )
    args = parser.parse_args()

    written = fetch_and_extract(args.event, args.catalog, args.analysis, args.outdir)
    for ifo, path in written.items():
        print(f"  {ifo}: {path}")


if __name__ == "__main__":
    main()
