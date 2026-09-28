# Heron Training Pipeline

This document described the old two-stage training pipeline (training data generation + GPR model training via asimov/HTCondor). That pipeline has been replaced.

See the current training documentation:
- **Sphinx docs:** `docs/training.rst`
- **Example configs:** `examples/train_exact_gp.yaml`, `examples/train_sparse_active.yaml`, `examples/train_from_data.yaml`
- **Examples README:** `examples/README.md`

The new pipeline supports three modes (fixed grid, active learning, pre-existing data) and two model types (exact GP, sparse variational GP) via a single `heron train --settings config.yaml` command.
