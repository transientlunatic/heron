"""Version-robust save/load helpers for GPyTorch model state.

Two problems with a bare ``model.load_state_dict(state)``:

1. **Prior buffers are renamed between GPyTorch releases.** ``LogNormalPrior``
   stores its parameters as ``_transformed_loc``/``_transformed_scale`` in
   gpytorch 1.14.x and ``_buffered_loc``/``_buffered_scale`` in 1.15.x, so a
   checkpoint written under one fails to load under the other with
   "Missing/Unexpected key(s)". These buffers are not trained state: every
   surrogate's ``load()`` rebuilds its model from the saved training data and
   config, which reconstructs the priors exactly. So :func:`load_gp_state`
   sets prior entries aside (whatever their name) and keeps the reconstructed
   priors, cross-checking values where the two names can be matched.
2. **A load can "succeed" while restoring the wrong thing.**
   :func:`load_gp_state` stays strict on everything that is not a prior
   (trained raw parameters, constraint bounds, ``active_dims``, variational
   state), and :func:`hyperparameter_summary` / :func:`check_hyperparameters`
   let ``save()`` record the constrained hyperparameters by readable name and
   ``load()`` verify the restored model reproduces them.
"""

from __future__ import annotations

import logging
import math
import re

import torch

logger = logging.getLogger("heron.models.gp.checkpoint")

# A state-dict key belongs to a prior if any dotted path component is a
# registered prior module (gpytorch names them ``<something>_prior``).
_PRIOR_COMPONENT = re.compile(r"(^|\.)[A-Za-z0-9_]*_prior\.")
# Version-specific buffer-name prefixes, normalised away for cross-checking.
_PRIOR_BUFFER_PREFIX = re.compile(
    r"\._(?:transformed|buffered)_([A-Za-z0-9_]+)$"
)


def is_prior_key(key: str) -> bool:
    """True if ``key`` names a buffer/parameter inside a registered prior."""
    return bool(_PRIOR_COMPONENT.search(key))


def _normalise_prior_key(key: str) -> str:
    return _PRIOR_BUFFER_PREFIX.sub(r".\1", key)


def environment_versions() -> dict:
    """Library versions worth recording in a checkpoint."""
    import gpytorch

    return {"torch": torch.__version__, "gpytorch": gpytorch.__version__}


def load_gp_state(module: torch.nn.Module, state: dict, *, name: str = "model",
                  rtol: float = 1e-6) -> None:
    """Load ``state`` into ``module``, tolerant only of prior-buffer renames.

    Parameters
    ----------
    module
        The freshly constructed model (or likelihood) to restore into. Its
        priors must already be built from the same training data/config.
    state
        The saved ``state_dict()``.
    name
        Label used in error and warning messages.
    rtol
        Relative tolerance for the saved-vs-reconstructed prior cross-check.

    Raises
    ------
    RuntimeError
        If any non-prior entry is missing from ``state`` or unexpected in it.
    """
    expected = module.state_dict()
    saved_core = {k: v for k, v in state.items() if not is_prior_key(k)}
    expected_core = {k for k in expected if not is_prior_key(k)}

    missing = sorted(expected_core - set(saved_core))
    unexpected = sorted(set(saved_core) - expected_core)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint state for {name} does not match the model: "
            f"missing {missing}, unexpected {unexpected}. Only prior buffers "
            "are allowed to differ (they are rebuilt from the saved config)."
        )

    # Restore the trained state; prior buffers keep their reconstructed values.
    result = module.load_state_dict(saved_core, strict=False)
    leftover = [k for k in result.missing_keys if not is_prior_key(k)]
    if leftover or result.unexpected_keys:  # defensive; checked above
        raise RuntimeError(
            f"Inconsistent load for {name}: missing {leftover}, "
            f"unexpected {list(result.unexpected_keys)}"
        )

    # Cross-check priors whose names can be matched across versions.
    rebuilt = {
        _normalise_prior_key(k): v
        for k, v in expected.items()
        if is_prior_key(k)
    }
    renamed = 0
    for key, value in state.items():
        if not is_prior_key(key):
            continue
        norm = _normalise_prior_key(key)
        if key not in expected:
            renamed += 1
        ref = rebuilt.get(norm)
        if ref is None or not torch.is_tensor(value):
            continue
        if ref.shape != value.shape:
            continue
        if not torch.allclose(value.to(ref), ref, rtol=rtol, atol=0.0):
            logger.warning(
                "%s: saved prior %s=%s differs from the one rebuilt from the "
                "checkpoint config (%s); the model code or config may have "
                "changed since this checkpoint was written.",
                name, norm, value.tolist(), ref.tolist(),
            )
    if renamed:
        logger.info(
            "%s: %d prior buffer(s) saved under a different GPyTorch naming; "
            "using the reconstructed priors.", name, renamed,
        )


def hyperparameter_summary(module: torch.nn.Module) -> dict[str, list]:
    """Constrained (physical) hyperparameter values keyed by parameter path.

    Every ``raw_*`` parameter that has a registered constraint is reported
    through that constraint's transform (e.g. ``raw_lengthscale`` ->
    ``lengthscale``), so the record is independent of GPyTorch's internal
    parametrisation. Parameters without a constraint are reported as-is.
    Large tensors (e.g. inducing points, variational parameters) are skipped:
    this is a human-readable fingerprint, not a second copy of the state.
    """
    out: dict[str, list] = {}
    for pname, param in module.named_parameters():
        if param.numel() > 64:
            continue
        owner_path, _, leaf = pname.rpartition(".")
        owner = module.get_submodule(owner_path) if owner_path else module
        value = param.detach()
        label = pname
        if leaf.startswith("raw_"):
            constraint = getattr(owner, f"{leaf}_constraint", None)
            if constraint is not None:
                value = constraint.transform(value)
                prefix = f"{owner_path}." if owner_path else ""
                label = prefix + leaf[len("raw_"):]
        out[label] = value.double().flatten().tolist()
    return out


def check_hyperparameters(module: torch.nn.Module, saved: dict | None, *,
                          name: str = "model", rtol: float = 1e-5) -> None:
    """Verify ``module`` reproduces a :func:`hyperparameter_summary` record.

    A no-op when ``saved`` is ``None`` (checkpoints written before the
    record existed). Raises ``RuntimeError`` on any missing entry or value
    mismatch beyond ``rtol``.
    """
    if not saved:
        return
    current = hyperparameter_summary(module)
    problems = []
    for key, values in saved.items():
        if key not in current:
            problems.append(f"{key}: not present in the loaded model")
            continue
        for a, b in zip(values, current[key]):
            if not math.isclose(a, b, rel_tol=rtol, abs_tol=0.0):
                problems.append(
                    f"{key}: saved {values}, loaded {current[key]}"
                )
                break
    if problems:
        raise RuntimeError(
            f"Loaded {name} does not reproduce the saved hyperparameters: "
            + "; ".join(problems)
        )
