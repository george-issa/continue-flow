# Changelog

## Unreleased

- Bins that stop at β − Δτ are read: G(β) = 1 − G(0) is appended
  (`--endpoint`, decided from the data by default).
- A spin-summed G, a transposed file and values that are not numbers are
  refused with the reason.

## 0.1.0

First tagged version.

- `spevae continue` with a flow checkpoint prints the posterior report: how
  well the samples fit the bins, the spectral weight per frequency window with
  its 50% and 90% intervals, the range the data alone allow, and warnings.
- Sharp lines are fitted as zero-width lines when the data reject peaks of the
  model's minimum width. `--discrete` states that the system is an atom, a
  molecule or a finite cluster.
- Trained range: fermionic G(τ), β from 1 to 20, relative error bars from 1e-5
  to 3e-2, Δτ from 0.02 to 0.5. A warning is printed outside it.
- Tested against exact answers at twelve settings and on three DQMC runs of
  exactly solvable clusters (`docs/TRAINING.md`, `figures/`).
- The earlier VAE model, the MaxEnt and DEAC comparison and the evaluation
  suite are included.
- Stated intervals are never negative and never outside the range the data
  alone allow.
- Checkpoints are read with `weights_only=True`.
- No trained checkpoint is published yet.
