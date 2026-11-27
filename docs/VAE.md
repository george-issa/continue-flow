# The VAE model

The earlier model of this package. A variational autoencoder maps a noisy
G(τ) to one pole set: a transformer encoder reads G(τ) as a sequence of
time-slice tokens, and the decoder emits a simple pole expansion.

```
G(τ) from DQMC ──▶ transformer ──▶ q(z|G) ──▶ decoder ──▶ {ε_p − iγ_p,  a_p + i b_p}
  any β, any Δτ      encoder       (sample)                       │
                                                                  ▼
                                   A(ω) = −(1/π) Σ_p Im[ r_p / (ω − z_p) ]
```

The flow model described in the README uses the same encoder and the same pole
representation. It replaced the VAE as the model to use because it returns a
posterior with tested intervals. The VAE is kept as a baseline, and for
fine-tuning on a single dataset.

## Encoder

Every imaginary-time slice is one token: the value of G (standardised and
signed-log), its error bar, and position features of x = τ/β. A `CLS` token
adds β, L_τ, the noise level and the leading Matsubara coefficients.
Self-attention over the slices, with a padding mask for different lengths, lets
one set of weights read any grid. q(z|G) is read off the `CLS` output
(`src/spevae/models/encoder.py`, [`DESIGN.md`](DESIGN.md) §6).

With x = τ/β and ν = βω the fermionic kernel is K(x, ν) = 1/(e^{xν} +
e^{(x−1)ν}), which does not contain β. One set of weights serves every
simulation cell, and the number of time slices is a sequence length for the
encoder ([`DESIGN.md`](DESIGN.md) §3).

## Constraints

Built into the decoder's output layer (`src/spevae/models/constraints.py`,
asserted in `tests/`):

| property | mechanism |
|---|---|
| causality: all poles in the lower half-plane | `γ = γ_min + softplus(·)` |
| positive weights, Re r_p = a_p > 0 | `softmax` |
| sum rule, Σ Re r_p = ∫A dω = measured norm | `softmax` × N |
| zero-sum asymmetry, Σ Im r_p = 0 with b_p ≠ 0 | `b = a·(t − ⟨t⟩_a)`, `t = tanh(·)/2` |
| bounded asymmetry, \|b_p\| ≤ a_p | same map |
| 1/ω² tail with the physical coefficient | follows from Σ b_p = 0 |
| width floor γ ≥ 0.1 π/β | additive floor |
| G ≡ 0 ⇒ A ≡ 0, and A[cG] = c·A[G] | encoder sees G/N, decoder rescales by N |

Positivity of A(ω) at every ω cannot be built in per pole when the residues
are complex without giving up asymmetry. It is checked exactly instead:
`spevae.physics.positivity` reduces A ≥ 0 on the whole real axis to a real-root
count of a polynomial built from the pole parameters. Where A dips below zero
it locates the dip and integrates its weight in closed form. Every reported
spectrum carries the result. `residue_mode="real"` gives real residues, where
positivity holds by construction.

## Usage

```bash
# Stage 1: pretrain on the synthetic generator
spevae pretrain --preset fast --out out/model.pt --device mps

# continue your own bins
spevae continue --bins mybins.txt --beta 8.0 --checkpoint out/model.pt \
                --out result.npz --figure result.png

# Stage 2: adapt the weights to one dataset (no ground truth needed)
spevae finetune --bins mybins.txt --beta 8.0 --checkpoint out/model.pt \
                --out out/model-tuned.pt

# refine the latent code for this input (weights frozen), then merge poles
# below the resolution and refit the few that remain to the data
spevae continue --bins mybins.txt --beta 8.0 --checkpoint out/model-tuned.pt \
                --refine --consolidate --out result.npz
```

[`TUTORIAL.md`](TUTORIAL.md) walks through this on a real half-filled Hubbard
dataset.

`--figure` also writes the pole/residue-plane figure and the latent-space
collapse diagnostic.

From Python:

```python
import torch
from spevae import load_model

model = load_model("out/model.pt")
out = model.continue_(G, beta=10.0, tau=tau,
                      omega=torch.linspace(-8, 8, 2001), n_samples=200)
out["poles"]     # the result: evaluate .spectral(any_omega) later
out["A"]         # on the grid given;  out["A_std"] = spread of the latent samples
```

## Three training stages

1. **Pretrain** (`spevae.training.pretrain`) — supervised, on nine spectral
   families generated on the fly over wide ranges of β, Δτ, bandwidth and
   noise. There is no fixed training set. Held-out evaluation uses a frozen
   batch stream, so the curves are comparable across runs and checkpoints.
2. **Fine-tune** (`spevae.training.finetune`) — Stage 2 on your own bins:
   calibrated per-bin χ², free-bits KL, noise consistency across bins (the bins
   are independent noise realisations of one spectrum), the positivity penalty,
   and complete-monotonicity penalties on G, G″ and G⁗. By default only the
   encoder is trained. A divergence guard keeps outlier batches from wrecking a
   pretrained model.
3. **Refine** (`spevae.refine`) — MAP optimisation of z for one input with the
   weights frozen, stopping at the calibrated noise floor.

Different noise realisations of the same physics map to overlapping posteriors
(a symmetric-KL consistency term), and `A_std` shows where the data stop
determining the answer. Report `active_units` next to any band: a collapsed
encoder gives tight bands that mean nothing.

## Comparing with MaxEnt and DEAC

```bash
python scripts/benchmark.py --synthetic --checkpoint out/model.pt        # known answer
python scripts/benchmark.py --holstein 19 1 5 --checkpoint out/model.pt  # Holstein cube
python scripts/benchmark.py --bins mybins.txt --beta 8.0 --figure cmp.png
```

Three solvers with one interface, one forward map for scoring and one set of
metrics:

- **MaxEnt** — [`ana_cont`][anacont], vendored (MIT). Nothing to install.
- **SmoQyDEAC** — [`SmoQyDEAC.jl`][deac] through a Julia subprocess
  (`$SPEVAE_JULIA_PROJECT`, default `@deac`).
- **spevae** — this package, with the columns `certified_positive`,
  `min_A_exact` and `negative_weight` from the positivity check.

The three kernels are asserted equal to 1e-12 in the tests. Every result is
scored through one whitened covariance, so χ² means the same in every row.

## Reading the results

- **χ² at its target does not mean the spectrum is right.** It means the fit
  is consistent with the data at the noise level. The kernel maps a large
  family of spectra to the same G(τ) within the error bars.
- **The χ² of a perfect fit is not 1** with a shrunk covariance estimator.
  `spevae.data.chi2_target` returns the calibrated value (0.112 on a real
  site-Holstein cell).
- **`effective_poles`** (the participation ratio) is the pole count to report.
  `num_poles` is an upper bound.
- **`active_units`** belongs next to every uncertainty band.

## History

This is the third version. `SPE-AC-VAE` (v1) established the ansatz and the
calibrated χ² protocol. `SPE-AC-VAE-2` rewrote it without a frequency grid and
for any β. v3 started from an audit of v2 (13 defects found and fixed), made
complex residues the default with the exact positivity check, and added
Stage-2 fine-tuning, the readers for real data, the command line and the
pole and latent figures. The design and its measurements are in
[`DESIGN.md`](DESIGN.md).

[anacont]: https://doi.org/10.1016/j.cpc.2022.108519
[deac]: https://scipost.org/SciPostPhysCodeb.39
