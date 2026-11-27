# continue-flow

[![tests](https://github.com/george-issa/continue-flow/actions/workflows/tests.yml/badge.svg)](https://github.com/george-issa/continue-flow/actions/workflows/tests.yml)

Analytic continuation of imaginary-time Green's functions G(τ) from quantum
Monte Carlo to real-frequency spectral functions A(ω). The result is a set of
spectra that fit the data, reported with intervals on the spectral weight that
were tested against exact answers.

The Python package and the command are called `spevae`.

## Overview

Extracting A(ω) from G(τ) is an ill-posed inverse problem: many different
spectra reproduce the same G(τ) within its error bars. A method that returns
one curve hides this. This package returns many spectra that all fit the data,
and reports what they agree on.

```
G(τ) bins ──▶ flow model ──▶ 200 pole sets ──▶ fit each to the data ──▶ report
 any β, Δτ    (trained)       (samples)        (least squares)
```

1. A trained flow model draws 200 spectra given the measured G(τ). Each
   spectrum is a set of poles.
2. Each one is fitted to the data by least squares, starting from where the
   model put it.
3. The samples that fit within the noise are kept.
4. The report gives the spectral weight in each frequency window with its 50%
   and 90% intervals. Next to it is the range of weight that any non-negative,
   normalised spectrum consistent with the data could have. That second range
   uses no trained model.

## Model

- **Spectrum** — a sum of up to 12 poles in the lower half-plane,

  $$A(\omega) = -\frac{1}{\pi} \sum_p \text{Im}\!\left(\frac{r_p}{\omega - z_p}\right),$$

  evaluated at any ω. There is no frequency grid in training or in inference.
  A sharp line is a pole of zero width: a position and a weight.
- **Encoder** — a transformer with one token per time slice (the value of G,
  its error bar, and τ/β). With x = τ/β and ν = βω the fermionic kernel is
  1/(e^{xν} + e^{(x−1)ν}), which does not contain β, so one set of weights
  serves every β and Δτ in the trained range.
- **Flow** — a transformer over the pole tokens that attends to the encoder
  output, trained by flow matching on pole mixtures, smooth bands and exactly
  solvable models. About one million parameters.
- **Fit** — Levenberg–Marquardt on the pole parameters against the
  covariance-whitened χ² of the data.
- **Intervals** — quantiles of the samples, corrected by a table fitted on
  held-out draws of the training prior so that the 50% and 90% intervals hold
  the truth that often.
- **Range allowed by the data alone** — two linear programs per window over
  all non-negative, normalised spectra whose G(τ) stays within the error bars.

Every sample satisfies these by construction: causality (all poles in the
lower half-plane), positive weights that sum to the measured norm, a zero sum
of the imaginary parts of the residues (the 1/ω² tail), and a minimum width of
0.1π/β for every pole that is not a zero-width line. Positivity of A(ω) at
every ω is not imposed on the samples.

## Setup

```bash
conda env create -f environment.yml
conda activate spevae
pip install -e ".[dev]"
```

## Usage

```bash
spevae continue --bins mybins.txt --beta 8.0 --checkpoint flow.pt \
                --out report.npz --figure report.png
```

`--bins` is a whitespace-delimited text file with one row per bin and one
column per time slice, τ = 0 … β. A file that stops at β − Δτ is read too:
G(β) = 1 − G(0) is appended in every bin, and the printed summary says so. The
sign convention is detected from the sum rule G(0) + G(β) = 1. The run stops
with the reason if the data fit no convention: a spin-summed G, a transposed
file, a value that is not a number.

The output on real DQMC bins of the half-filled Hubbard dimer (U = 4, β = 8,
400 bins, the file in `src/spevae/evaluation/fixtures/`):

```
posterior from 200 of 200 samples (100% accepted, ...)
  fit      : chi2 / target median 0.62 [0.56, 0.70]
  weight per window: posterior 5-95%  |  allowed by the data alone  |  share of the bracket
    [-∞, -3)       0.000-0.097  |  0.000-0.226  |  0.43
    [-3, -1.5)     0.404-0.506  |  0.185-0.506  |  0.32
    [-1.5, -0.5)   0.000-0.017  |  0.000-0.155  |  0.11
    [-0.5, 0.5)    0.000-0.001  |  0.000-0.004  |  0.30
    [0.5, 1.5)     0.000-0.017  |  0.000-0.155  |  0.11
    [1.5, 3)       0.424-0.504  |  0.164-0.504  |  0.24
    [3, ∞)         0.000-0.079  |  0.000-0.237  |  0.33
  WARNING  : 97% of the spectral weight sat on poles at the model's minimum width, ...
```

The exact weights of this spectrum are 0.073, 0.426, 0.001, 0, 0.001, 0.426 and
0.073. All seven are inside their intervals.

- **fit** — how many samples fit the bins, and how well. A value of 1 is a fit
  as good as the noise allows.
- **weight per window** — the 5–95% interval of the samples, next to the range
  allowed by the data alone. The share is the ratio of the two widths. The
  smaller it is, the more of the answer comes from the training prior.
- **warnings** — narrow lines the model was not trained on, error bars smaller
  than the systematic errors, inputs outside the trained range, every pole in
  use.

`report.npz` holds the band of A(ω) on a grid and every accepted pole set.
`report.npz.json` holds the window table, the warnings and the diagnostics.

From Python:

```python
from spevae.data import read_text_bins
from spevae.models.flow import load_flow
from spevae.posterior import continue_measured

flow = load_flow("flow.pt")
bins = read_text_bins("mybins.txt", beta=8.0)
report = continue_measured(flow, bins)
print(report.summary())
```

Readers for other formats are in `spevae.data`: SmoQyDQMC simulation
directories, the Holstein cubes, and long-format `[tau_index, G, err]` text.

### Atoms, molecules and finite clusters

If the system has a discrete spectrum, say so:

```bash
spevae continue --bins mybins.txt --beta 8.0 --checkpoint flow.pt --discrete
```

The data often cannot tell a line from a narrow peak, or many close lines from
a continuous band. The statement settles it. Features at the minimum width are
fitted as zero-width lines, and the intervals allow every other feature to be
one sharp line. Do not use it for an extended system: it puts lines where
there are none and makes the intervals about ten times wider than needed.

Without the statement, zero-width lines are fitted only when the data reject
peaks of the model's minimum width. This happens at relative error bars near
1e-5, and the report says so in a warning.

### Checkpoint

No trained checkpoint is published yet. The training recipe is
`scripts/step3/run_lines.sh`, with the label bank from
`scripts/step3/build_physics_bank.py`. It takes about three hours on an Apple
GPU.

## Trained range

Fermionic G(τ), β from 1 to 20, relative error bars from 1e-5 to 3e-2, Δτ from
0.02 to 0.5, up to 401 time slices. Outside this range the report prints a
warning.

## Tests against exact answers

### Basic spectra

Lorentzians, Gaussians and pole sets with complex residues, with positions,
widths and weights drawn at random (10 spectra each). The error here is the
mean error of the weight in a frequency window, for this model / MaxEnt on the
same data. The last column is how often the stated 50% and 90% intervals hold
the exact weight, over four settings.

| spectrum | β = 8, error bar 1e-3 | β = 8, error bar 1e-4 | intervals hold |
|---|---|---|---|
| 1 Lorentzian | 0.0034 / 0.0080 | 0.0015 / 0.0073 | 0.64 / 0.98 |
| 2 Lorentzians | 0.0044 / 0.0138 | 0.0015 / 0.0091 | 0.63 / 0.99 |
| 3 Lorentzians | 0.0082 / 0.0130 | 0.0028 / 0.0088 | 0.62 / 0.95 |
| 1 Gaussian | 0.0130 / 0.0119 | 0.0079 / 0.0025 | 0.72 / 0.97 |
| 2 Gaussians | 0.0095 / 0.0113 | 0.0049 / 0.0031 | 0.68 / 0.99 |
| 3 Gaussians | 0.0109 / 0.0126 | 0.0055 / 0.0062 | 0.60 / 0.95 |
| pole set, 3 to 8 poles | 0.0075 / 0.0107 | 0.0040 / 0.0072 | 0.60 / 0.96 |

### Lattice and line spectra

The error is the area between the reconstructed and the exact cumulative
spectral weight (W₁, in units of ω; lower is better). MaxEnt is run on the same
data, with a flat default model. Lattice spectra are continuous (tight-binding
bands, BCS, Falicov–Kimball and others); line spectra are exactly diagonalised
atoms and clusters.

W₁ counts weight by how far it is misplaced, so it is sensitive to tails, which
the pole model has built in and MaxEnt does not. On the window-weight error of
the table above, this model has about half of MaxEnt's error on lattice spectra
at β = 8 and 16, and 0.8 of it at β = 2.

| setting | lattice: this model / MaxEnt | lower on | lines: this model / MaxEnt | lower on |
|---|---|---|---|---|
| β = 8, error bar 1e-3 | 0.090 / 0.229 | 18 of 20 | 0.209 / 0.280 | 10 of 12 |
| β = 8, error bar 1e-4 | 0.040 / 0.172 | 18 of 20 | 0.124 / 0.124 | 6 of 12 |
| β = 8, error bar 1e-2 | 0.302 / 0.578 | 20 of 20 | 0.507 / 0.702 | 12 of 12 |
| β = 2, error bar 1e-3 | 0.132 / 0.465 | 18 of 20 | 0.367 / 0.323 | 5 of 12 |
| β = 16, error bar 1e-3 | 0.108 / 0.213 | 17 of 20 | 0.185 / 0.333 | 11 of 12 |

Seven more settings at the edges of the trained range are in
`docs/TRAINING.md`. Over all twelve, the model has the lower error on 17 to 20
of 20 lattice spectra at every setting. On line spectra it is ahead of MaxEnt
at eight settings and level with it at four.

The stated 50% and 90% intervals hold the exact weight this often:

| spectra | 50% interval | 90% interval |
|---|---|---|
| held-out draws of the training prior | 0.50 | 0.90 |
| exact lattice spectra | 0.48 | 0.91 |
| exact line spectra that the report flags | 0.55 | 0.90 |
| exact line spectra that the report does not flag | 0.32 | 0.62 |
| the same, stated to be discrete | 0.68 | 0.95 |
| bands of 16 to 256 random lines, β = 8 | 0.35 | 0.75 |
| the same, stated to be discrete | 0.81 | 0.99 |

Three real DQMC runs of exactly solvable clusters (SmoQyDQMC, U = 4, 400 bins):

| run | this model | exact weight inside the 90% interval | with `--discrete` | inside | MaxEnt |
|---|---|---|---|---|---|
| Hubbard dimer, β = 8 | 0.319 | 7 of 7 windows | 0.291 | 5 of 7 | 0.777 |
| four-site Hubbard ring, β = 8 | 0.320 | 4 of 7 | 0.320 | 7 of 7 | 0.657 |
| Hubbard dimer, β = 2 | 0.413 | 7 of 7 | 0.342 | 7 of 7 | 0.350 |

The two misses of the dimer with `--discrete` are by 0.01 of the weight.

The figures for all of this are in [`figures/`](figures/README.md), and the
full record of what was measured is [`docs/TRAINING.md`](docs/TRAINING.md).

## Known limits

- **Spectra with many features that the data do not resolve.** A cluster, the
  local spectrum of a small lattice, a disordered system. The data look like a
  smooth band, the model was trained on smooth bands and on spectra of up to
  12 poles, and the report cannot tell. The intervals are too narrow and there
  is no warning. Measured on bands of 16 to 256 random lines at β = 8: the
  50% and 90% intervals hold the exact weight 0.35 and 0.75 of the time. With
  `--discrete` they hold it 0.81 and 0.99 of the time at four times the
  width. On the four-site ring the exact weight is inside the 90% interval in
  4 of 7 windows without `--discrete` and in 7 of 7 with it. The range allowed
  by the data alone held in every case.
- **Gaussian peaks at small error bars.** On a single Gaussian at relative
  error bars of 1e-4, MaxEnt is 2 to 3 times more accurate. At 1e-3 the two
  are level. The intervals hold in both cases. A Gaussian can also draw the
  warning about narrow lines, which is wrong for it: the fit uses its narrowest
  poles to make the core of the Gaussian.
- **A few sharp lines at β near 2 that are not stated to be discrete.** MaxEnt
  is about 1.1 times more accurate. With the statement the two are level.
- **Lines too close for the data to separate.** One fitted line stands in for
  the group, and MaxEnt's broad peak can be the closer of the two, by up to 2
  times on the worst case judged.
- **Continuous spectra at relative error bars of 1e-4 and below.** The 90%
  interval holds the exact weight 79–86% of the time. The misses are below
  0.005 of the total weight.
- **Scope.** Fermionic G(τ) only. No bosonic correlators, no self-energies.
- **The prior.** The intervals are a posterior under the training prior. They
  are tested on the spectra listed above and say nothing about a spectrum
  unlike all of them. The range allowed by the data alone does not have this
  limit: it held the exact weight in every window judged.

## Other parts of the package

- **The earlier VAE model** — a variational autoencoder that maps G(τ) to one
  pole set, with latent refinement and Stage-2 fine-tuning on a single dataset.
  See [`docs/VAE.md`](docs/VAE.md) and [`docs/TUTORIAL.md`](docs/TUTORIAL.md).
- **Comparison with MaxEnt and DEAC** — `scripts/benchmark.py` runs
  [`ana_cont`](https://doi.org/10.1016/j.cpc.2022.108519) MaxEnt (vendored)
  and [SmoQyDEAC](https://scipost.org/SciPostPhysCodeb.39) (through Julia) on
  the same data and scores all methods with the same code.
- **Evaluation suite** — `spevae eval` judges any method on a frozen set of
  synthetic truths and one real Holstein dataset. See
  [`docs/EVAL.md`](docs/EVAL.md).

## Repository layout

```
continue-flow/
├── src/spevae/
│   ├── posterior.py          # the posterior report
│   ├── models/               # flow model, encoder, VAE, constraint layer
│   ├── physics/              # kernel, poles → G(τ), least squares, positivity
│   ├── data/                 # readers, covariance, training priors, exact models
│   ├── evaluation/           # the evaluation suite and its two data files
│   ├── benchmark/            # MaxEnt and DEAC drivers
│   └── cli.py                # the spevae command
├── scripts/step3/            # training recipe and the judges of the current model
├── scripts/dqmc/             # DQMC runs of exactly solvable clusters (Julia)
├── scripts/figures/          # builds figures/
├── figures/                  # the figures, in reading order
├── docs/                     # TRAINING (what was measured), DESIGN, EVAL, VAE, TUTORIAL
└── tests/
```

## Tests

```bash
pytest -n auto
```

The suite checks the physics (kernel, sum rules, causality, the positivity
certificate), the readers, the fit, and the report. The tests of the report
need a trained checkpoint and are skipped without one.

## Method attribution

The pole–residue ansatz with the hardwired sum rule and zero-sum tail, the
covariance-whitened χ² calibration, the G/G″/G⁗ negativity penalties, and the
two-stage pretrain/fine-tune protocol of the VAE are due to **Benjamin
Cohen-Stead** (UT-Knoxville). References, including the closest published
prior work, are in [`docs/DESIGN.md`](docs/DESIGN.md) §10.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT.
