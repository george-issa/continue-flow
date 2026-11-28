r"""Reading real DQMC data.

The universal entry point is :func:`from_bins`: hand it an ``(N_bins, L_tau)``
array and :math:`\beta` and you get everything the model and the loss need --
the mean curve, per-slice standard deviations, a shrunk correlation estimate, its
whitener, and the *calibrated* :math:`\chi^2` target.  Everything else in this
module is a convenience reader that ends in the same object, so support for a new
file format is a function that returns bins, not a new code path through the
package.

Conventions checked, not assumed
--------------------------------
:class:`MeasuredGreens` verifies on construction that
:math:`G(0) + G(\beta) \approx 1` and reports the deviation.  That single check
catches the three mistakes that silently corrupt an analytic continuation:

* the **sign convention** (:math:`G = -\langle \mathcal{T} cc^\dagger\rangle`
  instead of :math:`+`),
* a **missing endpoint** (a grid running to :math:`\beta - \Delta\tau`, so the
  norm cannot be read off and the estimator's homogeneity is lost),
* a **normalisation** other than one (a correlator that is not a single-particle
  fermionic Green's function).

Endpoint-inclusive grids
------------------------
This package wants :math:`\tau = 0 \ldots \beta` inclusive.  Some DQMC codes
store :math:`L_\tau = \beta/\Delta\tau` slices ending at :math:`\beta-\Delta\tau`;
:func:`from_bins` will reconstruct the missing endpoint from antiperiodicity when
told it is absent, but supplying it is better.  Note that :math:`\tau=\beta` is
linearly dependent on :math:`\tau=0` through the sum rule, so it must be excluded
from the covariance whitening -- :attr:`MeasuredGreens.n_independent` carries
that bookkeeping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .noise import chi2_target, ledoit_wolf_correlation, whitener

__all__ = [
    "autocorrelation_report",
    "bin_statistics",
    "MeasuredGreens",
    "from_bins",
    "read_text_bins",
    "read_holstein_cube",
    "holstein_cube_axes",
]


@dataclass
class MeasuredGreens:
    r"""Binned DQMC :math:`G(\tau)` for one simulation cell, plus its noise model.

    Attributes
    ----------
    bins
        ``(N, L)`` raw bin measurements.
    tau
        ``(L,)`` imaginary times, ascending, normally ``0 … beta`` inclusive.
    beta
        Inverse temperature.
    includes_endpoints
        Whether ``tau[0] == 0`` and ``tau[-1] == beta``.
    reference
        Optional ``(omega, A)`` from another solver (e.g. a shipped SmoQyDEAC
        continuation).  Kept explicitly labelled as a *reference*, never as
        ground truth: comparing against it is a three-way agreement test between
        analytic-continuation methods.
    """

    bins: torch.Tensor
    tau: torch.Tensor
    beta: float
    label: str = ""
    includes_endpoints: bool = True
    reference: tuple[torch.Tensor, torch.Tensor] | None = None
    meta: dict = field(default_factory=dict)

    _cov: dict | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.bins.ndim != 2:
            raise ValueError(f"bins must be (N, L), got {tuple(self.bins.shape)}")
        if self.bins.shape[1] != self.tau.numel():
            raise ValueError(
                f"bins has {self.bins.shape[1]} time slices but tau has "
                f"{self.tau.numel()}")
        if self.bins.shape[0] < 2:
            raise ValueError("need at least two bins to estimate a covariance")

    # -- basic quantities ---------------------------------------------------- #

    @property
    def n_bins(self) -> int:
        return int(self.bins.shape[0])

    @property
    def n_tau(self) -> int:
        return int(self.bins.shape[1])

    @property
    def x(self) -> torch.Tensor:
        r"""Dimensionless times :math:`\tau/\beta`."""
        return self.tau / self.beta

    @property
    def mean(self) -> torch.Tensor:
        return self.bins.mean(0)

    @property
    def sigma(self) -> torch.Tensor:
        r"""Per-slice standard deviation of a *single bin*."""
        return self.bins.std(0, unbiased=True).clamp_min(1e-30)

    @property
    def sigma_mean(self) -> torch.Tensor:
        r"""Standard error of the bin mean, :math:`\sigma/\sqrt{N}`."""
        return self.sigma / math.sqrt(self.n_bins)

    @property
    def sigma_encoder(self) -> torch.Tensor:
        r"""Per-slice noise as the *encoder input*: floored at
        :math:`10^{-6}\max\sigma`.

        On PH-symmetric data the pinned endpoints have
        :math:`\sigma \sim 10^{-17}`, and the encoder's relative
        log-:math:`\sigma` channel would see a value ~30 natural logs outside
        anything in its training distribution.  The floor only touches pinned
        slices; use the raw :attr:`sigma`/:attr:`sigma_mean` for any
        :math:`\chi^2` (which excludes those slices instead).
        """
        return self.sigma.clamp_min(1e-6 * float(self.sigma.max()))

    @property
    def n_independent(self) -> int:
        r"""Time slices to whiten with.

        On an endpoint-inclusive grid :math:`\tau=\beta` is linearly dependent on
        :math:`\tau=0` through :math:`G(0)+G(\beta)=1`, which makes the full
        covariance singular; drop it.
        """
        return self.n_tau - 1 if self.includes_endpoints else self.n_tau

    @property
    def statistical_slices(self) -> torch.Tensor:
        r"""Indices of the slices that actually carry statistical noise.

        Beyond the :math:`\tau=\beta` redundancy, a slice whose bin variance is
        *numerically zero* is an exact constraint, not a measurement -- on
        particle-hole-symmetric data :math:`G(0) = G(\beta) = 1/2` is pinned to
        machine precision by the algorithm itself.  Dividing a residual by that
        :math:`\sigma \sim 10^{-17}` turns :math:`\chi^2` into
        :math:`10^{27}` of numerical noise, so pinned slices are excluded from
        the whitened block (their content already enters through the measured
        norm), and deviations there are an exact-constraint check, not a
        statistics one -- see :meth:`pinned_deviation`.
        """
        s = self.sigma[: self.n_independent]
        keep = s > 1e-8 * float(s.max())
        return torch.nonzero(keep).flatten()

    def pinned_deviation(self, G_recon: torch.Tensor) -> float:
        r"""Largest violation of the exactly-pinned slices, if any exist."""
        n = self.n_independent
        s = self.sigma[:n]
        pinned = torch.nonzero(s <= 1e-8 * float(s.max())).flatten()
        if pinned.numel() == 0:
            return 0.0
        g = G_recon.reshape(-1, self.n_tau)[:, :n]
        ref = self.mean[:n].to(g.dtype)
        return float((g[:, pinned] - ref[pinned]).abs().max())

    # -- noise model --------------------------------------------------------- #

    def covariance(self, min_eigenvalue: float = 0.0,
                   method: str = "lw") -> dict:
        r"""The noise model the reconstruction loss needs, plus its calibration.

        Returns ``sigma``, ``inv_sqrt_R``, ``rho``, ``chi2_target`` and
        ``n_independent``.

        ``chi2_target`` is the value a *perfect* fit gives under this whitener.
        It is **not 1** in general, and on real DQMC data it can be far from 1:
        with :math:`N_{\rm bins} \lesssim L_\tau` the sample correlation is
        rank-deficient, Ledoit--Wolf barely shrinks a strongly correlated matrix
        (its identity target is far away, so the optimal intensity is small), and
        the near-null directions dominate.  Compare a converged run against this
        number.

        ``min_eigenvalue`` floors the eigenvalues of the correlation estimate,
        bounding the whitener's amplification by :math:`1/\sqrt{\text{floor}}`.
        Raise it when the bin count is not comfortably above the number of time
        slices, and report the value used.
        """
        key = (float(min_eigenvalue), method)
        cache = self._cov or {}
        if key not in cache:
            slices = self.statistical_slices
            b = self.bins[:, slices]
            if method == "structured":
                from .noise import structured_shrunk_correlation
                sigma, R_lw, rho, xi = structured_shrunk_correlation(
                    b, self.tau[slices], min_eigenvalue=key[0])
            else:
                sigma, R_lw, rho = ledoit_wolf_correlation(b,
                                                           min_eigenvalue=key[0])
                xi = None
            Z = (b - b.mean(0, keepdim=True)) / sigma
            R_sample = (Z.T @ Z) / b.shape[0]
            cache[key] = {
                "slices": slices,
                "sigma": sigma,
                # Both R and its inverse square root are kept.  A consumer that
                # needs the covariance (ana_cont takes one) must not have to
                # invert the whitener to get it back: near-singular R makes that
                # round trip lose precision exactly where the matrix matters.
                "R": R_lw,
                "inv_sqrt_R": whitener(R_lw),
                "rho": rho,
                "chi2_target": chi2_target(R_lw, R_sample),
                # Count of whitened degrees of freedom.  Equal to the leading
                # block length when no slice is exactly pinned; smaller on
                # PH-symmetric data.  chi2 consumers should select columns
                # with "slices" rather than truncating to the leading n.
                "n_independent": int(slices.numel()),
                "min_eigenvalue": key[0],
                "method": method,
                "xi": xi,
            }
            object.__setattr__(self, "_cov", cache)
        return cache[key]

    def check_sum_rule(self, tol: float = 5e-3) -> dict:
        r"""Verify :math:`G(0) + G(\beta) = 1` and report the deviation.

        Returns a dict with ``norm``, ``deviation`` and ``ok``.  A large
        deviation almost always means a sign-convention or endpoint problem
        rather than bad data -- see the module docstring.
        """
        if not self.includes_endpoints:
            return {"norm": float("nan"), "deviation": float("nan"), "ok": False,
                    "note": "grid excludes tau=beta; norm not directly readable"}
        norm = float((self.mean[0] + self.mean[-1]).item())
        return {"norm": norm, "deviation": abs(norm - 1.0),
                "ok": abs(norm - 1.0) < tol}

    def summary(self, min_eigenvalue: float = 0.0) -> str:
        sr = self.check_sum_rule()
        cov = self.covariance(min_eigenvalue)
        rho, e_chi2 = cov["rho"], cov["chi2_target"]
        return (
            f"{self.label or 'MeasuredGreens'}\n"
            f"  beta      : {self.beta:g}   dtau = {float(self.tau[1]-self.tau[0]):g}\n"
            f"  grid      : {self.n_tau} slices, "
            f"{'endpoint-inclusive' if self.includes_endpoints else 'tau < beta'}\n"
            f"  bins      : {self.n_bins}\n"
            f"  G(0)      : {float(self.mean[0]):.6f} +- {float(self.sigma[0]):.2e}\n"
            f"  G(beta)   : {float(self.mean[-1]):.6f} +- {float(self.sigma[-1]):.2e}"
            f"{'   (appended as 1 - G(0): the bins stop at beta - dtau)' if self.meta.get('endpoint_appended') else ''}\n"
            f"  sum rule  : G(0)+G(beta) = {sr['norm']:.6f} "
            f"({'ok' if sr['ok'] else 'CHECK CONVENTION'})\n"
            f"  noise     : sigma(beta/2) = {float(self.sigma[self.n_tau//2]):.2e} "
            f"per bin, {float(self.sigma_mean[self.n_tau//2]):.2e} on the mean\n"
            f"  LW rho    : {rho:.4f}  (floor {min_eigenvalue:g})"
            f"  ->  chi2 at a perfect fit = {e_chi2:.4f}\n"
            f"  bins/slices: {self.n_bins}/{self.n_independent}"
            f"{'   *** N_bins <= L_tau: raise min_eigenvalue ***' if self.n_bins <= self.n_independent else ''}\n"
            f"  reference : {'yes' if self.reference is not None else 'none'}"
        )


def from_bins(
    bins,
    beta: float,
    tau=None,
    label: str = "",
    sign_flip: bool | None = None,
    includes_endpoints: bool | None = None,
    reference=None,
    dtype: torch.dtype = torch.float32,
    endpoint: str = "auto",
) -> MeasuredGreens:
    r"""Build a :class:`MeasuredGreens` from raw bins -- the universal entry point.

    Parameters
    ----------
    bins
        ``(N_bins, L_tau)`` array-like.
    beta
        Inverse temperature.
    tau
        Times; defaults to a uniform grid inferred from ``includes_endpoints``.
    sign_flip
        Negate the data, for sources that store
        :math:`-\langle \mathcal{T} c c^\dagger\rangle`.  ``None`` (default)
        **detects it**: the sum rule :math:`G(0)+G(\beta)=1` is satisfied by
        exactly one of the two signs, so there is nothing to guess and no reason
        to make the user guess.  Pass ``True``/``False`` to force it; the choice
        is recorded in ``meta['sign_flip']`` either way, and
        :meth:`MeasuredGreens.check_sum_rule` will still say if neither sign
        works (which means the problem is elsewhere -- a missing endpoint, or a
        correlator that is not a fermionic single-particle :math:`G`).
    includes_endpoints
        Whether the grid ends at :math:`\beta`.  ``None`` infers it: an
        endpoint-inclusive uniform grid has :math:`\Delta\tau = \beta/(L-1)`,
        and the two conventions are distinguished by which choice makes
        :math:`G(0)+G(\beta)` come out at 1.
    endpoint
        What to do about :math:`\tau = \beta` when neither ``tau`` nor
        ``includes_endpoints`` is given.  Many codes write
        :math:`\tau = 0 \dots \beta - \Delta\tau`.  For a fermionic
        single-particle :math:`G` the missing value is known exactly,
        :math:`G(\beta) = 1 - G(0)` in every bin, and with ``"missing"`` it is
        appended as a last column (``meta["endpoint_appended"]``).
        ``"included"`` takes the last column as :math:`\tau = \beta`.
        ``"auto"`` (default) decides from the data, see
        :func:`endpoint_convention`, and raises when they fit neither reading.
    """
    b = torch.as_tensor(np.asarray(bins), dtype=dtype)
    if b.ndim == 2 and not bool(torch.isfinite(b).all()):
        bad = (~torch.isfinite(b)).nonzero()
        raise ValueError(f"the bins hold {len(bad)} values that are not finite numbers (the first "
                         f"in bin {int(bad[0, 0])}, time slice {int(bad[0, 1])})")
    if sign_flip is None:
        norm = float((b[:, 0] + b[:, -1]).mean())
        sign_flip = abs(-norm - 1.0) < abs(norm - 1.0)
    if sign_flip:
        b = -b

    appended = False
    if includes_endpoints is None and tau is None:
        if endpoint not in ("auto", "included", "missing"):
            raise ValueError(f"endpoint must be 'auto', 'included' or 'missing', not {endpoint!r}")
        if (endpoint_convention(b) if endpoint == "auto" else endpoint) == "missing":
            b = torch.cat([b, 1.0 - b[:, :1]], dim=1)          # G(beta) = 1 - G(0), bin by bin
            appended = True
        includes_endpoints = True
    elif includes_endpoints is None:
        includes_endpoints = True

    if tau is None:
        L = b.shape[1]
        tau = (torch.linspace(0.0, beta, L, dtype=dtype) if includes_endpoints
               else torch.arange(L, dtype=dtype) * (beta / L))
    else:
        tau = torch.as_tensor(np.asarray(tau), dtype=dtype)

    if reference is not None:
        ws, A = reference
        reference = (torch.as_tensor(np.asarray(ws), dtype=dtype),
                     torch.as_tensor(np.asarray(A), dtype=dtype))

    mg = MeasuredGreens(bins=b, tau=tau, beta=float(beta), label=label,
                        includes_endpoints=bool(includes_endpoints),
                        reference=reference)
    mg.meta["sign_flip"] = bool(sign_flip)
    mg.meta["endpoint_appended"] = appended
    return mg


def endpoint_convention(bins: torch.Tensor) -> str:
    r"""Does the last column of the bins sit at :math:`\tau = \beta` or one step before it?

    Returns ``"included"`` or ``"missing"``; raises ``ValueError`` when the
    data fit neither.  The bins are in the positive convention.

    If the grid stops at :math:`\beta - \Delta\tau`, the value
    :math:`G(\beta) = 1 - G(0)` lies one step beyond the last column, and that
    step continues the curve: it has the sign of the step before it and between
    half and four times its size (:math:`G` is a sum of exponentials, so its
    steps change by a factor :math:`e^{\pm\omega\Delta\tau}` from one to the
    next).  If the grid ends at :math:`\beta`, the sum rule already holds and
    the step is zero.  Anything else is not a fermionic single-particle
    :math:`G` with unit norm on a uniform grid (a spin-summed :math:`G`,
    another correlator, a transposed file), or the grid is too coarse to tell,
    and guessing would give a confident, wrong spectrum.

    Data that are flat near :math:`\beta` within their noise satisfy both
    readings and are read as ``"included"``; state the convention for those.
    """
    b = bins.double()
    n, L = b.shape
    if L < 3:
        return "included"
    step = 1.0 - b[:, 0] - b[:, -1]                         # to G(beta) = 1 - G(0) from the last column
    last = b[:, -1] - b[:, -2]                              # the step before it
    err = 4.0 * float(step.std(unbiased=True)) / math.sqrt(n) + 1e-9
    step_m, last_m = float(step.mean()), float(last.mean())
    if abs(step_m) <= max(err, 0.25 * abs(last_m)):
        return "included"
    if last_m != 0.0 and 0.5 <= step_m / last_m <= 4.0:
        return "missing"
    norm = 1.0 - step_m
    hint = ""
    if abs(norm - 2.0) < 0.05:
        hint = " G(0) + G(beta) = 2 is what a spin-summed G gives: divide the bins by 2."
    elif float(b.std(0).mean()) > float(b.std(1).mean()):
        # rows that differ from one another more than a row varies along itself: rows are time slices
        hint = (f" The {n} rows differ from one another more than each varies along its {L} columns: "
                "one row per bin is expected, so the file may be transposed.")
    raise ValueError(
        f"G(0) + G(last column) = {norm:.6f}. The data fit neither a grid that ends at beta (the sum "
        f"would be 1) nor one that stops at beta - dtau (then 1 - G(0) = {1.0 - float(b[:, 0].mean()):.6f} "
        f"would continue the curve one step beyond the last column, {float(b[:, -1].mean()):.6f})."
        + hint + " Guessing would give a confident, wrong spectrum. If the grid is known, state it: "
        "--endpoint included or --endpoint missing (endpoint= in Python).")


def read_text_bins(path: str, beta: float, sign_flip: bool | None = None,
                   **kw) -> MeasuredGreens:
    r"""Whitespace-delimited bins, one row per bin.

    The sign convention is detected from the sum rule by default (see
    :func:`from_bins`); plain-text dumps come in both, and a wrong guess produces
    a confident, wrong spectrum rather than an error.
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return from_bins(data, beta, label=path, sign_flip=sign_flip, **kw)


# --------------------------------------------------------------------------- #
# Site-Holstein cube (Julia / JLD2, read through h5py)
# --------------------------------------------------------------------------- #

#: Parameter axes of the packed site-Holstein cube.
HOLSTEIN_DTAU = 0.1
HOLSTEIN_N = [round(0.05 * i, 10) for i in range(1, 21)]
HOLSTEIN_OMEGA = [0.5, 1.0, 1.5, 2.0]
HOLSTEIN_BETA = [float(b) for b in range(5, 21)]


def holstein_cube_axes() -> dict:
    """The cube's parameter grids, for building sweeps."""
    return {"n": list(HOLSTEIN_N), "omega": list(HOLSTEIN_OMEGA),
            "beta": list(HOLSTEIN_BETA), "dtau": HOLSTEIN_DTAU}


def read_holstein_cube(
    path: str,
    n_idx: int,
    omega_idx: int,
    beta_idx: int,
    with_reference: bool = True,
    dtype: torch.dtype = torch.float32,
) -> MeasuredGreens:
    r"""One cell of the packed site-Holstein cube.

    JLD2 is column-major, so ``h5py`` sees every axis reversed relative to
    Julia's view; the transpose on read restores the physical order
    ``(n, Omega, beta, tau, bin)``.

    The stored :math:`\tau` axis is zero-padded beyond :math:`10\beta + 1`
    points, so the padding is sliced off.  Unlike some earlier tooling this
    keeps :math:`\tau = \beta`: that endpoint is what makes the spectral norm
    readable and the estimator exactly 1-homogeneous.  It is excluded from the
    covariance instead, via :attr:`MeasuredGreens.n_independent`.

    ``with_reference`` attaches the shipped SmoQyDEAC continuation when the file
    has one -- as a *reference*, not ground truth.
    """
    import h5py

    beta = HOLSTEIN_BETA[beta_idx]
    n_valid = int(round(beta / HOLSTEIN_DTAU)) + 1        # endpoint-inclusive

    with h5py.File(path, "r") as f:
        G_r = f["G_r"][()].transpose()                    # (n, Om, beta, tau, bin)
        raw = np.array(G_r[n_idx, omega_idx, beta_idx, :n_valid, :])
        ref = None
        if with_reference and "dos" in f and "ws" in f:
            dos = f["dos"][()].transpose()
            ws = f["ws"][()]
            A = np.array(dos[n_idx, omega_idx, beta_idx])
            if np.any(A != 0.0):
                ref = (np.asarray(ws), A)

    label = (f"holstein n={HOLSTEIN_N[n_idx]:.2f} "
             f"Omega={HOLSTEIN_OMEGA[omega_idx]:.1f} beta={beta:g}")
    mg = from_bins(raw.T.copy(), beta, label=label, includes_endpoints=True,
                   reference=ref, dtype=dtype)
    mg.meta.update(n=HOLSTEIN_N[n_idx], omega=HOLSTEIN_OMEGA[omega_idx],
                   beta=beta, dtau=HOLSTEIN_DTAU, source=path)
    return mg


#: Excess kurtosis above which the bins of a time slice count as far from Gaussian.
HEAVY_KURTOSIS = 3.0


def bin_statistics(bins: torch.Tensor) -> dict:
    r"""Are the bins near enough to Gaussian for their error bars to mean something?

    An error bar taken from :math:`N` bins assumes that no few bins carry the
    variance.  In DQMC at low temperature they can: the estimator of
    :math:`G(\tau)` is then heavy-tailed, most bins sit close together and a
    rare one lies far out.  A run or a part of a run without such a bin has a
    small error bar and a mean that is off by many of them.  Measured on the
    runs of exactly solvable clusters (docs/TRAINING.md (xxxv)): at
    :math:`\beta = 8` the bins of 85% of the time slices have an excess
    kurtosis above 3 (median 28), blocks of 25 bins differ from the full mean
    by 6 of their own error bars, and intervals from subsets of the bins hold
    the exact answer 0.6 to 0.7 of the time instead of 0.9; at
    :math:`\beta = 2` and on the Holstein cell it is 2 to 15% of the slices
    and the intervals hold.

    Returns, over the time slices that have any variance: ``heavy_share`` (the
    share whose bins have excess kurtosis above :data:`HEAVY_KURTOSIS`; 0 for
    Gaussian bins at any ``N``), ``kurtosis_median``, ``largest_bin_share``
    (the largest share of one slice's variance carried by a single bin),
    ``n_bins`` and ``n_slices``.
    """
    b = torch.as_tensor(bins).detach().cpu().to(torch.float64)
    n = b.shape[0]
    d = b - b.mean(0, keepdim=True)
    m2 = (d * d).mean(0)
    live = m2 > 1e-18 * float(m2.max().clamp_min(1e-300))
    if n < 4 or not bool(live.any()):
        return {"n_bins": int(n), "n_slices": int(live.sum()), "heavy_share": 0.0,
                "kurtosis_median": 0.0, "largest_bin_share": 0.0}
    d, m2 = d[:, live], m2[live]
    kurt = (d ** 4).mean(0) / (m2 * m2) - 3.0
    return {"n_bins": int(n), "n_slices": int(live.sum()),
            "heavy_share": float((kurt > HEAVY_KURTOSIS).double().mean()),
            "kurtosis_median": float(kurt.median()),
            "largest_bin_share": float(((d * d).amax(0) / (d * d).sum(0)).max())}


def autocorrelation_report(bins: torch.Tensor, max_block: int = 64) -> dict:
    r"""Bin-autocorrelation diagnostics: blocking curve + Sokal :math:`\tau_{int}`.

    Requires bins in their original Monte-Carlo order.  Returns the blocking
    curve (error-of-mean inflation vs block size), the integrated
    autocorrelation time of the bins' leading principal component (Sokal
    self-consistent window, :math:`W \ge 5\tau_{int}`), the statistical
    inefficiency ``g`` from each method, and ``n_eff = N / g``.  The two
    estimates must roughly agree; quote the larger (conservative) one.
    """
    X = bins.detach().cpu().to(torch.float64)
    N, L = X.shape
    Xc = X - X.mean(0, keepdim=True)
    # leading principal component: the slowest collective mode of the bins
    _, _, Vt = torch.linalg.svd(Xc, full_matrices=False)
    pc = Xc @ Vt[0]
    pc = pc / pc.std().clamp_min(1e-300)

    # blocking curve on the PC (per-slice curves average to the same story)
    var1 = float(pc.var(unbiased=True)) / N
    curve = {}
    b = 1
    while b <= max_block and N // b >= 8:
        m = pc[: (N // b) * b].reshape(N // b, b).mean(1)
        curve[b] = (float(m.var(unbiased=True)) / (N // b)) / var1
        b *= 2
    g_block = max(curve.values())

    # Sokal: integrate rho(k) with the self-consistent window
    rho = []
    for k in range(1, N // 4):
        r = float((pc[:-k] * pc[k:]).mean())
        rho.append(r)
    tau = 0.5
    for k, r in enumerate(rho, start=1):
        tau += r
        if k >= 5.0 * tau:
            break
    g_sokal = max(2.0 * tau, 1.0)

    g = max(g_block, g_sokal)
    return {"blocking_curve": curve, "g_blocking": g_block,
            "tau_int": tau, "g_sokal": g_sokal, "g": g,
            "n_eff": N / g}
