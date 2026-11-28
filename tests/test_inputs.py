"""What the reader does with the files people actually have.

The bins of a fermionic single-particle G come with tau = 0 ... beta or with
tau = 0 ... beta - dtau, in either sign, and sometimes spin-summed, transposed
or with a NaN in them.  The first two are read; the rest are refused with a
message that says what is wrong.
"""
import numpy as np
import pytest
import torch

from poleflow.data import from_bins, read_text_bins
from poleflow.data.real import endpoint_convention

BETA = 8.0


def make_bins(n_bins=200, n_tau=81, noise=2e-3, seed=0, exact_sum_rule=True,
              poles=((-1.5, 0.45), (0.4, 0.2), (2.0, 0.35))):
    """Bins on tau = 0 ... beta of a three-line spectrum, with noise that is correlated in tau."""
    rng = np.random.default_rng(seed)
    tau = np.linspace(0.0, BETA, n_tau)
    G = sum(w * np.exp(-tau * e) / (1.0 + np.exp(-BETA * e)) for e, w in poles)
    white = rng.normal(size=(n_bins, n_tau)) * noise
    bins = G[None, :] + 0.5 * (white + np.roll(white, 1, axis=1))
    if exact_sum_rule:                                  # what DQMC gives: G(beta) = 1 - G(0) in every bin
        bins[:, -1] = 1.0 - bins[:, 0]
    return bins


def test_bins_on_the_full_grid_are_read_as_they_are():
    bins = make_bins()
    mg = from_bins(bins, BETA, dtype=torch.float64)
    assert mg.n_tau == bins.shape[1] and not mg.meta["endpoint_appended"]
    assert float(mg.tau[-1]) == BETA and mg.check_sum_rule()["ok"]


def test_bins_that_stop_before_beta_get_the_endpoint_appended():
    bins = make_bins()
    full = from_bins(bins, BETA, dtype=torch.float64)
    short = from_bins(bins[:, :-1], BETA, dtype=torch.float64)
    assert short.meta["endpoint_appended"] and short.includes_endpoints
    assert torch.equal(short.tau, full.tau)
    assert torch.allclose(short.bins, full.bins, atol=1e-15)       # the same data as the full file
    assert "appended as 1 - G(0)" in short.summary()


def test_the_sign_convention_is_detected_on_either_grid():
    bins = make_bins()
    for b in (-bins, -bins[:, :-1]):
        mg = from_bins(b, BETA, dtype=torch.float64)
        assert mg.meta["sign_flip"] and mg.n_tau == bins.shape[1]
        assert float(mg.mean[0]) > 0 and mg.check_sum_rule()["ok"]


def test_a_measured_endpoint_that_obeys_the_sum_rule_within_noise_is_not_replaced():
    bins = make_bins(exact_sum_rule=False, seed=3)       # G(beta) measured on its own
    assert endpoint_convention(torch.tensor(bins)) == "included"
    assert not from_bins(bins, BETA, dtype=torch.float64).meta["endpoint_appended"]


@pytest.mark.parametrize("n_tau", [17, 21, 41, 161, 401])
def test_the_grid_is_recognised_from_coarse_to_fine(n_tau):
    bins = make_bins(n_tau=n_tau, noise=1e-3, seed=n_tau)
    assert endpoint_convention(torch.tensor(bins)) == "included"
    assert endpoint_convention(torch.tensor(bins[:, :-1])) == "missing"


def test_a_spin_summed_g_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="spin-summed"):
        from_bins(2.0 * make_bins(), BETA)


def test_a_transposed_file_is_refused_with_the_reason():
    with pytest.raises(ValueError, match="transposed"):
        from_bins(make_bins(n_bins=30, n_tau=81).T, BETA)


def test_values_that_are_not_numbers_are_refused_with_their_place(tmp_path):
    bins = make_bins()
    bins[3, 10] = np.nan
    path = tmp_path / "bins.txt"
    np.savetxt(path, bins)
    with pytest.raises(ValueError, match="not finite numbers.*bin 3, time slice 10"):
        read_text_bins(str(path), BETA)


def test_the_grid_can_be_stated_when_the_data_do_not_decide():
    # a grid so coarse that the step to beta is more than four times the step before it
    bins = make_bins(n_tau=6, noise=1e-4, poles=((-3.0, 0.5), (3.0, 0.5)))
    with pytest.raises(ValueError, match="--endpoint"):
        from_bins(bins[:, :-1], BETA)
    mg = from_bins(bins[:, :-1], BETA, endpoint="missing", dtype=torch.float64)
    assert mg.meta["endpoint_appended"] and mg.n_tau == 6 and float(mg.tau[-1]) == BETA
    mg = from_bins(bins, BETA, endpoint="included", dtype=torch.float64)
    assert not mg.meta["endpoint_appended"] and mg.n_tau == 6
    with pytest.raises(ValueError, match="endpoint must be"):
        from_bins(bins, BETA, endpoint="maybe")


def test_the_command_reads_a_file_that_stops_before_beta_and_refuses_a_spin_summed_one(tmp_path, capsys):
    import argparse
    from poleflow.cli import _load_measured
    bins = make_bins()
    short, doubled = tmp_path / "short.txt", tmp_path / "doubled.txt"
    np.savetxt(short, bins[:, :-1])
    np.savetxt(doubled, 2.0 * bins)
    args = lambda path: argparse.Namespace(bins=str(path), beta=BETA, sign_flip=None, endpoint="auto")
    mg = _load_measured(args(short))
    assert mg.meta["endpoint_appended"] and "appended as 1 - G(0)" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        _load_measured(args(doubled))
    assert "spin-summed" in str(e.value)


def test_gaussian_bins_pass_the_check_and_heavy_tailed_bins_do_not():
    from poleflow.data.real import bin_statistics
    rng = np.random.default_rng(0)
    for n in (16, 25, 400):
        s = bin_statistics(torch.tensor(rng.normal(size=(n, 80))))
        assert s["heavy_share"] < 0.1 and s["n_slices"] == 80 and s["n_bins"] == n
    # mostly quiet bins and a rare one far out: what DQMC gives at low temperature
    quiet = rng.normal(size=(400, 80)) * 1e-3
    quiet[rng.random((400, 80)) < 0.01] += 0.05
    s = bin_statistics(torch.tensor(quiet))
    assert s["heavy_share"] > 0.8 and s["largest_bin_share"] > 0.1
    # time slices pinned by a symmetry have no variance and are left out
    pinned = rng.normal(size=(50, 10))
    pinned[:, 0] = 0.5
    assert bin_statistics(torch.tensor(pinned))["n_slices"] == 9
    assert bin_statistics(torch.tensor(rng.normal(size=(3, 10))))["heavy_share"] == 0.0     # too few to say


def test_the_check_of_the_bins_on_the_two_versioned_dqmc_runs():
    """The dimer at beta = 8 has heavy-tailed bins (TRAINING.md (xxxv)); the Holstein cell does not."""
    from poleflow.data.real import bin_statistics
    from poleflow.evaluation import cases as C
    from poleflow.posterior import HEAVY_BINS_SHARE
    dimer_bins = C.hubbard_dimer_cell()[0].bins                 # (measured data, exact spectrum)
    dimer, holstein = bin_statistics(dimer_bins), bin_statistics(C.holstein_cell().bins)
    assert dimer["heavy_share"] > 0.7 > HEAVY_BINS_SHARE and dimer["largest_bin_share"] > 0.5
    assert holstein["heavy_share"] < 0.1 < HEAVY_BINS_SHARE
    # and the check already sees it in the first 25 bins of the dimer
    assert bin_statistics(dimer_bins[:25])["heavy_share"] > HEAVY_BINS_SHARE
