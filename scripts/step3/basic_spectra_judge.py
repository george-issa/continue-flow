"""The basic spectra: Lorentzians, Gaussians and exact pole sets, through the report.

One, two and three Lorentzians; one, two and three Gaussians; pole sets with
complex residues.  Positions, widths and weights are drawn at random, the data
are exact with noise added, and each spectrum is continued with
`continue_posterior` at its defaults (the call a user makes) and with MaxEnt on
the same data.  Scored: the error (area between the reconstructed and the
exact cumulative weight) and how often the stated intervals hold the exact
window weight.

    python scripts/step3/basic_spectra_judge.py run OUT FLOW BETA NOISE [PER=10]
    python scripts/step3/basic_spectra_judge.py fig OUT

Two measures of the error are kept, because they say different things on
Lorentzians.  The area between the cumulative weights counts weight by how far
it is misplaced, so it is dominated by the tails: the pole model has the tail
of a Lorentzian built in, and MaxEnt with a flat default model on a bounded
grid does not.  The mean error of the weight in the seven frequency windows
does not care where inside a window the weight sits.

`run` writes OUT/basic_b<beta>_n<noise>.json and the curves of the first two
spectra of each family beside it; `fig` writes OUT/basic_spectra.{json,png}
and OUT/basic_gallery_b<beta>_n<noise>.png.
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/step3")
sys.path.insert(0, "scripts/eval")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import exact_models_judge as J
import floor_option_judge as F

FAMILIES = ("1 Lorentzian", "2 Lorentzians", "3 Lorentzians", "1 Gaussian", "2 Gaussians", "3 Gaussians",
            "pole set")
BLUE, GREY, INK, RED = "#2a78d6", "#8a8f98", "#0b0b0b", "#b3261e"
SHOW = np.linspace(-6.0, 6.0, 1201)


def centres(rng, n, gap):
    """n positions in [-3, 3], no two closer than gap."""
    while True:
        e = np.sort(rng.uniform(-3.0, 3.0, n))
        if n == 1 or np.diff(e).min() >= gap:
            return e


def tup(v):
    """Plain Python floats (the suite builds single-precision tensors from them)."""
    return tuple(np.asarray(v, float).tolist())


def weights(rng, n):
    w = rng.uniform(0.3, 1.0, n)
    return w / w.sum()


def draw_case(family, rng, k):
    """One random spectrum of a family, as a suite case."""
    from spevae.evaluation.cases import Case, PoleSpec, SmoothSpec
    n = int(family[0]) if family[0].isdigit() else None
    if "Lorentzian" in family:
        spec = PoleSpec(tup(centres(rng, n, 0.8)), tup(rng.uniform(0.2, 1.0, n)), tup(weights(rng, n)), None)
    elif "Gaussian" in family:
        spec = SmoothSpec("gauss_mixture", tuple(sorted(dict(
            weights=tup(weights(rng, n)), mus=tup(centres(rng, n, 1.0)),
            sigmas=tup(rng.uniform(0.25, 1.0, n)), W=4.0).items())))
    else:                                                   # poles with complex residues, A >= 0
        while True:
            n = int(rng.integers(3, 9))
            spec = PoleSpec(tup(rng.uniform(-3.5, 3.5, n)), tup(rng.uniform(0.2, 0.9, n)),
                            tup(weights(rng, n)), tup(rng.uniform(-0.4, 0.4, n)))
            from spevae.evaluation.cases import truth_on_grid
            A = truth_on_grid(Case("try", spec), SHOW, 8.0)
            if A.min() >= -1e-4 * A.max():
                break
    return Case(f"{family} #{k}", spec)


def exact_windows(item, windows):
    """Exact weight per window, in closed form (a grid would cut the Lorentzian tails)."""
    import math
    from scipy.special import erf
    p = item["params"]
    if "eps" in p:                                          # poles: absorptive atan, dispersive log
        e, g, a = (np.array(p[k], float) for k in ("eps", "gam", "a"))
        t = np.zeros_like(a) if p["t"] is None else np.array(p["t"], float)
        b = a * (t - (a * t).sum() / a.sum())

        def cdf(w):
            if math.isinf(w):
                return 0.0 if w < 0 else float(a.sum())
            u = w - e
            return float((a * (np.arctan(u / g) / math.pi + 0.5) - b * np.log(u * u + g * g) / (2 * math.pi)).sum())
        tot = float(a.sum())
    else:
        w_, mu, sg = (np.array(p[k], float) for k in ("weights", "mus", "sigmas"))

        def cdf(w):
            if math.isinf(w):
                return 0.0 if w < 0 else float(w_.sum())
            return float((w_ * 0.5 * (1.0 + erf((w - mu) / (sg * math.sqrt(2.0))))).sum())
        tot = float(w_.sum())
    return [((cdf(hi) - cdf(lo)) / tot, False) for lo, hi in windows]


def items(beta, per):
    from spevae.evaluation import cases as C
    out = []
    ntau = int(round(beta / J.DTAU))
    for fi, family in enumerate(FAMILIES):
        rng = np.random.default_rng(7000 + fi)
        for k in range(per):
            case = draw_case(family, rng, k)
            A = C.truth_on_grid(case, J.WG, beta)
            spec = case.spec
            params = ({"eps": spec.eps, "gam": spec.gam, "a": spec.a, "t": spec.t} if hasattr(spec, "eps")
                      else dict(spec.params))
            out.append({"family": family, "kind": "continuum", "params": params, "A_grid": A,
                        "cdf": np.cumsum(A) * J.DW, "truth": C.realize(case, beta, ntau), "k": k})
    return out


def run(out, flow_path, beta, noise, per):
    from spevae.evaluation import cases as C
    from spevae.evaluation import methods as M
    from spevae.models.flow import load_flow
    from spevae.posterior import DEFAULT_WINDOWS, continue_posterior
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    flow = load_flow(flow_path).eval()
    gphys = flow.coords.gamma_min / beta
    rows, curves = [], {}
    pick = np.searchsorted(J.WG, SHOW)
    for i, item in enumerate(items(beta, per)):
        inst = C.draw(item["truth"], noise, 100 + i)
        truth_w = exact_windows(item, DEFAULT_WINDOWS)
        t0 = time.time()
        rep = continue_posterior(flow, inst.G, inst.truth.x, beta, inst.sigma, inst.isr, n_samples=F.S,
                                 omega=J.WG, bounds=True, generator=torch.Generator().manual_seed(9))
        fit = F.score(rep, item, DEFAULT_WINDOWS, truth_w, gphys)
        fit["warnings"] = list(rep.warnings)
        me = M.make_method("maxent").synthetic(inst)
        A_me = np.interp(J.WG, me.omega, me.A, left=0.0, right=0.0)
        row = {"index": i, "family": item["family"], "params": item["params"], "fit": fit,
               "maxent_w1": J.w1_cdf(np.cumsum(A_me) * J.DW, item["cdf"]),
               "maxent_windows": grid_windows(A_me, DEFAULT_WINDOWS), "seconds": time.time() - t0}
        rows.append(row)
        if item["k"] < 2:
            b = rep.band
            curves[f"c{i}"] = np.stack([item["A_grid"][pick], b["q50"][pick], b["q05"][pick], b["q95"][pick],
                                        A_me[pick]])
        q, t = np.array(fit["reported"]), np.array(fit["truth"])
        print(f"{item['family']:14s} #{item['k']:2d}  W1 {fit['w1']:.4f}  MaxEnt {row['maxent_w1']:.4f}  inside the "
              f"90% interval: {int(((q[:, 0] <= t) & (t <= q[:, 3])).sum())} of {len(t)}  "
              f"{'flagged' if fit['line_shape_readings'] else ''}  {row['seconds']:.0f}s", flush=True)
    tag = f"basic_b{beta:g}_n{noise:g}"
    with open(out / f"{tag}.json", "w") as fh:
        json.dump({"flow": flow_path, "beta": beta, "noise": noise, "dtau": J.DTAU, "rows": rows}, fh,
                  indent=1, default=float)
    np.savez_compressed(out / f"{tag}.curves.npz", **curves)
    print(f"-> {out}/{tag}.json")


def grid_windows(A, windows):
    """Weight per window of a spectrum on the fine grid, as shares of its total."""
    c = np.concatenate([[0.0], np.cumsum(A) * J.DW])
    at = lambda b: 0.0 if b == -np.inf else c[-1] if b == np.inf else float(np.interp(b, J.WG, c[1:]))
    return [(at(hi) - at(lo)) / c[-1] for lo, hi in windows]


def window_error(rows):
    """Mean absolute error of the window weights: the model's (mean over its samples) and MaxEnt's."""
    t = np.array([r["fit"]["truth"] for r in rows])
    mine = np.array([np.mean(r["fit"]["w"], axis=0) for r in rows])
    out = {"model": float(np.abs(mine - t).mean())}
    if all("maxent_windows" in r for r in rows):
        out["maxent"] = float(np.abs(np.array([r["maxent_windows"] for r in rows]) - t).mean())
    return out


def backfill(out):
    """MaxEnt's window weights for result files written before they were stored (the data are redrawn)."""
    from spevae.evaluation import cases as C
    from spevae.evaluation import methods as M
    from spevae.posterior import DEFAULT_WINDOWS
    for path in sorted(Path(out).glob("basic_b*_n*.json")):
        d = json.load(open(path))
        if all("maxent_windows" in r for r in d["rows"]):
            continue
        its = items(d["beta"], max(r["index"] for r in d["rows"]) // len(FAMILIES) + 1)
        for r in d["rows"]:
            inst = C.draw(its[r["index"]]["truth"], d["noise"], 100 + r["index"])
            me = M.make_method("maxent").synthetic(inst)
            A_me = np.interp(J.WG, me.omega, me.A, left=0.0, right=0.0)
            w1 = J.w1_cdf(np.cumsum(A_me) * J.DW, its[r["index"]]["cdf"])
            assert abs(w1 - r["maxent_w1"]) < 1e-6 * max(1.0, w1), (path, r["index"], w1, r["maxent_w1"])
            r["maxent_windows"] = grid_windows(A_me, DEFAULT_WINDOWS)
        with open(path, "w") as fh:
            json.dump(d, fh, indent=1, default=float)
        print(f"MaxEnt window weights added to {path}")


def summary(rows):
    w1 = np.array([r["fit"]["w1"] for r in rows])
    me = np.array([r["maxent_w1"] for r in rows])
    q = np.concatenate([np.array(r["fit"]["reported"]) for r in rows])
    t = np.concatenate([np.array(r["fit"]["truth"]) for r in rows])
    mid, lo, hi = F.ratio_ci(w1, me)
    return {"spectra": len(rows), "w1": float(np.exp(np.log(w1).mean())), "maxent_w1": float(np.exp(np.log(me).mean())),
            "over_maxent": [mid, lo, hi], "lower_than_maxent": int((w1 < me).sum()),
            "held50": float(((q[:, 1] <= t) & (t <= q[:, 2])).mean()),
            "held90": float(((q[:, 0] <= t) & (t <= q[:, 3])).mean()), "windows": int(len(t)),
            "flagged": int(sum(r["fit"]["line_shape_readings"] for r in rows)),
            "with_warning": int(sum(bool(r["fit"]["warnings"]) for r in rows)),
            "window_error": window_error(rows)}


def gallery(d, curves, path):
    rows = [r for r in d["rows"] if f"c{r['index']}" in curves]
    f, axes = plt.subplots(2, len(FAMILIES), figsize=(3.1 * len(FAMILIES), 5.6), squeeze=False)
    for col, family in enumerate(FAMILIES):
        for ax, r in zip(axes[:, col], [r for r in rows if r["family"] == family][:2]):
            A, q50, q05, q95, me = curves[f"c{r['index']}"]
            ax.fill_between(SHOW, q05, q95, color=BLUE, alpha=0.25, lw=0)
            ax.plot(SHOW, me, color=GREY, lw=1.2, label="MaxEnt")
            ax.plot(SHOW, q50, color=BLUE, lw=1.4, label="this model: median, 90% band")
            ax.plot(SHOW, A, color=INK, lw=1.0, ls="--", label="exact")
            ax.set_ylim(0, 1.25 * A.max())
            worse = r["fit"]["w1"] > r["maxent_w1"]
            ax.set_title(f"{family}\nerror {r['fit']['w1']:.3f}, MaxEnt {r['maxent_w1']:.3f}", fontsize=8.5,
                         color=RED if worse else INK)
            ax.set_yticks([])
            for s_ in ("top", "right", "left"):
                ax.spines[s_].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    f.suptitle(f"$\\beta$ = {d['beta']:g}, relative error bar {d['noise']:g}.  A red title marks a spectrum where "
               "MaxEnt has the lower error.", fontsize=10)
    f.tight_layout()
    f.savefig(path, dpi=150)
    plt.close(f)


def fig(out):
    out = Path(out)
    runs = sorted((json.load(open(p)) for p in out.glob("basic_b*_n*.json")), key=lambda d: (d["beta"], -d["noise"]))
    res = []
    f, axes = plt.subplots(len(runs), 3, figsize=(18, 3.4 * len(runs)), squeeze=False)
    for (a0, a2, a1), d in zip(axes, runs):
        tag = f"basic_b{d['beta']:g}_n{d['noise']:g}"
        gallery(d, np.load(out / f"{tag}.curves.npz"), out / f"basic_gallery_b{d['beta']:g}_n{d['noise']:g}.png")
        fams = {fam: summary([r for r in d["rows"] if r["family"] == fam]) for fam in FAMILIES}
        res.append({"beta": d["beta"], "noise": d["noise"], "families": fams})
        y = np.arange(len(FAMILIES))
        for i, fam in enumerate(FAMILIES):
            ratio = [r["fit"]["w1"] / r["maxent_w1"] for r in d["rows"] if r["family"] == fam]
            a0.plot(ratio, [i] * len(ratio), "o", color=BLUE, alpha=0.35, ms=4)
            mid, lo, hi = fams[fam]["over_maxent"]
            a0.plot([lo, hi], [i, i], color=INK, lw=1.6)
            a0.plot(mid, i, "o", color=INK, ms=6)
        a0.axvline(1.0, color=INK, lw=0.8, ls=":")
        a0.set_xscale("log")
        a0.set_yticks(y)
        a0.set_yticklabels(FAMILIES)
        a0.invert_yaxis()
        a0.set_xlabel("cumulative-weight error over MaxEnt's (left of 1: this model is better)")
        we = [fams[fam]["window_error"] for fam in FAMILIES]
        a2.barh(y - 0.2, [w["model"] for w in we], 0.38, color=BLUE, label="this model")
        if all("maxent" in w for w in we):
            a2.barh(y + 0.2, [w["maxent"] for w in we], 0.38, color=GREY, label="MaxEnt")
        a2.set_yticks(y)
        a2.set_yticklabels([])
        a2.invert_yaxis()
        a2.set_xlabel("mean error of the weight in a frequency window")
        if a2 is axes[0][1]:
            a2.legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        a0.set_title(f"$\\beta$ = {d['beta']:g}, relative error bar {d['noise']:g}", fontsize=10)
        a1.barh(y - 0.2, [fams[fam]["held50"] for fam in FAMILIES], 0.38, color=BLUE, alpha=0.45, label="50% interval")
        a1.barh(y + 0.2, [fams[fam]["held90"] for fam in FAMILIES], 0.38, color=BLUE, label="90% interval")
        for lv in (0.5, 0.9):
            a1.axvline(lv, color=INK, lw=0.8, ls="--")
        a1.set_xlim(0, 1)
        a1.set_yticks(y)
        a1.set_yticklabels([])
        a1.invert_yaxis()
        a1.set_xlabel("share of windows whose exact weight is inside the stated interval")
        for ax in (a0, a1, a2):
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    axes[0, 2].legend(frameon=False, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    f.tight_layout()
    f.savefig(out / "basic_spectra.png", dpi=150)
    with open(out / "basic_spectra.json", "w") as fh:
        json.dump(res, fh, indent=1)
    for s in res:
        print(f"beta {s['beta']:g}  error bar {s['noise']:g}")
        for fam, h in s["families"].items():
            we = h["window_error"]
            print(f"  {fam:14s} W1 {h['w1']:.4f}  MaxEnt {h['maxent_w1']:.4f}  ratio {h['over_maxent'][0]:.2f} "
                  f"[{h['over_maxent'][1]:.2f}, {h['over_maxent'][2]:.2f}]  lower on {h['lower_than_maxent']:2d} of "
                  f"{h['spectra']}  window error {we['model']:.4f} / {we.get('maxent', float('nan')):.4f}  "
                  f"intervals {h['held50']:.2f} / {h['held90']:.2f}  flagged {h['flagged']}  "
                  f"with a warning {h['with_warning']}")


def main():
    if sys.argv[1] == "fig":
        backfill(sys.argv[2])
        return fig(sys.argv[2])
    run(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 10)


if __name__ == "__main__":
    main()
