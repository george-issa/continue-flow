"""How many close lines does it take before the report can treat them as a continuous band?

The report cannot tell many close lines from a continuous band, and without the
statement that the spectrum is discrete its intervals are too narrow for the
lines (TRAINING.md (xxii), (xxx)).  This judge measures where that stops: bands
of N lines, N from a few to hundreds, whose envelope is a semicircle, so that
N -> infinity is a spectrum the model was trained on.  Each is continued at the
defaults and with `discrete=True`, and the stated intervals are scored against
the exact window weights.

    python scripts/step3/line_density_judge.py run OUT FLOW BETA NOISE N [PER=10] [ARMS=default,stated]
    python scripts/step3/line_density_judge.py fig OUT
    python scripts/step3/line_density_judge.py soft OUT

`run` writes OUT/density_b<beta>_n<noise>_N<N>.json and the accepted samples'
poles of the default arm beside it; N = 0 runs the exact lattice spectra of the
known-answer judge instead (the continuous case).  `fig` writes
OUT/line_density.{json,png}.

`soft` asks whether the failure is the sharp edge of the windows.  The weight
in a window with a sharp edge depends on which side of the edge each line
falls, which the data cannot resolve.  It rescores the stored samples on
windows whose edges are smoothed over a width s (the same as the weight in the
sharp window of the spectrum broadened by a Lorentzian of half-width s), for
s from 0 to pi/beta, and writes OUT/soft_windows.{json,png}.
"""
import json
import math
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

OFF, ON, INK = "#2a78d6", "#0b7a75", "#0b0b0b"


def band_of_lines(n, beta, rng):
    """N lines under a semicircle of random width and centre; weights scatter around the envelope."""
    from poleflow.data.exact_models import DiscreteSpectrum
    W, c = rng.uniform(3.0, 6.0), rng.uniform(-1.0, 1.0)
    u = np.sort(rng.uniform(-1.0, 1.0, n))
    w = np.sqrt(np.clip(1.0 - u * u, 0.0, None)) * rng.gamma(2.0, 0.5, n)
    return DiscreteSpectrum(c + 0.5 * W * u, w / w.sum(), beta, "band_of_lines",
                            {"n": n, "W": W, "c": c, "spacing": W / n})


def items(n, beta, per):
    from poleflow.evaluation.cases import Truth
    from poleflow.physics import SpectralQuadrature
    quad = SpectralQuadrature.for_beta(beta)
    x = torch.linspace(0.0, 1.0, int(round(beta / J.DTAU)) + 1)
    rng = np.random.default_rng(4000 + n)
    out = []
    for _ in range(per):
        sp = band_of_lines(n, beta, rng)
        Gc = torch.tensor(sp.greens(x.double().numpy()), dtype=torch.float32).unsqueeze(0)
        out.append({"family": sp.family, "kind": "discrete", "params": sp.params, "spectrum": sp,
                    "cdf": sp.cdf(J.WG),
                    "truth": Truth(None, beta, x, quad, torch.zeros(1, quad.nodes.numel()), Gc, None)})
    return out


def lattice_items(beta):
    """The exact lattice spectra of the known-answer judge: the continuous case."""
    return [it for it in J.make_truths(beta) if it["kind"] == "continuum"]


def run(out, flow_path, beta, noise, n, per, arms=("default", "stated")):
    from poleflow.evaluation import cases as C
    from poleflow.models.flow import load_flow
    from poleflow.posterior import DEFAULT_WINDOWS, continue_posterior
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    flow = load_flow(flow_path).eval()
    gphys = flow.coords.gamma_min / beta
    rows, poles = [], {}
    for i, item in enumerate(items(n, beta, per) if n else lattice_items(beta)):
        inst = C.draw(item["truth"], noise, J.SEED)
        truth_w = F.true_windows(item, DEFAULT_WINDOWS)
        row = {"index": i, "n": n, "params": item["params"], "kind": item["kind"], "family": item["family"]}
        t0 = time.time()
        for arm in arms:
            rep = continue_posterior(flow, inst.G, inst.truth.x, beta, inst.sigma, inst.isr, n_samples=F.S,
                                     omega=J.WG, bounds=True, generator=torch.Generator().manual_seed(9),
                                     discrete=arm == "stated")
            row[arm] = F.score(rep, item, DEFAULT_WINDOWS, truth_w, gphys)
            if arm == "default":                 # accepted samples, physical units: other windows without refitting
                poles[f"p{i}"] = np.stack([getattr(rep.poles, k).double().numpy()
                                           for k in ("eps", "gamma", "a", "b")])
        row["seconds"] = time.time() - t0
        rows.append(row)
        msg = []
        for arm in arms:
            q, t = np.array(row[arm]["reported"]), np.array(row[arm]["truth"])
            msg.append(f"{arm}: {int(((q[:, 0] <= t) & (t <= q[:, 3])).sum())} of {len(t)} windows inside "
                       f"the 90% interval, W1 {row[arm]['w1']:.3f}")
        print(f"N {n:4d} #{i} {item['family']:14s} "
              f"{'flagged' if row[arms[0]]['line_shape_readings'] else '       '}  " + " | ".join(msg)
              + f" | {row['seconds']:.0f}s", flush=True)
    path = out / f"density_b{beta:g}_n{noise:g}_N{n}.json"
    with open(path, "w") as fh:
        json.dump({"flow": flow_path, "beta": beta, "noise": noise, "dtau": J.DTAU, "n": n, "rows": rows},
                  fh, indent=1, default=float)
    if poles:
        np.savez_compressed(str(path)[:-5] + ".poles.npz", **poles)
    print(f"-> {path}")


SOFT = (0.0, 0.125, 0.25, 0.5, 1.0)                 # edge widths, in units of pi / beta


def soft_truth(item, windows, s):
    """Exact weight per window of the truth broadened by a Lorentzian of half-width s."""
    def step(e, edge):                                # share of a line at e below the edge
        if math.isinf(edge):
            return np.zeros_like(e) if edge < 0 else np.ones_like(e)
        return (e <= edge).astype(float) if s == 0 else np.arctan((edge - e) / s) / math.pi + 0.5
    if item["kind"] == "discrete":
        sp = item["spectrum"]
        e, w = sp.energies, sp.weights / sp.weights.sum()
    else:                                             # a continuous spectrum on the fine grid
        e, w = J.WG, item["A_grid"] * J.DW
        w = w / w.sum()
    return [float((w * (step(e, hi) - step(e, lo))).sum()) for lo, hi in windows]


def soft(out):
    from poleflow.physics.poles import PoleSet
    from poleflow.posterior import DEFAULT_WINDOWS, central_interval
    out = Path(out)
    res = {"edge_widths_over_resolution": list(SOFT), "settings": []}
    runs = [(json.load(open(f)), f) for f in sorted(out.glob("density_b*_N*.json"))
            if Path(str(f)[:-5] + ".poles.npz").exists()]
    settings = sorted({(d["beta"], d["noise"]) for d, _ in runs})
    f, axes = plt.subplots(len(settings), 2, figsize=(12.5, 4.2 * len(settings)), squeeze=False)
    shades = plt.cm.viridis(np.linspace(0.0, 0.85, len(SOFT)))
    for row_ax, (beta, noise) in zip(axes, settings):
        sel = sorted(((d, f_) for d, f_ in runs if (d["beta"], d["noise"]) == (beta, noise)),
                     key=lambda p: p[0]["n"] or 10 ** 9)                  # lattice (N = 0) last
        table = []
        for d, path in sel:
            n = d["n"]
            poles = np.load(str(path)[:-5] + ".poles.npz")
            its = items(n, beta, len(d["rows"])) if n else lattice_items(beta)
            hits = {s: [[], []] for s in SOFT}
            for r in d["rows"]:
                if f"p{r['index']}" not in poles:
                    continue
                eps, gam, a, b = (torch.tensor(v) for v in poles[f"p{r['index']}"])
                lev = r["default"]["levels"]
                for s in SOFT:
                    width = s * math.pi / beta
                    w = F.sample_windows(PoleSet(eps, gam + width, a, b), DEFAULT_WINDOWS)
                    t = soft_truth(its[r["index"]], DEFAULT_WINDOWS, width)
                    for k, tk in enumerate(t):
                        (l9, h9), (l5, h5) = central_interval(w[:, k], lev["0.9"]), central_interval(w[:, k], lev["0.5"])
                        hits[s][0].append(l5 <= tk <= h5)
                        hits[s][1].append(l9 <= tk <= h9)
            table.append({"n": n, "held": {str(s): [float(np.mean(h[0])), float(np.mean(h[1]))]
                                           for s, h in hits.items()}, "windows": len(hits[SOFT[0]][0])})
        res["settings"].append({"beta": beta, "noise": noise, "rows": table})
        labels = [str(t["n"]) if t["n"] else "lattice\n(continuous)" for t in table]
        for ax, k, lvl in ((row_ax[0], 0, 0.5), (row_ax[1], 1, 0.9)):
            for s, col in zip(SOFT, shades):
                ax.plot(range(len(table)), [t["held"][str(s)][k] for t in table], color=col, marker="o", lw=1.6,
                        label="sharp edge" if s == 0 else f"edge width {s:g} $\\pi/\\beta$")
            ax.axhline(lvl, color=INK, lw=0.8, ls="--")
            ax.set_ylim(0, 1.02)
            ax.set_xticks(range(len(table)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_xlabel("number of lines in the band")
            ax.set_title(f"exact weight inside the {int(lvl * 100)}% interval of the samples", fontsize=10)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        row_ax[0].set_ylabel(f"$\\beta$ = {beta:g}, error bar {noise:g}", fontsize=10)
        row_ax[1].legend(frameon=False, fontsize=8.5, loc="lower left")
    f.suptitle("The same samples scored on windows with smoothed edges", fontsize=11)
    f.tight_layout()
    f.savefig(out / "soft_windows.png", dpi=150)
    with open(out / "soft_windows.json", "w") as fh:
        json.dump(res, fh, indent=1)
    for s_ in res["settings"]:
        print(f"beta {s_['beta']:g} noise {s_['noise']:g}   (50% / 90% per edge width in units of pi/beta: {SOFT})")
        for t in s_["rows"]:
            print(f"  N {t['n'] or 'lattice':>7}  " + "   ".join(f"{t['held'][str(s)][0]:.2f} / {t['held'][str(s)][1]:.2f}"
                                                              for s in SOFT) + f"   ({t['windows']} windows)")


def held(rows, arm):
    q = np.concatenate([np.array(r[arm]["reported"]) for r in rows])
    t = np.concatenate([np.array(r[arm]["truth"]) for r in rows])
    h50, h90 = (q[:, 1] <= t) & (t <= q[:, 2]), (q[:, 0] <= t) & (t <= q[:, 3])
    se = lambda h: float(np.sqrt(max(h.mean() * (1 - h.mean()), 1e-9) / len(h)))
    return {"windows": int(len(t)), "held50": float(h50.mean()), "held90": float(h90.mean()),
            "se50": se(h50), "se90": se(h90), "width90": float(np.mean(q[:, 3] - q[:, 0])),
            "w1": float(np.exp(np.mean(np.log([r[arm]["w1"] for r in rows]))))}


def fig(out):
    out = Path(out)
    runs = sorted((json.load(open(f)) for f in out.glob("density_b*_N*.json")),
                  key=lambda d: (d["beta"], d["noise"], d["n"]))
    settings = sorted({(d["beta"], d["noise"]) for d in runs})
    res = {"settings": []}
    f, axes = plt.subplots(len(settings), 4, figsize=(17, 3.9 * len(settings)), squeeze=False)
    for row_ax, (beta, noise) in zip(axes, settings):
        sel = [d for d in runs if (d["beta"], d["noise"]) == (beta, noise)]
        ns = [d["n"] for d in sel]
        spacing = [float(np.mean([r["params"]["spacing"] for r in d["rows"]])) * beta / math.pi for d in sel]
        arms = {a: [held(d["rows"], a) for d in sel] for a in ("default", "stated")}
        flagged = [float(np.mean([r["default"]["line_shape_readings"] for r in d["rows"]])) for d in sel]
        res["settings"].append({"beta": beta, "noise": noise, "n": ns, "spacing_over_resolution": spacing,
                                "flagged": flagged, **arms})
        for ax, lvl, key, se in ((row_ax[0], 0.5, "held50", "se50"), (row_ax[1], 0.9, "held90", "se90")):
            for a, col, lab in (("default", OFF, "at the defaults"), ("stated", ON, "stated to be discrete")):
                ax.errorbar(ns, [h[key] for h in arms[a]], yerr=[h[se] for h in arms[a]], color=col,
                            marker="o", lw=1.6, capsize=2, label=lab)
            ax.axhline(lvl, color=INK, lw=0.8, ls="--")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"exact weight inside the stated {int(lvl * 100)}% interval", fontsize=10)
        row_ax[2].plot(ns, flagged, color=INK, marker="o", lw=1.6)
        row_ax[2].set_ylim(-0.02, 1.02)
        row_ax[2].set_title("share of spectra the report flags as narrow lines", fontsize=10)
        for a, col in (("default", OFF), ("stated", ON)):
            row_ax[3].plot(ns, [h["w1"] for h in arms[a]], color=col, marker="o", lw=1.6)
        row_ax[3].set_yscale("log")
        row_ax[3].set_title("error W$_1$ (geometric mean)", fontsize=10)
        for ax in row_ax:
            ax.set_xscale("log", base=2)
            ax.set_xticks(ns)
            ax.set_xticklabels([f"{n}\n{s:.2g}" for n, s in zip(ns, spacing)], fontsize=8)
            ax.set_xlabel("number of lines in the band\nmean spacing in units of $\\pi/\\beta$", fontsize=9)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        row_ax[0].set_ylabel(f"$\\beta$ = {beta:g}, error bar {noise:g}", fontsize=10)
        row_ax[0].legend(frameon=False, fontsize=9, loc="lower right")
    f.suptitle("Bands of N lines under a semicircle: where do the default intervals start to hold?",
               fontsize=11)
    f.tight_layout()
    f.savefig(out / "line_density.png", dpi=150)
    with open(out / "line_density.json", "w") as fh:
        json.dump(res, fh, indent=1)
    for s in res["settings"]:
        print(f"beta {s['beta']:g} noise {s['noise']:g}")
        for i, n in enumerate(s["n"]):
            d, st = s["default"][i], s["stated"][i]
            print(f"  N {n:4d} spacing {s['spacing_over_resolution'][i]:5.2f} pi/beta  flagged {s['flagged'][i]:.0%}  "
                  f"default {d['held50']:.2f} / {d['held90']:.2f}  stated {st['held50']:.2f} / {st['held90']:.2f}  "
                  f"W1 {d['w1']:.3f} / {st['w1']:.3f}")


def main():
    if sys.argv[1] in ("fig", "soft"):
        return {"fig": fig, "soft": soft}[sys.argv[1]](sys.argv[2])
    out, flow, beta, noise, n = sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
    run(out, flow, beta, noise, n, int(sys.argv[7]) if len(sys.argv) > 7 else 10,
        tuple(sys.argv[8].split(",")) if len(sys.argv) > 8 else ("default", "stated"))


if __name__ == "__main__":
    main()
