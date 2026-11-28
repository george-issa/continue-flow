"""Does the report hold when a DQMC run has few bins?

The DQMC runs with known answers have 400 bins each.  A run with 25 or 50 bins
has the same data with larger error bars, and a covariance matrix estimated
from fewer bins than there are time slices.  This judge takes subsets of the
bins of a run of an exactly solvable cluster, continues each subset as a user
would (`posterior.continue_measured` at its defaults, and with the statement
that the spectrum is discrete), and scores the stated intervals against the
exact window weights.

    python scripts/dqmc/bin_count_judge.py run OUT FLOW SIM_DIR U [SUBSETS=6]
    python scripts/dqmc/bin_count_judge.py fig OUT

`run` writes OUT/bins_<run>.json; `fig` writes OUT/bin_count.{json,png}.
"""
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

COUNTS = (25, 50, 100, 200)
OFF, ON, INK = "#2a78d6", "#0b7a75", "#0b0b0b"
WG = np.linspace(-30.0, 30.0, 60001)
DW = WG[1] - WG[0]


def load(sim, U):
    from poleflow.data.exact_models import hubbard_cluster
    from poleflow.data.readers import read_smoqy_dir
    name = os.path.basename(os.path.normpath(sim))
    ring = re.search(r"hubbard_ring(\d+)", name)
    if ring:
        n = int(ring.group(1))
        mg = read_smoqy_dir(sim)
        hop = np.zeros((n, n))
        for i in range(n):
            hop[i, (i + 1) % n] = hop[(i + 1) % n, i] = -1.0
        tag = f"ring{n}_b{mg.beta:g}"
    else:
        mg = read_smoqy_dir(sim, pair=(0, 1))
        hop = np.array([[0.0, -1.0], [-1.0, 0.0]])
        tag = f"dimer_b{mg.beta:g}"
    return mg, hubbard_cluster(hop, U, U / 2.0, mg.beta), tag


def subset(mg, idx):
    from poleflow.data.real import MeasuredGreens
    return MeasuredGreens(bins=mg.bins[idx].clone(), tau=mg.tau, beta=mg.beta, label=mg.label,
                          includes_endpoints=mg.includes_endpoints, meta=dict(mg.meta))


def score(rep, exact):
    w = exact.weights / exact.weights.sum()
    wins = []
    for r in rep.windows:
        lo, hi = r["window"]
        t = float(w[(exact.energies >= lo) & (exact.energies < hi)].sum())
        wins.append({"exact": t, "q": [r["q05"], r["q25"], r["q75"], r["q95"]], "data": [r["lo"], r["hi"]]})
    d = rep.diagnostics
    cdf = np.cumsum(rep.band["mean"]) * DW
    return {"windows": wins, "w1": float(np.abs(cdf - exact.cdf(WG)).sum() * DW),
            "accepted": d["accepted_fraction"], "chi2_median": float(np.median(rep.chi2)),
            "relative_error_bar": d.get("relative_error_bar"), "warnings": list(rep.warnings),
            "flagged": bool(d.get("line_shape_readings"))}


def run(out, flow_path, sim, U, k):
    from poleflow.models.flow import load_flow
    from poleflow.posterior import continue_measured
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    flow = load_flow(flow_path).eval()
    mg, exact, tag = load(sim, U)
    N = mg.n_bins
    rows = []
    for n in COUNTS + (N,):
        starts = [0] if n == N else [int(round(s)) for s in np.linspace(0, N - n, k)]
        for s in starts:
            sub = subset(mg, torch.arange(s, s + n))
            row = {"bins": n, "start": s}
            t0 = time.time()
            for arm, kw in (("default", {}), ("stated", {"discrete": True})):
                try:
                    rep = continue_measured(flow, sub, omega=WG, generator=torch.Generator().manual_seed(9), **kw)
                    row[arm] = score(rep, exact)
                except Exception as e:                       # a failure is a result too
                    row[arm] = {"error": f"{type(e).__name__}: {e}"}
            row["seconds"] = time.time() - t0
            rows.append(row)
            msg = []
            for arm in ("default", "stated"):
                a = row[arm]
                if "error" in a:
                    msg.append(f"{arm}: {a['error'][:60]}")
                else:
                    hit = sum(wn["q"][0] - 1e-6 <= wn["exact"] <= wn["q"][3] + 1e-6 for wn in a["windows"])
                    msg.append(f"{arm}: {hit} of {len(a['windows'])} inside 90%, W1 {a['w1']:.3f}, "
                               f"accepted {a['accepted']:.0%}, {len(a['warnings'])} warnings")
            print(f"{tag} bins {n:3d} from {s:3d} | " + " | ".join(msg) + f" | {row['seconds']:.0f}s", flush=True)
    with open(out / f"bins_{tag}.json", "w") as fh:
        json.dump({"flow": flow_path, "run": tag, "sim": sim, "U": U, "n_bins": N, "rows": rows}, fh,
                  indent=1, default=float)
    print(f"-> {out}/bins_{tag}.json")


def held(rows, arm):
    ok = [r[arm] for r in rows if "error" not in r[arm]]
    w = [wn for a in ok for wn in a["windows"]]
    if not w:
        return None
    h50 = np.array([wn["q"][1] - 1e-6 <= wn["exact"] <= wn["q"][2] + 1e-6 for wn in w])
    h90 = np.array([wn["q"][0] - 1e-6 <= wn["exact"] <= wn["q"][3] + 1e-6 for wn in w])
    hd = np.array([wn["data"][0] - 1e-6 <= wn["exact"] <= wn["data"][1] + 1e-6 for wn in w
                   if wn["data"][0] == wn["data"][0]])
    return {"reports": len(ok), "failed": len(rows) - len(ok), "windows": len(w),
            "held50": float(h50.mean()), "held90": float(h90.mean()),
            "data_alone": [int(hd.sum()), int(len(hd))],
            "w1": float(np.exp(np.mean(np.log([a["w1"] for a in ok])))),
            "accepted": float(np.mean([a["accepted"] for a in ok])),
            "width90": float(np.mean([wn["q"][3] - wn["q"][0] for wn in w]))}


def fig(out):
    out = Path(out)
    runs = [json.load(open(f)) for f in sorted(out.glob("bins_*.json"))]
    res = {}
    f, axes = plt.subplots(len(runs), 3, figsize=(14, 3.7 * len(runs)), squeeze=False)
    for row_ax, d in zip(axes, runs):
        counts = sorted({r["bins"] for r in d["rows"]})
        res[d["run"]] = {}
        for arm, col, lab in (("default", OFF, "at the defaults"), ("stated", ON, "stated to be discrete")):
            hs = [held([r for r in d["rows"] if r["bins"] == n], arm) for n in counts]
            res[d["run"]][arm] = dict(zip(map(str, counts), hs))
            xs = [n for n, h in zip(counts, hs) if h]
            hs = [h for h in hs if h]
            row_ax[0].plot(xs, [h["held90"] for h in hs], color=col, marker="o", lw=1.6, label=f"90%, {lab}")
            row_ax[0].plot(xs, [h["held50"] for h in hs], color=col, marker="o", mfc="white", lw=1.0, ls=":",
                           label=f"50%, {lab}")
            row_ax[1].plot(xs, [h["w1"] for h in hs], color=col, marker="o", lw=1.6, label=lab)
            row_ax[2].plot(xs, [h["width90"] for h in hs], color=col, marker="o", lw=1.6, label=lab)
        for lv in (0.5, 0.9):
            row_ax[0].axhline(lv, color=INK, lw=0.8, ls="--")
        row_ax[0].set_ylim(0, 1.02)
        row_ax[0].set_ylabel(d["run"].replace("_", ", ").replace("b", "$\\beta$ = "), fontsize=10)
        row_ax[0].set_title("exact weight inside the stated interval (share of windows)", fontsize=10)
        row_ax[1].set_title("error W$_1$ (geometric mean over subsets)", fontsize=10)
        row_ax[2].set_title("mean width of the 90% interval", fontsize=10)
        for ax in row_ax:
            ax.set_xscale("log", base=2)
            ax.set_xticks(counts)
            ax.set_xticklabels([str(c) for c in counts])
            ax.set_xlabel("number of bins used")
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        if row_ax[0] is axes[0][0]:
            row_ax[0].legend(frameon=False, fontsize=8, loc="lower right")
    f.suptitle("Real DQMC runs of exactly solvable clusters, continued from subsets of their bins", fontsize=11)
    f.tight_layout()
    f.savefig(out / "bin_count.png", dpi=150)
    with open(out / "bin_count.json", "w") as fh:
        json.dump(res, fh, indent=1)
    for run_, arms in res.items():
        for arm, by in arms.items():
            for n, h in by.items():
                if h:
                    print(f"{run_:12s} {arm:8s} bins {n:>3s}: {h['held50']:.2f} / {h['held90']:.2f} over {h['windows']} "
                          f"windows, data-alone {h['data_alone'][0]} of {h['data_alone'][1]}, W1 {h['w1']:.3f}, "
                          f"accepted {h['accepted']:.0%}, failed {h['failed']}")


def main():
    if sys.argv[1] == "fig":
        return fig(sys.argv[2])
    run(sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), int(sys.argv[6]) if len(sys.argv) > 6 else 6)


if __name__ == "__main__":
    main()
