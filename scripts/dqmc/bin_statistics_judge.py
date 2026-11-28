"""Does the check of the bins separate the runs whose error bars hold from those whose do not?

`poleflow.data.real.bin_statistics` gives the share of time slices whose bins
are heavy-tailed; the report warns from `posterior.HEAVY_BINS_SHARE`.  Here it
is computed on the DQMC runs with known answers and on the Holstein cell, on
all bins and on subsets of consecutive bins, and set next to how often the
stated 90% intervals held the exact weight from the same subsets
(`bin_count_judge.py`).  No model is run.

    python scripts/dqmc/bin_statistics_judge.py OUT BIN_COUNT_DIR

Writes OUT/bin_statistics.{json,png}.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/dqmc")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bin_count_judge as B

COUNTS = (16, 25, 50, 100, 200, 400)
RUNS = (("dimer_b8", "out/dqmc/hubbard_dimer_U4.00_b8.00_dt0.050-1", "#b3261e"),
        ("ring4_b8", "out/dqmc/hubbard_ring4_U4.00_b8.00_dt0.050-1", "#c9710b"),
        ("dimer_b2", "out/dqmc/hubbard_dimer_U4.00_b2.00_dt0.050-1", "#2a78d6"),
        ("holstein", None, "#0b7a75"))
INK = "#0b0b0b"


def main():
    from poleflow.data.real import bin_statistics
    from poleflow.evaluation import cases as C
    from poleflow.posterior import HEAVY_BINS_SHARE
    out, held_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    held = json.load(open(held_dir / "bin_count.json"))
    res = {"threshold": HEAVY_BINS_SHARE, "runs": {}}
    f, (a0, a1) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for tag, sim, col in RUNS:
        if sim is None:
            cell = C.holstein_cell()
            bins = (cell[0] if isinstance(cell, tuple) else cell).bins
        else:
            bins = B.load(sim, 4.0)[0].bins
        N = bins.shape[0]
        rows = {}
        for n in COUNTS:
            starts = [0] if n == N else [int(round(s)) for s in np.linspace(0, N - n, 12)]
            sh = [bin_statistics(bins[s:s + n])["heavy_share"] for s in starts]
            rows[n] = {"heavy_share": sh, "flagged": float(np.mean([v >= HEAVY_BINS_SHARE for v in sh]))}
            a0.plot([n] * len(sh), sh, "o", color=col, alpha=0.35, ms=4)
        a0.plot(COUNTS, [float(np.median(rows[n]["heavy_share"])) for n in COUNTS], color=col, lw=1.6, label=tag)
        res["runs"][tag] = {"bins": N, "by_count": rows}
        h = held.get(tag.replace("ring4_b8", "ring4_b8"), {}).get("default")
        if h:
            xs = [int(k) for k, v in h.items() if v]
            a1.plot(xs, [h[str(k)]["held90"] for k in xs], color=col, marker="o", lw=1.6, label=tag)
    a0.axhline(HEAVY_BINS_SHARE, color=INK, lw=0.8, ls="--")
    a0.text(COUNTS[0], HEAVY_BINS_SHARE + 0.015, "the report warns above this line", fontsize=8.5)
    a0.set_ylabel("share of time slices with heavy-tailed bins")
    a0.set_title("the check: one point per subset of consecutive bins, line = median", fontsize=10)
    a1.axhline(0.9, color=INK, lw=0.8, ls="--")
    a1.set_ylim(0, 1.02)
    a1.set_ylabel("exact weight inside the stated 90% interval")
    a1.set_title("what the report did on the same runs (defaults)", fontsize=10)
    for ax in (a0, a1):
        ax.set_xscale("log", base=2)
        ax.set_xticks(COUNTS)
        ax.set_xticklabels([str(c) for c in COUNTS])
        ax.set_xlabel("number of bins used")
        ax.legend(frameon=False, fontsize=9)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    f.tight_layout()
    f.savefig(out / "bin_statistics.png", dpi=150)
    with open(out / "bin_statistics.json", "w") as fh:
        json.dump(res, fh, indent=1)
    for tag, r in res["runs"].items():
        print(f"{tag:9s} " + "  ".join(f"{n}: median {np.median(v['heavy_share']):.2f}, warned {v['flagged']:.0%}"
                                        for n, v in r["by_count"].items()))


if __name__ == "__main__":
    main()
