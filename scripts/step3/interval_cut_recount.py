"""Stated intervals cut to the possible range: a recount on stored judge rows.

The report cuts the stated interval of a window weight to what is possible:
not below zero, and not outside the range the data alone allow
(`posterior.continue_posterior`).  This script applies the same cut to the
intervals the judges stored (stated interval, exact weight and, where it was
kept, the data-alone range per window) and counts how often the exact weight
is inside, before and after.  No model is run.

    python scripts/step3/interval_cut_recount.py OUT JUDGE_DIR [JUDGE_DIR ...]

Writes OUT/interval_cut.json and OUT/interval_cut.png.
"""
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BEFORE, AFTER, INK = "#9aa0a6", "#2a78d6", "#0b0b0b"
KIND = {"continuum": "lattice spectra", "discrete": "line spectra"}


def cut(q, lo, hi):
    """The four quantiles of one window after the cut; lo, hi = data-alone range or None."""
    q05, q95 = q[0], q[3]
    have = lo is not None and lo == lo and hi == hi
    if have and (q95 < lo - 1e-6 or q05 > hi + 1e-6):       # wholly outside: left as it is, warned about
        return list(q)
    c_lo, c_hi = (max(lo, 0.0), hi) if have else (0.0, math.inf)
    return [min(max(v, c_lo), c_hi) for v in q]


def recount(dirs):
    groups = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            try:
                rows = json.load(open(f)).get("rows")
            except (ValueError, AttributeError):
                continue
            for r in rows or []:
                for arm, fit in r.items():
                    if not (isinstance(fit, dict) and fit.get("reported") and fit.get("truth")):
                        continue
                    br = fit.get("bracket") or [[None, None]] * len(fit["truth"])
                    key = f"{os.path.basename(os.path.normpath(d))} | {KIND.get(r.get('kind'), '?')} | {arm}"
                    g = groups.setdefault(key, dict(windows=0, with_data_range=0, changed=0,
                                                    hit50=[0, 0], hit90=[0, 0], width90=[0.0, 0.0]))
                    for q, t, (lo, hi) in zip(fit["reported"], fit["truth"], br):
                        c = cut(q, lo, hi)
                        g["windows"] += 1
                        g["with_data_range"] += lo is not None and lo == lo
                        g["changed"] += c != list(q)
                        for i, v in enumerate((q, c)):
                            g["hit50"][i] += v[1] <= t <= v[2]
                            g["hit90"][i] += v[0] <= t <= v[3]
                            g["width90"][i] += v[3] - v[0]
    for g in groups.values():
        n = g["windows"]
        g["held50"] = [h / n for h in g["hit50"]]
        g["held90"] = [h / n for h in g["hit90"]]
        g["width90"] = [w / n for w in g["width90"]]
    return groups


def figure(groups, path):
    keys = list(groups)
    f, axes = plt.subplots(1, 3, figsize=(15, 0.42 * len(keys) + 1.8), sharey=True)
    y = range(len(keys))
    for ax, field, nominal, title in ((axes[0], "held50", 0.5, "exact weight inside the 50% interval"),
                                      (axes[1], "held90", 0.9, "inside the 90% interval"),
                                      (axes[2], "width90", None, "mean width of the 90% interval")):
        ax.barh([i + 0.2 for i in y], [groups[k][field][0] for k in keys], 0.38, color=BEFORE, label="as stored")
        ax.barh([i - 0.2 for i in y], [groups[k][field][1] for k in keys], 0.38, color=AFTER, label="with the cut")
        if nominal:
            ax.axvline(nominal, color=INK, lw=0.8, ls="--")
        ax.set_title(title, fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels([f"{k}  ({groups[k]['windows']} windows)" for k in keys], fontsize=8)
    axes[0].invert_yaxis()
    axes[2].legend(frameon=False, fontsize=9)
    f.suptitle("Stated intervals cut to the possible range (not below zero, not outside the range the data "
               "alone allow): recount on stored judge rows", fontsize=10.5)
    f.tight_layout()
    f.savefig(path, dpi=150)


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    os.makedirs(out, exist_ok=True)
    groups = recount(dirs)
    json.dump(groups, open(os.path.join(out, "interval_cut.json"), "w"), indent=1)
    figure(groups, os.path.join(out, "interval_cut.png"))
    for k, g in groups.items():
        print(f"{k:62s} windows {g['windows']:5d}  50%: {g['held50'][0]:.3f} -> {g['held50'][1]:.3f}  "
              f"90%: {g['held90'][0]:.3f} -> {g['held90'][1]:.3f}  width {g['width90'][0]:.4f} -> "
              f"{g['width90'][1]:.4f}  lost {g['hit90'][0] - g['hit90'][1]} / {g['hit50'][0] - g['hit50'][1]}")


if __name__ == "__main__":
    main()
