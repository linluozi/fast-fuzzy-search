"""Regenerate the figures in docs/ from live measurements.

    python docs/make_figures.py

Every figure is plotted as a *ratio* to the naive full scan, never as absolute
milliseconds. Wall-clock times on a shared machine can drift 2x between runs
while the ratios stay put, so ratios are the part worth publishing and the only
part comparable across hardware. Per-round ratios are taken before averaging,
so background load hits every arm in a round equally.
"""
import os
import random
import string
import sys
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from fast_fuzzy import _full_distance, _levenshtein_banded, levenshtein_bounded  # noqa: E402
import benchmark as bm  # noqa: E402

INK = "#2b2420"
MUTED = "#9a9188"
FADE = "#e8e2da"
LENF = "#c8955a"
LIB = "#6b3f24"
ACCENT = "#b34a2f"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def plain(a, b, k):
    d = _full_distance(a, b)
    return d if d <= k else None


def plain_early(a, b, k):
    if len(a) < len(b):
        a, b = b, a
    m = len(b)
    prev = list(range(m + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > k:
            return None
        prev = cur
    return prev[m] if prev[m] <= k else None


VARIANTS = (
    ("plain", plain),
    ("early", plain_early),
    ("banded", _levenshtein_banded),
    ("library", levenshtein_bounded),
)


def measure_lengths(k=2, lengths=(2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 26, 32),
                    n=4000, rounds=11):
    """Per-round time for each variant; median of per-round speedups vs plain."""
    rng = random.Random(2026)
    out = {"len": [], "sp_early": [], "sp_banded": [], "sp_library": [],
           "banded_vs_early": []}
    for L in lengths:
        words = [''.join(rng.choice(string.ascii_lowercase) for _ in range(L)) for _ in range(n)]
        q = ''.join(rng.choice(string.ascii_lowercase) for _ in range(L))
        rounds_data = []
        for _ in range(rounds):
            timings = {}
            for name, fn in VARIANTS:
                t = time.perf_counter()
                hits = sum(1 for w in words if fn(q, w, k) is not None)
                timings[name] = time.perf_counter() - t
            timings["hits"] = hits
            rounds_data.append(timings)

        out["len"].append(L)
        out["sp_early"].append(statistics.median(d["plain"] / d["early"] for d in rounds_data))
        out["sp_banded"].append(statistics.median(d["plain"] / d["banded"] for d in rounds_data))
        out["sp_library"].append(statistics.median(d["plain"] / d["library"] for d in rounds_data))
        out["banded_vs_early"].append(
            statistics.median(d["early"] / d["banded"] for d in rounds_data))
        print(f"  L={L:>3}  speedup  early={out['sp_early'][-1]:6.2f}x  "
              f"banded={out['sp_banded'][-1]:6.2f}x  lib={out['sp_library'][-1]:6.2f}x",
              flush=True)
    return out


def measure_scenarios(reps=3):
    """Run the benchmark scenarios `reps` times, keep the median of each ratio."""
    rows = []
    for label, factory in bm.CASES:
        runs = [bm.run(label, factory) for _ in range(reps)]
        rows.append({
            "label": label,
            "hits": runs[0]["hits"],
            "a": statistics.median(r["a"] for r in runs),
            "b": statistics.median(r["b"] for r in runs),
            "c": statistics.median(r["c"] for r in runs),
            "lenf": statistics.median(r["lenf"] for r in runs),
            "band": statistics.median(r["band"] for r in runs),
            "total": statistics.median(r["total"] for r in runs),
        })
        print(f"  {label:<20} hits={rows[-1]['hits']:>5}  "
              f"B/A={rows[-1]['lenf']:5.2f}x  C/B={rows[-1]['band']:5.2f}x  "
              f"C/A={rows[-1]['total']:6.2f}x", flush=True)
    return rows


def fig_crossover(data, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 9.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.55, 1], "hspace": 0.16})

    for key, color, label, lw, ls, z in (
        ("sp_early", LENF, "plain DP + early exit", 2.2, "-", 3),
        ("sp_banded", LIB, "banded DP (core, no dispatch)", 2.0, (0, (4, 3)), 4),
        ("sp_library", ACCENT, "this library (dispatched)", 2.6, "-", 5),
    ):
        ax1.plot(data["len"], data[key], marker="o", ms=4.2, lw=lw, ls=ls, color=color,
                 label=label, zorder=z, solid_capstyle="round")

    ax1.axhline(1.0, color=INK, lw=1.2, ls=(0, (5, 4)), zorder=2)
    ax1.text(21.5, 1.05, "parity with naive", fontsize=9, color=MUTED)
    ax1.axhspan(0.35, 1.0, color=ACCENT, alpha=0.06, zorder=1)
    ax1.text(21.5, 0.46, "slower than naive", fontsize=9,
             color=ACCENT, alpha=0.9)

    ax1.set_yscale("log")
    ax1.set_ylim(0.35, 60)
    ax1.margins(x=0.05)
    ax1.set_ylabel("speedup vs naive full scan  (higher is better)")
    ax1.set_title("The band is slower than doing nothing, until it isn't",
                  fontsize=13.5, weight="bold", pad=14, loc="left")
    ax1.grid(axis="y", color=FADE, lw=0.9, zorder=0)
    ax1.legend(frameon=False, loc="upper left", fontsize=9.5)
    ax1.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax1.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))

    ratio = data["banded_vs_early"]
    ax2.plot(data["len"], ratio, marker="o", ms=4.2, lw=2.2, color=LIB, zorder=4)
    ax2.axhline(1.0, color=INK, lw=1.2, ls=(0, (5, 4)), zorder=3)

    # The measured crossover wanders by a character or two between runs, so mark
    # the plateau where it lands rather than a single false-precision point.
    ax2.axvspan(9, 14, color=ACCENT, alpha=0.10, zorder=1)
    ax2.text(11.0, 2.16, "crossover ≈ 9", fontsize=9.2, color=ACCENT,
             ha="center", va="top")
    ax2.text(0.028, 0.93, "shipped cutoff 6k+2 = 14\nkept late on purpose",
             transform=ax2.transAxes, fontsize=9, color=ACCENT, va="top")

    ax2.axhspan(0.3, 1.0, color=ACCENT, alpha=0.06, zorder=1)
    ax2.text(0.012, 0.05, "shaded: the band loses here", transform=ax2.transAxes,
             fontsize=9, color=ACCENT, alpha=0.8)

    ax2.set_yscale("log")
    ax2.set_ylim(0.3, 2.3)
    ax2.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax2.set_ylabel("banded ÷ plain+early")
    ax2.set_xlabel("string length (query and candidate equal length)")
    ax2.grid(axis="y", color=FADE, lw=0.9, zorder=0)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


SHORT = {
    "zh 2-4 chars, k=2": "Chinese words, 2-4 chars",
    "en 5-8, k=2": "English, 5-8 chars, dense hits",
    "en 8-18, k=2": "English, 8-18 chars",
    "en 20-40, k=2": "English, 20-40 chars",
    "en 8-18, k=5": "English, 8-18 chars, k=5",
}


def fig_range(rows, path):
    rows = sorted(rows, key=lambda r: r["total"])
    labels = [SHORT.get(r["label"], r["label"]) for r in rows]
    vals = [r["total"] for r in rows]
    colors = [ACCENT if v < 1.15 else (LENF if v < 10 else LIB) for v in vals]

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    y = list(range(len(rows)))
    bars = ax.barh(y, vals, color=colors, height=0.6, zorder=3)
    ax.axvline(1.0, color=INK, lw=1.2, ls=(0, (5, 4)), zorder=4)

    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("this library ÷ naive full scan  (log, higher is better)")
    ax.set_title("One library, 130× spread — input length decides everything",
                 fontsize=13.5, weight="bold", pad=14, loc="left")
    ax.grid(axis="x", color=FADE, lw=0.9, zorder=0)
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlim(0.4, max(vals) * 2.2)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() * 1.12, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}×", va="center", fontsize=10,
                color=ACCENT if v < 1.15 else INK, weight="bold")

    ax.text(0.995, -0.155, "left of the dashed line = slower than the naive loop",
            ha="right", va="top", fontsize=9, color=MUTED,
            transform=ax.transAxes)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def fig_arms(rows, path):
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    n = len(rows)
    w = 0.34
    idx = list(range(n))

    # Use the per-round ratio medians, so these bars agree with benchmark.py.
    rel_b = [1.0 / r["lenf"] for r in rows]
    rel_c = [1.0 / r["total"] for r in rows]

    ax.bar([i - w / 2 for i in idx], rel_b, width=w, color=LENF,
           label="+ length filter", zorder=3)
    ax.bar([i + w / 2 for i in idx], rel_c, width=w, color=LIB,
           label="this library", zorder=3)
    ax.axhline(1.0, color=INK, lw=1.2, ls=(0, (5, 4)), zorder=4)

    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([SHORT.get(r["label"], r["label"]) for r in rows],
                       fontsize=9.3, rotation=12, ha="right")
    ax.set_ylabel("time relative to naive  (log, lower is better)")
    ax.set_title("Two changes, two very different amounts of work",
                 fontsize=13.5, weight="bold", pad=14, loc="left")
    ax.grid(axis="y", color=FADE, lw=0.9, zorder=0)
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper right")
    ax.set_ylim(0.003, 6)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))

    for i, (b, c) in enumerate(zip(rel_b, rel_c)):
        ax.text(i - w / 2, b * 1.18, f"{b:.2f}", ha="center", fontsize=8.6, color=INK)
        ax.text(i + w / 2, c * 1.18, f"{c:.2f}", ha="center", fontsize=8.6,
                color=INK, weight="bold")

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    print("measuring length sweep (k=2)...")
    data = measure_lengths()

    print("measuring benchmark scenarios (3 reps, median)...")
    rows = measure_scenarios()

    print("writing figures...")
    fig_crossover(data, os.path.join(HERE, "crossover.png"))
    fig_range(rows, os.path.join(HERE, "speedup-range.png"))
    fig_arms(rows, os.path.join(HERE, "three-arms.png"))


if __name__ == "__main__":
    main()
