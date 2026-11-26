"""Chart: speed of every kernel against torch.matmul, per batch size and GPU.

Reads the benchmark CSVs in results/ and writes results/speed_vs_torch.png.
Needs only matplotlib, no GPU.

    python plot.py
"""
import csv
import glob
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M_VALUES = [1, 16, 128, 512, 4096]
# One fixed colour and marker per kernel, the same in both panels.
STYLE = {
    "v1_basic": ("#257de0", "o", "v1 basic (Triton)"),
    "v2_grouped": ("#f47230", "s", "v2 grouped (Triton)"),
    "v3_autotuned": ("#00b894", "D", "v3 autotuned (Triton)"),
    "v4_gluon": ("#f2a600", "^", "v4 Gluon, mma.sync"),
    "v5_gluon_hopper": ("#ee7fac", "v", "v5 Gluon, TMA + wgmma"),
}
PANELS = [
    ("RTX 4090", "results/bench_nvidia_geforce_rtx_4090*.csv", ["v1_basic", "v2_grouped", "v3_autotuned"]),
    ("H100 NVL", "results/bench_nvidia_h100_nvl*.csv", ["v3_autotuned", "v4_gluon", "v5_gluon_hopper"]),
]
INK, MUTED, GRID, AXIS = "#191919", "#666666", "#e7e7e7", "#c2c2c2"


def load(pattern):
    """{kernel: {M: [speedup in each run]}} over every CSV matching the pattern."""
    data = defaultdict(lambda: defaultdict(list))
    files = sorted(glob.glob(pattern))
    for path in files:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                data[row["kernel"]][int(row["M"])].append(float(row["speedup_vs_torch"]))
    return data, len(files)


def spread_labels(ys, gap):
    """Nudge label heights apart so they don't overlap, keeping their order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    xs = range(len(M_VALUES))

    for ax, (gpu, pattern, kernels) in zip(axes, PANELS):
        data, runs = load(pattern)
        ax.plot([-0.3, len(M_VALUES) - 0.7], [1.0, 1.0], color=INK, linewidth=1, zorder=1)

        ends = []
        for kernel in kernels:
            color, marker, _ = STYLE[kernel]
            mean = [sum(data[kernel][m]) / len(data[kernel][m]) for m in M_VALUES]
            if runs > 1:  # thin bar from the slowest to the fastest run
                lo = [min(data[kernel][m]) for m in M_VALUES]
                hi = [max(data[kernel][m]) for m in M_VALUES]
                ax.vlines(xs, lo, hi, color=color, linewidth=1.2, zorder=2)
            ax.plot(xs, mean, color=color, linewidth=2, marker=marker, markersize=8,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            ends.append(mean[-1])

        # Labels at the right end of each line, plus one for the 1.0 reference line.
        names = [STYLE[k][2] for k in kernels] + ["torch.matmul (cuBLAS)"]
        heights = spread_labels(ends + [1.0], 0.045)
        for name, y_text in zip(names, heights):
            ax.text(len(M_VALUES) - 0.62, y_text, name, fontsize=9, va="center",
                    color=MUTED if name.startswith("torch") else INK)

        note = f"mean of {runs} runs, bars show min to max" if runs > 1 else "one run"
        ax.set_title(f"{gpu}  ({note})", loc="left", fontsize=10.5, color=INK, pad=10)
        ax.set_xticks(list(xs), [str(m) for m in M_VALUES])
        ax.set_xlabel("batch size M (rows of x)", color=MUTED)
        ax.set_xlim(-0.3, len(M_VALUES) + 1.4)
        ax.set_ylim(0.5, 1.25)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=MUTED, length=0)

    axes[0].set_ylabel("speed relative to torch.matmul\n(above 1.0 = faster)", color=MUTED)
    axes[0].set_yticks([0.6, 0.8, 1.0, 1.2], ["0.6×", "0.8×", "1.0×", "1.2×"])
    fig.suptitle("Matmul kernel speed against cuBLAS:  x[M, 4096] @ W[4096, 14336], bf16",
                 x=0.012, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.012, 0.012, "Each panel is tuned and measured on its own GPU; do not compare heights across panels. "
             "v1 and v2 on the H100 (0.46–0.72×) are left out.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig("results/speed_vs_torch.png", dpi=170, facecolor="white")
    print("saved results/speed_vs_torch.png")


if __name__ == "__main__":
    main()
