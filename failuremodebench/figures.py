"""Figures for the paper. Non-interactive Agg backend (rolf has no X server)."""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import taxonomy


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.reader(f))


def stacked_failure_bar(family_csv: str, out_png: str, title="Failure modes by task family"):
    rows = _read_csv(family_csv)
    header, data = rows[0], rows[1:]
    codes = taxonomy.CODES
    pct_start = 2 + len(codes)
    families = [r[0] for r in data]
    series = {c: [float(r[pct_start + i]) for r in data] for i, c in enumerate(codes)}

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(families)), 5))
    bottom = [0.0] * len(families)
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(codes):
        vals = series[c]
        ax.bar(families, vals, bottom=bottom, label=f"{c} {taxonomy.BY_CODE[c].name}",
               color=cmap(i % 10))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("% of annotated errors")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def accuracy_heatmap(accuracy_csv: str, out_png: str):
    rows = _read_csv(accuracy_csv)[1:]
    models = sorted({r[0] for r in rows})
    datasets = sorted({r[1] for r in rows})
    grid = [[float("nan")] * len(datasets) for _ in models]
    mi = {m: i for i, m in enumerate(models)}
    di = {d: i for i, d in enumerate(datasets)}
    for r in rows:
        grid[mi[r[0]]][di[r[1]]] = float(r[4])
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(datasets)), 1 + 0.6 * len(models)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(datasets)), datasets, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    for i in range(len(models)):
        for j in range(len(datasets)):
            v = grid[i][j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="w" if v < 0.6 else "k", fontsize=7)
    fig.colorbar(im, ax=ax, label="accuracy")
    ax.set_title("Accuracy: model x dataset")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def make_all(out_dir: str, fig_dir: str) -> list[str]:
    made = []
    fam = os.path.join(out_dir, "failure_by_family.csv")
    acc = os.path.join(out_dir, "accuracy.csv")
    if os.path.exists(fam):
        made.append(stacked_failure_bar(fam, os.path.join(fig_dir, "failure_by_family.png")))
    if os.path.exists(acc):
        made.append(accuracy_heatmap(acc, os.path.join(fig_dir, "accuracy_heatmap.png")))
    return made
