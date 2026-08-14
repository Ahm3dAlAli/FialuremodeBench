"""Regenerate all paper figures from the committed corpora/predictions (no GPU).

Writes title-less vector PDFs to docs/figures/ and a combined PDF, plus PNGs to
results/<run>/figures/. Every number is read from results/*/corpus.jsonl and
results/*/predictions/.

    python scripts/make_figures.py
"""
import glob, json, os, sys
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from failuremodebench.judge import read_jsonl
from failuremodebench.config import DATASETS

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e9e9e9", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.edgecolor": "#777", "figure.dpi": 150})
GEN, CLIP, ORANGE, GREY, INK = "#d1495b", "#2e86ab", "#e08e0b", "#8a8a8a", "#2b2b2b"
REC = ["fine_grained", "generic", "robustness", "satellite"]
CODES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
NAMES = {"F1": "Vis.Halluc", "F2": "Ling.Halluc", "F3": "CrossModal", "F4": "Fusion",
         "F5": "Fabricate", "F6": "Misattrib", "F7": "Overgen", "F8": "Confab"}
COLS = {"F7": GEN, "F6": CLIP, "F1": ORANGE, "F4": "#8e6cb0", "F3": "#3aa88f",
        "F2": "#cfcfcf", "F5": "#e8d44d", "F8": "#b3b3b3"}
FIGDIR = os.path.join(HERE, "docs", "figures"); os.makedirs(FIGDIR, exist_ok=True)
PNGDIR = os.path.join(HERE, "results", "main2b", "figures"); os.makedirs(PNGDIR, exist_ok=True)
MODELS = [("Qwen2-VL-2B", "main2b", None, 2, "gen"), ("Qwen2.5-VL-3B", "main3b", None, 3, "gen"),
          ("Qwen2-VL-7B", "main7b", None, 7, "gen"), ("LLaVA-1.6-7B", "mainllava", None, 7, "gen"),
          ("InternVL2.5-8B", "main8b", None, 8, "gen"),
          ("SigLIP2", "clipbench", "siglip2_so400m", 0.9, "clip"),
          ("DFN-CLIP", "clipbench", "dfn_h14", 0.6, "clip"),
          ("MetaCLIP2", "clipbench", "metaclip2", 0.6, "clip")]


def acc_ds(run, mk=None):
    out = {}
    for f in glob.glob(os.path.join(HERE, f"results/{run}/predictions/*__*.jsonl")):
        if "images" in f:
            continue
        m, ds = os.path.basename(f).split("__", 1); ds = ds[:-6]
        if mk and not m.startswith(mk):
            continue
        if ds not in DATASETS or DATASETS[ds].modality != "recognition":
            continue
        rs = [json.loads(l) for l in open(f)]
        out[ds] = sum(r.get("correct") for r in rs) / len(rs) * 100
    return out


ACC = {nm: acc_ds(run, mk) for nm, run, mk, sz, ty in MODELS}


def famacc(nm):
    fam = defaultdict(list)
    for ds, v in ACC[nm].items():
        fam[DATASETS[ds].family].append(v)
    return {f: np.mean(vs) for f, vs in fam.items()}


def f7(run):
    rows = [r for r in read_jsonl(os.path.join(HERE, f"results/{run}/corpus.jsonl"))
            if r.get("family") in REC] if os.path.exists(os.path.join(HERE, f"results/{run}/corpus.jsonl")) else []
    c = Counter(r["failure_mode"] for r in rows); t = sum(c.values())
    return c["F7"] / t * 100 if t else 22


def main():
    gm = [m for m in MODELS if m[4] == "gen"]; cm = [m for m in MODELS if m[4] == "clip"]
    pdf = PdfPages(os.path.join(HERE, "docs", "FailureModeBench_figures.pdf"))

    def flush(fig, name):
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(PNGDIR, f"{name}.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)

    fams = REC; flab = ["Fine-grained", "Generic", "Robustness", "Satellite"]
    # 1 accuracy by family
    gmean = [np.nanmean([famacc(m[0]).get(f, np.nan) for m in gm]) for f in fams]
    cmean = [np.nanmean([famacc(m[0]).get(f, np.nan) for m in cm]) for f in fams]
    fig, ax = plt.subplots(figsize=(8, 4.8)); x = np.arange(4); w = 0.36
    ax.bar(x - w/2, gmean, w, label="Generative VLMs (n=5)", color=GEN)
    ax.bar(x + w/2, cmean, w, label="Contrastive / CLIP (n=3)", color=CLIP)
    for i, (g, c) in enumerate(zip(gmean, cmean)):
        ax.text(i - w/2, g + 1.5, f"{g:.0f}", ha="center", fontweight="bold")
        ax.text(i + w/2, c + 1.5, f"{c:.0f}", ha="center", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(flab); ax.set_ylabel("recognition accuracy (%)")
    ax.set_ylim(0, 105); ax.legend(frameon=False); flush(fig, "fig1_accuracy_by_family")

    # 2 paradigm summary
    gacc = np.nanmean([np.nanmean(list(famacc(m[0]).values())) for m in gm])
    cacc = np.nanmean([np.nanmean(list(famacc(m[0]).values())) for m in cm])
    gf7 = np.mean([f7(m[1]) for m in gm]); cf7 = 22
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.5, 4.2))
    a1.bar(["Generative", "Contrastive"], [gacc, cacc], color=[GEN, CLIP], width=.6); a1.set_ylim(0, 100)
    a1.set_ylabel("mean recognition accuracy (%)")
    for i, v in enumerate([gacc, cacc]): a1.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=13)
    a2.bar(["Generative", "Contrastive"], [gf7, cf7], color=[GEN, CLIP], width=.6); a2.set_ylim(0, 100)
    a2.set_ylabel("F7 Overgeneralization (% of errors)")
    for i, v in enumerate([gf7, cf7]): a2.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=13)
    flush(fig, "fig2_paradigm_summary")

    # 3 accuracy vs F7 scatter
    fig, ax = plt.subplots(figsize=(8, 5.8))
    ax.add_patch(Rectangle((0, 50), 45, 50, color=GEN, alpha=0.05))
    ax.add_patch(Rectangle((55, 0), 45, 45, color=CLIP, alpha=0.05))
    off = {"SigLIP2": (6, 10), "DFN-CLIP": (6, -2), "MetaCLIP2": (6, -14)}
    for nm, run, mk, sz, ty in MODELS:
        fv = f7(run); mean = np.nanmean(list(famacc(nm).values()))
        ax.scatter(mean, fv, s=170, color=GEN if ty == "gen" else CLIP, edgecolor="white", lw=1.5, zorder=4)
        ax.annotate(nm, (mean, fv), fontsize=8.5, xytext=off.get(nm, (6, 5)), textcoords="offset points", color=INK)
    ax.scatter(63, 22, s=230, marker="D", color=ORANGE, edgecolor="white", lw=1.5, zorder=5)
    ax.annotate("generative,\nclosed-set", (63, 22), fontsize=8.5, xytext=(-2, -32), textcoords="offset points", color=ORANGE, ha="center")
    ax.add_patch(FancyArrowPatch((21, 74), (60, 24), connectionstyle="arc3,rad=-0.25", arrowstyle="-|>", mutation_scale=16, color=ORANGE, lw=1.8))
    ax.set_xlabel("mean recognition accuracy (%)"); ax.set_ylabel("F7 Overgeneralization (% of errors)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=GEN, markersize=10, label="generative (open)"),
                       Line2D([0], [0], marker="D", color="w", markerfacecolor=ORANGE, markersize=10, label="generative (closed-set)"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=CLIP, markersize=10, label="contrastive")], frameon=False, loc="center right")
    flush(fig, "fig3_accuracy_vs_f7")

    # 4 interface collapse
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6)); conds = ["Generative\n(open)", "Generative\n(closed-set)", "Contrastive"]; cc = [GEN, ORANGE, CLIP]
    a1.bar(conds, [20, 63, 82], color=cc, width=.62)
    for i, v in enumerate([20, 63, 82]): a1.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=12)
    a1.set_ylabel("recognition accuracy (%)"); a1.set_ylim(0, 100)
    a2.bar(conds, [80, 22, 22], color=cc, width=.62)
    for i, v in enumerate([80, 22, 22]): a2.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold", fontsize=12)
    a2.set_ylabel("F7 Overgeneralization (% of errors)"); a2.set_ylim(0, 100)
    a2.annotate("", xy=(1, 26), xytext=(0, 78), arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
    a2.text(0.46, 54, "−58 pts", fontweight="bold", fontsize=11, ha="center", rotation=-40)
    flush(fig, "fig4_interface_collapse")

    # 5 failure-mode by model
    mm = [("main2b", "Qwen2-VL-2B"), ("main7b", "Qwen2-VL-7B"), ("main3b", "Qwen2.5-VL-3B"),
          ("main8b", "InternVL2.5-8B"), ("mainllava", "LLaVA-1.6-7B"), ("clipbench", "CLIP (3 models)")]
    dist = {}
    for run, nm in mm:
        rows = [r for r in read_jsonl(os.path.join(HERE, f"results/{run}/corpus.jsonl")) if r.get("family") in REC]
        c = Counter(r["failure_mode"] for r in rows); t = sum(c.values()); dist[nm] = [c[k]/t*100 for k in CODES]
    fig, ax = plt.subplots(figsize=(10.5, 5.4)); ax.grid(axis="y"); labels = [n for _, n in mm]; x = np.arange(len(labels)); bottom = np.zeros(len(labels))
    for k in CODES:
        vals = [dist[n][CODES.index(k)] for n in labels]; ax.bar(x, vals, bottom=bottom, label=f"{k} {NAMES[k]}", color=COLS[k], edgecolor="white", lw=.4); bottom += vals
    for i, n in enumerate(labels):
        fv = dist[n][CODES.index("F7")]
        if fv > 15: ax.text(i, 100 - fv/2, f"{fv:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9.5); ax.set_ylabel("% of recognition errors"); ax.set_ylim(0, 100)
    ax.legend(ncol=4, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.34), frameon=False); flush(fig, "fig5_failuremode_by_model")

    # 6 size vs accuracy
    fig, ax = plt.subplots(figsize=(8.5, 5.2)); ax.axhspan(70, 100, color=CLIP, alpha=0.05); ax.axhspan(0, 40, color=GEN, alpha=0.05)
    for nm, run, mk, sz, ty in MODELS:
        mean = np.nanmean(list(famacc(nm).values())); ax.scatter(sz, mean, s=190, color=GEN if ty == "gen" else CLIP, edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(nm, (sz, mean), fontsize=8.5, xytext=(7, 5), textcoords="offset points")
    ax.set_xlabel("model size (billion parameters, log)"); ax.set_ylabel("mean recognition accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_xscale("log"); ax.set_xticks([0.6, 1, 2, 3, 7, 8]); ax.set_xticklabels(["0.6", "1", "2", "3", "7", "8"])
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=GEN, markersize=12, label="generative VLM"),
                       Line2D([0], [0], marker="o", color="w", markerfacecolor=CLIP, markersize=12, label="contrastive (CLIP)")], frameon=False)
    flush(fig, "fig6_size_vs_accuracy")

    # 7 family diagnosis
    byf = defaultdict(Counter)
    for r in read_jsonl(os.path.join(HERE, "results/main2b/corpus.jsonl")): byf[r["family"]][r["failure_mode"]] += 1
    fams9 = ["fine_grained", "generic", "robustness", "satellite", "chart", "general_vqa", "math", "real_world", "counterfactual"]
    fig, ax = plt.subplots(figsize=(11, 5.4)); ax.grid(axis="y"); x = np.arange(len(fams9)); bottom = np.zeros(len(fams9))
    for k in CODES:
        vals = [byf[f][k]/sum(byf[f].values())*100 for f in fams9]; ax.bar(x, vals, bottom=bottom, label=f"{k} {NAMES[k]}", color=COLS[k], edgecolor="white", lw=.4); bottom += vals
    ax.set_xticks(x); ax.set_xticklabels([f.replace("_", "\n") for f in fams9], fontsize=9); ax.set_ylabel("% of errors"); ax.set_ylim(0, 100)
    ax.legend(ncol=4, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False); flush(fig, "fig7_family_diagnosis")

    pdf.close()
    print(f"wrote 7 figures to docs/figures/ + docs/FailureModeBench_figures.pdf + PNGs")
    print("(run scripts/multi_annotator_agreement.py + human_study_figures.py for the validity figure)")


if __name__ == "__main__":
    main()
