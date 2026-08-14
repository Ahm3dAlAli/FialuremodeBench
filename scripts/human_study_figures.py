"""Figures for the human-study / judge-validation section.

Generates (into results/<run>/figures/):
  human_agreement_by_family.png  -- per-family human-vs-judge agreement bar
  human_judge_confusion.png      -- 8x8 human->judge confusion heatmap
  judge_vs_human_kappa.png       -- per-judge Cohen's kappa vs human (if panel present)

Uses only the LABELED errors (unlabeled excluded; GW/mislabel excluded from kappa).

    python scripts/human_study_figures.py --run main2b \
        --annotations human_annotations-11_first500.json
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from failuremodebench import taxonomy
from failuremodebench.judge import read_jsonl
from failuremodebench.multijudge import cohen_kappa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main2b")
    ap.add_argument("--annotations", required=True)
    a = ap.parse_args()
    run_dir = os.path.join(HERE, "results", a.run)
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    corp = {r["error_key"]: r for r in read_jsonl(os.path.join(run_dir, "corpus.jsonl"))}
    ann = json.load(open(a.annotations))

    # pairs (exclude unlabeled, GW, NA)
    pairs = []
    for r in ann:
        h = r.get("human"); j = corp.get(r["error_key"], {}).get("failure_mode")
        if h and h != "GW" and j and j != "NA":
            pairs.append((h, j, corp[r["error_key"]].get("family", "?")))
    n = len(pairs)
    kap = cohen_kappa([p[0] for p in pairs], [p[1] for p in pairs])
    raw = sum(p[0] == p[1] for p in pairs) / n
    gw = sum(1 for r in ann if r.get("human") == "GW")
    lab = sum(1 for r in ann if r.get("human"))
    print(f"n={n} labeled pairs, kappa={kap:.3f}, raw={raw*100:.1f}%, "
          f"mislabel={gw}/{lab}={gw/lab*100:.1f}%")

    codes = taxonomy.CODES
    CO = "#2a6fb0"

    # 1) per-family agreement
    byf = defaultdict(lambda: [0, 0])
    for h, j, f in pairs:
        byf[f][1] += 1; byf[f][0] += int(h == j)
    fams = sorted(byf, key=lambda f: byf[f][0] / byf[f][1])
    vals = [byf[f][0] / byf[f][1] * 100 for f in fams]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(fams, vals, color=CO)
    for i, (f, v) in enumerate(zip(fams, vals)):
        ax.text(v + 1, i, f"{v:.0f}% (n={byf[f][1]})", va="center", fontsize=9)
    ax.axvline(raw * 100, color="#c0392b", ls="--", lw=1,
               label=f"overall {raw*100:.0f}% (κ={kap:.2f})")
    ax.set_xlim(0, 105); ax.set_xlabel("human–judge agreement (%)")
    ax.set_title(f"Human vs judge agreement by family (n={n})")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "human_agreement_by_family.png"), dpi=140)
    plt.close(fig)

    # 2) confusion heatmap
    conf = Counter((h, j) for h, j, _ in pairs)
    M = [[conf.get((hc, jc), 0) for jc in codes] for hc in codes]
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(M, cmap="Blues")
    ax.set_xticks(range(len(codes))); ax.set_xticklabels(codes)
    ax.set_yticks(range(len(codes))); ax.set_yticklabels(codes)
    ax.set_xlabel("judge (deepseek-v4-flash)"); ax.set_ylabel("human")
    ax.set_title("Human → judge confusion (counts)")
    for i in range(len(codes)):
        for k in range(len(codes)):
            if M[i][k]:
                ax.text(k, i, M[i][k], ha="center", va="center", fontsize=8,
                        color="white" if M[i][k] > max(max(r) for r in M) / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "human_judge_confusion.png"), dpi=140)
    plt.close(fig)

    # 3) per-judge kappa vs human (from the panel verdict files, if present)
    mj = os.path.join(run_dir, "multijudge_human")
    human = {r["error_key"]: r["human"] for r in ann if r.get("human") and r["human"] != "GW"}
    rows = []
    for f in glob.glob(os.path.join(mj, "corpus_*.jsonl")):
        name = os.path.basename(f)[len("corpus_"):-6].replace("_", "/")
        v = {r["error_key"]: r["failure_mode"] for r in read_jsonl(f)}
        pp = [(human[k], v[k]) for k in human if k in v and v[k] != "NA"]
        if len(pp) < 20:
            continue
        rows.append((name, cohen_kappa([x for x, _ in pp], [y for _, y in pp]), len(pp)))
    if rows:
        rows.sort(key=lambda r: r[1])
        fig, ax = plt.subplots(figsize=(7, 4))
        names = [r[0] for r in rows]; ks = [r[1] for r in rows]
        ax.barh(names, ks, color=["#c0392b" if "deepseek-v4" in n else CO for n in names])
        for i, r in enumerate(rows):
            ax.text(r[1] + 0.005, i, f"{r[1]:.3f} (n={r[2]})", va="center", fontsize=9)
        ax.set_xlabel("Cohen's κ vs human"); ax.set_title("Which judge best matches humans")
        fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "judge_vs_human_kappa.png"), dpi=140)
        plt.close(fig)
        print("wrote judge_vs_human_kappa.png (%d judges)" % len(rows))

    print("figures ->", fig_dir)


if __name__ == "__main__":
    main()
