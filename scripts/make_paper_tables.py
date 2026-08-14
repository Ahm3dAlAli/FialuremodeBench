"""Generate paper-ready LaTeX tables + a filled results narrative from the run's
CSVs. Reproducible: reads results/<run>/tables/*.csv and the multijudge summary.

    python scripts/make_paper_tables.py --run main2b
Writes docs/paper_tables.tex and docs/PAPER_RESULTS.md.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from failuremodebench import taxonomy
from failuremodebench.config import DATASETS, FAMILY_DISPLAY


def _read(path):
    with open(path) as f:
        return list(csv.reader(f))


def table3_latex(run_dir) -> str:
    """Main results: per-model recognition acc / VQA score / macro-F1."""
    rows = _read(os.path.join(run_dir, "tables", "main_results.csv"))[1:]
    body = ""
    for m, ra, vs, mf in rows:
        f = lambda x: "--" if x in ("", "None") else f"{float(x)*100:.1f}"
        body += f"    {m} & {f(ra)} & {f(vs)} & {f(mf)} \\\\\n"
    return ("\\begin{table}[t]\\centering\n\\caption{Main results (\\%). "
            "Recognition = mean top-1; VQA = mean VLMEvalKit score; Macro-F1 on "
            "fine-grained sets.}\\label{tab:main}\n"
            "\\begin{tabular}{lccc}\\toprule\n"
            "Model & Recog.\\ Acc.\\ $\\uparrow$ & VQA Score $\\uparrow$ & "
            "Macro-F1 $\\uparrow$ \\\\\\midrule\n" + body +
            "\\bottomrule\\end{tabular}\\end{table}\n")


def table4_latex(run_dir) -> str:
    """Failure-mode distribution by family."""
    rows = _read(os.path.join(run_dir, "tables", "failure_by_family.csv"))
    h = rows[0]; pi = h.index("F1_pct")
    body = ""
    for r in rows[1:]:
        fam, n = r[0], r[1]
        pcts = " & ".join(f"{float(r[pi+i]):.0f}" for i in range(8))
        body += f"    {fam} & {n} & {pcts} \\\\\n"
    cols = " & ".join(taxonomy.CODES)
    return ("\\begin{table}[t]\\centering\n\\caption{Failure-mode distribution by "
            "task family (\\% of judged errors, rows sum to $\\approx$100). Judge: "
            "deepseek-v4-flash.}\\label{tab:failure}\n"
            "\\begin{tabular}{lr" + "c"*8 + "}\\toprule\n"
            f"Family & $n$ & {cols} \\\\\\midrule\n" + body +
            "\\bottomrule\\end{tabular}\\end{table}\n")


def multijudge_latex(run_dir) -> str:
    p = os.path.join(run_dir, "multijudge", "multijudge_summary.json")
    if not os.path.exists(p):
        return "% (no multi-judge summary found)\n"
    s = json.load(open(p))
    body = ""
    for pair, v in s["pairwise_agreement"].items():
        body += f"    {pair} & {v['raw_agreement']*100:.1f} & {v['cohen_kappa']:.2f} \\\\\n"
    return ("\\begin{table}[t]\\centering\n\\caption{Inter-judge agreement over "
            f"{s['n_common_errors']} errors (8-way F1--F8).}}\\label{{tab:judges}}\n"
            "\\begin{tabular}{lcc}\\toprule\n"
            "Judge pair & Raw agr.\\ (\\%) & Cohen's $\\kappa$ \\\\\\midrule\n"
            + body + "\\bottomrule\\end{tabular}\\end{table}\n")


def narrative(run_dir) -> str:
    rows = _read(os.path.join(run_dir, "tables", "failure_by_family.csv"))
    h = rows[0]; pi = h.index("F1_pct")
    dom = {}
    for r in rows[1:]:
        pct = [float(r[pi+i]) for i in range(8)]
        top = max(range(8), key=lambda i: pct[i])
        dom[r[0]] = (taxonomy.CODES[top], pct[top])
    recog_fams = ["Generic", "Robustness", "Fine-grained", "Satellite"]
    recog = [d for d in dom if d in recog_fams]
    f7 = [dom[d][1] for d in recog if dom[d][0] == "F7"]
    lines = ["## Filled result sentences (replace the paper's [TODO]s)\n"]
    if f7:
        lines.append(f"- **Recognition is dominated by Overgeneralization (F7):** "
                     f"{len(f7)}/{len(recog)} recognition families have F7 as their "
                     f"dominant mode ({min(f7):.0f}–{max(f7):.0f}\\%).")
    lines.append("- **Recognition and reasoning occupy different taxonomy regions:** "
                 "recognition families concentrate in F7, while VQA families "
                 f"(General, Real-World, Chart) are dominated by "
                 f"{', '.join(sorted(set(dom[d][0] for d in dom if d not in recog_fams)))}.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="main2b")
    a = ap.parse_args()
    run_dir = os.path.join(HERE, "results", a.run)
    tex = "\n".join([table3_latex(run_dir), table4_latex(run_dir),
                     multijudge_latex(run_dir)])
    os.makedirs(os.path.join(HERE, "docs"), exist_ok=True)
    with open(os.path.join(HERE, "docs", "paper_tables.tex"), "w") as f:
        f.write(tex)
    with open(os.path.join(HERE, "docs", "PAPER_RESULTS.md"), "w") as f:
        f.write("# Paper-ready results\n\n"
                "LaTeX tables are in `paper_tables.tex` (Tables 3, 4, and the "
                "inter-judge agreement table).\n\n" + narrative(run_dir))
    print("wrote docs/paper_tables.tex and docs/PAPER_RESULTS.md")
    print("\n--- Table 3 + Table 4 + judges (LaTeX) ---\n")
    print(tex)


if __name__ == "__main__":
    main()
