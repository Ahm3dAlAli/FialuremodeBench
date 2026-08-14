#!/usr/bin/env python3
"""FailureModeBench — single reproduction entry point.

Regenerates every ANALYSIS artifact (tables, figures, judge-validity) from the
committed corpora in results/*/. No GPU and no API key required — model inference
and LLM-judging are already cached in results/. To re-run those heavy stages, see
the pipeline documented in README.md (uses failuremodebench.cli).

    python reproduce.py            # run all analysis stages
    python reproduce.py figures    # just the figures
    python reproduce.py agreement  # just the judge-validity numbers
    python reproduce.py tables     # aggregate the per-family tables
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=HERE, check=True)


STAGES = {
    "tables":    [PY, "-m", "failuremodebench.cli", "--run", "main2b", "aggregate"],
    "figures":   [PY, "scripts/make_figures.py"],
    "agreement": [PY, "scripts/multi_annotator_agreement.py", "--run", "main2b"],
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    order = ["tables", "figures", "agreement"] if which == "all" else [which]
    for s in order:
        if s not in STAGES:
            print(f"unknown stage {s!r}; choose from {list(STAGES)} or 'all'"); sys.exit(1)
        run(STAGES[s])
    print("\n[reproduce] done. Figures -> docs/figures/ ; tables -> results/main2b/tables/ ;"
          " judge-validity printed above.")


if __name__ == "__main__":
    main()
