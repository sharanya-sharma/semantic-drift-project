"""
merge_ldt.py
------------
Merges drift_scores.csv with the Hindi LDT reaction time dataset.

The LDT file uses 'stimulus' as the word column.
The drift file uses 'word' as the word column.
We merge on these, keeping only words present in both.

Output: results/merged_ldt_drift.csv
Columns: word | RT | drift_score

Usage:
    python src/analysis/merge_ldt.py
"""

import pandas as pd
from pathlib import Path

DRIFT_PATH = "drift/drift_scores.csv"
LDT_PATH   = "data/ldt/hindi_ldt.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = "results/merged_ldt_drift.csv"


def main():
    # ── Load drift scores ──────────────────────────────────────────
    print("Loading drift scores...")
    drift = pd.read_csv(DRIFT_PATH)
    print(f"  Drift words: {len(drift):,}")

    # ── Load LDT data ──────────────────────────────────────────────
    print("Loading LDT data...")
    ldt = pd.read_csv(LDT_PATH, index_col=0)
    print(f"  LDT words: {len(ldt):,}")
    print(f"  LDT columns: {ldt.columns.tolist()}")

    # ── Merge on word column ───────────────────────────────────────
    # LDT uses 'stimulus', drift uses 'word'
    print("\nMerging on word column...")
    merged = ldt.merge(drift, left_on="stimulus", right_on="word", how="inner")
    print(f"  Words after merge: {len(merged):,}")

    # ── Keep only needed columns ───────────────────────────────────
    merged = merged[["word", "RT", "drift_score"]].dropna()
    print(f"  Words after dropping NaN: {len(merged):,}")

    # ── Save ───────────────────────────────────────────────────────
    merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved → {OUTPUT_PATH}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n── Summary statistics ──────────────────────────")
    print(merged[["RT", "drift_score"]].describe().round(4).to_string())

    print("\n── Sample rows ──────────────────────────────────")
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()