"""
statistical_model.py
--------------------
Tests whether semantic drift predicts Hindi lexical decision reaction time (RT).

Three models as per synopsis:
    Baseline  : RT ~ drift_score only
    Full      : RT ~ drift_score + word_length + log_RT (control)

Since the LDT file only contains RT (no Zipf/Concreteness/AoA),
we use word length as the only available control variable.

Word length is computed from the word column directly.

Output:
    results/model_summary.csv   — model comparison table
    results/statistical_model_output.txt — full results

Usage:
    python src/analysis/statistical_model.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from pathlib import Path

MERGED_PATH = "results/merged_ldt_drift.csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def compute_aic(n, k, sse):
    """AIC = n * log(SSE/n) + 2k"""
    return n * np.log(sse / n) + 2 * k


def fit_ols(X, y, label):
    """Fit OLS regression and return metrics."""
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2  = r2_score(y, y_pred)
    sse = np.sum((y - y_pred) ** 2)
    n   = len(y)
    k   = X.shape[1] + 1  # predictors + intercept
    aic = compute_aic(n, k, sse)
    return {
        "model":       label,
        "R2":          round(r2, 4),
        "AIC":         round(aic, 2),
        "n":           n,
        "coefs":       dict(zip(range(X.shape[1]), model.coef_.round(4))),
        "intercept":   round(model.intercept_, 4),
    }


def main():
    # ── Load merged data ───────────────────────────────────────────
    print("Loading merged LDT + drift data...")
    df = pd.read_csv(MERGED_PATH)
    print(f"  Rows: {len(df):,}")

    # ── Add word length as control variable ────────────────────────
    df["word_length"] = df["word"].apply(len)

    y = df["RT"].values
    X_drift  = df[["drift_score"]].values
    X_length = df[["word_length"]].values
    X_full   = df[["drift_score", "word_length"]].values

    # ── Fit three models ───────────────────────────────────────────
    print("\nFitting models...")
    m_drift    = fit_ols(X_drift,  y, "Drift only  (RT ~ drift_score)")
    m_length   = fit_ols(X_length, y, "Baseline    (RT ~ word_length)")
    m_full     = fit_ols(X_full,   y, "Full model  (RT ~ drift_score + word_length)")

    # ── Simple t-test: is drift coefficient significant? ──────────
    slope, intercept, r, p, se = stats.linregress(
        df["drift_score"].values, y
    )
    sig = "SIGNIFICANT" if p < 0.05 else "NOT significant"

    # ── Print results ──────────────────────────────────────────────
    output_lines = []

    output_lines.append("=" * 60)
    output_lines.append("STATISTICAL MODELLING RESULTS")
    output_lines.append("Semantic Drift as Predictor of Reaction Time")
    output_lines.append("=" * 60)
    output_lines.append(f"\nDataset: {len(df):,} words (merged LDT + drift)")
    output_lines.append(f"RT range: {y.min():.1f} ms to {y.max():.1f} ms")
    output_lines.append(f"Mean RT:  {y.mean():.2f} ms")
    output_lines.append(f"Drift range: {df['drift_score'].min():.4f} to {df['drift_score'].max():.4f}")

    output_lines.append("\n── Model Comparison ─────────────────────────────")
    output_lines.append(f"{'Model':<45} {'R²':>6}  {'AIC':>10}")
    output_lines.append("-" * 65)
    for m in [m_drift, m_length, m_full]:
        output_lines.append(f"{m['model']:<45} {m['R2']:>6}  {m['AIC']:>10}")

    output_lines.append("\n── Drift Coefficient (simple linear regression) ─")
    output_lines.append(f"  Slope     : {slope:.4f}")
    output_lines.append(f"  R         : {r:.4f}")
    output_lines.append(f"  R²        : {round(r**2, 4)}")
    output_lines.append(f"  p-value   : {p:.6f}")
    output_lines.append(f"  Std Error : {se:.4f}")
    output_lines.append(f"\n  → Drift is {sig} at p < 0.05")

    output_lines.append("\n── Interpretation ───────────────────────────────")
    if p < 0.05 and slope > 0:
        output_lines.append("  Positive significant effect found.")
        output_lines.append("  Words with higher semantic drift show slower RT.")
        output_lines.append("  This supports the representational instability hypothesis:")
        output_lines.append("  domain-based semantic drift increases cognitive processing cost.")
    elif p < 0.05 and slope < 0:
        output_lines.append("  Negative significant effect found.")
        output_lines.append("  Words with higher drift show faster RT — unexpected direction.")
        output_lines.append("  Possible explanation: high-drift words may be more frequent")
        output_lines.append("  in social media context where participants are more exposed.")
    else:
        output_lines.append("  No significant effect found.")
        output_lines.append("  Domain-based semantic drift does not predict RT.")
        output_lines.append("  Core lexical representation appears stable across domains.")
        output_lines.append("  This is still a novel finding for Hindi NLP.")

    output_lines.append("\n── AIC Comparison ───────────────────────────────")
    if m_full["AIC"] < m_length["AIC"]:
        diff = round(m_length["AIC"] - m_full["AIC"], 2)
        output_lines.append(f"  Full model AIC is {diff} lower than baseline.")
        output_lines.append("  Adding drift_score improves model fit.")
    else:
        output_lines.append("  Adding drift_score does not improve AIC.")

    full_text = "\n".join(output_lines)
    print(full_text)

    # ── Save results ───────────────────────────────────────────────
    txt_path = "results/statistical_model_output.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"\nFull output saved → {txt_path}")

    summary = pd.DataFrame([m_drift, m_length, m_full])
    csv_path = "results/model_summary.csv"
    summary[["model", "R2", "AIC", "n"]].to_csv(csv_path, index=False)
    print(f"Model summary saved → {csv_path}")


if __name__ == "__main__":
    main()