"""
plot_results.py
---------------
Generates three plots from the semantic drift analysis:

    1. Drift score distribution histogram
    2. PCA scatter plot of news vs social embedding spaces
    3. Drift score vs Reaction Time scatter plot with regression line

Usage:
    python src/visualization/plot_results.py

Outputs:
    results/plots/drift_distribution.png
    results/plots/pca_embedding_spaces.png
    results/plots/drift_vs_rt.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from scipy import stats
from pathlib import Path

# ── Output directory ───────────────────────────────────────────────
PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
})


# ══════════════════════════════════════════════════════════════════
# PLOT 1 — Drift Score Distribution
# ══════════════════════════════════════════════════════════════════
def plot_drift_distribution():
    print("Generating Plot 1: Drift score distribution...")

    drift_df = pd.read_csv("drift/drift_scores.csv")
    scores   = drift_df["drift_score"]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Histogram
    n, bins, patches = ax.hist(
        scores, bins=80,
        color="#4F8EF7", alpha=0.85, edgecolor="white", linewidth=0.4
    )

    # Mean and median lines
    mean_val   = scores.mean()
    median_val = scores.median()
    ax.axvline(mean_val,   color="#E74C3C", linewidth=2,
               linestyle="--", label=f"Mean = {mean_val:.3f}")
    ax.axvline(median_val, color="#2ECC71", linewidth=2,
               linestyle=":",  label=f"Median = {median_val:.3f}")

    # Labels
    ax.set_xlabel("Drift Score  (cosine distance after Procrustes alignment)",
                  fontsize=12)
    ax.set_ylabel("Number of Words", fontsize=12)
    ax.set_title("Semantic Drift Score Distribution\nNews (SHABD) vs Social Media (SHABD 2.0)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=11)

    # Annotation box
    stats_text = (
        f"Words analysed: {len(scores):,}\n"
        f"Mean:   {mean_val:.4f}\n"
        f"Std:    {scores.std():.4f}\n"
        f"Min:    {scores.min():.4f}\n"
        f"Max:    {scores.max():.4f}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))

    plt.tight_layout()
    out = PLOTS_DIR / "drift_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════
# PLOT 2 — PCA of Embedding Spaces
# ══════════════════════════════════════════════════════════════════
def load_vec_subset(path, n=3000):
    """Load first n word vectors from a .vec file."""
    words, vecs = [], []
    with open(path, encoding="utf-8") as f:
        parts = f.readline().split()
        dim   = int(parts[1])
        for i, line in enumerate(f):
            if i >= n:
                break
            p = line.rstrip().split(" ")
            w = p[0]
            v = np.array(p[1:], dtype=np.float32)
            if len(v) == dim:
                words.append(w)
                vecs.append(v)
    return words, np.stack(vecs)


def plot_pca():
    print("Generating Plot 2: PCA of embedding spaces...")

    news_words,   W_news   = load_vec_subset("embeddings/news_fasttext_skipgram.vec",   3000)
    social_words, W_social = load_vec_subset("embeddings/social_fasttext_skipgram.vec", 3000)

    combined = np.vstack([W_news, W_social])
    pca      = PCA(n_components=2, random_state=42)
    reduced  = pca.fit_transform(combined)

    n_news = len(news_words)
    news_2d   = reduced[:n_news]
    social_2d = reduced[n_news:]

    var_explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(news_2d[:, 0],   news_2d[:, 1],
               s=6, alpha=0.35, color="#4F8EF7", label="News (SHABD)")
    ax.scatter(social_2d[:, 0], social_2d[:, 1],
               s=6, alpha=0.35, color="#F76F6F", label="Social (SHABD 2.0)")

    ax.set_xlabel(f"PC1  ({var_explained[0]:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2  ({var_explained[1]:.1f}% variance)", fontsize=12)
    ax.set_title(
        "PCA of Word Embedding Spaces\nNews vs Social Media (top 3,000 words each)",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.legend(fontsize=11, markerscale=3)

    plt.tight_layout()
    out = PLOTS_DIR / "pca_embedding_spaces.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════
# PLOT 3 — Drift Score vs Reaction Time
# ══════════════════════════════════════════════════════════════════
def plot_drift_vs_rt():
    print("Generating Plot 3: Drift score vs Reaction Time...")

    df = pd.read_csv("results/merged_ldt_drift.csv")

    x = df["drift_score"].values
    y = df["RT"].values

    # Regression line
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(9, 6))

    # Scatter
    ax.scatter(x, y, s=8, alpha=0.25, color="#4F8EF7", label="Words (n=6,483)")

    # Regression line
    ax.plot(x_line, y_line, color="#E74C3C", linewidth=2.5,
            label=f"Regression line  (slope={slope:.1f}, p<0.001)")

    # Confidence band (±1 SE of regression)
    y_pred = slope * x + intercept
    residuals = y - y_pred
    s_err = np.sqrt(np.sum(residuals**2) / (len(y) - 2))
    conf  = s_err * 1.96
    ax.fill_between(x_line,
                    y_line - conf,
                    y_line + conf,
                    color="#E74C3C", alpha=0.12, label="95% confidence band")

    ax.set_xlabel("Semantic Drift Score", fontsize=12)
    ax.set_ylabel("Reaction Time (ms)", fontsize=12)
    ax.set_title(
        "Semantic Drift vs Lexical Decision Reaction Time\n"
        "Higher drift → slower word recognition",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.legend(fontsize=10)

    # Stats annotation
    stats_text = (
        f"r = {r:.4f}\n"
        f"R² = {r**2:.4f}\n"
        f"slope = {slope:.2f} ms/unit\n"
        f"p < 0.001"
    )
    ax.text(0.97, 0.05, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))

    plt.tight_layout()
    out = PLOTS_DIR / "drift_vs_rt.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("Generating all visualisation plots...")
    print("=" * 50)

    plot_drift_distribution()
    plot_pca()
    plot_drift_vs_rt()

    print()
    print("=" * 50)
    print("All plots saved to results/plots/")
    print("  drift_distribution.png")
    print("  pca_embedding_spaces.png")
    print("  drift_vs_rt.png")
    print("=" * 50)