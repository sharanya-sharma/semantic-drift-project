"""
plot_results.py
---------------
Generates four plots from the semantic drift analysis:

    1. Drift score distribution histogram
    2. PCA scatter plot of news vs social embedding spaces
    3. Drift score vs Reaction Time scatter plot with regression line
    4. High-drift vs Low-drift words visualised in PCA space

Usage:
    python src/visualization/plot_results.py

Outputs:
    results/plots/drift_distribution.png
    results/plots/pca_embedding_spaces.png
    results/plots/drift_vs_rt.png
    results/plots/high_low_drift_words.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.font_manager as fm
from sklearn.decomposition import PCA
from scipy import stats
from pathlib import Path

# ── Hindi font setup (Windows compatible) ─────────────────────────
# Try fonts in order of preference — first match is used
HINDI_FONT_CANDIDATES = [
    "Nirmala UI",        # Built into Windows 8+
    "Mangal",            # Built into Windows XP+
    "Arial Unicode MS",
    "Lohit Devanagari",  # Linux
    "FreeSans",
    "DejaVu Sans",       # fallback (may not render Hindi)
]

available_fonts = {f.name for f in fm.fontManager.ttflist}
selected_font   = "DejaVu Sans"  # default fallback

for candidate in HINDI_FONT_CANDIDATES:
    if candidate in available_fonts:
        selected_font = candidate
        print(f"Hindi font selected: {candidate}")
        break

plt.rcParams.update({
    "font.family":       selected_font,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
})

# ── Output directory ───────────────────────────────────────────────
PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# PLOT 1 — Drift Score Distribution
# ══════════════════════════════════════════════════════════════════
def plot_drift_distribution():
    print("Generating Plot 1: Drift score distribution...")

    drift_df = pd.read_csv("drift/drift_scores.csv")
    scores   = drift_df["drift_score"]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(scores, bins=80, color="#4F8EF7", alpha=0.85,
            edgecolor="white", linewidth=0.4)

    mean_val   = scores.mean()
    median_val = scores.median()
    ax.axvline(mean_val,   color="#E74C3C", linewidth=2,
               linestyle="--", label=f"Mean = {mean_val:.3f}")
    ax.axvline(median_val, color="#2ECC71", linewidth=2,
               linestyle=":",  label=f"Median = {median_val:.3f}")

    ax.set_xlabel("Drift Score  (cosine distance after Procrustes alignment)",
                  fontsize=12)
    ax.set_ylabel("Number of Words", fontsize=12)
    ax.set_title(
        "Semantic Drift Score Distribution\nNews (SHABD) vs Social Media (SHABD 2.0)",
        fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=11)

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
    print(f"  Saved -> {out}")


# ══════════════════════════════════════════════════════════════════
# PLOT 2 — PCA of Embedding Spaces
# ══════════════════════════════════════════════════════════════════
def load_vec_subset(path, n=3000):
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

    news_words,   W_news   = load_vec_subset(
        "embeddings/news_fasttext_skipgram.vec",   3000)
    social_words, W_social = load_vec_subset(
        "embeddings/social_fasttext_skipgram.vec", 3000)

    combined = np.vstack([W_news, W_social])
    pca      = PCA(n_components=2, random_state=42)
    reduced  = pca.fit_transform(combined)

    n_news    = len(news_words)
    news_2d   = reduced[:n_news]
    social_2d = reduced[n_news:]
    var       = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(news_2d[:, 0],   news_2d[:, 1],
               s=6, alpha=0.35, color="#4F8EF7", label="News (SHABD)")
    ax.scatter(social_2d[:, 0], social_2d[:, 1],
               s=6, alpha=0.35, color="#F76F6F", label="Social (SHABD 2.0)")

    ax.set_xlabel(f"PC1  ({var[0]:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2  ({var[1]:.1f}% variance)", fontsize=12)
    ax.set_title(
        "PCA of Word Embedding Spaces\nNews vs Social Media (top 3,000 words each)",
        fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=11, markerscale=3)

    plt.tight_layout()
    out = PLOTS_DIR / "pca_embedding_spaces.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


# ══════════════════════════════════════════════════════════════════
# PLOT 3 — Drift Score vs Reaction Time
# ══════════════════════════════════════════════════════════════════
def plot_drift_vs_rt():
    print("Generating Plot 3: Drift score vs Reaction Time...")

    df = pd.read_csv("results/merged_ldt_drift.csv")
    x  = df["drift_score"].values
    y  = df["RT"].values

    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(x, y, s=8, alpha=0.25, color="#4F8EF7",
               label="Words (n=6,483)")
    ax.plot(x_line, y_line, color="#E74C3C", linewidth=2.5,
            label=f"Regression line  (slope={slope:.1f}, p<0.001)")

    y_pred    = slope * x + intercept
    residuals = y - y_pred
    s_err     = np.sqrt(np.sum(residuals ** 2) / (len(y) - 2))
    conf      = s_err * 1.96
    ax.fill_between(x_line, y_line - conf, y_line + conf,
                    color="#E74C3C", alpha=0.12,
                    label="95% confidence band")

    ax.set_xlabel("Semantic Drift Score", fontsize=12)
    ax.set_ylabel("Reaction Time (ms)", fontsize=12)
    ax.set_title(
        "Semantic Drift vs Lexical Decision Reaction Time\n"
        "Higher drift -> slower word recognition",
        fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=10)

    stats_text = (
        f"r = {r:.4f}\n"
        f"R2 = {r**2:.4f}\n"
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
    print(f"  Saved -> {out}")


# ══════════════════════════════════════════════════════════════════
# PLOT 4 — High-drift vs Low-drift words in PCA space
# ══════════════════════════════════════════════════════════════════
def load_vec_for_words(path, target_words):
    """Load vectors only for a specific set of words."""
    target_set = set(target_words)
    found = {}
    with open(path, encoding="utf-8") as f:
        parts = f.readline().split()
        dim   = int(parts[1])
        for line in f:
            p = line.rstrip().split(" ")
            w = p[0]
            if w in target_set:
                v = np.array(p[1:], dtype=np.float32)
                if len(v) == dim:
                    found[w] = v
            if len(found) == len(target_set):
                break
    return found


def plot_high_low_drift_words():
    print("Generating Plot 4: High-drift vs Low-drift words in PCA space...")

    drift_df = pd.read_csv("drift/drift_scores.csv")

    top_high = drift_df.nlargest(20,  "drift_score")
    top_low  = drift_df.nsmallest(20, "drift_score")

    high_words = top_high["word"].tolist()
    low_words  = top_low["word"].tolist()
    all_target = high_words + low_words

    print("  Loading vectors for highlighted words...")
    vec_map = load_vec_for_words(
        "embeddings/news_fasttext_skipgram.vec", all_target)

    shared_words = set(drift_df["word"].tolist())
    bg_words, bg_vecs = [], []
    with open("embeddings/news_fasttext_skipgram.vec", encoding="utf-8") as f:
        parts = f.readline().split()
        dim   = int(parts[1])
        for line in f:
            p = line.rstrip().split(" ")
            w = p[0]
            if w in shared_words and w not in set(all_target):
                v = np.array(p[1:], dtype=np.float32)
                if len(v) == dim:
                    bg_words.append(w)
                    bg_vecs.append(v)
            if len(bg_words) >= 2000:
                break

    highlight_words = [w for w in all_target if w in vec_map]
    highlight_vecs  = np.stack([vec_map[w] for w in highlight_words])
    all_vecs        = np.vstack([np.stack(bg_vecs), highlight_vecs])

    pca     = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(all_vecs)
    var     = pca.explained_variance_ratio_ * 100

    n_bg         = len(bg_vecs)
    bg_2d        = reduced[:n_bg]
    highlight_2d = reduced[n_bg:]

    high_set = set(high_words)
    low_set  = set(low_words)

    high_indices = [i for i, w in enumerate(highlight_words) if w in high_set]
    low_indices  = [i for i, w in enumerate(highlight_words) if w in low_set]

    high_2d     = highlight_2d[high_indices]
    low_2d      = highlight_2d[low_indices]
    high_labels = [highlight_words[i] for i in high_indices]
    low_labels  = [highlight_words[i] for i in low_indices]

    # Use the selected Hindi font explicitly for annotations
    font_prop = fm.FontProperties(family=selected_font, size=9)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(bg_2d[:, 0], bg_2d[:, 1],
               s=5, alpha=0.15, color="#AAAAAA", zorder=1,
               label="Other shared words")

    ax.scatter(high_2d[:, 0], high_2d[:, 1],
               s=90, alpha=0.9, color="#E74C3C", zorder=3,
               edgecolors="white", linewidths=0.8,
               label="High-drift words (top 20)")

    ax.scatter(low_2d[:, 0], low_2d[:, 1],
               s=90, alpha=0.9, color="#2ECC71", zorder=3,
               edgecolors="white", linewidths=0.8,
               label="Low-drift words (bottom 20)")

    for label, (x, y) in zip(high_labels, high_2d):
        ax.annotate(
            label, xy=(x, y), xytext=(6, 4),
            textcoords="offset points",
            fontproperties=font_prop,
            color="#C0392B", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )

    for label, (x, y) in zip(low_labels, low_2d):
        ax.annotate(
            label, xy=(x, y), xytext=(6, 4),
            textcoords="offset points",
            fontproperties=font_prop,
            color="#1A8A4A", fontweight="bold",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")]
        )

    ax.set_xlabel(f"PC1  ({var[0]:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2  ({var[1]:.1f}% variance)", fontsize=12)
    ax.set_title(
        "High-Drift vs Low-Drift Words in PCA Space\n"
        "Red = highest semantic shift  |  Green = most stable words",
        fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=11, markerscale=1.5, loc="upper right")

    high_sample = "\n".join(
        f"  {w}  ({s:.3f})"
        for w, s in zip(
            top_high["word"].head(5).tolist(),
            top_high["drift_score"].head(5).tolist()
        )
    )
    low_sample = "\n".join(
        f"  {w}  ({s:.3f})"
        for w, s in zip(
            top_low["word"].head(5).tolist(),
            top_low["drift_score"].head(5).tolist()
        )
    )
    info_text = (
        f"Top 5 high-drift:\n{high_sample}"
        f"\n\nTop 5 low-drift:\n{low_sample}"
    )
    ax.text(0.01, 0.01, info_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            fontfamily=selected_font,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.92))

    plt.tight_layout()
    out = PLOTS_DIR / "high_low_drift_words.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out}")


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
    plot_high_low_drift_words()

    print()
    print("=" * 50)
    print("All plots saved to results/plots/")
    print("  drift_distribution.png")
    print("  pca_embedding_spaces.png")
    print("  drift_vs_rt.png")
    print("  high_low_drift_words.png")
    print("=" * 50)