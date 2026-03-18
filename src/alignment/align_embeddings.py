import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────
NEWS_VEC   = "embeddings/news_fasttext_skipgram.vec"
SOCIAL_VEC = "embeddings/social_fasttext_skipgram.vec"
DRIFT_DIR  = Path("drift")
DRIFT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# STEP 1 — Load vectors
# ══════════════════════════════════════════════════════════════════
def load_vectors(path: str, top_n: int = None) -> dict:
    """
    Load a .vec file into a {word: np.array} dictionary.
    Streams line by line — safe for large files.

    Parameters
    ----------
    path  : path to .vec file
    top_n : if set, only load the first top_n words (ranked by frequency)
            useful when RAM is limited
    """
    log.info("Loading vectors from %s ...", path)
    vectors = {}

    with open(path, encoding="utf-8") as f:
        header     = f.readline().strip().split()
        total_words = int(header[0])
        dim         = int(header[1])
        log.info("  Header: %d words x %d dims", total_words, dim)

        for i, line in enumerate(f):
            if top_n and i >= top_n:
                break
            parts = line.rstrip().split(" ")
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            if len(vec) == dim:
                vectors[word] = vec

            if (i + 1) % 50_000 == 0:
                log.info("  Loaded %d / %d words ...", i + 1,
                         top_n if top_n else total_words)

    log.info("  Done — %d vectors loaded.", len(vectors))
    return vectors


# ══════════════════════════════════════════════════════════════════
# STEP 2 — Extract common vocabulary and build matrices
# ══════════════════════════════════════════════════════════════════
def build_shared_matrices(news_vecs: dict, social_vecs: dict):
    """
    Extracts shared vocabulary and builds aligned numpy matrices.

    Returns
    -------
    shared_words : list of words present in both corpora
    W_news       : np.ndarray shape (N, 300)
    W_social     : np.ndarray shape (N, 300)
    """
    shared_words = sorted(set(news_vecs.keys()) & set(social_vecs.keys()))
    N            = len(shared_words)
    dim          = next(iter(news_vecs.values())).shape[0]

    log.info("Shared vocabulary: %d words", N)
    log.info("Building matrices W_news and W_social (%d x %d) ...", N, dim)

    W_news   = np.stack([news_vecs[w]   for w in shared_words]).astype(np.float32)
    W_social = np.stack([social_vecs[w] for w in shared_words]).astype(np.float32)

    log.info("Matrices built.")
    log.info("  W_news   shape: %s", W_news.shape)
    log.info("  W_social shape: %s", W_social.shape)

    return shared_words, W_news, W_social


# ══════════════════════════════════════════════════════════════════
# STEP 3 — Orthogonal Procrustes Alignment
# ══════════════════════════════════════════════════════════════════
def procrustes_align(W_news: np.ndarray, W_social: np.ndarray):
    """
    Solves the Orthogonal Procrustes problem:
        R = argmin ||W_news @ R - W_social||_F   s.t. R^T R = I

    This finds the optimal rotation matrix R that maps the news
    embedding space onto the social embedding space.

    Returns
    -------
    R              : rotation matrix (300 x 300)
    W_news_aligned : W_news @ R  — rotated news space
    """
    log.info("Running Orthogonal Procrustes Alignment ...")
    log.info("  Solving: R = argmin ||W_news @ R - W_social||_F")
    t0 = time.time()

    R, _ = orthogonal_procrustes(W_news, W_social)

    log.info("  Solved in %.2f seconds.", time.time() - t0)
    log.info("  R shape: %s  (rotation matrix)", R.shape)

    # STEP 4 — Rotate news space
    log.info("Rotating news space: W_news_aligned = W_news @ R ...")
    W_news_aligned = W_news @ R
    log.info("  Done. Spaces are now comparable.")

    return R, W_news_aligned


# ══════════════════════════════════════════════════════════════════
# STEP 5 — Compute semantic drift scores
# ══════════════════════════════════════════════════════════════════
def compute_drift(
    shared_words: list,
    W_news_aligned: np.ndarray,
    W_social: np.ndarray,
) -> pd.DataFrame:
    """
    Computes cosine distance between aligned news and social vectors.

    drift(w) = 1 - cosine_similarity(W_news_aligned[w], W_social[w])

    High drift → word has different meaning/context across corpora.
    Low drift  → word is stable across corpora.

    Values range: [0, 2] theoretically, but practically [0, 1].
    """
    log.info("Computing semantic drift scores for %d words ...", len(shared_words))
    t0 = time.time()

    # Normalise rows to unit vectors for cosine similarity
    norm_n = np.linalg.norm(W_news_aligned, axis=1, keepdims=True)
    norm_s = np.linalg.norm(W_social,       axis=1, keepdims=True)

    # Avoid division by zero
    norm_n = np.where(norm_n == 0, 1e-9, norm_n)
    norm_s = np.where(norm_s == 0, 1e-9, norm_s)

    W_n_unit = W_news_aligned / norm_n
    W_s_unit = W_social       / norm_s

    # Vectorised dot product per word → cosine similarity
    cos_sim = np.einsum("ij,ij->i", W_n_unit, W_s_unit)

    # Drift = cosine distance
    drift = (1.0 - cos_sim).astype(float)

    log.info("  Computed in %.2f seconds.", time.time() - t0)

    # Build dataframe
    df = pd.DataFrame({
        "word":        shared_words,
        "drift_score": drift,
    })

    # Sort by drift score — highest drift first
    df = df.sort_values("drift_score", ascending=False).reset_index(drop=True)

    # Summary statistics
    log.info("")
    log.info("=== Drift Score Summary ===")
    log.info("  Words analysed : %d",    len(df))
    log.info("  Mean drift     : %.4f",  df["drift_score"].mean())
    log.info("  Median drift   : %.4f",  df["drift_score"].median())
    log.info("  Std deviation  : %.4f",  df["drift_score"].std())
    log.info("  Min drift      : %.4f",  df["drift_score"].min())
    log.info("  Max drift      : %.4f",  df["drift_score"].max())

    return df


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main(top_n: int = None):
    log.info("=" * 55)
    log.info("STEP 1 — Loading embedding spaces")
    log.info("=" * 55)
    news_vecs   = load_vectors(NEWS_VEC,   top_n)
    social_vecs = load_vectors(SOCIAL_VEC, top_n)

    log.info("")
    log.info("=" * 55)
    log.info("STEP 2 — Extracting common vocabulary")
    log.info("=" * 55)
    shared_words, W_news, W_social = build_shared_matrices(news_vecs, social_vecs)

    # Free memory — we no longer need the full dicts
    del news_vecs, social_vecs

    log.info("")
    log.info("=" * 55)
    log.info("STEPS 3 & 4 — Orthogonal Procrustes Alignment")
    log.info("=" * 55)
    R, W_news_aligned = procrustes_align(W_news, W_social)

    # Save rotation matrix
    r_path = str(DRIFT_DIR / "rotation_matrix_R.npy")
    np.save(r_path, R)
    log.info("Rotation matrix saved → %s", r_path)

    log.info("")
    log.info("=" * 55)
    log.info("STEP 5 — Computing semantic drift scores")
    log.info("=" * 55)
    drift_df = compute_drift(shared_words, W_news_aligned, W_social)

    # Save drift scores
    out_path = str(DRIFT_DIR / "drift_scores.csv")
    drift_df.to_csv(out_path, index=False, encoding="utf-8")
    log.info("")
    log.info("Drift scores saved → %s  (%d words)", out_path, len(drift_df))

    # Show top 20 high-drift words
    log.info("")
    log.info("=== Top 20 HIGH-DRIFT words ===")
    log.info("(Words with most different meaning across news vs social)")
    print(drift_df.head(20).to_string(index=False))

    log.info("")
    log.info("=== Top 20 LOW-DRIFT words ===")
    log.info("(Words with most stable meaning across corpora)")
    print(drift_df.tail(20).to_string(index=False))

    log.info("")
    log.info("=" * 55)
    log.info("Alignment complete.")
    log.info("Next step: python src/analysis/merge_ldt.py")
    log.info("=" * 55)


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Align embeddings and compute semantic drift scores"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Only load top-N words per corpus (use if RAM < 8 GB free)",
    )
    args = parser.parse_args()
    main(args.top_n)