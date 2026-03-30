# Semantic Drift Detection Across Corpora Using Word Embeddings

**Analyzing how Hindi word meanings shift between News and Social Media domains**

> B.Tech Major Project — Computer Science & Engineering  
> Sharanya Sharma (229301571) — Manipal University Jaipur  
> Under guidance of Mr. Vivek Singh Sikarwar

---

## Overview

This project measures **semantic drift** — how the meaning of Hindi words changes across two different text domains:

- **SHABD** — formal Hindi newspaper corpus
- **SHABD 2.0** — informal Hindi social media corpus (YouTube + Twitter)

Word embeddings are trained separately on each corpus, aligned using **Orthogonal Procrustes Alignment**, and drift scores are computed per word using cosine distance. The drift scores are then tested as predictors of **lexical decision reaction time (RT)** from the Hindi LDT Megastudy.

**Hypothesis:** Words with greater semantic drift across domains induce increased cognitive processing cost, resulting in slower lexical decision reaction times.

**Result: Hypothesis confirmed — drift is a significant predictor of RT (p < 0.001).**

---

## Key Results

| Metric | Value |
|---|---|
| Words analysed for drift | 32,820 |
| Mean drift score | 0.4909 |
| Max drift score | 0.9115 |
| Words in LDT analysis | 6,483 |
| Drift effect on RT | +144.6 ms per unit drift |
| p-value | < 0.001 |
| R² (full model) | 0.1762 |
| AIC improvement | −366 points |

---

## Project Structure

```
SEMANTIC-DRIFT-PROJECT/
│
├── data/
│   ├── raw/
│   │   ├── news/
│   │   │   ├── combined0.txt
│   │   │   ├── combined1.txt
│   │   │   ├── combined2.txt
│   │   │   ├── combined3.txt
│   │   │   └── combined4.txt
│   │   └── social/
│   │       ├── all_hindi_comments_doc_wise.txt
│   │       └── all_subtitles_doc_boundary.txt
│   ├── processed/
│   │   ├── news_clean.txt         (63.6M sentences)
│   │   └── social_clean.txt       (2.65M lines)
│   ├── ldt/
│   │   └── hindi_ldt.csv          (11,498 words with RT values)
│   └── resources/
│       └── hindi_stopwords.txt
│
├── embeddings/
│   ├── news_fasttext_skipgram.bin    (2.8 GB)
│   ├── news_fasttext_skipgram.vec    (646 MB)
│   ├── social_fasttext_skipgram.bin  (2.4 GB)
│   └── social_fasttext_skipgram.vec  (122 MB)
│
├── drift/
│   ├── drift_scores.csv              (32,820 words with drift scores)
│   └── rotation_matrix_R.npy         (300×300 Procrustes rotation matrix)
│
├── results/
│   ├── merged_ldt_drift.csv          (6,483 words — RT + drift)
│   ├── model_summary.csv             (model comparison table)
│   ├── statistical_model_output.txt  (full regression results)
│   └── plots/
│       ├── drift_distribution.png
│       ├── pca_embedding_spaces.png
│       └── drift_vs_rt.png
│
│
├── src/
│   ├── preprocessing/
│   │   ├── clean_news.py
│   │   ├── clean_social.py
│   │   └── inspect_corpus.py
│   ├── training/
│   │   └── train_embeddings.py
│   ├── alignment/
│   │   └── align_embeddings.py
│   ├── analysis/
│   │   ├── merge_ldt.py
│   │   └── statistical_model.py
│   └── visualization/
│       └── plot_results.py
│
├── venv/
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Raw Corpus
    ↓
Cleaning & Tokenization       clean_news.py / clean_social.py
    ↓
Word Embedding Training        train_embeddings.py  (fastText skip-gram)
    ↓
Procrustes Alignment           align_embeddings.py
    ↓
Drift Score Computation        align_embeddings.py  → drift/drift_scores.csv
    ↓
Merge with Hindi LDT           merge_ldt.py         → results/merged_ldt_drift.csv
    ↓
Statistical Modelling          statistical_model.py → results/model_summary.csv
    ↓
Visualisation                  plot_results.py      → results/plots/
```

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Inspect raw corpora
python src/preprocessing/inspect_corpus.py

# 3. Preprocess
python src/preprocessing/clean_news.py
python src/preprocessing/clean_social.py

# 4. Train embeddings (run both)
python src/training/train_embeddings.py --corpus news
python src/training/train_embeddings.py --corpus social

# 5. Align and compute drift
python src/alignment/align_embeddings.py

# 6. Merge with LDT data
python src/analysis/merge_ldt.py

# 7. Run statistical models
python src/analysis/statistical_model.py

# 8. Generate plots
python src/visualization/plot_results.py
```