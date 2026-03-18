import argparse
import logging
import os
import time
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CORPUS_MAP = {
    "news":   "data/processed/news_clean.txt",
    "social": "data/processed/social_clean.txt",
}
EMBEDDINGS_DIR = Path("embeddings")
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────
# These are tuned for speed while keeping academic validity.
# Skip-gram is used for BOTH corpora — required for valid comparison.

VECTOR_SIZE = 300    # synopsis: dimension = 300
WINDOW      = 5      # synopsis: window size = 5
MIN_COUNT   = 15     # min frequency threshold — raises from 10, shrinks vocab
EPOCHS      = 3      # reduced from 5 — 3 epochs standard in fastText literature
WORKERS     = 8      # all 8 cores — safe since we freed RAM
NEGATIVE    = 5      # reduced from 10 — halves computation per update
MINN        = 2      # subword min (Hindi morphology)
MAXN        = 4      # subword max — reduced from 5, speeds up subword hashing


# ── Streaming iterator (for Gensim fallback) ───────────────────────
class StreamSentences:
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) >= 2:
                    yield tokens


# ══════════════════════════════════════════════════════════════════
# fastText (PRIMARY)
# ══════════════════════════════════════════════════════════════════
def train_fasttext(corpus_name: str):
    import fasttext

    corpus_path = CORPUS_MAP[corpus_name]
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            "Corpus not found: {}\nRun preprocessing first.".format(corpus_path)
        )

    out_stem = str(EMBEDDINGS_DIR / "{}_fasttext_skipgram".format(corpus_name))

    log.info("=" * 55)
    log.info("METHOD     : fastText skip-gram")
    log.info("Corpus     : {}".format(corpus_path))
    log.info("Dimensions : {}  (synopsis: 300)".format(VECTOR_SIZE))
    log.info("Window     : {}  (synopsis: 5)".format(WINDOW))
    log.info("Min-count  : {}  (freq threshold)".format(MIN_COUNT))
    log.info("Epochs     : {}  (3 = standard, faster than 5)".format(EPOCHS))
    log.info("Workers    : {}  (all cores)".format(WORKERS))
    log.info("Neg samples: {}  (reduced for speed)".format(NEGATIVE))
    log.info("Subwords   : [{}, {}]".format(MINN, MAXN))
    log.info("Output     : {}".format(out_stem))
    log.info("=" * 55)

    t0 = time.time()
    log.info("Training started...")

    model = fasttext.train_unsupervised(
        input    = corpus_path,
        model    = "skipgram",
        dim      = VECTOR_SIZE,
        ws       = WINDOW,
        minCount = MIN_COUNT,
        epoch    = EPOCHS,
        thread   = WORKERS,
        neg      = NEGATIVE,
        minn     = MINN,
        maxn     = MAXN,
        verbose  = 2,
    )

    elapsed = time.time() - t0
    log.info("Training complete in {:.1f} min  ({:.1f} hrs)".format(
        elapsed / 60, elapsed / 3600))

    # ── Save .bin ──────────────────────────────────────────────────
    bin_path = out_stem + ".bin"
    model.save_model(bin_path)
    log.info("Binary saved → {}".format(bin_path))

    # ── Save .vec ──────────────────────────────────────────────────
    words = model.get_words()
    vec_path = out_stem + ".vec"
    log.info("Saving {} vectors to .vec ...".format(len(words)))

    with open(vec_path, "w", encoding="utf-8") as vf:
        vf.write("{} {}\n".format(len(words), VECTOR_SIZE))
        for w in words:
            vec = model.get_word_vector(w)
            vf.write("{} {}\n".format(
                w, " ".join("{:.6f}".format(v) for v in vec)
            ))

    log.info("Vectors saved → {}  ({:,} words)".format(vec_path, len(words)))

    # ── Sanity check ───────────────────────────────────────────────
    log.info("")
    log.info("Nearest neighbour check:")
    for word in ["देश", "सरकार", "लोग", "पानी", "दिल्ली"]:
        n = model.get_nearest_neighbors(word, k=3)
        log.info("  {} → {}".format(word, n))

    log.info("")
    log.info("Files saved:")
    log.info("  {}  (binary)".format(bin_path))
    log.info("  {}  (text vectors for alignment)".format(vec_path))


# ══════════════════════════════════════════════════════════════════
# Gensim Word2Vec (fallback)
# ══════════════════════════════════════════════════════════════════
def train_gensim(corpus_name: str):
    from gensim.models import Word2Vec
    from gensim.models.callbacks import CallbackAny2Vec

    class EpochLogger(CallbackAny2Vec):
        def __init__(self, total):
            self.epoch    = 0
            self.total    = total
            self.start    = time.time()
            self.ep_start = time.time()

        def on_epoch_begin(self, model):
            self.ep_start = time.time()
            log.info("Epoch {}/{} started".format(self.epoch + 1, self.total))

        def on_epoch_end(self, model):
            self.epoch += 1
            ep  = (time.time() - self.ep_start) / 60
            tot = (time.time() - self.start) / 60
            eta = (tot / self.epoch) * (self.total - self.epoch)
            log.info("Epoch {}/{} — {:.1f} min | ETA {:.1f} min".format(
                self.epoch, self.total, ep, eta))

    corpus_path = CORPUS_MAP[corpus_name]
    out_stem    = str(EMBEDDINGS_DIR / "{}_word2vec_skipgram".format(corpus_name))
    sentences   = StreamSentences(corpus_path)

    log.info("Building vocab...")
    model = Word2Vec(
        vector_size = VECTOR_SIZE,
        window      = WINDOW,
        min_count   = MIN_COUNT,
        workers     = WORKERS,
        sg          = 1,
        negative    = NEGATIVE,
        epochs      = EPOCHS,
        batch_words = 5000,
        seed        = 42,
    )
    model.build_vocab(sentences, progress_per=1_000_000)
    log.info("Vocab: {:,} tokens".format(len(model.wv)))

    log.info("Training...")
    model.train(
        sentences,
        total_examples = model.corpus_count,
        epochs         = model.epochs,
        callbacks      = [EpochLogger(EPOCHS)],
    )

    model.save(out_stem + ".model")
    model.wv.save_word2vec_format(out_stem + ".vec", binary=False)
    log.info("Saved → {}".format(out_stem))


# ── CLI ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        choices=["news", "social"],
        required=True,
    )
    parser.add_argument(
        "--use_gensim",
        action="store_true",
        help="Use Gensim Word2Vec instead of fastText",
    )
    args = parser.parse_args()

    if args.use_gensim:
        train_gensim(args.corpus)
    else:
        train_fasttext(args.corpus)