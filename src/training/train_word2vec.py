import os
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

NEWS_FILE    = "data/processed/news_clean.txt"
SOCIAL_FILE  = "data/processed/social_clean.txt"

W2V_PARAMS = dict(
    vector_size = 300,
    window      = 5,
    min_count   = 5,
    workers     = os.cpu_count(),
    sg          = 1,   # skip-gram
    epochs      = 5,
)

def train_single(corpus_path: str, model_path: str, label: str) -> Word2Vec:
    print(f"\n[{label}] Training on: {corpus_path}")
    sentences = LineSentence(corpus_path)
    model = Word2Vec(sentences=sentences, **W2V_PARAMS)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"[{label}] Vocab size : {len(model.wv)}")
    print(f"[{label}] Saved to   : {model_path}")
    return model

def train_all():
    news_model   = train_single(NEWS_FILE,   "models/news_word2vec.model",   "SHABD-News")
    social_model = train_single(SOCIAL_FILE, "models/social_word2vec.model", "SHABD-Social")

    # Quick sanity check — shared vocab size matters for Procrustes alignment later
    news_vocab   = set(news_model.wv.index_to_key)
    social_vocab = set(social_model.wv.index_to_key)
    shared       = news_vocab & social_vocab
    print(f"\nShared vocab (alignment candidates): {len(shared):,} words")
    print("Training complete. Ready for Procrustes alignment.")

if __name__ == "__main__":
    train_all()