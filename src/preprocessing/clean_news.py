import os
import re

NEWS_FOLDER = "data/raw/news"
OUTPUT_FILE = "data/processed/news_clean.txt"

# ── Sentence splitter ──────────────────────────────────────────────
# Split on:
#   । — Hindi danda (primary sentence boundary)
#   \n — newlines within a paragraph
#   . ! ? — English sentence endings (appear in news text)
# We keep the splitting BEFORE cleaning so boundaries are visible.
pattern_sentence_split = re.compile(r"[।\n]+|(?<=[.!?])\s+")

# ── Cleaning patterns ──────────────────────────────────────────────

# Zero-width and invisible Unicode characters
pattern_zero_width = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060]"
)

# Punctuation → replace with SPACE (after splitting, so we don't lose boundaries)
pattern_punctuation = re.compile(
    r"[।|,\.!\?;:\"\'\(\)\[\]\{\}/\\@#\$%\^&\*\-_=\+~`]"
)

# Digits
pattern_numbers = re.compile(r"\d+")

# Keep only Devanagari Unicode block + whitespace
pattern_non_hindi = re.compile(r"[^\u0900-\u097F\s]")

# Collapse multiple spaces
pattern_spaces = re.compile(r"\s+")


def split_sentences(text: str) -> list:
    """
    Split a paragraph into individual sentences.
    Returns list of raw sentence strings.
    """
    sentences = pattern_sentence_split.split(text)
    return [s.strip() for s in sentences if s.strip()]


def clean_sentence(text: str) -> str:
    """Clean a single sentence."""
    text = pattern_zero_width.sub("", text)       # zero-width chars
    text = pattern_punctuation.sub(" ", text)     # punctuation → space
    text = pattern_numbers.sub(" ", text)         # digits
    text = pattern_non_hindi.sub(" ", text)       # non-Devanagari
    text = pattern_spaces.sub(" ", text).strip()  # whitespace
    return text


def is_valid(tokens: list) -> bool:
    """Keep only lines with >= 3 tokens of >= 2 chars each."""
    valid = [t for t in tokens if len(t) >= 2]
    return len(valid) >= 3


def process_news():
    total_lines   = 0
    written_lines = 0
    skipped_lines = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        for file in sorted(os.listdir(NEWS_FOLDER)):
            file_path = os.path.join(NEWS_FOLDER, file)
            print("Processing:", file)

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    total_lines += 1

                    # Split paragraph into sentences first
                    sentences = split_sentences(line)

                    for sentence in sentences:
                        cleaned = clean_sentence(sentence)
                        tokens  = cleaned.split()

                        if is_valid(tokens):
                            out_file.write(cleaned + "\n")
                            written_lines += 1
                        else:
                            skipped_lines += 1

                    if total_lines % 1_000_000 == 0:
                        print(f"  Lines: {total_lines:,} | "
                              f"Written: {written_lines:,} | "
                              f"Skipped: {skipped_lines:,}")

    print("\n── Summary ──────────────────────────────────")
    print(f"Total lines processed : {total_lines:,}")
    print(f"Sentences written     : {written_lines:,}")
    print(f"Sentences skipped     : {skipped_lines:,}")
    print(f"Output                : {OUTPUT_FILE}")


if __name__ == "__main__":
    process_news()