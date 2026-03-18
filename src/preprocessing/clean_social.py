import os
import re

SOCIAL_FOLDER = "data/raw/social"
OUTPUT_FILE   = "data/processed/social_clean.txt"

# ── Compiled patterns ──────────────────────────────────────────────

# 1. HTML/XML tags
pattern_tags = re.compile(r"<[^>]+>")

# 2. Zero-width and invisible Unicode characters
#    Fixes phantom mid-word spaces like चाह​हते in YouTube data
pattern_zero_width = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060]"
)

# 3. Punctuation → replace with SPACE to prevent word merging
#    है।देश → है देश  (NOT हैदेश)
pattern_punctuation = re.compile(
    r"[।|,\.!\?;:\"\'\(\)\[\]\{\}/\\@#\$%\^&\*\-_=\+~`]"
)

# 4. Digits
pattern_numbers = re.compile(r"\d+")

# 5. Keep only Devanagari Unicode block + whitespace
pattern_non_hindi = re.compile(r"[^\u0900-\u097F\s]")

# 6. Collapse multiple spaces/tabs → single space
pattern_spaces = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = pattern_tags.sub(" ", text)        # 1. tags
    text = pattern_zero_width.sub("", text)   # 2. zero-width chars
    text = pattern_punctuation.sub(" ", text) # 3. punctuation → space
    text = pattern_numbers.sub(" ", text)     # 4. digits
    text = pattern_non_hindi.sub(" ", text)   # 5. non-Devanagari
    text = pattern_spaces.sub(" ", text).strip()  # 6. whitespace
    return text


def is_valid_line(tokens: list) -> bool:
    valid = [t for t in tokens if len(t) >= 2]
    return len(valid) >= 3


def process_social():
    total_lines   = 0
    written_lines = 0
    skipped_lines = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        for file in sorted(os.listdir(SOCIAL_FOLDER)):
            file_path = os.path.join(SOCIAL_FOLDER, file)
            print("Processing:", file)

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    total_lines += 1
                    cleaned = clean_text(line)
                    tokens  = cleaned.split()

                    if is_valid_line(tokens):
                        out_file.write(cleaned + "\n")
                        written_lines += 1
                    else:
                        skipped_lines += 1

                    if total_lines % 500_000 == 0:
                        print(f"  Lines: {total_lines:,} | "
                              f"Written: {written_lines:,} | "
                              f"Skipped: {skipped_lines:,}")

    print("\n── Summary ──────────────────────────────────")
    print(f"Total lines processed : {total_lines:,}")
    print(f"Lines written         : {written_lines:,}")
    print(f"Lines skipped (short) : {skipped_lines:,}")
    print(f"Output                : {OUTPUT_FILE}")


if __name__ == "__main__":
    process_social()