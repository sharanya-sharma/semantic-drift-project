import os
import re

SOCIAL_FOLDER = "data/raw/social"
OUTPUT_FILE = "data/processed/social_clean.txt"

pattern_numbers = re.compile(r"\d+")
pattern_tags = re.compile(r"<.*?>")
pattern_non_hindi = re.compile(r"[^\u0900-\u097F\s]")
pattern_spaces = re.compile(r"\s+")


def clean_text(text):

    text = text.lower()

    text = pattern_tags.sub(" ", text)
    text = pattern_numbers.sub(" ", text)
    text = pattern_non_hindi.sub(" ", text)
    text = pattern_spaces.sub(" ", text).strip()

    return text


def process_social():

    total_lines = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:

        for file in os.listdir(SOCIAL_FOLDER):

            file_path = os.path.join(SOCIAL_FOLDER, file)

            print("Processing:", file)

            with open(file_path, "r", encoding="utf-8") as f:

                for line in f:

                    cleaned = clean_text(line)

                    if len(cleaned) > 2:
                        out_file.write(cleaned + "\n")

                    total_lines += 1

                    if total_lines % 500000 == 0:
                        print("Processed", total_lines, "lines")


if __name__ == "__main__":
    process_social()