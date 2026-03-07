import os
import re

NEWS_FOLDER = "data/raw/news"
OUTPUT_FILE = "data/processed/news_clean.txt"

pattern_numbers = re.compile(r"\d+")
pattern_non_hindi = re.compile(r"[^\u0900-\u097F\s]")
pattern_spaces = re.compile(r"\s+")


def clean_text(text):

    text = text.lower()

    text = pattern_numbers.sub(" ", text)
    text = pattern_non_hindi.sub(" ", text)
    text = pattern_spaces.sub(" ", text).strip()

    return text


def process_news():

    total_lines = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:

        for file in os.listdir(NEWS_FOLDER):

            file_path = os.path.join(NEWS_FOLDER, file)

            print("Processing:", file)

            with open(file_path, "r", encoding="utf-8") as f:

                for i, line in enumerate(f):

                    cleaned = clean_text(line)

                    if len(cleaned) > 2:
                        out_file.write(cleaned + "\n")

                    total_lines += 1

                    if total_lines % 1000000 == 0:
                        print("Processed", total_lines, "lines")


if __name__ == "__main__":
    process_news()