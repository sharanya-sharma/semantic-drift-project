import os

NEWS_PATH = "data/raw/news"
SOCIAL_PATH = "data/raw/social"


def inspect_folder(folder):

    print("\nInspecting:", folder)

    for file in os.listdir(folder):

        file_path = os.path.join(folder, file)

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()

        print("File:", file)
        print("Total lines:", len(lines))
        print("Sample:", lines[:3])
        print()


inspect_folder(NEWS_PATH)
inspect_folder(SOCIAL_PATH)