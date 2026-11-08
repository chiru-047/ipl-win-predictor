"""Download the raw IPL data (ball-by-ball + match info, 2008 onwards).

The CSVs are built from Cricsheet (https://cricsheet.org) data and published at
https://github.com/ritesh-ojha/IPL-DATASET. They're ~25 MB, so they are not
committed to this repo.

Run:  python src/download_data.py
"""

from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://raw.githubusercontent.com/ritesh-ojha/IPL-DATASET/main/csv/"
FILES = {"Ball_By_Ball_Match_Data.csv": "ball_by_ball.csv", "Match_Info.csv": "match_info.csv"}
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for remote, local in FILES.items():
        print(f"Downloading {remote} ...")
        urlretrieve(BASE + remote, RAW / local)
    print(f"Saved to {RAW}")


if __name__ == "__main__":
    main()
