from __future__ import annotations

import argparse
import zipfile
import datetime
import subprocess
from pathlib import Path

import requests

URL = "https://raw.githubusercontent.com/Arrietti-li/Dataset/main/yoochoose.zip"
PERIOD_URL = "https://raw.githubusercontent.com/doublemul/ADER/master/data/YOOCHOOSE/period_{}.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/yoochoose"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "yoochoose.zip"
    if not target.exists():
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        target.write_bytes(response.content)
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        for name in names:
            if name.endswith("yoochoose-buys.dat"):
                destination = args.output_dir / Path(name).name
                destination.write_bytes(archive.read(name))
    clicks = args.output_dir / "yoochoose-clicks.dat"
    if not clicks.exists():
        session_id = -1
        with clicks.open("w", encoding="utf-8") as output:
            for period in range(3):
                response = subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--max-time", "180", PERIOD_URL.format(period)], check=True, capture_output=True, text=True)
                day = datetime.datetime(2014, 4, 1) + datetime.timedelta(days=period)
                for line in response.stdout.splitlines():
                    items = [x for x in line.split() if x.isdigit()]
                    for position, item in enumerate(items):
                        output.write(f"{session_id},{day.isoformat()}Z,{item},0\n")
                    session_id -= 1
    missing = [name for name in ("yoochoose-clicks.dat", "yoochoose-buys.dat") if not (args.output_dir / name).exists()]
    if missing:
        raise RuntimeError(f"archive lacks {missing}; contents={names}")


if __name__ == "__main__":
    main()
