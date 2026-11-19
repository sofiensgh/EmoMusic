"""Listening test for the report: can people guess the mood of a generated song?

    python -m src.survey make --n 12              # creates output/survey/clip_XX.wav + answer key
    python -m src.survey score answers.csv        # answers.csv: clip,guess  (one row per listener answer)
"""
import argparse
import csv
import random
from collections import Counter

from .config import MOODS, OUTPUT_DIR
from .generate import compose, save

SURVEY = OUTPUT_DIR / "survey"


def make(n, engine):
    SURVEY.mkdir(parents=True, exist_ok=True)
    moods = [MOODS[i % len(MOODS)] for i in range(n)]
    random.shuffle(moods)
    with open(SURVEY / "answer_key.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "mood", "engine"])
        for i, m in enumerate(moods, 1):
            pm, _, used = compose(m, bars=8, seed=i, engine=engine)
            save(pm, f"clip_{i:02d}", SURVEY)
            w.writerow([f"clip_{i:02d}", m, used])
    print(f"{n} clips in {SURVEY} - play them to listeners WITHOUT showing answer_key.csv")


def score(path):
    with open(SURVEY / "answer_key.csv") as f:
        key = {r["clip"]: r["mood"] for r in csv.DictReader(f)}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    correct = sum(key.get(r["clip"]) == r["guess"].strip().lower() for r in rows)
    print(f"accuracy {correct}/{len(rows)} = {correct / max(len(rows), 1):.0%} (chance = 25%)")
    confusions = Counter((key[r["clip"]], r["guess"].strip().lower()) for r in rows if r["clip"] in key)
    for true in MOODS:
        print(f"  {true:6s} ->", {g: confusions[(true, g)] for g in MOODS})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--n", type=int, default=12)
    m.add_argument("--engine", default="auto", choices=["auto", "lstm", "rules"])
    s = sub.add_parser("score"); s.add_argument("answers")
    a = p.parse_args()
    make(a.n, a.engine) if a.cmd == "make" else score(a.answers)
