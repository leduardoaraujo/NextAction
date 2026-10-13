"""Build paired English/Portuguese SGD data for the bilingual agent scenario.

The translated train/test dialogues must come from the official SGD train/test splits.
Training repeats the smaller Portuguese sample so both languages affect one classifier.
All outputs stay under data/external/ (git-ignored).
"""

import argparse
import csv
from pathlib import Path

EXTERNAL = Path("data/external")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"state", "action", "dialogue"} <= set(reader.fieldnames or ()):
            raise ValueError(f"Expected state,action,dialogue columns: {path}")
        return list(reader)


def write(path: Path, examples: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("state", "action", "dialogue"))
        writer.writerows((r["state"], r["action"], r["dialogue"]) for r in examples)


def build(en_train: Path, pt_train: Path, en_test: Path, pt_test: Path,
          train_out: Path, en_test_out: Path, pt_repeat: int) -> dict:
    if pt_repeat < 1:
        raise ValueError("--pt-repeat must be at least 1")
    english_train, portuguese_train = rows(en_train), rows(pt_train)
    english_test, portuguese_test = rows(en_test), rows(pt_test)
    paired_ids = {r["dialogue"] for r in portuguese_test}
    paired_english = [r for r in english_test if r["dialogue"] in paired_ids]
    if {r["dialogue"] for r in paired_english} != paired_ids:
        raise ValueError("Portuguese test dialogue IDs are not all in the English test split")
    # SGD dialogue IDs are reused across official splits; only pair IDs within test.
    write(train_out, english_train + portuguese_train * pt_repeat)
    write(en_test_out, paired_english)
    return {"train": str(train_out), "english_train": len(english_train),
            "portuguese_train": len(portuguese_train), "portuguese_repeat": pt_repeat,
            "english_paired_test": len(paired_english), "portuguese_test": len(portuguese_test)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--en-train", type=Path, default=EXTERNAL / "sgd_generic_train.csv")
    parser.add_argument("--pt-train", type=Path, default=EXTERNAL / "sgd_pt_train.csv")
    parser.add_argument("--en-test", type=Path, default=EXTERNAL / "sgd_generic_test.csv")
    parser.add_argument("--pt-test", type=Path, default=EXTERNAL / "sgd_pt_test.csv")
    parser.add_argument("--train-out", type=Path, default=EXTERNAL / "sgd_bilingual_train.csv")
    parser.add_argument("--en-test-out", type=Path, default=EXTERNAL / "sgd_en_paired_test.csv")
    parser.add_argument("--pt-repeat", type=int, default=4)
    args = parser.parse_args()
    print(build(args.en_train, args.pt_train, args.en_test, args.pt_test,
                args.train_out, args.en_test_out, args.pt_repeat))


if __name__ == "__main__":
    main()
