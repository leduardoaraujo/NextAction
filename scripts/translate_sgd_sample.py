"""Translate a deterministic sample of SGD dialogue utterances to Portuguese.

Requires ``pip install -e .[translate]`` and the separately downloaded SGD data.
The schema, act labels and dialogue-state keys remain unchanged.

Dialogues are written in blocks of ``--block`` dialogues (``dialogues_001.json``, ...), so
an interrupted run keeps finished blocks and resumes after them. Follow the progress with
``nextaction jobs --watch``.
"""

import argparse
import json
import random
from pathlib import Path

from nextaction.progress import Job, stage


def dialogue_ids(folders: list[Path]) -> set[str]:
    return {dialogue["dialogue_id"] for folder in folders
            for file in folder.glob("dialogues_*.json")
            for dialogue in json.loads(file.read_text(encoding="utf-8"))}


def translate_split(source: Path, destination: Path, count: int, model_name: str,
                    batch_size: int = 16, seed: int = 42, exclude: list[Path] = (),
                    block: int = 0) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    files = sorted(source.glob("dialogues_*.json"))
    if not files:
        raise ValueError(f"No SGD dialogues in {source}")
    skipped = dialogue_ids(list(exclude))
    dialogues = [dialogue for file in files
                 for dialogue in json.loads(file.read_text(encoding="utf-8"))
                 if dialogue["dialogue_id"] not in skipped]
    if count > len(dialogues) or count < 1:
        raise ValueError(f"count must be between 1 and {len(dialogues)}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    sample = random.Random(seed).sample(dialogues, count)
    block = block or count
    blocks = [sample[start:start + block] for start in range(0, count, block)]
    pending = [(number, dialogues) for number, dialogues in enumerate(blocks, 1)
               if not (destination / f"dialogues_{number:03d}.json").exists()]
    if not pending:
        raise ValueError(f"Destination already contains every block: {destination}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    target_id = tokenizer.convert_tokens_to_ids("por_Latn")
    if target_id == tokenizer.unk_token_id:
        raise ValueError(f"{model_name} does not support por_Latn")

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "schema.json").write_bytes((source / "schema.json").read_bytes())
    total_turns = sum(len(d["turns"]) for _, chunk in pending for d in chunk)
    translated_turns = 0
    with stage("translate", total=total_turns) as progress:
        for number, chunk in pending:
            # Sorting by length keeps padding, and so wasted computation, low within a batch.
            turns = sorted((turn for dialogue in chunk for turn in dialogue["turns"]),
                           key=lambda turn: len(turn["utterance"]))
            for start in range(0, len(turns), batch_size):
                batch = turns[start:start + batch_size]
                encoded = tokenizer([turn["utterance"] for turn in batch], return_tensors="pt",
                                    padding=True, truncation=True, max_length=256).to(device)
                with torch.no_grad():
                    output = model.generate(**encoded, forced_bos_token_id=target_id,
                                            max_new_tokens=256)
                for turn, tokens in zip(batch, output):
                    turn["utterance"] = tokenizer.decode(tokens, skip_special_tokens=True)
                translated_turns += len(batch)
                progress.update(translated_turns)
            path = destination / f"dialogues_{number:03d}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
    return {"source": str(source), "output": str(destination), "dialogues": count,
            "blocks_written": len(pending), "turns": translated_turns,
            "translation_model": model_name}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="official SGD train or test directory")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[],
                        help="folders of already translated dialogues to leave out of the sample")
    parser.add_argument("--block", type=int, default=0,
                        help="dialogues per output file; 0 writes one file at the end")
    args = parser.parse_args()
    with Job(f"translate {args.count} SGD dialogues to Portuguese"):
        print(json.dumps(translate_split(args.source, args.destination, args.count, args.model,
                                         args.batch_size, args.seed, args.exclude, args.block)))


if __name__ == "__main__":
    main()
