from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml
from transformers import AutoProcessor


CHECKED_ROOT = Path(__file__).resolve().parents[1]
SHARED_SCRIPTS = CHECKED_ROOT.parent / "partial_observation" / "scripts"
sys.path.insert(0, str(SHARED_SCRIPTS))

from common import load_prompt_template, resolve_model_reference, resolve_path  # noqa: E402
from train_generator import PartialCommentCollator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the text-heaviest CHECKED samples.")
    parser.add_argument(
        "--config",
        type=Path,
        default=CHECKED_ROOT / "config" / "config.yaml",
    )
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_dir = resolve_path(config_path, config["data"]["output_dir"])
    train_path = data_dir / config["data"]["train_jsonl"]
    with train_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    records.sort(
        key=lambda record: (
            len(record["model_input"]["news_text"])
            + sum(len(value) for value in record["model_input"]["observed_comments"])
            + len(record["target_comment"])
        ),
        reverse=True,
    )
    records = records[: args.limit]
    model_path = resolve_model_reference(config_path, config["model"]["name_or_path"])
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        min_pixels=int(config["model"]["min_pixels"]),
        max_pixels=int(config["model"]["max_pixels"]),
    )
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    template = load_prompt_template(resolve_path(config_path, config["prompt"]["template_path"]))
    system_prompt = resolve_path(
        config_path, config["prompt"]["system_prompt_path"]
    ).read_text(encoding="utf-8").strip()
    collator = PartialCommentCollator(
        processor,
        template,
        int(config["training"]["max_sequence_length"]),
        system_prompt,
    )
    token_lengths: list[int] = []
    for index, record in enumerate(records, start=1):
        batch = collator([record])
        token_lengths.append(int(batch["attention_mask"].sum().item()))
        if index % 20 == 0:
            print(f"checked={index}/{len(records)}")
    report = {
        "checked_samples": len(records),
        "selection": "largest combined news/observed/target character counts",
        "max_encoded_tokens": max(token_lengths, default=0),
        "configured_max_sequence_length": int(config["training"]["max_sequence_length"]),
        "all_targets_preserved": True,
    }
    path = CHECKED_ROOT / "reports" / "longest_sample_preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
