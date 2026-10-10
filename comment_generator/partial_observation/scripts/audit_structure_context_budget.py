"""Check real image+text chat-template token length before Qwen generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps
from transformers import AutoProcessor

from common import load_config, read_jsonl, resolve_model_reference, resolve_path
from structure_utils import build_structure_messages, load_structure_template, validate_structure_input


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    model_path = resolve_model_reference(config_path, config["model"]["name_or_path"])
    processor = AutoProcessor.from_pretrained(
        model_path, local_files_only=True,
        min_pixels=int(config["model"]["min_pixels"]),
        max_pixels=int(config["model"]["max_pixels"]),
    )
    template = load_structure_template(resolve_path(config_path, config["prompt"]["template_path"]))
    max_tokens = int(config["training"]["max_sequence_length"])
    records = []
    for row in read_jsonl(args.input_jsonl):
        model_input = row["model_input"]
        validate_structure_input(model_input)
        allowed = list(range(len(model_input["observed_nodes"]) + 1))
        messages = build_structure_messages(template, model_input, allowed_parent_indices=allowed)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(model_input["image_path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tokens = processor(text=[text], images=[image], return_tensors="pt", truncation=False)
        length = int(tokens["input_ids"].shape[1])
        records.append({"news_id": str(row["metadata"]["news_id"]),
                        "prompt_tokens_first_step": length,
                        "observed_nodes": len(model_input["observed_nodes"]),
                        "over_budget_first_step": length > max_tokens})
    lengths = sorted(record["prompt_tokens_first_step"] for record in records)
    result = {"input": str(args.input_jsonl.resolve()), "records": len(records),
              "max_sequence_length": max_tokens,
              "first_step_over_budget": sum(record["over_budget_first_step"] for record in records),
              "p50_first_step": lengths[len(lengths) // 2],
              "p95_first_step": lengths[round(.95 * (len(lengths) - 1))],
              "max_first_step": lengths[-1],
              "warning": "Later generation steps append up to five comments and may exceed this first-step estimate.",
              "cases": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
