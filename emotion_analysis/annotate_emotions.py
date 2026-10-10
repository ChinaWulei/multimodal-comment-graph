from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import load_config, read_jsonl, resolve_path, write_jsonl


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().with_name("config.yaml")
    parser = argparse.ArgumentParser(description="Annotate comments with GoEmotions.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Relative paths are resolved from the config directory.",
    )
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    data_dir = resolve_path(config_path, config["data"]["directory"])
    output_dir = resolve_path(config_path, config["output"]["directory"])
    input_path = data_dir / config["data"]["extracted_comments"]
    configured_output = Path(config["output"]["comment_emotions"])
    if args.output:
        output_path = args.output if args.output.is_absolute() else config_path.parent / args.output
    elif args.limit > 0:
        output_path = output_dir / (
            f"{configured_output.stem}.smoke{configured_output.suffix}"
        )
    else:
        output_path = output_dir / configured_output
    records = read_jsonl(input_path)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError(f"No comments found in {input_path}")

    model_config = config["emotion_model"]
    device = choose_device(args.device or str(model_config.get("device", "auto")))
    model_name = str(model_config["name_or_path"])
    local_only = bool(args.local_files_only or model_config.get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=model_config.get("cache_dir"),
        local_files_only=local_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        cache_dir=model_config.get("cache_dir"),
        local_files_only=local_only,
    ).to(device)
    model.eval()
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    labels = [id2label[index] for index in range(len(id2label))]
    threshold = float(model_config["threshold"])
    batch_size = int(model_config["batch_size"])
    max_length = int(model_config["max_length"])
    annotated = []

    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        tokens = tokenizer(
            [record["body"] for record in batch_records],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            probabilities = torch.sigmoid(model(**tokens).logits).cpu()
        for record, probability in zip(batch_records, probabilities):
            values = {label: round(float(probability[index]), 8) for index, label in enumerate(labels)}
            selected = [label for label, value in values.items() if value >= threshold]
            top_index = int(probability.argmax().item())
            output_record = dict(record)
            output_record.update(
                {
                    "emotion_model": model_name,
                    "emotion_threshold": threshold,
                    "emotion_probabilities": values,
                    "predicted_emotions": selected,
                    "top_emotion": labels[top_index],
                    "top_emotion_probability": round(float(probability[top_index]), 8),
                }
            )
            annotated.append(output_record)
        print(f"annotated={len(annotated):,}/{len(records):,}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, annotated)
    metadata = {
        "model": model_name,
        "device": str(device),
        "comments": len(annotated),
        "labels": labels,
        "threshold": threshold,
        "input": str(input_path),
        "output": str(output_path),
    }
    metadata_name = (
        "emotion_annotation_metadata.smoke.json"
        if args.limit > 0
        else "emotion_annotation_metadata.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    (output_path.parent / metadata_name).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
