from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from common import (
    build_messages,
    load_config,
    read_jsonl,
    resolve_model_reference,
    resolve_path,
    seed_everything,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().with_name("config.yaml")
    parser = argparse.ArgumentParser(description="Generate comments under every candidate label.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--output-jsonl", type=Path, default=Path("generated_comments.jsonl"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--comments-per-label", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_for_inference(
    config: dict[str, Any], config_path: Path, adapter_dir: Path, local_only: bool
):
    model_config = config["model"]
    model_path = resolve_model_reference(config_path, model_config["name_or_path"])
    cache_dir = model_config.get("cache_dir")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = None
    device_map = None
    if bool(model_config.get("use_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(model_config.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(model_config.get("double_quant", True)),
            bnb_4bit_compute_dtype=dtype,
        )
        device_map = {"": 0}
    processor_source = adapter_dir if (adapter_dir / "preprocessor_config.json").exists() else model_path
    processor = AutoProcessor.from_pretrained(
        processor_source,
        cache_dir=cache_dir,
        local_files_only=local_only,
        min_pixels=int(model_config["min_pixels"]),
        max_pixels=int(model_config["max_pixels"]),
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        local_files_only=local_only,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        attn_implementation=str(model_config.get("attn_implementation", "sdpa")),
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return model, processor


def unique_news(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    selected = []
    seen = set()
    for record in records:
        news_id = str(record["news_id"])
        if news_id in seen:
            continue
        seen.add(news_id)
        selected.append(record)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Qwen-VL generation.")
    config, config_path = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    prepared_dir = resolve_path(config_path, config["data"]["prepared_dir"])
    assert prepared_dir is not None
    input_path = args.input_jsonl.resolve() if args.input_jsonl else prepared_dir / "validate.jsonl"
    if not input_path.is_file() or input_path.stat().st_size == 0:
        fallback = prepared_dir / "train.jsonl"
        print(f"No prepared validation samples; using {fallback} for qualitative generation.")
        input_path = fallback
    records = unique_news(read_jsonl(input_path), args.limit)
    if not records:
        raise RuntimeError(f"No input records in {input_path}")

    adapter_dir = args.adapter_dir.resolve()
    model, processor = load_for_inference(
        config, config_path, adapter_dir, args.local_files_only
    )
    generation_config = config["generation"]
    label_column = str(config["data"]["label_column"])
    candidate_labels = list(generation_config["candidate_labels"])
    comments_per_label = args.comments_per_label or int(generation_config["comments_per_label"])
    outputs = []

    for record in records:
        image_path = Path(record["image_path"])
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB").copy()
        for candidate_label in candidate_labels:
            messages = build_messages(record["title"], label_column, candidate_label)
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                truncation=True,
                max_length=int(config["data"]["max_sequence_length"]),
            ).to(model.device)
            generated_comments = []
            for index in range(comments_per_label):
                torch.manual_seed(seed + index)
                generated = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=float(generation_config["temperature"]),
                    top_p=float(generation_config["top_p"]),
                    max_new_tokens=int(generation_config["max_new_tokens"]),
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
                new_tokens = generated[:, inputs["input_ids"].shape[1] :]
                comment = processor.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0].strip()
                generated_comments.append(comment)
            outputs.append(
                {
                    "news_id": record["news_id"],
                    "title": record["title"],
                    "image_path": str(image_path),
                    "candidate_label": int(candidate_label),
                    "generated_comments": generated_comments,
                    "model_name": str(config["model"]["name_or_path"]),
                    "adapter_dir": str(adapter_dir),
                    "generation_config": {
                        "temperature": generation_config["temperature"],
                        "top_p": generation_config["top_p"],
                        "max_new_tokens": generation_config["max_new_tokens"],
                        "seed": seed,
                    },
                }
            )
            print(f"generated news_id={record['news_id']} candidate_label={candidate_label}")
    output_path = args.output_jsonl.resolve()
    write_jsonl(output_path, outputs)
    print(f"output={output_path} rows={len(outputs)}")


if __name__ == "__main__":
    main()
