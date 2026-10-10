from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from common import (
    SYSTEM_PROMPT,
    build_messages,
    load_config,
    load_prompt_template,
    read_jsonl,
    resolve_model_reference,
    resolve_path,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator.yaml"
    parser = argparse.ArgumentParser(description="Generate comments from image, news text, and observed comments.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--split", choices=["train", "validate", "test"], default="validate")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--comments-per-news", type=int)
    parser.add_argument(
        "--all-input-records",
        action="store_true",
        help="Use every record in --input-jsonl without filtering its official split.",
    )
    parser.add_argument(
        "--unique-news",
        action="store_true",
        help="Keep at most one input record per news_id before applying --limit.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sample_seed(base_seed: int, news_id: str, index: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{news_id}:{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def load_model(config: dict, config_path: Path, adapter_dir: Path, local_only: bool):
    model_config = config["model"]
    model_path = resolve_model_reference(config_path, model_config["name_or_path"])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = None
    device_map = None
    if bool(model_config["use_4bit"]):
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(model_config["quant_type"]),
            bnb_4bit_use_double_quant=bool(model_config["double_quant"]),
            bnb_4bit_compute_dtype=dtype,
        )
        device_map = {"": 0}
    processor_source = adapter_dir if (adapter_dir / "preprocessor_config.json").exists() else model_path
    processor = AutoProcessor.from_pretrained(
        str(processor_source),
        local_files_only=local_only,
        min_pixels=int(model_config["min_pixels"]),
        max_pixels=int(model_config["max_pixels"]),
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=local_only,
        dtype=dtype,
        quantization_config=quantization,
        device_map=device_map,
        attn_implementation=str(model_config["attn_implementation"]),
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return model, processor


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen-VL generation")
    config, config_path = load_config(args.config)
    data_dir = resolve_path(config_path, config["data"]["output_dir"])
    input_path = args.input_jsonl.resolve() if args.input_jsonl else data_dir / config["data"]["news_contexts_jsonl"]
    contexts = read_jsonl(input_path)
    selected = (
        contexts
        if args.all_input_records
        else [record for record in contexts if record["metadata"]["split"] == args.split]
    )
    if not selected:
        available = sorted({record["metadata"]["split"] for record in contexts})
        raise RuntimeError(
            f"No {args.split} contexts. Available splits: {available}. "
            "Do not silently use train for validation or test claims."
        )
    if args.unique_news:
        unique = {}
        for record in selected:
            unique.setdefault(record["metadata"]["news_id"], record)
        selected = list(unique.values())
    if args.limit > 0:
        selected = selected[: args.limit]
    adapter_dir = args.adapter_dir.resolve()
    model, processor = load_model(
        config,
        config_path,
        adapter_dir,
        bool(args.local_files_only or config["model"]["local_files_only"]),
    )
    template = load_prompt_template(resolve_path(config_path, config["prompt"]["template_path"]))
    system_prompt = (
        resolve_path(config_path, config["prompt"]["system_prompt_path"])
        .read_text(encoding="utf-8")
        .strip()
        if config["prompt"].get("system_prompt_path")
        else SYSTEM_PROMPT
    )
    generation = config["generation"]
    count = args.comments_per_news or int(generation["comments_per_news"])
    base_seed = int(config["seed"])
    outputs = []
    for context in selected:
        metadata = context["metadata"]
        model_input = context["model_input"]
        if set(model_input) != {"image_path", "news_text", "observed_comments"}:
            raise ValueError(f"Forbidden model input field for {metadata['news_id']}")
        with Image.open(model_input["image_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB").copy()
        messages = build_messages(template, model_input, system_prompt=system_prompt)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            truncation=True,
            max_length=int(config["training"]["max_sequence_length"]),
        ).to(model.device)
        comments = []
        seeds = []
        pad_token_id = (
            processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else processor.tokenizer.eos_token_id
        )
        for index in range(count):
            seed = sample_seed(base_seed, metadata["news_id"], index)
            seeds.append(seed)
            torch.manual_seed(seed)
            generated = model.generate(
                **inputs,
                do_sample=True,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                repetition_penalty=float(generation["repetition_penalty"]),
                max_new_tokens=int(generation["max_new_tokens"]),
                pad_token_id=pad_token_id,
            )
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            comments.append(
                processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
            )
        outputs.append(
            {
                "news_id": metadata["news_id"],
                "split": metadata["split"],
                "image_path": model_input["image_path"],
                "news_text": model_input["news_text"],
                "observed_comments": model_input["observed_comments"],
                "generated_comments": comments,
                "adapter_dir": str(adapter_dir),
                "generation_config": {
                    "comments_per_news": count,
                    "temperature": generation["temperature"],
                    "top_p": generation["top_p"],
                    "repetition_penalty": generation["repetition_penalty"],
                    "max_new_tokens": generation["max_new_tokens"],
                    "seeds": seeds,
                },
                **(
                    {"real_target_comment": context["target_comment"]}
                    if "target_comment" in context
                    else {}
                ),
            }
        )
        print(f"generated news_id={metadata['news_id']} comments={count}")
    output_path = (
        args.output_jsonl.resolve()
        if args.output_jsonl
        else config_path.parent.parent / "outputs" / "generated_comments.jsonl"
    )
    write_jsonl(output_path, outputs)
    print(json.dumps({"news": len(outputs), "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
