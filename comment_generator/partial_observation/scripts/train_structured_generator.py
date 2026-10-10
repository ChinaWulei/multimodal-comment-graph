from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from common import load_config, read_jsonl, resolve_model_reference, resolve_path, write_jsonl
from structure_utils import (
    STRUCTURE_SYSTEM_PROMPT,
    build_structure_messages,
    load_structure_template,
    render_structure_prompt,
    validate_structure_input,
)
from train_generator import GpuMetricsCallback, apply_lora, inspect_trainable_parameters, load_model


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator_structure.yaml"
    parser = argparse.ArgumentParser(description="Train a QLoRA generator for comment content and reply parent JSON.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument("--processor-smoke-test", action="store_true")
    parser.add_argument("--smoke-train", action="store_true")
    parser.add_argument("--formal-train", action="store_true", help="Train on official train split and validate on official validate split.")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


class StructuredDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class StructuredCollator:
    def __init__(self, processor, template: str, max_length: int) -> None:
        self.processor = processor
        self.template = template
        self.max_length = max_length

    @staticmethod
    def _image(path: str) -> Image.Image:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images, prompt_texts, full_texts = [], [], []
        for feature in features:
            validate_record(feature)
            model_input = feature["model_input"]
            images.append(self._image(model_input["image_path"]))
            prompt_messages = build_structure_messages(self.template, model_input)
            full_messages = build_structure_messages(
                self.template, model_input, feature["target_json"]
            )
            prompt_texts.append(
                self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            )
            full_texts.append(
                self.processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
            )
        batch = self.processor(
            text=full_texts, images=images, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        prompt_batch = self.processor(
            text=prompt_texts, images=images, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        labels = batch["input_ids"].clone()
        pad = self.processor.tokenizer.pad_token_id
        if pad is not None:
            labels[batch["input_ids"].eq(pad)] = -100
        for index, length in enumerate(prompt_batch["attention_mask"].sum(dim=1).tolist()):
            prompt_length = int(length)
            full_length = int(batch["attention_mask"][index].sum().item())
            if prompt_length >= full_length:
                raise ValueError("Structured target was truncated")
            if not torch.equal(prompt_batch["input_ids"][index, :prompt_length], batch["input_ids"][index, :prompt_length]):
                raise ValueError("Structured prompt is not a prefix of supervised sequence")
            labels[index, :prompt_length] = -100
        batch["labels"] = labels
        return dict(batch)


def validate_record(record: dict[str, Any]) -> None:
    expected = {"sample_id", "metadata", "model_input", "target", "target_json"}
    if set(record) != expected:
        raise ValueError(f"Unexpected structured record schema: {record.get('sample_id')}")
    validate_structure_input(record["model_input"])
    target = record["target"]
    if set(target) != {"parent_node_index", "content"}:
        raise ValueError("Structured target has unexpected fields")
    if int(target["parent_node_index"]) < 0 or int(target["parent_node_index"]) > len(record["model_input"]["observed_nodes"]):
        raise ValueError("Structured target parent is outside the visible graph")
    if json.loads(record["target_json"]) != target:
        raise ValueError("Structured target JSON does not equal structured target")


def smoke_split(records: list[dict[str, Any]], seed: int, validation_news: int, per_news: int):
    by_news: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_news.setdefault(record["metadata"]["news_id"], []).append(record)
    if len(by_news) <= validation_news:
        raise RuntimeError("Not enough unique news for a smoke validation holdout")
    held_out = set(random.Random(seed).sample(sorted(by_news), validation_news))
    train = [record for record in records if record["metadata"]["news_id"] not in held_out]
    evaluation = []
    for news_id in sorted(held_out):
        evaluation.extend(by_news[news_id][:per_news])
    return train, evaluation, sorted(held_out)


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    torch.manual_seed(int(config["seed"]))
    data_dir = resolve_path(config_path, config["data"]["output_dir"])
    source = data_dir / config["data"]["structure_train_jsonl"]
    records = read_jsonl(source)
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No structured SFT records available")
    for record in records:
        validate_record(record)
    template = load_structure_template(resolve_path(config_path, config["prompt"]["template_path"]))
    for record in records[:20]:
        render_structure_prompt(template, record["model_input"])
    overview = {
        "records": len(records),
        "unique_news": len({record["metadata"]["news_id"] for record in records}),
        "model_input_fields": ["image_path", "news_text", "observed_nodes"],
        "supervised_target_fields": ["parent_node_index", "content"],
        "truthfulness_label_in_model_input": False,
    }
    print(json.dumps(overview, ensure_ascii=False, indent=2))
    model_path = args.model_path or resolve_model_reference(config_path, config["model"]["name_or_path"])
    if args.check_data_only:
        if args.processor_smoke_test:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(
                model_path, local_files_only=True,
                min_pixels=int(config["model"]["min_pixels"]), max_pixels=int(config["model"]["max_pixels"])
            )
            processor.tokenizer.padding_side = "right"
            if processor.tokenizer.pad_token_id is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            collator = StructuredCollator(processor, template, int(config["training"]["max_sequence_length"]))
            lengths = [int(collator([record])["attention_mask"].sum().item()) for record in records[:20]]
            report = {"processor_smoke_samples": len(lengths), "max_encoded_tokens": max(lengths), "training_started": False}
            report_path = config_path.parent.parent / "reports" / "structure_processor_smoke.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.smoke_train and args.formal_train:
        raise ValueError("Choose either --smoke-train or --formal-train, not both")
    if not args.smoke_train and not args.formal_train:
        raise RuntimeError("Run --smoke-train first, then explicitly pass --formal-train for official split training.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen-VL structured training")
    if args.smoke_train:
        smoke = config["smoke_training"]
        train_records, eval_records, held_out = smoke_split(
            records, int(config["seed"]), int(smoke["validation_news"]), int(smoke["max_validation_samples_per_news"])
        )
        output_dir = args.output_dir.resolve() if args.output_dir else resolve_path(config_path, smoke["output_dir"])
        write_jsonl(data_dir / "structure_smoke_train.jsonl", train_records)
        write_jsonl(data_dir / "structure_smoke_val.jsonl", eval_records)
        training_mode = "smoke"
    else:
        formal = config.get("formal_training", {})
        validation_name = str(formal.get("validation_jsonl", "structure_validate.jsonl"))
        validation_path = data_dir / validation_name
        if not validation_path.is_file():
            raise FileNotFoundError(f"Official validation SFT file is missing: {validation_path}")
        train_records = records
        eval_records = read_jsonl(validation_path)
        for record in eval_records:
            validate_record(record)
        if not train_records or not eval_records:
            raise RuntimeError("Official train/validate SFT data must both be non-empty")
        if any(record["metadata"]["official_split"] != "train" for record in train_records):
            raise ValueError("Training SFT file contains a non-train record")
        if any(record["metadata"]["official_split"] != "validate" for record in eval_records):
            raise ValueError("Validation SFT file contains a non-validate record")
        train_news = {record["metadata"]["news_id"] for record in train_records}
        val_news = {record["metadata"]["news_id"] for record in eval_records}
        if train_news & val_news:
            raise ValueError("News leakage between official train and validate SFT files")
        output_dir = args.output_dir.resolve() if args.output_dir else resolve_path(
            config_path, formal.get("output_dir", config["training"]["output_dir"])
        )
        held_out = []
        training_mode = "official"
    model, processor = load_model(config, model_path, bool(args.local_files_only or config["model"]["local_files_only"]))
    model = apply_lora(model, config)
    model.config.use_cache = False
    trainable = inspect_trainable_parameters(model, output_dir)
    parameters = config["smoke_training"] if training_mode == "smoke" else config["training"]
    argument_kwargs = dict(
        output_dir=str(output_dir), per_device_train_batch_size=int(parameters["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(parameters.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(parameters["gradient_accumulation_steps"]),
        learning_rate=float(parameters["learning_rate"]), logging_steps=int(parameters["logging_steps"]),
        eval_strategy="steps", eval_steps=int(parameters["eval_steps"]), save_strategy="steps", save_steps=int(parameters["save_steps"]),
        save_total_limit=int(parameters.get("save_total_limit", 2)), bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=bool(parameters["gradient_checkpointing"]), optim=str(config["training"]["optimizer"]),
        max_grad_norm=float(parameters["max_grad_norm"]), remove_unused_columns=False, report_to=[]
    )
    # Validation is the sole model-selection signal.  Because save/eval share
    # the same fixed step interval, Trainer can restore the exact best adapter
    # before the final root-level save instead of silently exporting the last
    # update.
    argument_kwargs.update(
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    if training_mode == "smoke":
        argument_kwargs["max_steps"] = int(parameters["max_steps"])
    else:
        argument_kwargs["num_train_epochs"] = float(parameters["epochs"])
    arguments = TrainingArguments(**argument_kwargs)
    callback = GpuMetricsCallback()
    trainer = Trainer(
        model=model, args=arguments, train_dataset=StructuredDataset(train_records),
        eval_dataset=StructuredDataset(eval_records),
        data_collator=StructuredCollator(processor, template, int(config["training"]["max_sequence_length"])), callbacks=[callback]
    )
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    result.metrics["peak_gpu_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    result.metrics["peak_gpu_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    (output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    split_record = {"training_mode": training_mode, "train_news": len({r['metadata']['news_id'] for r in train_records}), "validation_news": len({r['metadata']['news_id'] for r in eval_records}), "news_overlap": 0,
                    "best_model_checkpoint": trainer.state.best_model_checkpoint,
                    "best_validation_loss": trainer.state.best_metric}
    if training_mode == "smoke":
        split_record.update({"held_out_news_ids": held_out, "not_official_validation": True})
    (output_dir / ("smoke_split.json" if training_mode == "smoke" else "official_split.json")).write_text(json.dumps(split_record, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / ("smoke_metrics.json" if training_mode == "smoke" else "train_metrics.json")).write_text(json.dumps({"train_metrics": result.metrics, "history": callback.history, "trainable": trainable}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"structured_adapter_saved_to={output_dir}")


if __name__ == "__main__":
    main()
