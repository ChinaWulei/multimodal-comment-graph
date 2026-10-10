from __future__ import annotations

import argparse
import json
import math
import platform
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from common import (
    SYSTEM_PROMPT,
    build_messages,
    load_config,
    load_prompt_template,
    read_jsonl,
    render_user_prompt,
    resolve_model_reference,
    resolve_path,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    default = Path(__file__).resolve().parents[1] / "config" / "generator.yaml"
    parser = argparse.ArgumentParser(description="Label-free QLoRA comment generator training.")
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--check-data-only", action="store_true")
    parser.add_argument(
        "--processor-smoke-test",
        action="store_true",
        help="Load only the local Processor and collate the selected records; no model or training.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--smoke-train",
        action="store_true",
        help="Run the configured short QLoRA training and validation check.",
    )
    return parser.parse_args()


class CommentDataset(Dataset):
    def __init__(
        self,
        path: Path,
        limit: int = 0,
        records: list[dict[str, Any]] | None = None,
    ):
        self.path = path
        self.records = (
            list(records)
            if records is not None
            else (read_jsonl(path) if path.is_file() and path.stat().st_size else [])
        )
        if limit > 0:
            self.records = self.records[:limit]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class GpuMetricsCallback(TrainerCallback):
    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return control
        event = {"step": int(state.global_step), **dict(logs)}
        if torch.cuda.is_available():
            event["gpu_allocated_gib"] = round(
                torch.cuda.memory_allocated() / 1024**3, 3
            )
            event["gpu_reserved_gib"] = round(
                torch.cuda.memory_reserved() / 1024**3, 3
            )
        self.history.append(event)
        return control


class PartialCommentCollator:
    def __init__(
        self,
        processor,
        prompt_template: str,
        max_length: int,
        system_prompt: str,
    ):
        self.processor = processor
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.system_prompt = system_prompt

    @staticmethod
    def open_image(path: str) -> Image.Image:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = []
        prompt_texts = []
        full_texts = []
        for feature in features:
            model_input = feature["model_input"]
            if set(model_input) != {"image_path", "news_text", "observed_comments"}:
                raise ValueError("Refusing a model_input containing metadata or labels")
            images.append(self.open_image(model_input["image_path"]))
            prompt_messages = build_messages(
                self.prompt_template, model_input, system_prompt=self.system_prompt
            )
            full_messages = build_messages(
                self.prompt_template,
                model_input,
                feature["target_comment"],
                system_prompt=self.system_prompt,
            )
            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )
            )

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[batch["input_ids"].eq(pad_token_id)] = -100
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for index, prompt_length in enumerate(prompt_lengths):
            prompt_length = int(prompt_length)
            full_length = int(batch["attention_mask"][index].sum().item())
            if prompt_length >= full_length:
                raise ValueError("Target was truncated; reduce input length or raise max_sequence_length")
            if not torch.equal(
                prompt_batch["input_ids"][index, :prompt_length],
                batch["input_ids"][index, :prompt_length],
            ):
                raise ValueError("Prompt is not a prefix of the supervised conversation")
            labels[index, :prompt_length] = -100
            if not torch.any(labels[index].ne(-100)):
                raise ValueError("No target tokens remain after loss masking")
        batch["labels"] = labels
        return dict(batch)


def validate_records(dataset: CommentDataset) -> None:
    if not dataset.records:
        raise RuntimeError(f"No records in {dataset.path}")
    seen = set()
    for record in dataset.records:
        if set(record) != {"sample_id", "metadata", "model_input", "target_comment"}:
            raise ValueError(f"Unexpected schema: {record.get('sample_id')}")
        if record["sample_id"] in seen:
            raise ValueError(f"Duplicate sample_id: {record['sample_id']}")
        seen.add(record["sample_id"])
        if set(record["model_input"]) != {"image_path", "news_text", "observed_comments"}:
            raise ValueError(f"Label leakage risk in model_input: {record['sample_id']}")
        if not str(record["target_comment"]).strip():
            raise ValueError(f"Empty target: {record['sample_id']}")
        with Image.open(record["model_input"]["image_path"]) as image:
            image.verify()


def prepare_smoke_datasets(
    train_data: CommentDataset,
    eval_data: CommentDataset,
    config: dict,
    data_dir: Path,
) -> tuple[CommentDataset, CommentDataset, dict[str, Any]]:
    smoke = config["smoke_training"]
    if eval_data.records:
        by_news: dict[str, list[dict[str, Any]]] = {}
        for record in eval_data.records:
            by_news.setdefault(record["metadata"]["news_id"], []).append(record)
        requested = int(smoke["validation_news"])
        selected_ids = sorted(by_news)[:requested] if requested > 0 else sorted(by_news)
        per_news_limit = int(smoke["max_validation_samples_per_news"])
        smoke_eval_records: list[dict[str, Any]] = []
        for news_id in selected_ids:
            records = by_news[news_id]
            smoke_eval_records.extend(
                records[:per_news_limit] if per_news_limit > 0 else records
            )
        smoke_eval = CommentDataset(
            data_dir / "smoke_val.jsonl", records=smoke_eval_records
        )
        write_jsonl(data_dir / "smoke_val.jsonl", smoke_eval_records)
        split_info = {
            "validation_scope": "subset_of_official_validate_for_smoke_only",
            "held_out_news_ids": selected_ids,
            "train_news": len({r["metadata"]["news_id"] for r in train_data.records}),
            "validation_news": len(selected_ids),
            "train_samples": len(train_data),
            "validation_samples": len(smoke_eval),
            "news_overlap": 0,
        }
        return train_data, smoke_eval, split_info

    by_news: dict[str, list[dict[str, Any]]] = {}
    for record in train_data.records:
        by_news.setdefault(record["metadata"]["news_id"], []).append(record)
    news_ids = sorted(by_news)
    requested = int(smoke["validation_news"])
    if len(news_ids) <= requested:
        raise RuntimeError("Not enough news IDs for a news-level smoke validation holdout")
    held_out_ids = set(random.Random(int(config["seed"])).sample(news_ids, requested))
    smoke_train_records = [
        record
        for record in train_data.records
        if record["metadata"]["news_id"] not in held_out_ids
    ]
    per_news_limit = int(smoke["max_validation_samples_per_news"])
    smoke_eval_records: list[dict[str, Any]] = []
    for news_id in sorted(held_out_ids):
        records = by_news[news_id]
        smoke_eval_records.extend(records[:per_news_limit] if per_news_limit > 0 else records)

    train_ids = {r["metadata"]["news_id"] for r in smoke_train_records}
    eval_ids = {r["metadata"]["news_id"] for r in smoke_eval_records}
    if train_ids & eval_ids:
        raise AssertionError("Smoke train/validation news_id leakage")
    write_jsonl(data_dir / "smoke_train.jsonl", smoke_train_records)
    write_jsonl(data_dir / "smoke_val.jsonl", smoke_eval_records)
    split_info = {
        "validation_scope": "internal_news_level_holdout_from_official_train",
        "warning": "This validates the training pipeline only; it is not an official validation score.",
        "held_out_news_ids": sorted(held_out_ids),
        "train_news": len(train_ids),
        "validation_news": len(eval_ids),
        "train_samples": len(smoke_train_records),
        "validation_samples": len(smoke_eval_records),
        "news_overlap": len(train_ids & eval_ids),
    }
    (data_dir / "smoke_split.json").write_text(
        json.dumps(split_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return (
        CommentDataset(data_dir / "smoke_train.jsonl", records=smoke_train_records),
        CommentDataset(data_dir / "smoke_val.jsonl", records=smoke_eval_records),
        split_info,
    )


def audit_final_prompts(
    processor,
    template: str,
    system_prompt: str,
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    audited = []
    for record in records[:20]:
        model_input = record["model_input"]
        if set(model_input) != {"image_path", "news_text", "observed_comments"}:
            raise AssertionError("Prompt audit rejected metadata or label fields")
        messages = build_messages(template, model_input, system_prompt=system_prompt)
        final_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        expected_user_text = render_user_prompt(template, model_input)
        actual_user_text = messages[1]["content"][1]["text"]
        if actual_user_text != expected_user_text:
            raise AssertionError("Final Processor prompt differs from model_input-only prompt")
        audited.append(
            {
                "sample_id": record["sample_id"],
                "model_input_fields": sorted(model_input),
                "metadata_passed_to_prompt_builder": False,
                "final_prompt": final_prompt,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audited, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def inspect_trainable_parameters(model, output_dir: Path) -> dict[str, Any]:
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    forbidden = [
        name
        for name in trainable_names
        if "visual" in name.lower() or "vision_tower" in name.lower()
    ]
    non_lora = [name for name in trainable_names if "lora_" not in name.lower()]
    if forbidden:
        raise RuntimeError(f"Vision parameters unexpectedly trainable: {forbidden[:5]}")
    if non_lora:
        raise RuntimeError(f"Non-LoRA parameters unexpectedly trainable: {non_lora[:5]}")
    report = {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percentage": 100.0 * trainable / total,
        "trainable_parameter_names": trainable_names,
        "vision_trainable_parameter_names": forbidden,
        "non_lora_trainable_parameter_names": non_lora,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trainable_parameters.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "total_parameters": report["total_parameters"],
                "trainable_parameters": report["trainable_parameters"],
                "trainable_percentage": report["trainable_percentage"],
                "trainable_tensor_names": len(trainable_names),
                "trainable_modules": sorted(
                    {name.rsplit(".", 1)[0] for name in trainable_names}
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return report


def write_smoke_report(
    report_path: Path,
    history: list[dict[str, Any]],
    train_metrics: dict[str, Any],
    trainable: dict[str, Any],
    split_info: dict[str, Any],
    output_dir: Path,
) -> None:
    by_step: dict[int, dict[str, Any]] = {}
    for event in history:
        step = int(event["step"])
        row = by_step.setdefault(step, {"step": step})
        for key in [
            "loss",
            "eval_loss",
            "learning_rate",
            "grad_norm",
            "gpu_allocated_gib",
            "gpu_reserved_gib",
        ]:
            if key in event:
                row[key] = event[key]
    loss_rows = [row for row in by_step.values() if "loss" in row]
    eval_rows = [row for row in by_step.values() if "eval_loss" in row]
    values = [
        float(row[key])
        for row in by_step.values()
        for key in ["loss", "eval_loss", "grad_norm"]
        if key in row
    ]
    has_non_finite = any(not math.isfinite(value) for value in values)
    train_decreased = bool(
        len(loss_rows) >= 2 and float(loss_rows[-1]["loss"]) < float(loss_rows[0]["loss"])
    )
    eval_increased = bool(
        len(eval_rows) >= 2
        and float(eval_rows[-1]["eval_loss"]) > float(eval_rows[0]["eval_loss"]) * 1.05
    )
    exploding = any(float(row.get("grad_norm", 0)) > 100 for row in by_step.values())
    lines = [
        "# QLoRA Smoke Training Report",
        "",
        "## Scope",
        "",
        f"- Validation scope: `{split_info['validation_scope']}`",
        f"- Train: {split_info['train_news']} news / {split_info['train_samples']} samples",
        f"- Validation: {split_info['validation_news']} news / {split_info['validation_samples']} samples",
        f"- News ID overlap: {split_info.get('news_overlap', 0)}",
        "- The internal holdout is for pipeline validation only and is not an official Fakeddit validation score.",
        "",
        "## Parameters",
        "",
        f"- Total parameters: {trainable['total_parameters']:,}",
        f"- Trainable LoRA parameters: {trainable['trainable_parameters']:,}",
        f"- Trainable percentage: {trainable['trainable_percentage']:.4f}%",
        "- Vision trainable parameters: 0",
        "",
        "## Logged metrics",
        "",
        "| step | train loss | eval loss | learning rate | grad norm | GPU allocated GiB | GPU reserved GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for step in sorted(by_step):
        row = by_step[step]
        lines.append(
            "| {step} | {loss} | {eval_loss} | {lr} | {grad} | {alloc} | {reserved} |".format(
                step=step,
                loss=row.get("loss", ""),
                eval_loss=row.get("eval_loss", ""),
                lr=row.get("learning_rate", ""),
                grad=row.get("grad_norm", ""),
                alloc=row.get("gpu_allocated_gib", ""),
                reserved=row.get("gpu_reserved_gib", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- 训练 loss 整体下降：{'是' if train_decreased else '否'}",
            f"- eval loss 较首次上升超过 5%：{'是' if eval_increased else '否'}",
            f"- NaN / Inf：{'有' if has_non_finite else '无'}",
            f"- 梯度范数超过 100：{'有' if exploding else '无'}",
            "- CUDA OOM：无（训练完整结束）",
            f"- Peak allocated: {train_metrics.get('peak_gpu_allocated_gib', 'n/a')} GiB",
            f"- Peak reserved: {train_metrics.get('peak_gpu_reserved_gib', 'n/a')} GiB",
            f"- Adapter: `{output_dir}`",
            "",
            "## Interpretation",
            "",
            "- 两个 eval 点只能检查训练流程和明显异常，不能据此判断模型泛化能力。",
            "- 当前本地图片没有官方 validate/test，因此不能把内部留出结果写成正式验证性能。",
            "",
            "## Next stage after full generator training",
            "",
            "1. 分别为 train / validate / test 生成 K=5 条 synthetic comments。",
            "2. 评价长度、重复率、语义相似度、BERTScore、unsupported claims，并进行人工抽样。",
            "3. 生成 detector_input.jsonl。",
            "4. 后续分别评估 Image+Text、加入真实评论、加入生成评论、同时加入两类评论；另做 Random/Shuffled Comments 控制实验。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_model(config: dict, model_path: str, local_only: bool):
    model_config = config["model"]
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=local_only,
        min_pixels=int(model_config["min_pixels"]),
        max_pixels=int(model_config["max_pixels"]),
    )
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=local_only,
        dtype=dtype,
        quantization_config=quantization,
        device_map=device_map,
        attn_implementation=str(model_config["attn_implementation"]),
    )
    return model, processor


def apply_lora(model, config: dict):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if bool(config["model"]["use_4bit"]):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config["training"]["gradient_checkpointing"]),
        )
    else:
        model.requires_grad_(False)
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora["rank"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    if bool(config["model"]["freeze_vision"]):
        for name, parameter in model.named_parameters():
            if "visual" in name.lower() or "vision_tower" in name.lower():
                parameter.requires_grad = False
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    if args.smoke_test and args.smoke_train:
        raise ValueError("Choose either --smoke-test or --smoke-train")
    config, config_path = load_config(args.config)
    torch.manual_seed(int(config["seed"]))
    data_dir = resolve_path(config_path, config["data"]["output_dir"])
    expected_pool_source = config["data"].get(
        "comment_pool_source", "comment_pool_jsonl"
    )
    stats_path = data_dir / config["data"]["stats_json"]
    if expected_pool_source != "comment_pool_jsonl":
        if not stats_path.is_file():
            raise RuntimeError(
                f"Missing {stats_path}; rebuild SFT data from {expected_pool_source} first"
            )
        data_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        actual_pool_source = data_stats.get("protocol", {}).get("comment_pool_source")
        if actual_pool_source != expected_pool_source:
            raise RuntimeError(
                "Training-data provenance mismatch: "
                f"expected {expected_pool_source}, found {actual_pool_source!r}"
            )
    train_data = CommentDataset(data_dir / config["data"]["train_jsonl"], args.limit)
    eval_data = CommentDataset(data_dir / config["data"]["val_jsonl"])
    validate_records(train_data)
    template = load_prompt_template(resolve_path(config_path, config["prompt"]["template_path"]))
    system_prompt = (
        resolve_path(config_path, config["prompt"]["system_prompt_path"])
        .read_text(encoding="utf-8")
        .strip()
        if config["prompt"].get("system_prompt_path")
        else SYSTEM_PROMPT
    )
    for record in train_data.records[: min(20, len(train_data))]:
        render_user_prompt(template, record["model_input"])
    report = {
        "platform": platform.platform(),
        "train_samples": len(train_data),
        "train_news": len({r["metadata"]["news_id"] for r in train_data.records}),
        "validate_samples": len(eval_data),
        "label_fields_read_by_collator": [],
        "model_input_fields": ["image_path", "news_text", "observed_comments"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.check_data_only:
        if args.processor_smoke_test:
            model_path = args.model_path or resolve_model_reference(
                config_path, config["model"]["name_or_path"]
            )
            processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True,
                min_pixels=int(config["model"]["min_pixels"]),
                max_pixels=int(config["model"]["max_pixels"]),
            )
            processor.tokenizer.padding_side = "right"
            if processor.tokenizer.pad_token_id is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            collator = PartialCommentCollator(
                processor,
                template,
                int(config["training"]["max_sequence_length"]),
                system_prompt,
            )
            token_lengths = []
            for record in train_data.records:
                batch = collator([record])
                token_lengths.append(int(batch["attention_mask"].sum().item()))
            smoke_report = {
                "processor_smoke_samples": len(token_lengths),
                "max_encoded_tokens": max(token_lengths, default=0),
                "model_loaded": False,
                "training_started": False,
            }
            smoke_path = config_path.parent.parent / "reports" / "processor_smoke.json"
            smoke_path.parent.mkdir(parents=True, exist_ok=True)
            smoke_path.write_text(
                json.dumps(smoke_report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(smoke_report, ensure_ascii=False, indent=2))
            print(f"processor_smoke_report={smoke_path}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Qwen-VL training")

    smoke_split_info: dict[str, Any] | None = None
    if args.smoke_train:
        train_data, eval_data, smoke_split_info = prepare_smoke_datasets(
            train_data, eval_data, config, data_dir
        )
        validate_records(train_data)
        validate_records(eval_data)

    model_path = args.model_path or resolve_model_reference(
        config_path, config["model"]["name_or_path"]
    )
    local_only = bool(args.local_files_only or config["model"]["local_files_only"])
    model, processor = load_model(config, model_path, local_only)
    training = config["training"]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_path(
            config_path,
            config["smoke_training"]["output_dir"]
            if args.smoke_train
            else training["output_dir"],
        )
    )
    audit_final_prompts(
        processor,
        template,
        system_prompt,
        train_data.records,
        config_path.parent.parent / "reports" / "prompt_audit.json",
    )
    model = apply_lora(model, config)
    model.config.use_cache = False
    trainable_report = inspect_trainable_parameters(model, output_dir)
    if args.smoke_test:
        output_dir = output_dir.parent / "partial_smoke_test"
        train_data = Subset(train_data, range(min(2, len(train_data))))
    smoke = config.get("smoke_training", {})
    has_eval = len(eval_data) > 0 and not args.smoke_test
    max_steps = 1 if args.smoke_test else int(smoke["max_steps"]) if args.smoke_train else -1
    train_batch_size = int(
        smoke["per_device_train_batch_size"]
        if args.smoke_train
        else training["per_device_train_batch_size"]
    )
    grad_accumulation = int(
        smoke["gradient_accumulation_steps"]
        if args.smoke_train
        else (1 if args.smoke_test else training["gradient_accumulation_steps"])
    )
    learning_rate = float(
        smoke["learning_rate"] if args.smoke_train else training["learning_rate"]
    )
    logging_steps = int(
        smoke["logging_steps"]
        if args.smoke_train
        else (1 if args.smoke_test else training["logging_steps"])
    )
    eval_steps = int(smoke["eval_steps"] if args.smoke_train else training["eval_steps"])
    save_steps = int(smoke["save_steps"] if args.smoke_train else training["save_steps"])
    gradient_checkpointing = bool(
        smoke["gradient_checkpointing"]
        if args.smoke_train
        else training["gradient_checkpointing"]
    )
    max_grad_norm = float(
        smoke["max_grad_norm"] if args.smoke_train else training["max_grad_norm"]
    )
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training["epochs"]),
        max_steps=max_steps,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training["warmup_ratio"]),
        logging_steps=logging_steps,
        save_strategy="no" if args.smoke_test else "steps",
        save_steps=save_steps,
        save_total_limit=int(training["save_total_limit"]),
        eval_strategy="steps" if has_eval else "no",
        eval_steps=eval_steps if has_eval else None,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=gradient_checkpointing,
        optim=str(training["optimizer"]),
        max_grad_norm=max_grad_norm,
        dataloader_num_workers=int(training["num_workers"]),
        remove_unused_columns=False,
        report_to=[],
    )
    gpu_callback = GpuMetricsCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data if has_eval else None,
        data_collator=PartialCommentCollator(
            processor,
            template,
            int(training["max_sequence_length"]),
            system_prompt,
        ),
        callbacks=[gpu_callback],
    )
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    result.metrics["peak_gpu_allocated_gib"] = round(
        torch.cuda.max_memory_allocated() / 1024**3, 3
    )
    result.metrics["peak_gpu_reserved_gib"] = round(
        torch.cuda.max_memory_reserved() / 1024**3, 3
    )
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    (output_dir / "prompt_template.txt").write_text(template + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "logged_metrics.json").write_text(
        json.dumps(gpu_callback.history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.smoke_train and smoke_split_info is not None:
        (output_dir / "smoke_split.json").write_text(
            json.dumps(smoke_split_info, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_smoke_report(
            config_path.parent.parent / "reports" / "smoke_train_report.md",
            gpu_callback.history,
            result.metrics,
            trainable_report,
            smoke_split_info,
            output_dir,
        )
    print(f"adapter_saved_to={output_dir}")


if __name__ == "__main__":
    main()
