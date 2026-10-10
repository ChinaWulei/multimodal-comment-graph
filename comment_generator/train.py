from __future__ import annotations

import argparse
import json
import platform
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
    TrainingArguments,
)

from common import (
    build_messages,
    load_config,
    read_jsonl,
    resolve_model_reference,
    resolve_path,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    default_config = Path(__file__).resolve().with_name("config.yaml")
    parser = argparse.ArgumentParser(description="QLoRA SFT for Fakeddit comment generation.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--model-path", help="Override model.name_or_path.")
    parser.add_argument("--output-dir", type=Path, help="Override training.output_dir.")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--check-data-only",
        action="store_true",
        help="Validate prepared data and one real image without loading the model.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load the model and run one optimizer step on at most two samples.",
    )
    return parser.parse_args()


class CommentSFTDataset(Dataset):
    def __init__(self, path: Path):
        self.path = path
        self.records = read_jsonl(path) if path.is_file() else []

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class QwenCommentCollator:
    def __init__(self, processor, max_length: int):
        self.processor = processor
        self.max_length = max_length

    @staticmethod
    def _open_image(path: str) -> Image.Image:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [self._open_image(feature["image_path"]) for feature in features]
        prompt_texts = []
        full_texts = []
        for feature in features:
            prompt_messages = build_messages(
                feature["title"], feature["label_column"], feature["label"]
            )
            full_messages = build_messages(
                feature["title"],
                feature["label_column"],
                feature["label"],
                feature["target_comment"],
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
            full_nonpad = int(batch["attention_mask"][index].sum().item())
            if prompt_length >= full_nonpad:
                raise ValueError(
                    "Target comment was fully truncated. Increase data.max_sequence_length "
                    "or reduce image resolution/comment length."
                )
            if not torch.equal(
                prompt_batch["input_ids"][index, :prompt_length],
                batch["input_ids"][index, :prompt_length],
            ):
                raise ValueError(
                    "Prompt tokens are not a prefix of the supervised conversation; "
                    "refusing to apply an incorrect loss mask."
                )
            labels[index, :prompt_length] = -100
            if not torch.any(labels[index].ne(-100)):
                raise ValueError("No target tokens remain after prompt masking.")
        batch["labels"] = labels
        return dict(batch)


def environment_report() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gib": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if torch.cuda.is_available()
            else None
        ),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }


def validate_data(train_dataset: CommentSFTDataset, eval_dataset: CommentSFTDataset) -> None:
    if not train_dataset:
        raise RuntimeError(
            f"No training examples in {train_dataset.path}. Run prepare_data.py first."
        )
    required = {
        "sample_id",
        "news_id",
        "title",
        "image_path",
        "label_column",
        "label",
        "target_comment",
    }
    seen_samples: set[str] = set()
    split_news: dict[str, set[str]] = {"train": set(), "validate": set()}
    for split, dataset in (("train", train_dataset), ("validate", eval_dataset)):
        for record in dataset.records:
            missing = required.difference(record)
            if missing:
                raise ValueError(f"Missing fields in {dataset.path}: {sorted(missing)}")
            if record["sample_id"] in seen_samples:
                raise ValueError(f"Duplicate sample_id: {record['sample_id']}")
            seen_samples.add(record["sample_id"])
            split_news[split].add(str(record["news_id"]))
            image_path = Path(record["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not str(record["target_comment"]).strip():
                raise ValueError(f"Empty target comment: {record['sample_id']}")
    overlap = split_news["train"] & split_news["validate"]
    if overlap:
        raise ValueError(f"News leakage across train/validate: {sorted(overlap)[:10]}")
    first_image = Path(train_dataset.records[0]["image_path"])
    with Image.open(first_image) as image:
        image.load()
        print(
            f"first_image={first_image} format={image.format} size={image.size} mode={image.mode}"
        )


def load_model_and_processor(config: dict[str, Any], model_path: str, local_only: bool):
    model_config = config["model"]
    cache_dir = model_config.get("cache_dir")
    min_pixels = int(model_config["min_pixels"])
    max_pixels = int(model_config["max_pixels"])
    processor = AutoProcessor.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        local_files_only=local_only,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization_config = None
    device_map = None
    if bool(model_config.get("use_4bit", True)):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(model_config.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(model_config.get("double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        device_map = {"": 0}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        local_files_only=local_only,
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        attn_implementation=str(model_config.get("attn_implementation", "sdpa")),
    )
    return model, processor


def apply_lora(model, config: dict[str, Any]):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if bool(config["model"].get("use_4bit", True)):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config["training"].get("gradient_checkpointing", True)),
        )
    else:
        model.requires_grad_(False)

    lora_config = config["lora"]
    peft_config = LoraConfig(
        r=int(lora_config["rank"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=list(lora_config["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if bool(config["model"].get("freeze_vision", True)):
        for name, parameter in model.named_parameters():
            if "visual" in name.lower() or "vision_tower" in name.lower():
                parameter.requires_grad = False
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    config, config_path = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    prepared_dir = resolve_path(config_path, config["data"]["prepared_dir"])
    assert prepared_dir is not None
    train_dataset = CommentSFTDataset(prepared_dir / "train.jsonl")
    eval_dataset = CommentSFTDataset(prepared_dir / "validate.jsonl")
    validate_data(train_dataset, eval_dataset)

    report = environment_report()
    report.update(
        {
            "train_examples": len(train_dataset),
            "train_news": len({row["news_id"] for row in train_dataset.records}),
            "validate_examples": len(eval_dataset),
            "validate_news": len({row["news_id"] for row in eval_dataset.records}),
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.check_data_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Refusing to start formal Qwen-VL training on CPU.")

    model_path = args.model_path or resolve_model_reference(
        config_path, config["model"]["name_or_path"]
    )
    local_only = bool(args.local_files_only or config["model"].get("local_files_only", False))
    model, processor = load_model_and_processor(config, model_path, local_only)
    model = apply_lora(model, config)
    model.config.use_cache = False

    training_config = config["training"]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_path(config_path, training_config["output_dir"])
    )
    assert output_dir is not None
    if args.smoke_test:
        output_dir = output_dir.parent / "smoke_test"
        train_dataset = Subset(train_dataset, range(min(2, len(train_dataset))))

    has_eval = len(eval_dataset) > 0 and not args.smoke_test
    use_bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(training_config["epochs"]),
        max_steps=1 if args.smoke_test else -1,
        per_device_train_batch_size=int(training_config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=(
            1 if args.smoke_test else int(training_config["gradient_accumulation_steps"])
        ),
        learning_rate=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
        warmup_ratio=float(training_config["warmup_ratio"]),
        logging_steps=1 if args.smoke_test else int(training_config["logging_steps"]),
        save_strategy="no" if args.smoke_test else "steps",
        save_steps=int(training_config["save_steps"]),
        save_total_limit=int(training_config["save_total_limit"]),
        eval_strategy="steps" if has_eval else "no",
        eval_steps=int(training_config["eval_steps"]) if has_eval else None,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False if has_eval else None,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=bool(training_config["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=str(training_config["optimizer"]),
        max_grad_norm=float(training_config["max_grad_norm"]),
        dataloader_num_workers=int(training_config["num_workers"]),
        remove_unused_columns=False,
        report_to=[],
        seed=seed,
        data_seed=seed,
    )
    collator = QwenCommentCollator(processor, int(config["data"]["max_sequence_length"]))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if has_eval else None,
        data_collator=collator,
        processing_class=processor,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if torch.cuda.is_available():
        train_result.metrics["peak_gpu_allocated_gib"] = round(
            torch.cuda.max_memory_allocated() / (1024**3), 3
        )
        train_result.metrics["peak_gpu_reserved_gib"] = round(
            torch.cuda.max_memory_reserved() / (1024**3), 3
        )
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    if has_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    (output_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"adapter_saved_to={output_dir}")


if __name__ == "__main__":
    main()
