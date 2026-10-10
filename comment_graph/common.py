"""Shared configuration and dataset construction utilities."""

from __future__ import annotations

import random
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from munch import DefaultMunch
from torchvision import transforms
from transformers import AutoTokenizer

from comment_graph.dataset import CheckedCommentGraphDataset


def load_config(path: str | Path):
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = DefaultMunch.fromDict(yaml.safe_load(handle))
    model_reference = str(config.encoder.text_model_name)
    if bool(config.encoder.local_files_only) and not Path(model_reference).exists():
        # Passing only a Hub ID to older model wrappers can still trigger a HEAD
        # request. Resolve the already-cached snapshot once and pass its concrete
        # local directory to both tokenizer and encoder.
        config.encoder.text_model_name = snapshot_download(
            repo_id=model_reference,
            local_files_only=True,
        )
    return config, config_path


def resolve_path(config_path: Path, value: str | Path) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_tokenizer_and_transform(config):
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.encoder.text_model_name),
        local_files_only=bool(config.encoder.local_files_only),
        use_fast=True,
    )
    image_transform = transforms.Compose(
        [
            transforms.Resize(int(config.data.image_size) + 32),
            transforms.CenterCrop(int(config.data.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return tokenizer, image_transform


def generated_path_for_split(config, config_path: Path, split: str) -> Path:
    value = config.data.generated_comments.get(split)
    if not value:
        raise FileNotFoundError(f"No generated-comments path configured for split={split}")
    return resolve_path(config_path, value)


def generated_coverage_ids(config, config_path: Path, split: str) -> set[str]:
    """Return news IDs with a persisted generated graph for a formal paired arm."""
    path = generated_path_for_split(config, config_path, split)
    if not path.is_file():
        raise FileNotFoundError(
            f"Paired generated-coverage control requires completed graph file: {path}"
        )
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            news_id = str(row.get("news_id", ""))
            if not news_id or news_id in ids:
                raise ValueError(f"Invalid generated coverage news_id in {path}: {news_id!r}")
            if str(row.get("official_split", "")) != split:
                raise ValueError(f"Generated graph split mismatch in {path}: {news_id}")
            ids.add(news_id)
    if not ids:
        raise RuntimeError(f"No generated graph coverage in {path}")
    return ids


def make_dataset(
    config,
    config_path: Path,
    split: str,
    mode: str,
    tokenizer,
    image_transform,
    max_samples=None,
    allowed_news_ids=None,
):
    if (config.graph.get("structure_mode") == "generated_root" and
            config.data.get("dataset_name", "checked") != "fakeddit_structure"):
        raise ValueError("generated_root placement ablation requires the Fakeddit structure adapter")
    if config.data.get("dataset_name", "checked") == "weibo":
        if mode != "baseline":
            raise ValueError("This local Weibo dataset has no comments; use baseline, not fabricated comment graphs")
        from comment_graph.weibo_dataset import WeiboImageTextDataset
        return WeiboImageTextDataset(resolve_path(config_path, config.data.audit_json), split,
                                     tokenizer, image_transform, config, max_samples, allowed_news_ids)
    if config.data.get("dataset_name", "checked") == "fakeddit_structure":
        from comment_graph.fakeddit_structure_dataset import FakedditStructureGraphDataset
        generated_path = None
        if mode in {"generated_only", "real_generated"} or config.data.get("real_node_selection", "prefix") == "observed_context":
            generated_path = generated_path_for_split(config, config_path, split)
        return FakedditStructureGraphDataset(
            manifest_csv=resolve_path(config_path, config.data.manifest_csv),
            real_trees_jsonl=resolve_path(config_path, config.data.real_trees_jsonl),
            generated_graphs_jsonl=generated_path,
            split=split, mode=mode, tokenizer=tokenizer, image_transform=image_transform,
            config=config, max_samples=max_samples, allowed_news_ids=allowed_news_ids,
        )
    generated_path = None
    if mode in {"generated_only", "real_generated"}:
        generated_path = generated_path_for_split(config, config_path, split)
    return CheckedCommentGraphDataset(
        manifest_csv=resolve_path(config_path, config.data.manifest_csv),
        real_comments_jsonl=resolve_path(config_path, config.data.real_comments_jsonl),
        generated_comments_jsonl=generated_path,
        split=split,
        mode=mode,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_real_comments=int(config.data.max_real_comments),
        max_generated_comments=int(config.data.max_generated_comments),
        max_news_length=int(config.data.max_news_length),
        max_comment_length=int(config.data.max_comment_length),
        strict_generated_coverage=bool(config.data.strict_generated_coverage),
        max_samples=max_samples,
        allowed_news_ids=allowed_news_ids,
    )
