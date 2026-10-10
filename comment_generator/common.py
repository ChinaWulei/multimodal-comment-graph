from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import yaml


LABEL_MAPPINGS = {
    "2_way_label": {
        "0": "Fake",
        "1": "True",
    },
    "3_way_label": {
        "0": "True",
        "1": "Fake with true text",
        "2": "Fake with false text",
    },
    "6_way_label": {
        "0": "True",
        "1": "Satire/Parody",
        "2": "False Connection",
        "3": "Imposter Content",
        "4": "Manipulated Content",
        "5": "Misleading Content",
    },
}

SYSTEM_PROMPT = (
    "You write one natural Reddit-style user comment in response to a post. "
    "The comment must be relevant to the title and image. It may express agreement, "
    "doubt, a question, surprise, or useful context. The dataset authenticity condition "
    "is metadata about the post, not the commenter's stance. Do not mechanically repeat "
    "the label, do not force support or opposition, and do not invent verified facts or sources. "
    "Return only the comment body."
)


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config, config_path


def resolve_path(config_path: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def resolve_model_reference(config_path: Path, value: str | Path) -> str:
    """Resolve a config-relative local model directory, or keep a Hub model ID."""
    raw = str(value)
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    local_candidate = (config_path.parent / path).resolve()
    return str(local_candidate) if local_candidate.exists() else raw


def label_text(label_column: str, value: object) -> str:
    key = str(value).strip()
    mapping = LABEL_MAPPINGS.get(label_column)
    if mapping is None:
        raise ValueError(
            f"Unsupported label column {label_column!r}. "
            f"Choose one of {sorted(LABEL_MAPPINGS)}."
        )
    if key not in mapping:
        raise ValueError(f"Unknown value {key!r} for {label_column}")
    return mapping[key]


def user_prompt(title: str, label_column: str, value: object) -> str:
    condition = label_text(label_column, value)
    return (
        f"Post title: {title.strip()}\n"
        f"Dataset authenticity condition ({label_column}): {condition}\n\n"
        "Write one plausible user comment responding to this post and its image."
    )


def build_messages(
    title: str,
    label_column: str,
    label_value: object,
    target_comment: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": user_prompt(title, label_column, label_value),
                },
            ],
        },
    ]
    if target_comment is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_comment.strip()}],
            }
        )
    return messages


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
