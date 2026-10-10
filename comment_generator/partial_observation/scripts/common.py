from __future__ import annotations

import hashlib
import json
import random
import re
import string
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
LABEL_COLUMNS = ["2_way_label", "3_way_label", "6_way_label"]
INVALID_COMMENT_BODIES = {"[deleted]", "[removed]"}
AUTOMATED_AUTHORS = {"automoderator", "fakehistorypornbot"}
AUTOMATED_BODY_MARKERS = {
    "i am a bot, and this action was performed automatically",
    "this action was performed automatically. please contact the moderators",
}
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE = re.compile(r"\s+")
COMMENT_RECORD_START = re.compile(r"^(\d+)\t([A-Za-z0-9]+)\t")

SYSTEM_PROMPT = (
    "Write one natural Reddit-style user comment in response to the supplied post and "
    "existing discussion. Use the image, news text, and observed comments. Do not claim "
    "external verification, invent unsupported personal experience, or output analysis "
    "about the task. Return only the comment body."
)


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config, config_path


def resolve_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def resolve_model_reference(config_path: Path, value: str | Path) -> str:
    raw = str(value)
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    candidate = (config_path.parent / path).resolve()
    return str(candidate) if candidate.exists() else raw


def normalize_reddit_id(value: object) -> str:
    text = "" if value is None else str(value).strip().lower()
    while text.startswith(("t1_", "t3_")):
        text = text[3:]
    return text


def decode_tsv_field(value: str) -> str:
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1].replace('""', '"')
    return value


def clean_comment(value: object) -> str:
    text = "" if value is None else str(value)
    text = CONTROL_CHARS.sub(" ", decode_tsv_field(text))
    return WHITESPACE.sub(" ", text).strip()


def is_obvious_automated_comment(author: object, body: object) -> bool:
    author_text = "" if author is None else str(author).strip().casefold()
    body_text = "" if body is None else str(body).strip().casefold()
    return author_text in AUTOMATED_AUTHORS or any(
        marker in body_text for marker in AUTOMATED_BODY_MARKERS
    )


def parse_comment_record(raw: str, source_index: int) -> dict[str, Any] | None:
    text = raw.rstrip("\r\n")
    left = text.split("\t", 3)
    if len(left) != 4:
        return None
    tail = left[3].rsplit("\t", 4)
    if len(tail) != 5:
        return None
    body, is_top_level, parent_id, submission_id, ups = tail
    try:
        ups_value: float | None = float(decode_tsv_field(ups).strip())
    except ValueError:
        ups_value = None
    return {
        "source_index": source_index,
        "comment_id": normalize_reddit_id(decode_tsv_field(left[1])),
        "author": decode_tsv_field(left[2]).strip(),
        "body": clean_comment(body),
        "is_top_level": decode_tsv_field(is_top_level).strip().lower()
        in {"true", "1", "yes"},
        "parent_id": normalize_reddit_id(decode_tsv_field(parent_id)),
        "submission_id": normalize_reddit_id(decode_tsv_field(submission_id)),
        "ups": ups_value,
    }


def iter_comment_records(path: Path) -> Iterator[tuple[dict[str, Any] | None, int]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        handle.readline()
        lines: list[str] = []
        current_index: int | None = None
        for line in handle:
            match = COMMENT_RECORD_START.match(line)
            candidate = int(match.group(1)) if match else None
            boundary = candidate is not None and (
                current_index is None or candidate > current_index
            )
            if boundary:
                if current_index is not None:
                    yield parse_comment_record("".join(lines), current_index), current_index
                lines = [line]
                current_index = candidate
            elif current_index is not None:
                lines.append(line)
        if current_index is not None:
            yield parse_comment_record("".join(lines), current_index), current_index


def validate_image(path: Path) -> dict[str, Any]:
    result = {
        "image_valid": False,
        "image_format": "",
        "image_width": 0,
        "image_height": 0,
        "image_error": "",
    }
    try:
        if path.stat().st_size <= 0:
            raise ValueError("zero-byte file")
        with Image.open(path) as image:
            result["image_format"] = image.format or ""
            result["image_width"], result["image_height"] = image.size
            image.verify()
        with Image.open(path) as image:
            image.load()
        result["image_valid"] = True
    except Exception as exc:
        result["image_error"] = f"{type(exc).__name__}: {exc}"
    return result


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def stable_random(seed: int, news_id: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{news_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def load_prompt_template(path: Path) -> str:
    template = path.read_text(encoding="utf-8").strip()
    fields = {
        field_name
        for _, field_name, _, _ in string.Formatter().parse(template)
        if field_name
    }
    allowed = {"news_text", "observed_comments"}
    if fields != allowed:
        raise ValueError(
            f"Prompt placeholders must be exactly {sorted(allowed)}; found {sorted(fields)}"
        )
    lowered = template.lower()
    forbidden_tokens = ["{label", "2_way_label", "3_way_label", "6_way_label", "authenticity condition"]
    if any(token in lowered for token in forbidden_tokens):
        raise ValueError("Prompt template contains a forbidden label field or condition")
    return template


def render_user_prompt(template: str, model_input: dict[str, Any]) -> str:
    if set(model_input) != {"image_path", "news_text", "observed_comments"}:
        raise ValueError(
            "model_input must contain only image_path, news_text, and observed_comments"
        )
    comments = model_input["observed_comments"]
    if not isinstance(comments, list) or not comments:
        raise ValueError("observed_comments must be a non-empty list")
    numbered = "\n".join(f"{index}. {body}" for index, body in enumerate(comments, 1))
    return template.format(
        news_text=str(model_input["news_text"]).strip(), observed_comments=numbered
    )


def build_messages(
    template: str,
    model_input: dict[str, Any],
    target_comment: str | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": render_user_prompt(template, model_input)},
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
