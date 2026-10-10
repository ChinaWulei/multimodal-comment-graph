"""CHECKED news/image/comment loader for star-shaped comment graphs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch_geometric.data import Data


GRAPH_MODES = {"baseline", "real_only", "generated_only", "real_generated"}
NODE_TYPE_NEWS = 0
NODE_TYPE_REAL = 1
NODE_TYPE_GENERATED = 2


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def load_real_comment_map(path: Path) -> Dict[str, List[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Real comment file not found: {path}")
    result: Dict[str, List[str]] = {}
    for record in read_jsonl(path):
        comments = []
        seen = set()
        for item in record.get("comments", []):
            text = str(item.get("body", "")).strip()
            key = text.casefold()
            if text and key not in seen:
                comments.append(text)
                seen.add(key)
        result[str(record["news_id"])] = comments
    return result


def load_generated_comment_map(path: Optional[Path]) -> Dict[str, List[str]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Generated comment file not found: {path}")
    result: Dict[str, List[str]] = {}
    for record in read_jsonl(path):
        comments = []
        seen = set()
        for value in record.get("generated_comments", []):
            text = str(value).strip()
            key = text.casefold()
            if text and key not in seen:
                comments.append(text)
                seen.add(key)
        result[str(record["news_id"])] = comments
    return result


class CheckedCommentGraphDataset(Dataset):
    """One CHECKED microblog per sample with an undirected star comment graph."""

    def __init__(
        self,
        manifest_csv: Path,
        real_comments_jsonl: Path,
        generated_comments_jsonl: Optional[Path],
        split: str,
        mode: str,
        tokenizer,
        image_transform,
        max_real_comments: int = 20,
        max_generated_comments: int = 5,
        max_news_length: int = 128,
        max_comment_length: int = 96,
        strict_generated_coverage: bool = True,
        max_samples: Optional[int] = None,
        allowed_news_ids: Optional[set[str]] = None,
    ):
        if mode not in GRAPH_MODES:
            raise ValueError(f"Unknown graph mode {mode!r}; expected one of {sorted(GRAPH_MODES)}")
        if not manifest_csv.is_file():
            raise FileNotFoundError(f"CHECKED manifest not found: {manifest_csv}")

        self.mode = mode
        self.num_classes = 2
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_real_comments = int(max_real_comments)
        self.max_generated_comments = int(max_generated_comments)
        self.max_news_length = int(max_news_length)
        self.max_comment_length = int(max_comment_length)
        self.real_comments = load_real_comment_map(real_comments_jsonl)

        needs_generated = mode in {"generated_only", "real_generated"}
        if needs_generated and generated_comments_jsonl is None:
            raise FileNotFoundError(f"Mode {mode} requires a generated-comment JSONL for split={split}")
        self.generated_comments = (
            load_generated_comment_map(generated_comments_jsonl) if needs_generated else {}
        )

        with manifest_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("split") == split]

        records = []
        missing_generated = []
        for row in rows:
            news_id = str(row["news_id"])
            if allowed_news_ids is not None and news_id not in allowed_news_ids:
                continue
            image_path = Path(row["primary_image_path"])
            if not image_path.is_file() or not str(row.get("news_text", "")).strip():
                continue
            real = self.real_comments.get(news_id, [])
            generated = self.generated_comments.get(news_id, [])
            if mode in {"real_only", "real_generated"} and not real:
                continue
            if needs_generated and not generated:
                missing_generated.append(news_id)
                if strict_generated_coverage:
                    continue
            records.append(row)
            if max_samples is not None and len(records) >= int(max_samples):
                break

        if needs_generated and strict_generated_coverage and not records:
            raise RuntimeError(
                f"No {split} records have generated comments in {generated_comments_jsonl}. "
                "Generate that split before formal training."
            )
        if not records:
            raise RuntimeError(f"No usable CHECKED records for split={split}, mode={mode}")
        self.records = records
        self.missing_generated_ids = missing_generated

    def __len__(self) -> int:
        return len(self.records)

    def labels(self) -> List[int]:
        return [int(row["binary_label"]) for row in self.records]

    def _select_comments(self, news_id: str) -> tuple[List[str], List[str]]:
        real = self.real_comments.get(news_id, [])[: self.max_real_comments]
        generated = self.generated_comments.get(news_id, [])[: self.max_generated_comments]
        if self.mode == "baseline":
            return [], []
        if self.mode == "real_only":
            return real, []
        if self.mode == "generated_only":
            return [], generated
        return real, generated

    def __getitem__(self, index: int) -> Data:
        row = self.records[index]
        news_id = str(row["news_id"])
        news_text = str(row["news_text"]).strip()
        real, generated = self._select_comments(news_id)
        node_texts = [news_text, *real, *generated]
        node_types = [NODE_TYPE_NEWS] + [NODE_TYPE_REAL] * len(real) + [NODE_TYPE_GENERATED] * len(generated)

        node_tokens = self.tokenizer(
            node_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_comment_length,
            return_tensors="pt",
        )
        root_tokens = self.tokenizer(
            news_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_news_length,
            return_tensors="pt",
        )

        edges = []
        for comment_index in range(1, len(node_texts)):
            edges.extend(((0, comment_index), (comment_index, 0)))
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )

        with Image.open(row["primary_image_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixel_values = self.image_transform(image).unsqueeze(0)

        data = Data(
            node_input_ids=node_tokens["input_ids"],
            node_attention_mask=node_tokens["attention_mask"],
            node_type=torch.tensor(node_types, dtype=torch.long),
            root_input_ids=root_tokens["input_ids"],
            root_attention_mask=root_tokens["attention_mask"],
            pixel_values=pixel_values,
            edge_index=edge_index,
            y=torch.tensor([int(row["binary_label"])], dtype=torch.long),
            num_real_comments=torch.tensor([len(real)], dtype=torch.long),
            num_generated_comments=torch.tensor([len(generated)], dtype=torch.long),
            sample_idx=torch.tensor([index], dtype=torch.long),
            news_id=news_id,
        )
        data.num_nodes = len(node_texts)
        return data
