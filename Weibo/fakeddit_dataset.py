"""Fakeddit raw-data loader for online multimodal encoding.

Tokenisation and image decoding happen in the dataset, while the actual text and
image feature extraction happens in ``OnlineFeatureEncoder`` during ``forward``.
The returned object is a PyG ``Data`` instance so variable-sized comment graphs
can still be batched by ``torch_geometric.loader.DataLoader``.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import torch
from PIL import Image
from torch_geometric.data import Data
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _normalise_reddit_id(value) -> str:
    value = str(value)
    if value.startswith(("t1_", "t3_")):
        return value[3:]
    return value


def collect_split_ids(paths: Iterable[str], id_column: str = "id") -> set:
    """Read only sample IDs, used to filter the much larger comments file."""
    ids = set()
    for path in paths:
        frame = pd.read_csv(path, sep="\t", usecols=[id_column], dtype={id_column: str})
        ids.update(_normalise_reddit_id(value) for value in frame[id_column].dropna())
    return ids


def load_comment_map(
    comments_tsv: Optional[str],
    allowed_submission_ids: Optional[set] = None,
    max_comments: int = 32,
) -> Dict[str, List[dict]]:
    """Load Fakeddit comments and group them by submission.

    The released comments files have changed slightly between mirrors. Column
    names are therefore detected from a small set of known alternatives. If a
    parent ID is present it is retained, otherwise the dataset builds a star
    graph from the source post to all comments.
    """
    if not comments_tsv:
        return {}

    header = pd.read_csv(comments_tsv, sep="\t", nrows=0).columns.tolist()

    def choose(candidates, required=True):
        column = next((name for name in candidates if name in header), None)
        if column is None and required:
            raise ValueError(
                f"None of {candidates} exists in {comments_tsv}. "
                f"Available columns: {header}"
            )
        return column

    submission_col = choose(("submission_id", "link_id", "post_id"))
    text_col = choose(("body", "comment", "comment_text", "text"))
    comment_id_col = choose(("comment_id", "id"), required=False)
    parent_col = choose(("parent_id", "parent_comment_id"), required=False)
    usecols = [submission_col, text_col]
    if comment_id_col:
        usecols.append(comment_id_col)
    if parent_col:
        usecols.append(parent_col)

    grouped = defaultdict(list)
    reader = pd.read_csv(
        comments_tsv,
        sep="\t",
        usecols=list(dict.fromkeys(usecols)),
        chunksize=200_000,
    )
    for frame in reader:
        frame = frame.dropna(subset=[submission_col, text_col])
        frame[submission_col] = frame[submission_col].map(_normalise_reddit_id)
        if allowed_submission_ids is not None:
            frame = frame[frame[submission_col].isin(allowed_submission_ids)]

        for row in frame.itertuples(index=False, name=None):
            values = dict(zip(frame.columns, row))
            submission_id = values[submission_col]
            if len(grouped[submission_id]) >= max_comments:
                continue
            text = str(values[text_col]).strip()
            if not text or text in {"[deleted]", "[removed]"}:
                continue
            comment_id = values[comment_id_col] if comment_id_col else None
            parent_id = values[parent_col] if parent_col else None
            grouped[submission_id].append(
                {
                    "id": None if pd.isna(comment_id) else _normalise_reddit_id(comment_id),
                    "parent_id": None if pd.isna(parent_id) else _normalise_reddit_id(parent_id),
                    "text": text,
                }
            )
    return dict(grouped)


def build_image_index(images_dir: str) -> Dict[str, str]:
    """Optionally index recursively stored Fakeddit images by file stem."""
    index = {}
    for path in Path(images_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(_normalise_reddit_id(path.stem), str(path))
    return index


class FakedditDataset(Dataset):
    """Create one source-post/comment propagation graph per Fakeddit sample."""

    def __init__(
        self,
        tsv_path: str,
        images_dir: str,
        tokenizer,
        image_transform,
        comments_by_submission: Optional[Dict[str, List[dict]]] = None,
        image_index: Optional[Dict[str, str]] = None,
        id_column: str = "id",
        text_column: str = "clean_title",
        label_column: str = "2_way_label",
        image_path_column: Optional[str] = None,
        max_text_length: int = 64,
        max_comment_length: int = 64,
        max_comments: int = 32,
        min_comments: int = 0,
        undirected_graph: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.tsv_path = Path(tsv_path)
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.comments = comments_by_submission or {}
        self.image_index = image_index or {}
        self.id_column = id_column
        self.text_column = text_column
        self.label_column = label_column
        self.image_path_column = image_path_column
        self.max_text_length = max_text_length
        self.max_comment_length = max_comment_length
        self.max_comments = max_comments
        self.undirected_graph = undirected_graph

        frame = pd.read_csv(self.tsv_path, sep="\t", dtype={id_column: str})
        required = {id_column, text_column, label_column}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {tsv_path}")
        frame = frame.dropna(subset=list(required)).copy()
        frame[id_column] = frame[id_column].map(_normalise_reddit_id)
        if min_comments > 0:
            frame = frame[frame[id_column].map(lambda value: len(self.comments.get(value, [])) >= min_comments)]

        records = []
        for record in frame.to_dict("records"):
            image_path = self._resolve_image(record)
            if image_path is None:
                continue
            record["_resolved_image_path"] = image_path
            records.append(record)
            if max_samples and len(records) >= max_samples:
                break
        self.records = records

        if not self.records:
            raise RuntimeError(
                f"No usable multimodal samples found in {tsv_path}. Check images_dir, "
                "ID column names and image_path_column."
            )

    def _resolve_image(self, record) -> Optional[str]:
        if self.image_path_column and self.image_path_column in record:
            value = record[self.image_path_column]
            if pd.notna(value):
                path = Path(str(value))
                if not path.is_absolute():
                    path = self.images_dir / path
                if path.is_file():
                    return str(path)

        sample_id = _normalise_reddit_id(record[self.id_column])
        if sample_id in self.image_index:
            return self.image_index[sample_id]
        for stem in (sample_id, f"t3_{sample_id}"):
            for extension in IMAGE_EXTENSIONS:
                path = self.images_dir / f"{stem}{extension}"
                if path.is_file():
                    return str(path)
        return None

    @staticmethod
    def _parse_label(value) -> int:
        if isinstance(value, str):
            normalised = value.strip().lower()
            # Fakeddit's published binary convention is 0=fake, 1=true.
            mapping = {"fake": 0, "false": 0, "misinformation": 0, "real": 1, "true": 1, "factual": 1}
            if normalised in mapping:
                return mapping[normalised]
        return int(value)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        sample_id = record[self.id_column]
        root_text = str(record[self.text_column])
        comments = self.comments.get(sample_id, [])[: self.max_comments]

        node_texts = [root_text] + [comment["text"] for comment in comments]
        node_tokens = self.tokenizer(
            node_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_comment_length,
            return_tensors="pt",
        )
        root_tokens = self.tokenizer(
            root_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )

        comment_indices = {
            comment["id"]: position
            for position, comment in enumerate(comments, start=1)
            if comment["id"] is not None
        }
        edges = []
        for child_index, comment in enumerate(comments, start=1):
            parent_index = comment_indices.get(comment["parent_id"], 0)
            edges.append((parent_index, child_index))
            if self.undirected_graph:
                edges.append((child_index, parent_index))
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty((2, 0), dtype=torch.long)
        )

        with Image.open(record["_resolved_image_path"]) as image:
            pixel_values = self.image_transform(image.convert("RGB")).unsqueeze(0)

        data = Data(
            node_input_ids=node_tokens["input_ids"],
            node_attention_mask=node_tokens["attention_mask"],
            root_input_ids=root_tokens["input_ids"],
            root_attention_mask=root_tokens["attention_mask"],
            pixel_values=pixel_values,
            edge_index=edge_index,
            y=torch.tensor([self._parse_label(record[self.label_column])], dtype=torch.long),
            sample_idx=torch.tensor([index], dtype=torch.long),
        )
        # PyG needs this to offset edge indices and to create ``batch`` even
        # though raw samples do not yet have an ``x`` tensor.
        data.num_nodes = len(node_texts)
        return data
