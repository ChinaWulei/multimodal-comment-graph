"""Fakeddit adapter that preserves observed reply and generated parent edges.

This is separate from the CHECKED star-graph loader.  It accepts only the
official-image manifest produced by the partial-observation pipeline and never
uses a Fakeddit label to construct a node, edge, or generated-comment prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch_geometric.data import Data

from comment_graph.dataset import GRAPH_MODES, NODE_TYPE_GENERATED, NODE_TYPE_NEWS, NODE_TYPE_REAL


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_real_nodes(
    tree: dict[str, Any], limit: int
) -> tuple[list[dict[str, Any]], list[tuple[int, int, int]]]:
    """Parent-first traversal, with roots already ranked deterministically upstream."""
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    stack = list(reversed(tree.get("roots", [])))
    while stack and len(result) < limit:
        node = stack.pop()
        comment_id = str(node.get("comment_id", ""))
        if not comment_id or comment_id in seen:
            continue
        seen.add(comment_id)
        result.append(node)
        stack.extend(reversed(node.get("children", [])))
    local = {str(node["comment_id"]): index + 1 for index, node in enumerate(result)}
    edges = []
    for index, node in enumerate(result, start=1):
        parent = local.get(str(node.get("parent_id", "")), 0)
        # Relation 0 is an observed Fakeddit parent-child edge.
        edges.append((parent, index, 0))
    return result, edges


def root_generated_edges(edges: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Placement ablation: preserve real edges; reparent only synthetic nodes."""
    return [(0 if relation == 1 else source, target, relation)
            for source, target, relation in edges]


def select_real_nodes(tree, limit, required_ids=None):
    """Keep observed context plus its ancestors, then fill in original order.

    Required IDs come only from the generator's INPUT context, never its
    predictions or a news label. The same policy must be used in all arms.
    """
    if required_ids is None:
        return flatten_real_nodes(tree, limit)
    all_nodes, _ = flatten_real_nodes(tree, 2 ** 63 - 1)
    by_id = {str(node["comment_id"]): node for node in all_nodes}
    selected = set()
    for required in required_ids:
        current = str(required)
        if current not in by_id:
            raise ValueError(f"Observed context comment missing from real tree: {current}")
        chain = set()
        while current in by_id:
            if current in chain:
                raise ValueError(f"Cycle in real parent IDs: {current}")
            chain.add(current)
            selected.add(current)
            current = str(by_id[current].get("parent_id", ""))
    if len(selected) > limit:
        raise ValueError(f"Observed context and ancestors require {len(selected)} nodes, limit={limit}")
    for node in all_nodes:
        if len(selected) >= limit:
            break
        comment_id = str(node["comment_id"])
        parent = str(node.get("parent_id", ""))
        if parent not in by_id or parent in selected:
            selected.add(comment_id)
    nodes = [node for node in all_nodes if str(node["comment_id"]) in selected]
    local = {str(node["comment_id"]): index + 1 for index, node in enumerate(nodes)}
    edges = [(local.get(str(node.get("parent_id", "")), 0), index, 0)
             for index, node in enumerate(nodes, start=1)]
    if any(source >= target for source, target, _ in edges):
        raise ValueError("Real tree traversal does not precede children with parents")
    return nodes, edges


def bidirectional(
    edges: list[tuple[int, int, int]], valid_nodes: set[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs, types = [], []
    seen = set()
    for source, target, relation in edges:
        if source not in valid_nodes or target not in valid_nodes or source == target:
            continue
        for pair in ((source, target), (target, source)):
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                types.append(int(relation))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous(), torch.tensor(types, dtype=torch.long)


class FakedditStructureGraphDataset(Dataset):
    """Three-way Fakeddit graph dataset with root-only/tree topology ablations."""

    def __init__(
        self,
        manifest_csv: Path,
        real_trees_jsonl: Path,
        generated_graphs_jsonl: Path | None,
        split: str,
        mode: str,
        tokenizer,
        image_transform,
        config,
        max_samples: int | None = None,
        allowed_news_ids: set[str] | None = None,
    ):
        if mode not in GRAPH_MODES:
            raise ValueError(f"Unknown graph mode {mode}")
        self.mode, self.tokenizer, self.image_transform = mode, tokenizer, image_transform
        self.num_classes = int(config.classification.num_classes)
        self.label_column = str(config.data.label_column)
        self.max_real_comments = int(config.data.max_real_comments)
        self.max_generated_comments = int(config.data.max_generated_comments)
        self.max_news_length = int(config.data.max_news_length)
        self.max_comment_length = int(config.data.max_comment_length)
        self.graph_structure = str(config.graph.structure_mode)
        self.strict_generated_coverage = bool(config.data.strict_generated_coverage)
        self.real_node_selection = str(config.data.get("real_node_selection", "prefix"))
        if self.real_node_selection not in {"prefix", "observed_context"}:
            raise ValueError(f"Unknown real node selection: {self.real_node_selection}")
        manifest = pd.read_csv(manifest_csv, dtype=str, keep_default_na=False)
        required = {"news_id", "split", "news_text", "image_path", "image_valid", self.label_column}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Fakeddit manifest misses columns: {sorted(missing)}")
        if manifest["news_id"].duplicated().any():
            raise ValueError("Fakeddit manifest has duplicate news IDs")
        tree_records = read_jsonl(real_trees_jsonl)
        trees = {str(row["news_id"]): row for row in tree_records}
        if len(trees) != len(tree_records):
            raise ValueError("Fakeddit tree JSONL has duplicate news IDs")
        needs_generated = mode in {"generated_only", "real_generated"} or self.real_node_selection == "observed_context"
        generated: dict[str, dict[str, Any]] = {}
        if needs_generated:
            if generated_graphs_jsonl is None or not generated_graphs_jsonl.is_file():
                raise FileNotFoundError("Generated-graph JSONL is required for a generated-comment mode")
            generated_rows = read_jsonl(generated_graphs_jsonl)
            generated = {str(row["news_id"]): row for row in generated_rows}
            if len(generated) != len(generated_rows):
                raise ValueError("Generated graph JSONL has duplicate news IDs")
        self.generated_graphs = generated
        self.tree_by_id = trees
        records, missing_generated = [], []
        for row in manifest.to_dict("records"):
            news_id = str(row["news_id"])
            if row["split"] != split or str(row["image_valid"]).lower() != "true":
                continue
            if allowed_news_ids is not None and news_id not in allowed_news_ids:
                continue
            if not str(row["news_text"]).strip() or not Path(row["image_path"]).is_file():
                continue
            label = str(row[self.label_column]).strip()
            if not label or not label.isdigit() or not 0 <= int(label) < self.num_classes:
                continue
            if mode in {"real_only", "real_generated"} and news_id not in trees:
                continue
            if needs_generated and news_id not in generated:
                missing_generated.append(news_id)
                if self.strict_generated_coverage:
                    continue
            records.append(row)
            if max_samples is not None and len(records) >= int(max_samples):
                break
        if not records:
            raise RuntimeError(f"No usable Fakeddit records for {split}/{mode}")
        self.records, self.missing_generated_ids = records, missing_generated

    def __len__(self) -> int:
        return len(self.records)

    def labels(self) -> list[int]:
        return [int(row[self.label_column]) for row in self.records]

    def _real_nodes_and_edges(self, news_id):
        required_ids = None
        if self.real_node_selection == "observed_context":
            required_ids = [str(node["source_comment_id"])
                            for node in self.generated_graphs[news_id]["graph_nodes"]
                            if node["node_type"] == "real_comment"]
        return select_real_nodes(self.tree_by_id[news_id], self.max_real_comments, required_ids)

    def _from_generated_graph(
        self, news_id: str
    ) -> tuple[list[str], list[int], list[tuple[int, int, int]]]:
        graph = self.generated_graphs[news_id]
        raw_nodes = sorted(graph.get("graph_nodes", []), key=lambda node: int(node.get("node_index", -1)))
        if not raw_nodes or int(raw_nodes[0].get("node_index", -1)) != 0:
            raise ValueError(f"Generated graph {news_id} has no news root at index 0")
        kept = [node for node in raw_nodes if node.get("node_type") != "news"]
        if self.mode == "generated_only":
            kept = [node for node in kept if node.get("node_type") == "generated_comment"][: self.max_generated_comments]
        elif self.mode == "real_generated":
            real = [node for node in kept if node.get("node_type") == "real_comment"][: self.max_real_comments]
            generated = [node for node in kept if node.get("node_type") == "generated_comment"][: self.max_generated_comments]
            kept = [*real, *generated]
        old_to_new = {0: 0}
        for index, node in enumerate(kept, start=1):
            old_to_new[int(node["node_index"])] = index
        texts = [str(node.get("content", "")).strip() for node in kept]
        types = [NODE_TYPE_REAL if node.get("node_type") == "real_comment" else NODE_TYPE_GENERATED for node in kept]
        if any(not text for text in texts):
            raise ValueError(f"Generated graph {news_id} has blank node content")
        edges: list[tuple[int, int, int]] = []
        for edge in graph.get("edges", []):
            source, target = int(edge["source"]), int(edge["target"])
            if source in old_to_new and target in old_to_new:
                target_type = types[old_to_new[target] - 1]
                relation = int(edge.get("edge_type", 1 if target_type == NODE_TYPE_GENERATED else 0))
                edges.append((old_to_new[source], old_to_new[target], relation))
        # A selected generated node whose original parent was excluded is safely
        # attached to root; the count is retained for later audit via Data.
        child_nodes = {target for _, target, _ in edges}
        for index in range(1, len(texts) + 1):
            if index not in child_nodes:
                edges.append((0, index, 1 if types[index - 1] == NODE_TYPE_GENERATED else 0))
        return texts, types, edges

    def _nodes_and_edges(
        self, news_id: str
    ) -> tuple[list[str], list[int], list[tuple[int, int, int]]]:
        if self.mode == "baseline":
            return [], [], []
        if self.mode == "generated_only":
            return self._from_generated_graph(news_id)
        nodes, edges = self._real_nodes_and_edges(news_id)
        real_texts = [str(node["body"]).strip() for node in nodes]
        if self.mode == "real_only":
            return real_texts, [NODE_TYPE_REAL] * len(nodes), edges
        # Formal generator outputs retain a parent_comment_id alongside the
        # model-facing parent_node_index.  Grafting by immutable raw ID keeps
        # the real-node set identical between root-only and tree experiments.
        generated = self.generated_graphs[news_id].get("generated_nodes", [])[: self.max_generated_comments]
        if not generated:
            # A fixed-depth/finite-retry generator can legitimately return no
            # valid node. Preserve the *same selected real tree* in paired
            # arms; falling back to graph_nodes would silently truncate it.
            return real_texts, [NODE_TYPE_REAL] * len(real_texts), edges
        if generated and all("parent_comment_id" in node for node in generated):
            local = {str(node["comment_id"]): index + 1 for index, node in enumerate(nodes)}
            generated_texts = [str(node.get("content", "")).strip() for node in generated]
            if any(not text for text in generated_texts):
                raise ValueError(f"Generated graph {news_id} has blank generated content")
            generated_edges: list[tuple[int, int, int]] = []
            for index, node in enumerate(generated, start=1):
                new_index = len(real_texts) + index
                # Relation 1 is a generated parent edge predicted by the
                # structure-aware generator (including generated-to-generated).
                parent_id = str(node["parent_comment_id"])
                if self.real_node_selection == "observed_context" and parent_id != "news" and parent_id not in local:
                    raise ValueError(f"Predicted parent missing after context-preserving selection: {news_id}/{parent_id}")
                generated_edges.append((local.get(parent_id, 0), new_index, 1))
                local[str(node.get("generated_node_id", f"generated:{index}"))] = new_index
            return (
                [*real_texts, *generated_texts],
                [NODE_TYPE_REAL] * len(real_texts) + [NODE_TYPE_GENERATED] * len(generated_texts),
                [*edges, *generated_edges],
            )
        if self.real_node_selection == "observed_context":
            raise ValueError(f"Generated graph {news_id} lacks immutable parent IDs")
        return self._from_generated_graph(news_id)

    def __getitem__(self, index: int) -> Data:
        row = self.records[index]
        news_id, news_text = str(row["news_id"]), str(row["news_text"]).strip()
        comment_texts, comment_types, observed_edges = self._nodes_and_edges(news_id)
        node_texts = [news_text, *comment_texts]
        node_types = [NODE_TYPE_NEWS, *comment_types]
        tokenized = self.tokenizer(node_texts, padding="max_length", truncation=True,
                                   max_length=self.max_comment_length, return_tensors="pt")
        root_tokens = self.tokenizer(news_text, padding="max_length", truncation=True,
                                     max_length=self.max_news_length, return_tensors="pt")
        if self.graph_structure == "star":
            observed_edges = [
                (0, node, 1 if comment_types[node - 1] == NODE_TYPE_GENERATED else 0)
                for node in range(1, len(node_texts))
            ]
        elif self.graph_structure == "generated_root":
            observed_edges = root_generated_edges(observed_edges)
        edge_index, edge_type = bidirectional(observed_edges, set(range(len(node_texts))))
        with Image.open(row["image_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            pixel_values = self.image_transform(image).unsqueeze(0)
        data = Data(
            node_input_ids=tokenized["input_ids"], node_attention_mask=tokenized["attention_mask"],
            root_input_ids=root_tokens["input_ids"], root_attention_mask=root_tokens["attention_mask"],
            node_type=torch.tensor(node_types, dtype=torch.long), edge_index=edge_index, edge_type=edge_type,
            pixel_values=pixel_values, y=torch.tensor([int(row[self.label_column])], dtype=torch.long),
            num_real_comments=torch.tensor([sum(item == NODE_TYPE_REAL for item in comment_types)], dtype=torch.long),
            num_generated_comments=torch.tensor([sum(item == NODE_TYPE_GENERATED for item in comment_types)], dtype=torch.long),
            sample_idx=torch.tensor([index], dtype=torch.long), news_id=news_id,
        )
        data.num_nodes = len(node_texts)
        return data
