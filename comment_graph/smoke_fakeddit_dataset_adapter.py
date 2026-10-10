"""End-to-end local smoke test for Fakeddit tree/generation dataset assembly."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.common import load_config, make_tokenizer_and_transform
from comment_graph.fakeddit_structure_dataset import FakedditStructureGraphDataset
from comment_graph.train import evaluate


class ThreeClassOracle:
    """Minimal model used only to prove the generic three-class metric path."""

    num_classes = 3
    decision_mode = "concat"

    def eval(self):
        return self

    def __call__(self, batch, return_generated_context=False):
        logits = torch.full((batch.y.numel(), 3), -8.0, device=batch.y.device)
        logits[torch.arange(batch.y.numel(), device=batch.y.device), batch.y.view(-1)] = 8.0
        return {"logits": logits, "global_logits": logits, "local_correction": {}}


def main() -> None:
    image_path = next((PROJECT_ROOT / "images").glob("*.*"), None)
    if image_path is None:
        raise RuntimeError("A local pilot image is required for the dataset adapter smoke test")
    config, _ = load_config(PROJECT_ROOT / "comment_graph" / "config_fakeddit_structure.yaml")
    tokenizer, transform = make_tokenizer_and_transform(config)
    with tempfile.TemporaryDirectory(prefix="fakeddit_structure_smoke_") as directory:
        root = Path(directory)
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["news_id", "split", "news_text", "image_path", "image_valid", "3_way_label"])
            writer.writeheader()
            writer.writerow({"news_id": "n1", "split": "train", "news_text": "A test news post", "image_path": str(image_path), "image_valid": "true", "3_way_label": "1"})
        tree = {"news_id": "n1", "roots": [
            {"comment_id": "r1", "parent_id": "n1", "body": "root one", "children": [
                {"comment_id": "c1", "parent_id": "r1", "body": "child one", "children": []}]},
            {"comment_id": "r2", "parent_id": "n1", "body": "root two", "children": []},
        ]}
        trees = root / "trees.jsonl"
        trees.write_text(json.dumps(tree) + "\n", encoding="utf-8")
        generated = root / "generated.jsonl"
        generated.write_text(json.dumps({
            "news_id": "n1",
            "generated_nodes": [
                {"generated_node_id": "g1", "parent_comment_id": "r1", "content": "generated reply one"},
                {"generated_node_id": "g2", "parent_comment_id": "g1", "content": "generated reply two"},
            ],
        }) + "\n", encoding="utf-8")
        config.data.max_real_comments = 3
        config.data.max_generated_comments = 2
        config.graph.structure_mode = "observed_tree"
        observed = FakedditStructureGraphDataset(manifest, trees, generated, "train", "real_generated", tokenizer, transform, config)[0]
        observed_edges = {tuple(pair) for pair in observed.edge_index.t().tolist()}
        expected_tree_edges = {(0, 1), (1, 0), (1, 2), (2, 1), (0, 3), (3, 0), (1, 4), (4, 1), (4, 5), (5, 4)}
        assert observed_edges == expected_tree_edges, observed_edges
        assert observed.node_type.tolist() == [0, 1, 1, 1, 2, 2]
        relation_by_edge = {
            tuple(edge): relation
            for edge, relation in zip(observed.edge_index.t().tolist(), observed.edge_type.tolist())
        }
        assert relation_by_edge[(1, 4)] == relation_by_edge[(4, 1)] == 1
        assert relation_by_edge[(0, 1)] == relation_by_edge[(1, 0)] == 0
        config.graph.structure_mode = "star"
        star = FakedditStructureGraphDataset(manifest, trees, generated, "train", "real_generated", tokenizer, transform, config)[0]
        star_edges = {tuple(pair) for pair in star.edge_index.t().tolist()}
        expected_star_edges = {(0, i) for i in range(1, 6)} | {(i, 0) for i in range(1, 6)}
        assert star_edges == expected_star_edges, star_edges
        assert star.node_type.tolist() == observed.node_type.tolist()
        # When all generation attempts fail, the paired real-only and
        # real+generated arms must still select exactly the same real nodes.
        generated.write_text(json.dumps({
            "news_id": "n1", "generated_nodes": [],
            "graph_nodes": [
                {"node_index": 0, "node_type": "news", "content": "A test news post"},
                {"node_index": 1, "node_type": "real_comment", "content": "root one",
                 "source_comment_id": "r1"},
            ],
            "edges": [{"source": 0, "target": 1, "edge_type": 0}],
        }) + "\n", encoding="utf-8")
        config.data.real_node_selection = "observed_context"
        config.graph.structure_mode = "observed_tree"
        no_generated_real = FakedditStructureGraphDataset(
            manifest, trees, generated, "train", "real_only", tokenizer, transform, config)[0]
        no_generated_augmented = FakedditStructureGraphDataset(
            manifest, trees, generated, "train", "real_generated", tokenizer, transform, config)[0]
        assert no_generated_real.node_type.tolist() == no_generated_augmented.node_type.tolist()
        assert no_generated_real.node_type.tolist() == [0, 1, 1, 1]
        assert torch.equal(no_generated_real.edge_index, no_generated_augmented.edge_index)
        assert torch.equal(no_generated_real.edge_type, no_generated_augmented.edge_type)
        assert int(no_generated_augmented.num_generated_comments) == 0
        metric_loader = DataLoader(
            [Data(y=torch.tensor([label]), news_id=f"metric_{label}") for label in range(3)],
            batch_size=3,
        )
        metric = evaluate(ThreeClassOracle(), metric_loader, torch.device("cpu"), False, None)
        assert metric["roc_auc"] is None and len(metric["confusion_matrix"]) == 3
        assert metric["macro_f1"] == 1.0
        print(json.dumps({
            "nodes": int(observed.num_nodes), "tree_directed_edges": int(observed.edge_index.size(1)),
            "star_directed_edges": int(star.edge_index.size(1)),
            "checks": ["same_nodes_across_ablation", "real_parent_edges", "generated_to_real_edge", "generated_chain_edge", "typed_edge_relations", "zero_generated_keeps_same_real_tree", "three_class_metrics"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
