"""Synthetic invariants and optional real Fakeddit forward/backward smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from comment_graph.common import load_config, make_dataset, make_tokenizer_and_transform, seed_everything
from comment_graph.fakeddit_structure_dataset import bidirectional
from comment_graph.generated_reliability import RELIABILITY_MODES, refine_generated_graph
from comment_graph.model import CommentGraphMFCL
from comment_graph.typed_graph import TypedCommentEncoder


def synthetic_checks() -> dict:
    features = torch.tensor([
        [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
        [0., .95, .312], [0., 1., 0.],
    ])
    node_type = torch.tensor([0, 1, 1, 2, 2])
    edge_index, edge_type = bidirectional(
        [(0, 1, 0), (0, 2, 0), (2, 3, 1), (3, 4, 1)], set(range(5)),
    )
    batch = torch.zeros(5, dtype=torch.long)
    real_edges = {tuple(pair) for pair, relation in zip(edge_index.t().tolist(), edge_type.tolist()) if relation == 0}
    results = {}
    for mode in ("quality", "placement", "both"):
        edges, types, reliability, audit = refine_generated_graph(
            features, node_type, edge_index, edge_type, batch, mode,
        )
        retained_real = {tuple(pair) for pair, relation in zip(edges.t().tolist(), types.tolist()) if relation == 0}
        assert retained_real == real_edges, "Observed real reply edges changed"
        assert len(audit) == 2
        assert all(int(node_type[i]) != 2 or 0 < reliability[i] <= 1 for i in range(5))
        assert all(reliability[i] == 1 for i in range(3))
        assert all(item["selected_parent"] < item["node_index"] for item in audit)
        if mode in {"placement", "both"}:
            assert audit[0]["original_parent"] == 2 and audit[0]["selected_parent"] == 1
            assert audit[0]["rewired"]
        else:
            assert torch.equal(edges, edge_index) and torch.equal(types, edge_type)
        if mode in {"quality", "both"}:
            assert reliability[4] < reliability[3], "Near duplicate was not downweighted"
        else:
            assert torch.all(reliability == 1)
        results[mode] = {"rewired": sum(item["rewired"] for item in audit),
                         "node_reliability": [round(float(x), 4) for x in reliability]}

    # An independent second news graph must never receive cross-news edges.
    second_features = torch.cat((features, features), dim=0)
    second_types = torch.cat((node_type, node_type), dim=0)
    second_edges = torch.cat((edge_index, edge_index + 5), dim=1)
    second_edge_types = torch.cat((edge_type, edge_type))
    second_batch = torch.tensor([0] * 5 + [1] * 5)
    edges, types, reliability, audit = refine_generated_graph(
        second_features, second_types, second_edges, second_edge_types,
        second_batch, "both",
    )
    assert len(audit) == 4
    assert all(second_batch[a] == second_batch[b] for a, b in edges.t().tolist())
    assert int((types == 0).sum()) == 8
    x = torch.nn.Parameter(torch.randn(10, 3))
    encoder = TypedCommentEncoder(input_dim=3, hidden_dim=2)
    h_graph, _ = encoder(x, edges, types, second_batch, reliability)
    h_graph.sum().backward()
    assert encoder.conv1.weight.grad is not None
    assert encoder.conv1.weight.grad.abs().sum() > 0
    assert encoder.conv2.weight.grad is not None
    # A depth bound must never *force* replacement of an otherwise good
    # supplied parent. It only limits newly chosen synthetic parents.
    deep_features = torch.cat((features, features[4:5]), dim=0)
    deep_types = torch.cat((node_type, torch.tensor([2])))
    deep_edges, deep_edge_types = bidirectional(
        [(0, 1, 0), (0, 2, 0), (2, 3, 1), (3, 4, 1), (4, 5, 1)],
        set(range(6)),
    )
    _, _, _, deep_audit = refine_generated_graph(
        deep_features, deep_types, deep_edges, deep_edge_types,
        torch.zeros(6, dtype=torch.long), "placement", max_generated_depth=2,
    )
    assert deep_audit[-1]["original_parent"] == deep_audit[-1]["selected_parent"] == 4
    assert deep_audit[-1]["original_parent_similarity"] is not None
    poor_features = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., -1.], [0., 0., 1.]])
    poor_types = torch.tensor([0, 1, 1, 2])
    poor_edges, poor_edge_types = bidirectional(
        [(0, 1, 0), (0, 2, 0), (2, 3, 1)], set(range(4)),
    )
    fallback_edges, fallback_types, fallback_quality, fallback_audit = refine_generated_graph(
        poor_features, poor_types, poor_edges, poor_edge_types,
        torch.zeros(4, dtype=torch.long), "confidence",
    )
    assert fallback_audit[0]["selected_parent"] == 0
    assert fallback_audit[0]["original_parent_percentile"] == 0
    assert fallback_quality[3] == .25
    assert [0, 3] in fallback_edges.t().tolist()
    assert int((fallback_types == 0).sum()) == 4
    anchor_features = torch.tensor([
        [1., 0.], [1., 0.], [0., 1.], [0., 1.], [0., 1.], [1., 0.],
    ])
    anchor_types = torch.tensor([0, 1, 1, 2, 2, 2])
    anchor_edges, anchor_edge_types = bidirectional(
        [(0, 1, 0), (0, 2, 0), (2, 3, 1), (3, 4, 1), (1, 5, 1)], set(range(6)),
    )
    _, _, anchor_weights, anchor_audit = refine_generated_graph(
        anchor_features, anchor_types, anchor_edges, anchor_edge_types,
        torch.zeros(6, dtype=torch.long), "ancestry_anchor",
    )
    assert anchor_audit[0]["news_anchor_factor"] == .2
    assert anchor_audit[1]["news_anchor_factor"] == .2
    assert anchor_audit[1]["anchor_source_node"] == 3
    assert anchor_audit[2]["news_anchor_factor"] == 1.0
    assert anchor_weights[3] < anchor_weights[5]
    assert anchor_weights[4] < anchor_weights[5]
    assert torch.all(anchor_weights[:3] == 1)
    return {"synthetic": results, "typed_rgcn_gradient": True,
            "batched_graphs_isolated": True, "depth_bound_does_not_force_rewire": True,
            "low_confidence_falls_back_to_news": True,
            "off_topic_chain_inherits_low_news_anchor": True}


def real_batch_check(config_path: Path, mode: str, decision_mode: str) -> dict:
    config, config_path = load_config(config_path)
    config.graph.structure_mode = "observed_tree"
    config.graph.message_passing = "typed_rgcn"
    config.graph.pooling_mode = "mean"
    config.graph.generated_reliability["mode"] = mode
    config.decision_mode = decision_mode
    seed_everything(42)
    tokenizer, transform = make_tokenizer_and_transform(config)
    dataset = make_dataset(config, config_path, "train", "real_generated", tokenizer, transform,
                           max_samples=2)
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CommentGraphMFCL(config, "real_generated").to(device)
    batch = batch.to(device)
    model.train()
    output = model(batch)
    assert output["logits"].shape == (2, int(config.classification.num_classes))
    assert len(output["generated_reliability_diagnostics"]) == int(batch.num_generated_comments.sum())
    loss = F.cross_entropy(output["logits"], batch.y.view(-1))
    loss.backward()
    conv = model.typed_graph_encoder.conv1
    assert conv.weight.grad is not None and conv.weight.grad.abs().sum() > 0
    quality_gradient = model.quality_gate.quality_head[-1].weight.grad
    if mode in {"quality", "both", "quality_confidence", "ancestry_anchor"}:
        assert quality_gradient is not None
    residual_gradient = None
    if decision_mode == "reliability_residual":
        residual_gradient = model.reliability_residual.correction.weight.grad
        assert residual_gradient is not None and residual_gradient.abs().sum() > 0
    return {"real_news_ids": list(batch.news_id), "generated_nodes": int(batch.num_generated_comments.sum()),
            "logits_shape": list(output["logits"].shape), "loss": float(loss.detach()),
            "gnn_gradient_norm": float(conv.weight.grad.norm()),
            "quality_head_gradient_norm": float(quality_gradient.norm()) if quality_gradient is not None else None,
            "residual_head_gradient_norm": float(residual_gradient.norm()) if residual_gradient is not None else None,
            "rewired": sum(item["rewired"] for item in output["generated_reliability_diagnostics"]),
            "device": str(device)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-batch", action="store_true")
    parser.add_argument("--reliability-mode", choices=sorted(RELIABILITY_MODES), default="both")
    parser.add_argument("--decision-mode", choices=["concat", "reliability_residual"], default="concat")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_fakeddit_structure_expanded_generateddepth.yaml"))
    args = parser.parse_args()
    result = synthetic_checks()
    if args.real_batch:
        result["real_batch"] = real_batch_check(args.config, args.reliability_mode, args.decision_mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
