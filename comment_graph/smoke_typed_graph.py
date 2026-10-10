"""CPU smoke test for real-edge/generated-edge R-GCN message passing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from comment_graph.typed_graph import TypedCommentEncoder


def main() -> None:
    torch.manual_seed(7)
    # graph 0: news -> real -> generated; graph 1: news -> real.
    features = torch.randn(5, 8, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3]])
    # 0 = genuine Fakeddit reply edge, 1 = generator-predicted edge.
    edge_type = torch.tensor([0, 0, 1, 1, 0, 0])
    batch = torch.tensor([0, 0, 0, 1, 1])
    model = TypedCommentEncoder(input_dim=8, hidden_dim=4, num_relations=2)
    embedding, nodes = model(features, edge_index, edge_type, batch)
    assert embedding.shape == (2, 12)
    assert nodes.shape == (5, 12)
    embedding.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    print(json.dumps({
        "graph_embedding_shape": list(embedding.shape),
        "node_embedding_shape": list(nodes.shape),
        "relations": {"0": "observed_real_edge", "1": "generated_parent_edge"},
        "gradient_check": "passed",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
