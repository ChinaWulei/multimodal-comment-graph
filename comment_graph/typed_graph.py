"""Relation-aware encoder for observed and generated comment edges.

Relation 0 denotes a real Fakeddit parent-child edge; relation 1 denotes a
generator-predicted attachment.  Keeping those channels separate prevents a
synthetic edge from being treated as equally reliable by message passing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv, global_mean_pool

from comment_graph.dynamic_graph import quality_weighted_pool


class TypedCommentEncoder(nn.Module):
    """Two-layer R-GCN with the same 512+256 pooled width as MFCL's GCN."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_relations: int = 2):
        super().__init__()
        if int(num_relations) < 2:
            raise ValueError("TypedCommentEncoder requires real and generated edge relations")
        self.num_relations = int(num_relations)
        self.conv1 = RGCNConv(input_dim, 2 * hidden_dim, self.num_relations)
        self.conv2 = RGCNConv(2 * hidden_dim, hidden_dim, self.num_relations)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch: torch.Tensor,
        node_reliability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_type.numel() != edge_index.size(1):
            raise ValueError("edge_type must have exactly one entry per directed edge")
        if edge_type.numel() and (edge_type.min() < 0 or edge_type.max() >= self.num_relations):
            raise ValueError("edge_type contains an unknown relation")
        if node_reliability is None:
            h1 = self.activation(self.conv1(x, edge_index, edge_type))
            h2 = self.activation(self.conv2(h1, edge_index, edge_type))
            return torch.cat((global_mean_pool(h1, batch), global_mean_pool(h2, batch)), dim=-1), torch.cat((h1, h2), dim=-1)
        if node_reliability.shape != (x.size(0),):
            raise ValueError("node_reliability must contain one weight per node")
        weights = node_reliability.unsqueeze(-1)
        h1 = self.activation(self.conv1(x * weights, edge_index, edge_type))
        h2 = self.activation(self.conv2(h1 * weights, edge_index, edge_type))
        graph_nodes = torch.cat((h1, h2), dim=-1)
        h_graph = quality_weighted_pool(graph_nodes, node_reliability, batch, int(batch.max()) + 1)
        return h_graph, graph_nodes
