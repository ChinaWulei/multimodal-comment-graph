"""Cross-modal comment attention and feature-driven dynamic graph propagation."""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax


# ``observed_tree`` consumes the explicit parent-child edges supplied by the
# Fakeddit structure adapter.  In contrast, ``dynamic_cross_modal`` rebuilds
# topology from features and must not be used to claim parent-edge ablations.
GRAPH_STRUCTURES = {"star", "observed_tree", "generated_root", "dynamic_cross_modal"}
POOLING_MODES = {"mean", "dual_modal_attention", "quality_gated_attention"}


def parent_semantic_consistency_gate(
    node_features: torch.Tensor,
    node_type: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    floor: float = 0.25,
) -> torch.Tensor:
    """Label-free reliability weights for generated reply nodes.

    The generated-edge relation is bidirectional in the model graph.  For a
    generated target, the directed parent -> child copy identifies its parent;
    the cosine similarity of the two text encodings then determines how much
    of the generated-node representation enters message passing.  Real/news
    nodes remain exactly at weight one.
    """
    if not 0.0 <= float(floor) <= 1.0:
        raise ValueError("semantic parent-gate floor must be in [0, 1]")
    gate = node_features.new_ones((node_features.size(0),))
    generated = torch.nonzero(node_type == 2, as_tuple=False).flatten()
    for child in generated.tolist():
        incoming = (edge_index[1] == child) & (edge_type == 1)
        parents = edge_index[0, incoming]
        if parents.numel() == 0:
            # Dataset construction guarantees an attachment edge, but leave
            # this generated node neutral rather than inventing an edge.
            continue
        parent = int(parents[0])
        similarity = F.cosine_similarity(
            node_features[child].unsqueeze(0),
            node_features[parent].unsqueeze(0),
            dim=-1,
        ).squeeze(0).clamp(-1.0, 1.0)
        normalized = 0.5 * (similarity + 1.0)
        gate[child] = float(floor) + (1.0 - float(floor)) * normalized
    return gate


def quality_weighted_pool(
    node_features: torch.Tensor,
    node_quality: torch.Tensor,
    batch: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Pool nodes without letting low-quality generated nodes dilute the graph."""
    weights = node_quality.clamp_min(1e-6).unsqueeze(-1)
    numerator = global_add_pool(node_features * weights, batch, size=batch_size)
    denominator = global_add_pool(weights, batch, size=batch_size).clamp_min(1e-6)
    return numerator / denominator


class CrossModalQualityGate(nn.Module):
    """Predict a soft quality score only for generated comments."""

    def __init__(self, node_dim: int = 768, query_dim: int = 768, score_dim: int = 128):
        super().__init__()
        self.node_projection = nn.Linear(node_dim, score_dim, bias=False)
        self.text_projection = nn.Linear(query_dim, score_dim, bias=False)
        self.image_projection = nn.Linear(query_dim, score_dim, bias=False)
        self.quality_head = nn.Sequential(
            nn.Linear(score_dim * 5 + 1, score_dim),
            nn.ReLU(),
            nn.Linear(score_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        node_type: torch.Tensor,
        batch: torch.Tensor,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
    ) -> torch.Tensor:
        quality = node_features.new_ones((node_features.size(0),))
        generated_indices = torch.nonzero(node_type == 2, as_tuple=False).flatten()
        if generated_indices.numel() == 0:
            return quality

        projected_nodes = self.node_projection(node_features)
        projected_text = self.text_projection(text_embedding)
        projected_image = self.image_projection(image_embedding)
        generated_nodes = projected_nodes[generated_indices]
        generated_batch = batch[generated_indices]
        text_query = projected_text[generated_batch]
        image_query = projected_image[generated_batch]

        redundancy_values = []
        for generated_index in generated_indices:
            graph_id = batch[generated_index]
            real_indices = torch.nonzero(
                (batch == graph_id) & (node_type == 1), as_tuple=False
            ).flatten()
            if real_indices.numel() == 0:
                redundancy_values.append(node_features.new_zeros(()))
            else:
                generated_vector = projected_nodes[generated_index].unsqueeze(0)
                similarities = F.cosine_similarity(
                    generated_vector,
                    projected_nodes[real_indices],
                    dim=-1,
                )
                redundancy_values.append(similarities.max())
        redundancy = torch.stack(redundancy_values).unsqueeze(-1)

        features = torch.cat(
            (
                generated_nodes,
                generated_nodes * text_query,
                generated_nodes * image_query,
                torch.abs(generated_nodes - text_query),
                torch.abs(generated_nodes - image_query),
                redundancy,
            ),
            dim=-1,
        )
        quality[generated_indices] = torch.sigmoid(self.quality_head(features).squeeze(-1))
        return quality


class SourceAwareQualityAttention(nn.Module):
    """Low-rank sigmoid attention with separate real/generated summaries."""

    def __init__(
        self,
        node_dim: int = 768,
        query_dim: int = 768,
        score_dim: int = 128,
        dropout: float = 0.1,
        real_residual_init: float = 0.10,
        generated_residual_init: float = 0.02,
    ):
        super().__init__()
        self.score_dim = int(score_dim)
        self.node_key = nn.Linear(node_dim, score_dim, bias=False)
        self.text_query = nn.Linear(query_dim, score_dim, bias=False)
        self.image_query = nn.Linear(query_dim, score_dim, bias=False)
        self.real_bias = nn.Parameter(torch.zeros(()))
        self.generated_bias = nn.Parameter(torch.zeros(()))
        self.real_residual_logit = nn.Parameter(
            torch.tensor(math.log(real_residual_init / (1.0 - real_residual_init)))
        )
        self.generated_residual_logit = nn.Parameter(
            torch.tensor(math.log(generated_residual_init / (1.0 - generated_residual_init)))
        )
        self.dropout = nn.Dropout(dropout)

    def _source_summary(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor,
        mask: torch.Tensor,
        scores: torch.Tensor,
        quality: torch.Tensor,
        batch_size: int,
    ):
        weights = torch.sigmoid(scores[mask]) * quality[mask]
        source_batch = batch[mask]
        if weights.numel() == 0:
            return node_features.new_zeros((batch_size, node_features.size(-1))), weights
        numerator = global_add_pool(
            self.dropout(node_features[mask]) * weights.unsqueeze(-1),
            source_batch,
            size=batch_size,
        )
        denominator = global_add_pool(
            weights.unsqueeze(-1), source_batch, size=batch_size
        ).clamp_min(1e-6)
        return numerator / denominator, weights

    def forward(
        self,
        node_features: torch.Tensor,
        node_type: torch.Tensor,
        node_quality: torch.Tensor,
        batch: torch.Tensor,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
    ):
        batch_size = text_embedding.size(0)
        keys = self.node_key(node_features)
        queries = (
            self.text_query(text_embedding) + self.image_query(image_embedding)
        ) * 0.5
        scores = (keys * queries[batch]).sum(dim=-1) / math.sqrt(self.score_dim)
        real_mask = node_type == 1
        generated_mask = node_type == 2
        real_summary, real_weights = self._source_summary(
            node_features,
            batch,
            real_mask,
            scores + self.real_bias,
            node_quality,
            batch_size,
        )
        generated_summary, generated_weights = self._source_summary(
            node_features,
            batch,
            generated_mask,
            scores + self.generated_bias,
            node_quality,
            batch_size,
        )
        alpha_real = torch.sigmoid(self.real_residual_logit)
        alpha_generated = torch.sigmoid(self.generated_residual_logit)
        summary = alpha_real * real_summary + alpha_generated * generated_summary
        all_weights = node_features.new_zeros((node_features.size(0),))
        all_weights[real_mask] = real_weights
        all_weights[generated_mask] = generated_weights
        return summary, all_weights, alpha_real, alpha_generated


class DualModalCommentAttention(nn.Module):
    """Pool comment nodes using independent text and image queries."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.text_query = nn.Linear(dim, dim, bias=False)
        self.image_query = nn.Linear(dim, dim, bias=False)
        self.text_key = nn.Linear(dim, dim, bias=False)
        self.image_key = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_features: torch.Tensor,
        node_type: torch.Tensor,
        batch: torch.Tensor,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        comment_mask = node_type != 0
        comment_nodes = node_features[comment_mask]
        comment_batch = batch[comment_mask]
        batch_size = text_embedding.size(0)

        if comment_nodes.numel() == 0:
            zeros = text_embedding.new_zeros((batch_size, self.dim))
            node_zeros = text_embedding.new_zeros((node_features.size(0),))
            return zeros, node_zeros, node_zeros, comment_mask

        scale = math.sqrt(self.dim)
        text_scores = (
            self.text_key(comment_nodes)
            * self.text_query(text_embedding)[comment_batch]
        ).sum(dim=-1) / scale
        image_scores = (
            self.image_key(comment_nodes)
            * self.image_query(image_embedding)[comment_batch]
        ).sum(dim=-1) / scale

        text_weights = softmax(text_scores, comment_batch, num_nodes=batch_size)
        image_weights = softmax(image_scores, comment_batch, num_nodes=batch_size)
        text_summary = global_add_pool(
            self.dropout(comment_nodes) * text_weights.unsqueeze(-1),
            comment_batch,
            size=batch_size,
        )
        image_summary = global_add_pool(
            self.dropout(comment_nodes) * image_weights.unsqueeze(-1),
            comment_batch,
            size=batch_size,
        )
        gate = torch.sigmoid(self.gate(torch.cat((text_embedding, image_embedding), dim=-1)))
        summary = gate * text_summary + (1.0 - gate) * image_summary

        all_text_weights = text_embedding.new_zeros((node_features.size(0),))
        all_image_weights = image_embedding.new_zeros((node_features.size(0),))
        all_text_weights[comment_mask] = text_weights
        all_image_weights[comment_mask] = image_weights
        return summary, all_text_weights, all_image_weights, comment_mask


class DynamicEdgeBuilder(nn.Module):
    """Rebuild weighted news-comment/comment-comment edges from current features."""

    def __init__(self, node_dim: int, query_dim: int, score_dim: int, top_k: int):
        super().__init__()
        self.top_k = int(top_k)
        self.node_query = nn.Linear(node_dim, score_dim, bias=False)
        self.node_key = nn.Linear(node_dim, score_dim, bias=False)
        self.text_query = nn.Linear(query_dim, score_dim, bias=False)
        self.image_query = nn.Linear(query_dim, score_dim, bias=False)
        self.type_bias = nn.Embedding(3, score_dim)
        self.edge_scale = nn.Parameter(torch.tensor(1.0))
        self.score_dim = int(score_dim)

    def _pair_score(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        graph_id: int,
        node_query: torch.Tensor,
        node_key: torch.Tensor,
        node_type: torch.Tensor,
        text_query: torch.Tensor,
        image_query: torch.Tensor,
    ) -> torch.Tensor:
        source_repr = node_key[source] + self.type_bias(node_type[source])
        target_repr = node_query[target]
        semantic = (source_repr * target_repr).sum(dim=-1)
        cross_modal = (
            source_repr * (text_query[graph_id] + image_query[graph_id]) * 0.5
        ).sum(dim=-1)
        return (semantic + cross_modal) / math.sqrt(self.score_dim)

    def forward(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor,
        batch: torch.Tensor,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        node_quality: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_query = self.node_query(x)
        node_key = self.node_key(x)
        text_query = self.text_query(text_embedding)
        image_query = self.image_query(image_embedding)
        edge_parts: List[torch.Tensor] = []
        score_parts: List[torch.Tensor] = []
        batch_size = int(text_embedding.size(0))

        for graph_id in range(batch_size):
            graph_nodes = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
            root_nodes = graph_nodes[node_type[graph_nodes] == 0]
            comment_nodes = graph_nodes[node_type[graph_nodes] != 0]
            if root_nodes.numel() == 0 or comment_nodes.numel() == 0:
                continue
            root = root_nodes[0]

            # Preserve bidirectional news-comment connectivity.
            roots = root.expand_as(comment_nodes)
            base_source = torch.cat((roots, comment_nodes))
            base_target = torch.cat((comment_nodes, roots))
            base_scores = self._pair_score(
                base_source, base_target, graph_id, node_query, node_key,
                node_type, text_query, image_query,
            )
            edge_parts.append(torch.stack((base_source, base_target), dim=0))
            score_parts.append(base_scores)

            # Select strongest comment-comment neighbours for each destination.
            if comment_nodes.numel() > 1 and self.top_k > 0:
                for target in comment_nodes:
                    candidates = comment_nodes[comment_nodes != target]
                    targets = target.expand_as(candidates)
                    scores = self._pair_score(
                        candidates, targets, graph_id, node_query, node_key,
                        node_type, text_query, image_query,
                    )
                    keep = min(self.top_k, int(candidates.numel()))
                    # Topology is discrete; weights on selected edges remain differentiable.
                    selected = torch.topk(scores.detach(), k=keep, largest=True).indices
                    edge_parts.append(torch.stack((candidates[selected], targets[selected]), dim=0))
                    score_parts.append(scores[selected])

        if not edge_parts:
            return (
                torch.empty((2, 0), dtype=torch.long, device=x.device),
                torch.empty((0,), dtype=x.dtype, device=x.device),
            )
        edge_index = torch.cat(edge_parts, dim=1)
        raw_scores = torch.cat(score_parts)
        edge_weight = torch.sigmoid(self.edge_scale * raw_scores).clamp_min(1e-4)
        if node_quality is not None:
            source_quality = node_quality[edge_index[0]]
            target_quality = node_quality[edge_index[1]]
            edge_weight = edge_weight * torch.sqrt(source_quality * target_quality).clamp_min(1e-4)
        return edge_index, edge_weight


class DynamicCrossModalEncoder(nn.Module):
    """Use MFCL's GCN layers while rebuilding edges before every layer."""

    def __init__(self, convs: nn.ModuleList, top_k: int = 3, score_dim: int = 128):
        super().__init__()
        if len(convs) != 2:
            raise ValueError("The current MFCL dynamic encoder expects exactly two GCN layers")
        # Keep non-registering references to the existing MFCL GCN layers.
        # They remain owned by graph_model, avoiding duplicate state-dict keys.
        object.__setattr__(self, "_convs", convs)
        self.edge_builders = nn.ModuleList(
            (
                DynamicEdgeBuilder(768, 768, score_dim, top_k),
                DynamicEdgeBuilder(512, 768, score_dim, top_k),
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor,
        batch: torch.Tensor,
        text_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        node_quality: torch.Tensor | None = None,
    ):
        layer_outputs = []
        edge_history = []
        current = x
        for conv, edge_builder in zip(self._convs, self.edge_builders):
            edge_index, edge_weight = edge_builder(
                current, node_type, batch, text_embedding, image_embedding, node_quality
            )
            current = F.relu(conv(current, edge_index, edge_weight=edge_weight))
            layer_outputs.append(current)
            edge_history.append((edge_index, edge_weight))
        if node_quality is None:
            pooled_values = [
                global_mean_pool(value, batch, size=text_embedding.size(0))
                for value in layer_outputs
            ]
        else:
            pooled_values = [
                quality_weighted_pool(
                    value, node_quality, batch, int(text_embedding.size(0))
                )
                for value in layer_outputs
            ]
        pooled = torch.cat(pooled_values, dim=-1)
        return pooled, torch.cat(layer_outputs, dim=-1), edge_history
