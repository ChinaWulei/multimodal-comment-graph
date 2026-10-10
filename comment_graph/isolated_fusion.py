"""Source-isolated graph propagation and an optional generated-comment residual.

No new comments, pseudo labels, or reply relationships are created here.
"""
import math

import torch
from torch import nn
from torch_geometric.data import Data

from Weibo.propogation import Encoder
from comment_graph.dynamic_graph import DynamicCrossModalEncoder


def select_nodes(data, keep):
    """Build a fresh view and remap edges; never leave stale PyG batch slices."""
    indices = keep.nonzero(as_tuple=False).flatten()
    mapping = torch.full((data.num_nodes,), -1, device=keep.device, dtype=torch.long)
    mapping[indices] = torch.arange(indices.numel(), device=keep.device)
    edge_keep = keep[data.edge_index[0]] & keep[data.edge_index[1]]
    edges = mapping[data.edge_index[:, edge_keep]]
    edge_types = data.edge_type[edge_keep] if hasattr(data, "edge_type") else None
    graph_ids = data.batch[indices]
    size = data.root_input_ids.size(0)
    counts = torch.bincount(graph_ids, minlength=size)
    if (counts == 0).any():
        raise ValueError("Every isolated view must retain the news root")
    view = Data(
        node_input_ids=data.node_input_ids[indices],
        node_attention_mask=data.node_attention_mask[indices],
        node_type=data.node_type[indices], batch=graph_ids, edge_index=edges,
        edge_type=edge_types,
        root_input_ids=data.root_input_ids, root_attention_mask=data.root_attention_mask,
        pixel_values=data.pixel_values, y=data.y, news_id=data.news_id,
        num_nodes=indices.numel(),
        ptr=torch.cat((counts.new_zeros(1), counts.cumsum(0))),
    )
    return view, indices


class GeneratedGraphCorrection(nn.Module):
    """A separate MFCL GCN; synthetic nodes never enter the base GCN."""

    def __init__(self, config, global_dim):
        super().__init__()
        options = config.get("isolated_fusion", {})
        dim = int(options.get("hidden_dim", 128))
        gate_init = float(options.get("gate_init", .1))
        self.branch_dropout = float(options.get("branch_dropout", .2))
        self.max_logit_delta = float(options.get("max_logit_delta", 2.0))
        self.detach_base_context = bool(options.get("detach_base_context", True))
        self.correction_reference = str(options.get("correction_reference", "none"))
        if self.correction_reference not in {"none", "null_generated"}:
            raise ValueError("Unknown generated correction reference")
        if not (dim > 0 and 0 < gate_init < 1 and 0 <= self.branch_dropout <= 1
                and self.max_logit_delta > 0):
            raise ValueError("Invalid isolated_fusion dimensions, gate, dropout or correction bound")
        self.encoder = Encoder(768, 256, k=2)
        self.dynamic_encoder = DynamicCrossModalEncoder(
            self.encoder.convs, top_k=int(config.graph.dynamic_top_k),
            score_dim=int(config.graph.edge_score_dim),
        )
        self.node_type_embedding = nn.Embedding(3, 768)
        nn.init.normal_(self.node_type_embedding.weight, std=.02)
        self.evidence = nn.Sequential(nn.Linear(768 + 1536, dim), nn.LayerNorm(dim), nn.GELU())
        self.base_projection = nn.Sequential(nn.Linear(global_dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.correction = nn.Linear(dim, 2, bias=False)
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.normal_(self.gate[-1].weight, std=.001)
        nn.init.constant_(self.gate[-1].bias, math.log(gate_init / (1 - gate_init)))

    def _graph_evidence(self, features, view, text, image, structure):
        typed = features + self.node_type_embedding(view.node_type)
        if structure == "dynamic_cross_modal":
            h_graph, _, edges = self.dynamic_encoder(typed, view.node_type, view.batch, text, image)
        else:
            h_graph, _ = self.encoder(typed, view.edge_index, view.batch)
            edges = [(view.edge_index, None)]
        evidence = self.evidence(torch.cat((h_graph, text, image), dim=-1))
        return evidence, h_graph, edges

    def forward(self, features, view, base_features, text, image, structure, apply_dropout=True):
        if self.detach_base_context:
            base_features, text, image = base_features.detach(), text.detach(), image.detach()
        evidence, h_graph, edges = self._graph_evidence(features, view, text, image, structure)
        reference_graph = None
        if self.correction_reference == "null_generated":
            # Retain root, types and cardinality; remove only projected generated
            # text content. Recompute dynamic edges for this null-content view.
            # Both passes share weights and receive gradients. No BERT/image
            # re-encoding, additional parameters, BN updates or RNG draws.
            null_features = features.masked_fill(view.node_type.eq(2)[:, None], 0)
            reference, reference_graph, _ = self._graph_evidence(
                null_features, view, text, image, structure)
            evidence = evidence - reference
            # This is an architectural reference subtraction, NOT an identified
            # causal effect or proof that the remaining content is relevant.
        local_logits = self.max_logit_delta * torch.tanh(self.correction(evidence))
        gate = self.gate(torch.cat((self.base_projection(base_features), evidence), dim=-1)).sigmoid()
        present = torch.bincount(view.batch[view.node_type == 2], minlength=text.size(0)) > 0
        active = present[:, None].to(gate.dtype)
        if self.training and apply_dropout and self.branch_dropout > 0:
            # Whole branch dropout per news, no inverted-dropout rescaling.
            active = active * (torch.rand_like(gate) >= self.branch_dropout)
        gate = gate * active
        return {"logit_delta": gate * local_logits, "local_logits": local_logits,
                "correction_gate": gate, "branch_active": active,
                "h_generated_graph": h_graph, "edge_history": edges,
                "h_reference_graph": reference_graph}


def classification_objective(output, labels, weights, base_loss_weight=0.0):
    """Only the caller enables auxiliary base supervision for the new mode."""
    if base_loss_weight < 0:
        raise ValueError("base_loss_weight must be nonnegative")
    final_ce = nn.functional.cross_entropy(output["logits"], labels, weight=weights)
    base_ce = nn.functional.cross_entropy(output["global_logits"], labels, weight=weights)
    # Preserve the original objective exactly when auxiliary supervision is off.
    total = final_ce if base_loss_weight == 0 else final_ce + base_loss_weight * base_ce
    return total, final_ce, base_ce
