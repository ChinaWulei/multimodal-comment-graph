"""GAHR-inspired, comment-guided local correction (not a GAHR reproduction).

Reference: sections 4.4, Eqs. 21-30 of arXiv:2609.36850v1.
We query BERT tokens and ResNet spatial cells using comment vectors, instead of
the paper's RoBERTa/DINOv2/CLIP representations and global directional queries.
"""

import math

import torch
from torch import nn
from torch_geometric.nn import global_add_pool


DECISION_MODES = {"concat", "global_local_correction", "source_isolated", "reliability_residual"}


def projection(input_dim, dim):
    return nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim), nn.GELU())


def relation(u, v):
    return torch.cat((u, v, (u - v).abs(), u * v), dim=-1)


class CommentLocalCorrection(nn.Module):
    def __init__(self, global_dim=2304, dim=128, temperature=1.0,
                 dropout=0.1, gate_init=0.1, num_classes=2):
        super().__init__()
        if num_classes < 2 or dim < 1 or temperature <= 0 or not 0 < gate_init < 1:
            raise ValueError("Require num_classes >= 2, dim > 0, temperature > 0 and 0 < gate_init < 1")
        self.temperature = float(temperature)
        self.comment_projection = projection(768, dim)
        self.text_projection = projection(768, dim)
        self.image_projection = projection(2048, dim)
        self.text_query = nn.Linear(dim, dim, bias=False)
        self.image_query = nn.Linear(dim, dim, bias=False)
        self.text_relation = projection(4 * dim, dim)
        self.image_relation = projection(4 * dim, dim)
        self.cross_relation = projection(4 * dim, dim)
        self.evidence_projection = projection(3 * dim, dim)
        # Provenance-specific reliability heads; provenance is not a veracity label.
        self.real_reliability = nn.Linear(dim, 1)
        self.generated_reliability = nn.Linear(dim, 1)
        # Do not normalize away the magnitude reduction caused by low reliability.
        self.source_fusion = nn.Sequential(nn.Linear(2 * dim, dim, bias=False), nn.GELU())
        self.global_projection = projection(global_dim, dim)
        # The original implementation was binary because it was first used on
        # CHECKED.  Fakeddit's official 3-way label is a legitimate
        # classification target, so the residual must have one logit per class
        # rather than forcing a binary reduction of the dataset.
        self.correction = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, int(num_classes), bias=False))
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1))
        # Small initial influence, but gradients reach local evidence on step one.
        nn.init.normal_(self.gate[-1].weight, std=0.001)
        nn.init.constant_(self.gate[-1].bias, math.log(gate_init / (1 - gate_init)))

    def attend(self, query, keys, mask):
        scores = torch.einsum("nd,nld->nl", query, keys)
        scores = scores / (math.sqrt(keys.size(-1)) * self.temperature)
        mask = mask.bool()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1) * mask.to(scores.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("nl,nld->nd", weights, keys), weights

    @staticmethod
    def source_pool(evidence, reliability, source_mask, graph_ids, size):
        # Divide by count, not sum(reliability): rejecting ALL comments must
        # reduce evidence magnitude, not be cancelled by renormalization.
        numerator = global_add_pool(
            evidence[source_mask] * reliability[source_mask, None],
            graph_ids[source_mask], size=size,
        )
        count = global_add_pool(
            evidence.new_ones((int(source_mask.sum()), 1)),
            graph_ids[source_mask], size=size,
        )
        return numerator / count.clamp_min(1), count

    def forward(self, global_features, node_features, node_type, batch,
                text_tokens, text_mask, image_map):
        size = global_features.size(0)
        indices = (node_type != 0).nonzero(as_tuple=False).flatten()
        text_length = text_tokens.size(1)
        image_length = image_map.size(-1) * image_map.size(-2)
        if indices.numel() == 0:
            zeros = global_features.new_zeros((size, self.correction[-1].out_features))
            return {
                "logit_delta": zeros,
                "local_logits": zeros,
                "correction_gate": zeros[:, :1],
                "comment_indices": indices,
                "text_weights": zeros.new_empty((0, text_length)),
                "image_weights": zeros.new_empty((0, image_length)),
                "reliability": zeros.new_empty((0,)),
            }

        graph_ids = batch[indices]
        types = node_type[indices]
        comments = self.comment_projection(node_features[indices])
        tokens = self.text_projection(text_tokens)
        regions = self.image_projection(image_map.flatten(2).transpose(1, 2))
        text_response, text_weights = self.attend(
            self.text_query(comments), tokens[graph_ids], text_mask[graph_ids],
        )
        image_response, image_weights = self.attend(
            self.image_query(comments), regions[graph_ids],
            torch.ones((indices.numel(), image_length), device=regions.device, dtype=torch.bool),
        )
        evidence = self.evidence_projection(torch.cat((
            self.text_relation(relation(comments, text_response)),
            self.image_relation(relation(comments, image_response)),
            self.cross_relation(relation(text_response, image_response)),
        ), dim=-1))
        reliability = torch.where(
            types == 2,
            self.generated_reliability(evidence).squeeze(-1),
            self.real_reliability(evidence).squeeze(-1),
        ).sigmoid()
        real, real_count = self.source_pool(evidence, reliability, types == 1, graph_ids, size)
        generated, generated_count = self.source_pool(evidence, reliability, types == 2, graph_ids, size)
        local_features = self.source_fusion(torch.cat((real, generated), dim=-1))
        local_logits = self.correction(local_features)
        gate = self.gate(torch.cat((self.global_projection(global_features), local_features), -1)).sigmoid()
        # Absent comments cannot produce a correction through learned biases.
        gate = gate * ((real_count + generated_count) > 0).to(gate.dtype)
        return {
            "logit_delta": gate * local_logits,
            "local_logits": local_logits,
            "correction_gate": gate,
            "comment_indices": indices,
            "text_weights": text_weights,
            "image_weights": image_weights,
            "reliability": reliability,
        }
