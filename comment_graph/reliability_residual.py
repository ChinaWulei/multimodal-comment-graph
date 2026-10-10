"""Bounded synthetic-comment residual on top of a real-discussion decision."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool


class ReliabilityResidual(nn.Module):
    """Only the difference caused by generated nodes can correct base logits."""

    def __init__(self, graph_dim: int, base_dim: int, num_classes: int,
                 hidden_dim: int = 128, gate_init: float = 0.2,
                 max_logit_delta: float = 2.0, branch_dropout: float = 0.1):
        super().__init__()
        if graph_dim < 1 or base_dim < 1 or num_classes < 2 or hidden_dim < 1:
            raise ValueError("Invalid reliability residual dimensions")
        if not 0 < gate_init < 1 or max_logit_delta <= 0 or not 0 <= branch_dropout <= 1:
            raise ValueError("Invalid gate, delta bound or dropout")
        self.max_logit_delta = float(max_logit_delta)
        self.branch_dropout = float(branch_dropout)
        self.generated_projection = nn.Sequential(
            nn.LayerNorm(graph_dim), nn.Linear(graph_dim, hidden_dim), nn.GELU(),
        )
        self.base_projection = nn.Sequential(nn.Linear(base_dim, hidden_dim), nn.GELU())
        self.correction = nn.Linear(hidden_dim, num_classes, bias=False)
        self.gate = nn.Sequential(nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, 1))
        nn.init.normal_(self.correction.weight, std=0.01)
        nn.init.normal_(self.gate[-1].weight, std=0.001)
        nn.init.constant_(self.gate[-1].bias, math.log(gate_init / (1 - gate_init)))

    def forward(self, base_logits: torch.Tensor, base_features: torch.Tensor,
                base_graph: torch.Tensor, full_graph: torch.Tensor,
                node_quality: torch.Tensor, node_type: torch.Tensor,
                batch: torch.Tensor) -> dict:
        generated = node_type == 2
        size = base_logits.size(0)
        graph_ids = batch[generated]
        quality_sum = global_add_pool(node_quality[generated, None], graph_ids, size=size)
        count = global_add_pool(node_quality.new_ones((int(generated.sum()), 1)),
                                graph_ids, size=size)
        mean_quality = quality_sum / count.clamp_min(1)
        present = (count > 0).to(base_logits.dtype)
        difference = full_graph - base_graph
        evidence = self.generated_projection(difference)
        context = self.base_projection(base_features)
        local_logits = self.max_logit_delta * torch.tanh(self.correction(evidence))
        gate = self.gate(torch.cat((context, evidence, mean_quality), dim=-1)).sigmoid()
        gate = gate * mean_quality * present
        if self.training and self.branch_dropout > 0:
            gate = gate * (torch.rand_like(gate) >= self.branch_dropout).to(gate.dtype)
        return {
            "logits": base_logits + gate * local_logits,
            "logit_delta": gate * local_logits,
            "local_logits": local_logits,
            "correction_gate": gate,
            "mean_generated_quality": mean_quality,
            "h_generated_difference": difference,
        }
