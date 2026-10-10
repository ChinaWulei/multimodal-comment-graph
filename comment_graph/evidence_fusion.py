"""Label-free entity evidence encoding and residual calibration for MFCL.

The evidence cache contains only retrieval-derived quantities: entity coverage,
text similarity, image similarity, reliability and a valid-item mask.  It
never exposes the retrieved news labels to this module or the classifier.
"""

from __future__ import annotations

import math

import torch
from torch import nn


EVIDENCE_MODES = {"none", "gated", "direct"}


class EntityEvidenceFusion(nn.Module):
    """Aggregate Top-K label-free evidence and add a small calibrated residual."""

    def __init__(self, mm_dim: int, evidence_dim: int = 256, dropout: float = .1,
                 gate_init: float = .10, mode: str = "gated"):
        super().__init__()
        if mode not in EVIDENCE_MODES - {"none"}:
            raise ValueError(f"Unsupported evidence mode: {mode}")
        if not 0 < gate_init < 1:
            raise ValueError("evidence gate_init must be in (0,1)")
        self.mode = mode
        self.evidence_dim = int(evidence_dim)
        # coverage, text similarity, image similarity, reliability, and the
        # coverage/image discrepancy. No retrieved class label is a feature.
        self.item_encoder = nn.Sequential(
            nn.Linear(5, evidence_dim), nn.LayerNorm(evidence_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(evidence_dim, evidence_dim),
        )
        self.query = nn.Sequential(nn.LayerNorm(mm_dim), nn.Linear(mm_dim, evidence_dim))
        self.value_projection = nn.Sequential(nn.LayerNorm(evidence_dim), nn.Linear(evidence_dim, mm_dim))
        self.gate = nn.Sequential(
            nn.LayerNorm(mm_dim + evidence_dim + 4), nn.Linear(mm_dim + evidence_dim + 4, evidence_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(evidence_dim, 1),
        )
        nn.init.normal_(self.gate[-1].weight, std=.001)
        nn.init.constant_(self.gate[-1].bias, math.log(gate_init / (1.0 - gate_init)))

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        logits = scores.masked_fill(~valid, -1e4)
        weights = logits.softmax(dim=1) * valid.float()
        return weights / weights.sum(1, keepdim=True).clamp_min(1e-8)

    def forward(self, h_mm: torch.Tensor, evidence_scores: torch.Tensor,
                evidence_text_sims: torch.Tensor, evidence_image_sims: torch.Tensor,
                evidence_reliability: torch.Tensor, evidence_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        valid = evidence_mask.bool()
        image_01 = ((evidence_image_sims.float() + 1.) / 2.).clamp(0., 1.)
        discrepancy = (evidence_scores.float() - image_01).abs()
        features = torch.stack((evidence_scores.float(), evidence_text_sims.float(),
                                evidence_image_sims.float(), evidence_reliability.float(), discrepancy), dim=-1)
        items = self.item_encoder(features)
        query = self.query(h_mm)
        similarity = (items * query[:, None]).sum(-1) / math.sqrt(self.evidence_dim)
        # Reliability is retrieval-derived, never a veracity label.
        attention = self._masked_softmax(similarity + evidence_reliability.float().clamp_min(1e-8).log(), valid)
        h_evidence = (items * attention.unsqueeze(-1)).sum(1)
        summary = torch.stack((
            (attention * evidence_scores.float()).sum(1),
            (attention * evidence_text_sims.float()).sum(1),
            (attention * evidence_image_sims.float()).sum(1),
            valid.float().mean(1),
        ), dim=-1)
        residual = self.value_projection(h_evidence)
        available = valid.any(1, keepdim=True).to(h_mm.dtype)
        if self.mode == "gated":
            gate = self.gate(torch.cat((h_mm, h_evidence, summary), dim=-1)).sigmoid() * available
        else:
            # Direct residual is the explicit no-gate ablation, not the main method.
            gate = available
        return {
            "h_mm": h_mm + gate * residual,
            "h_evidence": h_evidence,
            "evidence_attention": attention,
            "evidence_gate": gate,
            "evidence_summary": summary,
            "evidence_available": available,
        }
