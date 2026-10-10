"""MIMoE-style interaction routing adapted to the MFCL feature interface.

The public MIMoE-FND implementation (yliuaa/MIMoE-FND, commit
002e747829dffcb2925803bb617cd1f9dd2de20c) uses MAE patch tokens and
ChineseCLIP embeddings that are not part of this project.  This module retains
its central trainable mechanism while taking the existing MFCL pooled text and
image vectors instead:

* text/image unimodal predictions yield a Jensen--Shannon agreement signal;
* projected text/image vectors yield a semantic-agreement signal;
* their four combinations supervise a four-way interaction router; and
* each router condition owns a small mixture of fusion experts.

This is therefore an interface adaptation, not a claim that the authors'
end-to-end MAE/ChineseCLIP result is reproduced.  Unlike the released code,
the balancing loss below is differentiable and the router-z coefficient is
applied exactly once.  Both differences are documented so experiments remain
auditable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenAttention(nn.Module):
    """Stable attention pooling over the four MIMoE routing tokens."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.score(tokens).squeeze(-1).softmax(dim=1)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1), weights


def jensen_shannon_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-example JSD, matching the agreement signal in the authors' code."""
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    p, q = p / p.sum(dim=-1, keepdim=True), q / q.sum(dim=-1, keepdim=True)
    midpoint = .5 * (p + q)
    return .5 * ((p * (p.log() - midpoint.log())).sum(-1) +
                 (q * (q.log() - midpoint.log())).sum(-1))


class ConditionExperts(nn.Module):
    """Experts for one of the four agreement/semantic interaction conditions."""

    def __init__(self, dim: int, hidden_dim: int, num_experts: int, dropout: float):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.gate(x).softmax(dim=-1)
        candidates = torch.stack([expert(x) for expert in self.experts], dim=1)
        return (candidates * weights.unsqueeze(-1)).sum(dim=1), weights


class MIMoEFusion(nn.Module):
    """Four-condition mixture-of-experts fusion for pooled MFCL modalities."""

    def __init__(
        self,
        input_dim: int = 768,
        semantic_dim: int = 64,
        hidden_dim: int = 768,
        num_experts: int = 2,
        dropout: float = .1,
        agreement_threshold: float = .3,
        semantic_threshold: float = .3,
        interaction_loss_weight: float = .7,
        balance_loss_weight: float = .1,
        router_z_loss_weight: float = .01,
    ):
        super().__init__()
        if min(input_dim, semantic_dim, hidden_dim, num_experts) < 1:
            raise ValueError("MIMoE dimensions and expert count must be positive")
        self.output_dim = input_dim * 2
        self.agreement_threshold = float(agreement_threshold)
        self.semantic_threshold = float(semantic_threshold)
        self.interaction_loss_weight = float(interaction_loss_weight)
        self.balance_loss_weight = float(balance_loss_weight)
        self.router_z_loss_weight = float(router_z_loss_weight)

        self.text_unimodal = nn.Linear(input_dim, 2)
        self.image_unimodal = nn.Linear(input_dim, 2)
        self.text_route_projection = nn.Sequential(nn.Linear(input_dim, semantic_dim), nn.SiLU())
        self.image_route_projection = nn.Sequential(nn.Linear(input_dim, semantic_dim), nn.SiLU())
        self.text_semantic_projection = nn.Sequential(nn.Linear(input_dim, semantic_dim), nn.SiLU())
        self.image_semantic_projection = nn.Sequential(nn.Linear(input_dim, semantic_dim), nn.SiLU())
        self.router_attention = TokenAttention(semantic_dim)
        self.router = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim), nn.SiLU(), nn.Linear(semantic_dim, 4)
        )
        self.conditions = nn.ModuleList([
            ConditionExperts(self.output_dim, hidden_dim, num_experts, dropout) for _ in range(4)
        ])

    def _routing_targets(
        self, text_logits: torch.Tensor, image_logits: torch.Tensor,
        text_semantic: torch.Tensor, image_semantic: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Threshold targets are intentionally non-differentiable pseudo labels,
        # just as in the original interaction supervision.
        jsd = jensen_shannon_divergence(text_logits.softmax(-1), image_logits.softmax(-1))
        semantic = F.cosine_similarity(text_semantic, image_semantic, dim=-1)
        agreement = (jsd < self.agreement_threshold).long()
        semantic_match = (semantic > self.semantic_threshold).long()
        return 2 * agreement + semantic_match, jsd, semantic

    def forward(self, text: torch.Tensor, image: torch.Tensor) -> dict[str, torch.Tensor]:
        text_logits, image_logits = self.text_unimodal(text), self.image_unimodal(image)
        route_text, route_image = self.text_route_projection(text), self.image_route_projection(image)
        semantic_text, semantic_image = self.text_semantic_projection(text), self.image_semantic_projection(image)
        targets, jsd, semantic_similarity = self._routing_targets(
            text_logits, image_logits, semantic_text, semantic_image
        )
        routing_tokens = torch.stack((route_text, route_image, semantic_text, semantic_image), dim=1)
        router_input, router_token_attention = self.router_attention(routing_tokens)
        router_logits = self.router(router_input)
        router_weights = router_logits.softmax(dim=-1)

        # CE is the paper's relation pseudo-supervision.  The z-loss stabilizes
        # router scale.  The released implementation derives its balancing loss
        # from argmax assignments, so it cannot train the router; using average
        # soft dispatch makes the intended regularizer effective.
        interaction_ce = F.cross_entropy(router_logits, targets)
        router_z = torch.logsumexp(router_logits, dim=-1).square().mean()
        balance = F.mse_loss(router_weights.mean(dim=0), router_weights.new_full((4,), .25))
        routing_loss = (self.interaction_loss_weight * interaction_ce +
                        self.router_z_loss_weight * router_z +
                        self.balance_loss_weight * balance)

        fused_input = torch.cat((text, image), dim=-1)
        conditional_features, expert_weights = [], []
        for condition in self.conditions:
            feature, weights = condition(fused_input)
            conditional_features.append(feature)
            expert_weights.append(weights)
        conditional_features = torch.stack(conditional_features, dim=1)
        h_mm = (conditional_features * router_weights.unsqueeze(-1)).sum(dim=1)
        return {
            "h_mm": h_mm,
            "routing_loss": routing_loss,
            "text_unimodal_logits": text_logits,
            "image_unimodal_logits": image_logits,
            "router_logits": router_logits,
            "router_weights": router_weights,
            "router_targets": targets,
            "agreement_jsd": jsd,
            "semantic_similarity": semantic_similarity,
            "router_token_attention": router_token_attention,
            "expert_weights": torch.stack(expert_weights, dim=1),
        }
