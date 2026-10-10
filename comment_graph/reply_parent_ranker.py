"""A supervised reply-parent scorer trained only on observed real edges."""

from __future__ import annotations

import torch
import torch.nn as nn


class ReplyParentRanker(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.child = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.parent = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.score = nn.Sequential(
            nn.Linear(3 * hidden_dim + 1, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def pair_scores(self, child: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
        """Scores parent candidates for one or more child representations."""
        if child.ndim != 2 or parents.ndim != 2 or child.size(-1) != parents.size(-1):
            raise ValueError("Expected [children,dim] and [parents,dim]")
        c = self.child(child)[:, None, :].expand(-1, parents.size(0), -1)
        p = self.parent(parents)[None, :, :].expand(child.size(0), -1, -1)
        root_flag = c.new_zeros((*c.shape[:2], 1))
        root_flag[:, 0, 0] = 1.0
        return self.score(torch.cat((c, p, c * p, root_flag), dim=-1)).squeeze(-1)

    def real_parent_scores(self, features: torch.Tensor) -> torch.Tensor:
        """N-1 real reply rows; only preceding nodes are candidate parents."""
        if features.size(0) < 2:
            raise ValueError("Graph must include a news root and a real comment")
        scores = self.pair_scores(features[1:], features)
        child_indices = torch.arange(1, features.size(0), device=features.device)
        parent_indices = torch.arange(features.size(0), device=features.device)
        return scores.masked_fill(parent_indices[None, :] >= child_indices[:, None], -1e9)
