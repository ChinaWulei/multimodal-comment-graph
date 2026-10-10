"""Label-free news/comment compatibility head over frozen MFCL text features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NewsCommentRelevance(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.news_projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.comment_projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.classifier = nn.Sequential(
            nn.Linear(4 * hidden_dim + 1, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, news: torch.Tensor, comment: torch.Tensor) -> torch.Tensor:
        if news.shape != comment.shape or news.ndim != 2:
            raise ValueError("Expected matched [batch, dim] news/comment features")
        news = F.normalize(news.float(), dim=-1)
        comment = F.normalize(comment.float(), dim=-1)
        n = self.news_projection(news)
        c = self.comment_projection(comment)
        cosine = (news * comment).sum(dim=-1, keepdim=True)
        features = torch.cat((n, c, torch.abs(n - c), n * c, cosine), dim=-1)
        return self.classifier(features).squeeze(-1)
