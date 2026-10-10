"""Trainable raw text/image encoders used by the Fakeddit pipeline."""

from contextlib import nullcontext

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoModel


class OnlineFeatureEncoder(nn.Module):
    """Encode source text, comment nodes and pixels during every forward pass."""

    def __init__(
        self,
        text_model_name: str = "roberta-base",
        output_dim: int = 768,
        pretrained_image: bool = True,
        fine_tune_text: bool = True,
        fine_tune_image: bool = True,
        node_encode_batch_size: int = 128,
    ):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_model.config.hidden_size
        self.text_projection = nn.Identity() if text_dim == output_dim else nn.Linear(text_dim, output_dim)

        weights = ResNet50_Weights.DEFAULT if pretrained_image else None
        image_model = resnet50(weights=weights)
        self.image_backbone = nn.Sequential(*list(image_model.children())[:-2])

        self.fine_tune_text = fine_tune_text
        self.fine_tune_image = fine_tune_image
        self.node_encode_batch_size = node_encode_batch_size
        for parameter in self.text_model.parameters():
            parameter.requires_grad = fine_tune_text
        for parameter in self.image_backbone.parameters():
            parameter.requires_grad = fine_tune_image

    @staticmethod
    def _masked_mean(features, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(features.dtype)
        return (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def _text_forward(self, input_ids, attention_mask):
        context = nullcontext() if self.fine_tune_text else torch.no_grad()
        with context:
            return self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

    def forward(self, batch_data):
        root_tokens = self._text_forward(
            batch_data.root_input_ids,
            batch_data.root_attention_mask,
        )
        root_features = self.text_projection(root_tokens)

        node_chunks = []
        for start in range(0, batch_data.node_input_ids.size(0), self.node_encode_batch_size):
            end = start + self.node_encode_batch_size
            attention_mask = batch_data.node_attention_mask[start:end]
            token_features = self._text_forward(
                batch_data.node_input_ids[start:end],
                attention_mask,
            )
            node_chunks.append(self._masked_mean(token_features, attention_mask))
        node_features = self.text_projection(torch.cat(node_chunks, dim=0))

        image_context = nullcontext() if self.fine_tune_image else torch.no_grad()
        with image_context:
            image_features = self.image_backbone(batch_data.pixel_values)

        return root_features, image_features, node_features

    def train(self, mode=True):
        super().train(mode)
        # Frozen backbones should also have deterministic dropout/BatchNorm.
        if not self.fine_tune_text:
            self.text_model.eval()
        if not self.fine_tune_image:
            self.image_backbone.eval()
        return self
