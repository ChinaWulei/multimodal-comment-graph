"""Independent implementation of MTS equations 18-20 (WWW 2025).

Paper: https://doi.org/10.1145/3696410.3714719
Reference inspected: OneForAllSama/MTS, commit 77f6634b992b81efabbb6125e8797b74b154857e.
Retains both directions, linear/self-product/cross-product terms, batch infinity
scaling and identity-plus-Xavier initialization. Does NOT detach layer outputs
as the released notebook does. Integration uses MFCL features, not the paper's
ResNet34 end-to-end backbone. No claim of reproducing the published score.
"""
import torch
from torch import nn


class TaylorDirection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.coefficients = nn.ParameterList([nn.Parameter(torch.empty(dim, dim)) for _ in range(5)])
        for coefficient in self.coefficients:
            nn.init.xavier_uniform_(coefficient)
        with torch.no_grad():
            self.coefficients[0].add_(torch.eye(dim))

    def forward(self, own, other):
        linear, cross_other, cross_own, square_a, square_b = self.coefficients
        return (own @ linear + (other @ cross_other) * (own @ cross_own)
                + (own @ square_a) * (own @ square_b))


class TaylorLayer(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.image_direction = TaylorDirection(dim)
        self.text_direction = TaylorDirection(dim)
        self.eps = eps

    def forward(self, image, text):
        # One maximum over the entire batch and feature dimensions, as in the
        # authors' code. Evaluation therefore has a documented batch dependence.
        image = image / image.abs().amax().clamp_min(self.eps)
        text = text / text.abs().amax().clamp_min(self.eps)
        return self.image_direction(image, text), self.text_direction(text, image)


class MTSFusion(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=768, layers=8):
        super().__init__()
        if layers < 1 or hidden_dim < 1:
            raise ValueError('MTS needs positive dimensions and at least one layer')
        self.image_projection = nn.Linear(input_dim, hidden_dim)
        self.text_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([TaylorLayer(hidden_dim) for _ in range(layers)])
        self.output_dim = hidden_dim

    def forward(self, text, image):
        image = self.image_projection(image)
        text = self.text_projection(text)
        for layer in self.layers:
            image, text = layer(image, text)
        return image + text
