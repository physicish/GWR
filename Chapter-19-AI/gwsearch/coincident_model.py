"""Coincident transformer: a weight-shared encoder scores each detector.

The per-detector log-odds are combined afterwards by combined_usr. Because the
combination happens outside the network, a time slide can reuse cached
per-detector outputs instead of running the model again.
"""

import torch
from torch import nn


MODEL_DEFAULTS = dict(sample_rate=2048, n_samples=2048, d_model=90,
                      n_layers=6, n_heads=6, input_instance_norm=True)


class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model) * d_model ** -0.5)
        self.scale = d_model ** -0.5

    def forward(self, tokens):
        weights = torch.softmax((tokens @ self.query) * self.scale, dim=-1)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1)


class CoincidentTransformer(nn.Module):
    """Shared encoder and binary head for each IFO."""

    def __init__(self, sample_rate=2048, n_samples=2048, d_model=90,
                 n_layers=6, n_heads=6, input_instance_norm=True):
        super().__init__()
        if sample_rate != 2048 or n_samples != 2048 or d_model % n_heads:
            raise ValueError("Expected 2048 samples at 2048 Hz and d_model divisible by n_heads")
        self.config = dict(sample_rate=sample_rate, n_samples=n_samples,
                           d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                           input_instance_norm=input_instance_norm)
        self.input_norm = (nn.InstanceNorm1d(1, affine=False)
                           if input_instance_norm else nn.Identity())
        self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=3)
        self.conv3 = nn.Conv1d(64, d_model, kernel_size=5, stride=2, padding=2)
        self.act = nn.GELU()
        self.tok_norm = nn.LayerNorm(d_model)
        self.pos = nn.Parameter(torch.zeros(1, 64, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                     dim_feedforward=4*d_model, dropout=0.1, activation="gelu",
                     batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = AttnPool(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, 256), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(256, 2))
        self.n_tokens = 64

    def forward(self, strain):
        if strain.ndim != 3 or strain.shape[1:] != (2, 2048):
            raise ValueError(f"Expected (batch, 2, 2048); got {tuple(strain.shape)}")
        # Flatten detector into batch: the same weights process each IFO
        # independently, with no cross-detector attention or pooling.
        batch = strain.shape[0]
        x = self.input_norm(strain.reshape(batch * 2, 1, 2048))
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        tokens = self.conv3(x).transpose(1, 2)
        tokens = self.norm(self.encoder(self.tok_norm(tokens) + self.pos))
        logits = self.head(self.pool(tokens)).reshape(batch, 2, 2)
        return {"class_logits": logits}


def per_detector_usr(logits):
    """Binary signal-vs-noise log odds for each detector."""
    return logits[..., 1] - logits[..., 0]


def combined_usr(s_h, s_l):
    """Coincident network statistic: log(exp(sH)+exp(sL)+exp(sH+sL))."""
    return torch.logaddexp(torch.logaddexp(s_h, s_l), s_h + s_l)


def combined_usr_and(s_h, s_l):
    """AND log odds: both signal versus at least one detector containing noise.

    For independent per-detector signal-vs-noise log odds sH and sL, this is
    log[P(H signal, L signal) / P(H noise or L noise)].
    """
    return s_h + s_l - torch.logaddexp(
        torch.zeros_like(s_h + s_l), torch.logaddexp(s_h, s_l))

