import torch
import torch.nn as nn


class MLPConvModule(nn.Module):
    """Per-node MLP baseline that ignores graph edges."""

    def __init__(self, dim, hidden_dim=None, depth=2, bias=True, id=None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = dim if hidden_dim is None else hidden_dim
        self.depth = max(2, int(depth))
        self.id = id

        layers = [nn.Linear(self.dim, self.hidden_dim, bias=bias), nn.ReLU()]
        for _ in range(self.depth - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.dim, bias=bias))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs, adj):
        del adj
        if inputs.dim() not in [2, 3]:
            raise ValueError(f"Unsupported input shape: {inputs.shape}")
        return torch.tanh(self.mlp(inputs))
