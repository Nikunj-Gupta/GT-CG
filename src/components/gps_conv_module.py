import inspect
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GPSConv
    from torch_geometric.utils import dense_to_sparse
except ImportError as exc:  # pragma: no cover - runtime dependency
    GPSConv = None
    dense_to_sparse = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class GPSConvModule(nn.Module):
    """GPSConv wrapper that accepts dense adjacency."""

    def __init__(self, in_features, out_features, heads=1, dropout=0.0, bias=True, id=None):
        super().__init__()
        if GPSConv is None:  # pragma: no cover - runtime dependency
            raise ImportError("torch_geometric is required for GPSConvModule") from _IMPORT_ERROR
        self.in_features = in_features
        self.out_features = out_features
        self.id = id
        conv_kwargs = {
            "channels": out_features,
            "conv": None,
            "heads": heads,
            "dropout": dropout,
        }
        try:
            sig = inspect.signature(GPSConv.__init__)
            if "attn_dropout" in sig.parameters:
                conv_kwargs["attn_dropout"] = dropout
        except (TypeError, ValueError):
            pass
        self.conv = GPSConv(**conv_kwargs)
        if in_features != out_features:
            self.project = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.project = nn.Identity()

    def _forward_single(self, x, adj):
        edge_index, _ = dense_to_sparse(adj)
        x = self.project(x)
        if edge_index.numel() == 0:
            return torch.tanh(x)
        return torch.tanh(self.conv(x, edge_index))

    def forward(self, inputs, adj):
        if inputs.dim() == 2:
            return self._forward_single(inputs, adj)
        if inputs.dim() == 3:
            outputs = []
            for b in range(inputs.size(0)):
                outputs.append(self._forward_single(inputs[b], adj[b]))
            return torch.stack(outputs, dim=0)
        raise ValueError(f"Unsupported input shape: {inputs.shape}")
