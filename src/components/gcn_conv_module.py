import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import dense_to_sparse
except ImportError as exc:  # pragma: no cover - runtime dependency
    GCNConv = None
    dense_to_sparse = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class GCNConvModule(nn.Module):
    """GCNConv wrapper that accepts dense adjacency."""

    def __init__(self, in_features, out_features, bias=True, id=None):
        super().__init__()
        if GCNConv is None:  # pragma: no cover - runtime dependency
            raise ImportError("torch_geometric is required for GCNConvModule") from _IMPORT_ERROR
        self.in_features = in_features
        self.out_features = out_features
        self.id = id
        self.conv = GCNConv(
            in_channels=in_features,
            out_channels=out_features,
            bias=bias,
        )

    def _forward_single(self, x, adj):
        edge_index, _ = dense_to_sparse(adj)
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
