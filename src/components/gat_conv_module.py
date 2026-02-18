import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import dense_to_sparse
except ImportError as exc:  # pragma: no cover - runtime dependency
    GATConv = None
    dense_to_sparse = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class GATConvModule(nn.Module):
    """GATConv wrapper that accepts dense adjacency."""

    def __init__(self, in_features, out_features, heads=1, concat=False, bias=True, id=None):
        super().__init__()
        if GATConv is None:  # pragma: no cover - runtime dependency
            raise ImportError("torch_geometric is required for GATConvModule") from _IMPORT_ERROR
        self.in_features = in_features
        self.out_features = out_features
        self.id = id

        conv_out_features = out_features if not concat else max(1, out_features // heads)
        self.conv = GATConv(
            in_channels=in_features,
            out_channels=conv_out_features,
            heads=heads,
            concat=concat,
            bias=bias,
        )

        output_dim = conv_out_features * heads if concat else conv_out_features
        if output_dim != out_features:
            self.output_projector = nn.Linear(output_dim, out_features, bias=bias)
        else:
            self.output_projector = nn.Identity()

    def _forward_single(self, x, adj):
        edge_index, _ = dense_to_sparse(adj)
        if edge_index.numel() == 0:
            return torch.tanh(x)
        out = self.conv(x, edge_index)
        out = self.output_projector(out)
        return torch.tanh(out)

    def forward(self, inputs, adj):
        if inputs.dim() == 2:
            return self._forward_single(inputs, adj)
        if inputs.dim() == 3:
            outputs = []
            for b in range(inputs.size(0)):
                outputs.append(self._forward_single(inputs[b], adj[b]))
            return torch.stack(outputs, dim=0)
        raise ValueError(f"Unsupported input shape: {inputs.shape}")
