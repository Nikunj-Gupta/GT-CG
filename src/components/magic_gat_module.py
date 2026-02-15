import torch
import torch.nn as nn
import torch.nn.functional as F


class MagicGraphAttention(nn.Module):
    """Batched graph-attention layer used by MAGIC communication."""

    def __init__(
        self,
        in_features,
        out_features,
        dropout=0.0,
        negative_slope=0.2,
        num_heads=1,
        bias=True,
        self_loop_type=2,
        average=False,
        normalize=False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_heads = num_heads
        self.self_loop_type = self_loop_type
        self.average = average
        self.normalize = normalize

        self.w = nn.Parameter(torch.zeros(in_features, num_heads * out_features))
        self.a_i = nn.Parameter(torch.zeros(num_heads, out_features, 1))
        self.a_j = nn.Parameter(torch.zeros(num_heads, out_features, 1))
        if bias:
            bias_dim = out_features if average else num_heads * out_features
            self.bias = nn.Parameter(torch.zeros(bias_dim))
        else:
            self.register_parameter("bias", None)
        self.leaky_relu = nn.LeakyReLU(self.negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.w, gain=gain)
        nn.init.xavier_normal_(self.a_i, gain=gain)
        nn.init.xavier_normal_(self.a_j, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _apply_self_loops(self, adj):
        bs, n_agents, _ = adj.shape
        eye = torch.eye(n_agents, device=adj.device).unsqueeze(0).expand(bs, -1, -1)
        if self.self_loop_type == 0:
            adj = adj * (1.0 - eye)
        elif self.self_loop_type == 1:
            adj = eye + adj * (1.0 - eye)
        return adj

    def forward(self, inputs, adj):
        """
        inputs: [bs, n_agents, in_features]
        adj: [bs, n_agents, n_agents]
        """
        bs, n_agents, _ = inputs.shape
        adj = self._apply_self_loops(adj.float())

        h = torch.matmul(inputs, self.w).view(
            bs, n_agents, self.num_heads, self.out_features
        )
        coeff_i = torch.einsum("bnhd,hdo->bnho", h, self.a_i).squeeze(-1)
        coeff_j = torch.einsum("bnhd,hdo->bnho", h, self.a_j).squeeze(-1)

        e = self.leaky_relu(
            coeff_i.unsqueeze(2) + coeff_j.unsqueeze(1)
        )  # [bs, n, n, heads]
        adj_heads = adj.unsqueeze(-1).expand(-1, -1, -1, self.num_heads)
        masked_e = e.masked_fill(adj_heads <= 0, -1e9)
        attention = F.softmax(masked_e, dim=2)
        attention = attention * adj_heads

        if self.normalize:
            attention = attention + 1e-15
            denom = attention.sum(dim=2, keepdim=True).clamp(min=1e-8)
            attention = (attention / denom) * adj_heads

        attention = F.dropout(attention, self.dropout, training=self.training)

        outputs = []
        for head in range(self.num_heads):
            h_head = h[:, :, head, :]  # [bs, n, out]
            att_head = attention[:, :, :, head]  # [bs, n, n]
            outputs.append(torch.matmul(att_head, h_head))

        if self.average:
            out = torch.mean(torch.stack(outputs, dim=-1), dim=-1)
        else:
            out = torch.cat(outputs, dim=-1)

        if self.bias is not None:
            out = out + self.bias
        return out
