import math

import torch
import torch.nn as nn


class DGNRelationLayer(nn.Module):
    """DGN relation layer: masked multi-head self-attention over agents."""

    def __init__(self, embed_dim, n_heads=8, bias=True):
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, inputs, adj):
        """
        inputs: [bs, n_agents, embed_dim]
        adj: [bs, n_agents, n_agents] (0/1 or weighted)
        """
        bs, n_agents, _ = inputs.shape

        q = self.q_proj(inputs).view(bs, n_agents, self.n_heads, self.head_dim)
        k = self.k_proj(inputs).view(bs, n_agents, self.n_heads, self.head_dim)
        v = self.v_proj(inputs).view(bs, n_agents, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)  # [bs, h, n, d]
        k = k.transpose(1, 2)  # [bs, h, n, d]
        v = v.transpose(1, 2)  # [bs, h, n, d]

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask = adj.unsqueeze(1) > 0
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn = torch.softmax(attn_logits, dim=-1)

        outputs = torch.matmul(attn, v)  # [bs, h, n, d]
        outputs = outputs.transpose(1, 2).contiguous().view(bs, n_agents, self.embed_dim)
        outputs = self.out_proj(outputs)
        return torch.relu(outputs)
