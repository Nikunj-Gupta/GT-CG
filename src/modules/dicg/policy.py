from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from .layers import AttentionModule, CategoricalLSTMModule, GraphConvolutionModule, MLPEncoder


class DICGBase(nn.Module):
    """Backbone shared by DICG policies."""

    def __init__(
        self,
        input_dim: int,
        n_agents: int,
        embedding_dim: int,
        encoder_hidden_sizes: Sequence[int],
        attention_type: str,
        n_gcn_layers: int,
        gcn_bias: bool,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.encoder = MLPEncoder(
            input_dim=input_dim,
            output_dim=embedding_dim,
            hidden_sizes=encoder_hidden_sizes,
            output_nonlinearity=torch.tanh,
        )
        self.attention_layer = AttentionModule(dimensions=embedding_dim, attention_type=attention_type)
        self.gcn_layers = nn.ModuleList(
            [
                GraphConvolutionModule(
                    in_features=embedding_dim,
                    out_features=embedding_dim,
                    bias=gcn_bias,
                )
                for _ in range(n_gcn_layers)
            ]
        )

    def forward(self, obs_n: torch.Tensor):
        embeddings_0 = self.encoder(obs_n)
        attention_weights = self.attention_layer(embeddings_0)
        embeddings_collection = [embeddings_0]
        for gcn_layer in self.gcn_layers:
            embeddings_collection.append(gcn_layer(embeddings_collection[-1], attention_weights))
        return embeddings_collection, attention_weights


class DICGCategoricalPolicy(DICGBase):
    """DICG categorical policy adapted to PyMARL."""

    def __init__(
        self,
        input_dim: int,
        n_agents: int,
        n_actions: int,
        embedding_dim: int,
        encoder_hidden_sizes: Sequence[int],
        attention_type: str,
        n_gcn_layers: int,
        gcn_bias: bool,
        lstm_hidden_size: int,
        residual: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            n_agents=n_agents,
            embedding_dim=embedding_dim,
            encoder_hidden_sizes=encoder_hidden_sizes,
            attention_type=attention_type,
            n_gcn_layers=n_gcn_layers,
            gcn_bias=gcn_bias,
        )
        self.residual = residual
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.lstm_head = CategoricalLSTMModule(
            input_size=embedding_dim,
            output_size=n_actions,
            hidden_size=lstm_hidden_size,
        )

    def forward(
        self,
        obs_n: torch.Tensor,
        avail_actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ):
        embeddings_collection, _ = super().forward(obs_n)
        if self.residual:
            inputs = embeddings_collection[0] + embeddings_collection[-1]
        else:
            inputs = embeddings_collection[-1]

        batch_size = obs_n.shape[0]
        inputs = inputs.reshape(1, batch_size * self.n_agents, self.embedding_dim)
        if hidden_state is None:
            dist, next_h, next_c = self.lstm_head(inputs)
        else:
            dist, next_h, next_c = self.lstm_head(inputs, hidden_state[0], hidden_state[1])

        probs = dist.probs.reshape(batch_size, self.n_agents, self.n_actions)

        # Mask unavailable actions and re-normalize
        avail_actions = avail_actions.float()
        probs = probs * avail_actions
        norm = probs.sum(dim=-1, keepdim=True)
        probs = probs / norm.clamp(min=1e-8)

        zero_mask = norm.squeeze(-1) <= 0
        if zero_mask.any():
            uniform = avail_actions / avail_actions.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            fallback = torch.full_like(uniform, 1.0 / self.n_actions)
            uniform = torch.where(avail_actions > 0, uniform, fallback)
            probs = torch.where(zero_mask.unsqueeze(-1), uniform, probs)
        return probs, (next_h, next_c)

    @property
    def hidden_size(self) -> int:
        return self.lstm_head.hidden_size


class DICGSingleAgentPolicy(nn.Module):
    """DICG policy with per-agent parameters (one instance per agent)."""

    def __init__(
        self,
        agent_id: int,
        input_dim: int,
        n_agents: int,
        n_actions: int,
        embedding_dim: int,
        encoder_hidden_sizes: Sequence[int],
        attention_type: str,
        n_gcn_layers: int,
        gcn_bias: bool,
        lstm_hidden_size: int,
        residual: bool = True,
    ):
        super().__init__()
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.embedding_dim = embedding_dim
        self.residual = residual

        self.dicg_encoder = DICGBase(
            input_dim=input_dim,
            n_agents=n_agents,
            embedding_dim=embedding_dim,
            encoder_hidden_sizes=encoder_hidden_sizes,
            attention_type=attention_type,
            n_gcn_layers=n_gcn_layers,
            gcn_bias=gcn_bias,
        )
        self.lstm_head = CategoricalLSTMModule(
            input_size=embedding_dim,
            output_size=n_actions,
            hidden_size=lstm_hidden_size,
        )

    def forward(
        self,
        obs_n: torch.Tensor,
        avail_actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ):
        embeddings_collection, _ = self.dicg_encoder(obs_n)
        if self.residual:
            embeddings = embeddings_collection[0] + embeddings_collection[-1]
        else:
            embeddings = embeddings_collection[-1]

        agent_embedding = embeddings[:, self.agent_id]
        batch_size = obs_n.shape[0]
        inputs = agent_embedding.reshape(1, batch_size, self.embedding_dim)

        if hidden_state is None:
            dist, next_h, next_c = self.lstm_head(inputs)
        else:
            dist, next_h, next_c = self.lstm_head(inputs, hidden_state[0], hidden_state[1])

        probs = dist.probs.reshape(batch_size, self.n_actions)

        # Mask unavailable actions and re-normalize
        avail = avail_actions[:, self.agent_id].float()
        probs = probs * avail
        norm = probs.sum(dim=-1, keepdim=True)
        probs = probs / norm.clamp(min=1e-8)

        zero_mask = norm.squeeze(-1) <= 0
        if zero_mask.any():
            uniform = avail / avail.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            fallback = torch.full_like(uniform, 1.0 / self.n_actions)
            uniform = torch.where(avail > 0, uniform, fallback)
            probs = torch.where(zero_mask.unsqueeze(-1), uniform, probs)

        return probs, (next_h, next_c)

    @property
    def hidden_size(self) -> int:
        return self.lstm_head.hidden_size
