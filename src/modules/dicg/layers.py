import math
from typing import Iterable, Optional, Tuple
from typing import Callable

import torch
from torch import nn
from torch.distributions import Categorical

class _Lambda(nn.Module):
    """Wrap a callable inside an nn.Module for use in Sequential."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class MLPEncoder(nn.Module):
    """Simple MLP encoder used by DICG modules."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Iterable[int],
        output_nonlinearity: Optional[Callable[[torch.Tensor], torch.Tensor]] = nn.Tanh(),
    ):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.Tanh())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        if output_nonlinearity is not None:
            if isinstance(output_nonlinearity, nn.Module):
                layers.append(output_nonlinearity)
            else:
                layers.append(_Lambda(output_nonlinearity))
        self.model = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        leading_shape = inputs.shape[:-1]
        flat = inputs.reshape(-1, inputs.shape[-1])
        encoded = self.model(flat)
        return encoded.reshape(*leading_shape, self.output_dim)


class AttentionModule(nn.Module):
    """Self-attention over agent embeddings."""

    def __init__(self, dimensions: int, attention_type: str = "general"):
        super().__init__()
        self.attention_type = attention_type
        if attention_type == "general":
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        elif attention_type == "diff":
            self.linear_in = nn.Linear(dimensions, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        if self.attention_type in ["general", "dot"]:
            context = query.transpose(-2, -1).contiguous()
            if self.attention_type == "general":
                query = self.linear_in(query)
            attention_scores = torch.matmul(query, context)
            attention_weights = self.softmax(attention_scores)
        elif self.attention_type == "diff":
            n_agents = query.shape[-2]
            repeats = (1,) * (len(query.shape) - 2) + (n_agents, 1)
            augmented_shape = query.shape[:-1] + (n_agents,) + query.shape[-1:]
            query = query.repeat(*repeats).reshape(*augmented_shape)
            context = query.transpose(-3, -2).contiguous()
            attention_scores = torch.abs(query - context)
            attention_scores = self.linear_in(attention_scores).squeeze(-1)
            attention_scores = torch.tanh(attention_scores)
            attention_weights = self.softmax(attention_scores)
        elif self.attention_type == "identity":
            n_agents = query.shape[-2]
            attention_weights = query.new_zeros(query.shape[:-2] + (n_agents, n_agents))
            idx = torch.arange(n_agents, device=query.device)
            attention_weights[..., idx, idx] = 1.0
        else:  # uniform
            n_agents = query.shape[-2]
            attention_weights = query.new_ones(query.shape[:-2] + (n_agents, n_agents))
            attention_weights = attention_weights / n_agents
        return attention_weights


class GraphConvolutionModule(nn.Module):
    """Graph convolution used in DICG."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(inputs, self.weight)
        outputs = torch.matmul(adj, support)
        if self.bias is not None:
            outputs = outputs + self.bias
        return torch.tanh(outputs)


class CategoricalLSTMModule(nn.Module):
    """LSTM head that outputs categorical action distributions."""

    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size

    def forward(
        self,
        inputs: torch.Tensor,
        prev_hidden_state: Optional[torch.Tensor] = None,
        prev_cell_state: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        if prev_cell_state is None:
            outputs, (next_hidden_state, next_cell_state) = self.lstm(inputs)
        else:
            outputs, (next_hidden_state, next_cell_state) = self.lstm(
                inputs, (prev_hidden_state, prev_cell_state)
            )
        logits = self.decoder(outputs)
        return Categorical(logits=logits), next_hidden_state, next_cell_state
