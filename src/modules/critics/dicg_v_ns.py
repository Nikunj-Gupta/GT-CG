import torch as th
import torch.nn as nn

from modules.dicg.policy import DICGBase
from modules.dicg.layers import MLPEncoder


class DICGCriticNS(nn.Module):
    """Non-shared centralized critic using the DICG encoder (per-agent heads)."""

    def __init__(self, scheme, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.output_type = "v"
        self.residual = getattr(args, "dicg_critic_residual", True)

        input_shape = self._get_input_shape(scheme)

        self.dicg_encoder = DICGBase(
            input_dim=input_shape,
            n_agents=args.n_agents,
            embedding_dim=getattr(args, "dicg_embedding_dim", 128),
            encoder_hidden_sizes=getattr(args, "dicg_encoder_hidden_sizes", [128]),
            attention_type=getattr(args, "dicg_attention_type", "general"),
            n_gcn_layers=getattr(args, "dicg_n_gcn_layers", 2),
            gcn_bias=getattr(args, "dicg_gcn_bias", True),
        )

        self.aggregators = nn.ModuleList(
            [
                MLPEncoder(
                    input_dim=getattr(args, "dicg_embedding_dim", 128),
                    output_dim=1,
                    hidden_sizes=getattr(args, "dicg_critic_hidden_sizes", [128]),
                    output_nonlinearity=None,
                )
                for _ in range(self.n_agents)
            ]
        )

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        flat_inputs = inputs.reshape(bs * max_t, self.n_agents, -1)

        embeddings_collection, _ = self.dicg_encoder(flat_inputs)
        if self.residual:
            embeddings = embeddings_collection[0] + embeddings_collection[-1]
        else:
            embeddings = embeddings_collection[-1]

        values = []
        for i in range(self.n_agents):
            v = self.aggregators[i](embeddings[:, i])
            values.append(v.view(bs, max_t, 1, 1))

        return th.cat(values, dim=2)

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []

        inputs.append(batch["state"][:, ts])

        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1))

        if self.args.obs_last_action:
            if t == 0:
                last_actions = th.zeros_like(batch["actions_onehot"][:, 0:1]).view(
                    bs, max_t, 1, -1
                )
            elif isinstance(t, int):
                last_actions = batch["actions_onehot"][:, slice(t - 1, t)].view(
                    bs, max_t, 1, -1
                )
            else:
                last_actions = th.cat(
                    [
                        th.zeros_like(batch["actions_onehot"][:, 0:1]),
                        batch["actions_onehot"][:, :-1],
                    ],
                    dim=1,
                )
                last_actions = last_actions.view(bs, max_t, 1, -1)
            inputs.append(last_actions)

        inputs = th.cat([x.reshape(bs * max_t, -1) for x in inputs], dim=1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        return input_shape
