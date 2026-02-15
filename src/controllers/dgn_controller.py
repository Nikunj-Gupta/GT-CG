import itertools

import torch as th
import torch.nn as nn

from components.dgn_module import DGNRelationLayer
from components.action_selectors import REGISTRY as action_REGISTRY


class DGNMAC:
    """Multi-agent controller implementing DGN (encoder + 2 relation layers + Q head)."""

    def __init__(self, scheme, groups, args):
        del groups
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        obs_shape = scheme["obs"]["vshape"]
        input_dim = obs_shape if isinstance(obs_shape, int) else int(th.tensor(obs_shape).prod().item())
        if args.obs_last_action:
            input_dim += args.n_actions
        if args.obs_agent_id:
            input_dim += args.n_agents

        self.embed_dim = getattr(args, "dgn_embed_dim", 128)
        self.encoder_hidden = getattr(args, "dgn_encoder_hidden", [128, 128])
        self.n_heads = getattr(args, "dgn_n_heads", 8)

        self.encoder = self._mlp(input_dim, self.encoder_hidden, self.embed_dim)
        self.relation1 = DGNRelationLayer(self.embed_dim, n_heads=self.n_heads)
        self.relation2 = DGNRelationLayer(self.embed_dim, n_heads=self.n_heads)
        self.q_head = nn.Linear(self.embed_dim * 3, args.n_actions)

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        del test_mode
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        graph = ep_batch["graph"].float()

        if graph.dim() == 2:
            graph = graph.unsqueeze(0).expand(ep_batch.batch_size, -1, -1)

        eye = th.eye(self.n_agents, device=graph.device).unsqueeze(0)
        graph = th.clamp(graph + eye, max=1.0)

        features = self.encoder(agent_inputs)
        relation1 = self.relation1(features, graph)
        relation2 = self.relation2(relation1, graph)
        q_inputs = th.cat([features, relation1, relation2], dim=-1)
        q_values = self.q_head(q_inputs)

        if self.args.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                q_values = q_values.view(ep_batch.batch_size * self.n_agents, -1)
                q_values[reshaped_avail_actions == 0] = -1e10
                q_values = q_values.view(ep_batch.batch_size, self.n_agents, -1)
            q_values = th.nn.functional.softmax(q_values, dim=-1)

        return q_values

    def init_hidden(self, batch_size):
        del batch_size
        self.hidden_states = None

    def parameters(self):
        return itertools.chain(
            self.encoder.parameters(),
            self.relation1.parameters(),
            self.relation2.parameters(),
            self.q_head.parameters(),
        )

    def load_state(self, other_mac):
        self.encoder.load_state_dict(other_mac.encoder.state_dict())
        self.relation1.load_state_dict(other_mac.relation1.state_dict())
        self.relation2.load_state_dict(other_mac.relation2.state_dict())
        self.q_head.load_state_dict(other_mac.q_head.state_dict())

    def cuda(self):
        self.encoder.cuda()
        self.relation1.cuda()
        self.relation2.cuda()
        self.q_head.cuda()

    def save_models(self, path):
        th.save(self.encoder.state_dict(), f"{path}/dgn_encoder.th")
        th.save(self.relation1.state_dict(), f"{path}/dgn_relation1.th")
        th.save(self.relation2.state_dict(), f"{path}/dgn_relation2.th")
        th.save(self.q_head.state_dict(), f"{path}/dgn_q_head.th")

    def load_models(self, path):
        self.encoder.load_state_dict(
            th.load(f"{path}/dgn_encoder.th", map_location=lambda storage, loc: storage)
        )
        self.relation1.load_state_dict(
            th.load(f"{path}/dgn_relation1.th", map_location=lambda storage, loc: storage)
        )
        self.relation2.load_state_dict(
            th.load(f"{path}/dgn_relation2.th", map_location=lambda storage, loc: storage)
        )
        self.q_head.load_state_dict(
            th.load(f"{path}/dgn_q_head.th", map_location=lambda storage, loc: storage)
        )

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            )
        return th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)

    @staticmethod
    def _mlp(input_dim, hidden_dims, output_dim):
        hidden_dims = [] if hidden_dims is None else hidden_dims
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        layers = []
        dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)
