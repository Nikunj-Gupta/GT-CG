import numpy as np
import torch as th
from torch import nn

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.dicg import DICGSingleAgentPolicy


class DICGNonSharedMAC:
    """Non-shared variant of the DICG controller (separate params per agent)."""

    def __init__(self, scheme, groups, args):
        del groups  # unused
        self.n_agents = args.n_agents
        self.args = args

        obs_shape = scheme["obs"]["vshape"]
        if isinstance(obs_shape, (list, tuple)):
            base_obs_dim = int(np.prod(obs_shape))
        else:
            base_obs_dim = obs_shape

        input_dim = base_obs_dim
        if args.obs_last_action:
            input_dim += args.n_actions
        if args.obs_agent_id:
            input_dim += args.n_agents

        self.policies = nn.ModuleList(
            [
                DICGSingleAgentPolicy(
                    agent_id=i,
                    input_dim=input_dim,
                    n_agents=args.n_agents,
                    n_actions=args.n_actions,
                    embedding_dim=getattr(args, "dicg_embedding_dim", 128),
                    encoder_hidden_sizes=getattr(args, "dicg_encoder_hidden_sizes", [128]),
                    attention_type=getattr(args, "dicg_attention_type", "general"),
                    n_gcn_layers=getattr(args, "dicg_n_gcn_layers", 2),
                    gcn_bias=getattr(args, "dicg_gcn_bias", True),
                    lstm_hidden_size=getattr(args, "dicg_lstm_hidden_size", 128),
                    residual=getattr(args, "dicg_residual", True),
                )
                for i in range(self.n_agents)
            ]
        )

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = [None for _ in range(self.n_agents)]

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        del test_mode  # unused
        obs = ep_batch["obs"][:, t]
        inputs = [obs]

        if self.args.obs_last_action:
            if t == 0:
                last_action = th.zeros_like(ep_batch["actions_onehot"][:, t])
            else:
                last_action = ep_batch["actions_onehot"][:, t - 1]
            inputs.append(last_action)

        if self.args.obs_agent_id:
            agent_ids = (
                th.eye(self.n_agents, device=obs.device)
                .unsqueeze(0)
                .expand(obs.shape[0], -1, -1)
            )
            inputs.append(agent_ids)

        obs_in = th.cat(inputs, dim=-1)
        avail_actions = ep_batch["avail_actions"][:, t]

        outputs = []
        next_hidden = []
        for i, policy in enumerate(self.policies):
            probs, hid = policy(obs_in, avail_actions, self.hidden_states[i])
            outputs.append(probs.unsqueeze(1))
            next_hidden.append(hid)

        self.hidden_states = next_hidden
        return th.cat(outputs, dim=1)

    def init_hidden(self, batch_size):
        device = next(self.policies.parameters()).device
        self.hidden_states = []
        for policy in self.policies:
            h = th.zeros(1, batch_size, policy.hidden_size, device=device)
            c = th.zeros_like(h)
            self.hidden_states.append((h, c))

    def parameters(self):
        return self.policies.parameters()

    def load_state(self, other_mac):
        self.policies.load_state_dict(other_mac.policies.state_dict())

    def cuda(self):
        self.policies.cuda()

    def save_models(self, path):
        th.save(self.policies.state_dict(), f"{path}/dicg_policy_ns.th")

    def load_models(self, path):
        self.policies.load_state_dict(
            th.load(f"{path}/dicg_policy_ns.th", map_location=lambda storage, loc: storage)
        )
