import numpy as np
import torch as th

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.dicg import DICGCategoricalPolicy


class DICGMAC:
    """Multi-agent controller for Deep Implicit Coordination Graphs."""

    def __init__(self, scheme, groups, args):
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

        self.policy = DICGCategoricalPolicy(
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

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_state = None

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
        agent_outs, self.hidden_state = self.policy(obs_in, avail_actions, self.hidden_state)
        return agent_outs

    def init_hidden(self, batch_size):
        device = next(self.policy.parameters()).device
        h = th.zeros(1, batch_size * self.n_agents, self.policy.hidden_size, device=device)
        c = th.zeros_like(h)
        self.hidden_state = (h, c)

    def parameters(self):
        return self.policy.parameters()

    def load_state(self, other_mac):
        self.policy.load_state_dict(other_mac.policy.state_dict())

    def cuda(self):
        self.policy.cuda()

    def save_models(self, path):
        th.save(self.policy.state_dict(), f"{path}/dicg_policy.th")

    def load_models(self, path):
        self.policy.load_state_dict(
            th.load(f"{path}/dicg_policy.th", map_location=lambda storage, loc: storage)
        )
