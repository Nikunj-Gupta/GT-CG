import itertools

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from components.action_selectors import REGISTRY as action_REGISTRY
from components.magic_gat_module import MagicGraphAttention


class MAGICMAC:
    """PyMARL-style controller for MAGIC (two-round graph-attention communication)."""

    def __init__(self, scheme, groups, args):
        del groups
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        obs_shape = scheme["obs"]["vshape"]
        input_dim = (
            obs_shape
            if isinstance(obs_shape, int)
            else int(th.tensor(obs_shape).prod().item())
        )
        if args.obs_last_action:
            input_dim += args.n_actions
        if args.obs_agent_id:
            input_dim += args.n_agents

        self.hidden_dim = getattr(args, "magic_hidden_dim", 128)
        self.gat_hidden_dim = getattr(args, "magic_gat_hidden_dim", 128)
        self.gat_heads = getattr(args, "magic_gat_heads", 4)
        self.gat_out_heads = getattr(args, "magic_gat_out_heads", 4)
        self.use_gat_encoder = getattr(args, "magic_use_gat_encoder", False)
        self.encoder_out_dim = getattr(args, "magic_gat_encoder_out_dim", 64)
        self.directed = getattr(args, "magic_directed", True)
        self.first_graph_complete = getattr(args, "magic_first_graph_complete", False)
        self.second_graph_complete = getattr(args, "magic_second_graph_complete", False)
        self.learn_second_graph = getattr(args, "magic_learn_second_graph", True)

        self.obs_encoder = nn.Linear(input_dim, self.hidden_dim)
        self.lstm_cell = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.sub_processor1 = MagicGraphAttention(
            self.hidden_dim,
            self.gat_hidden_dim,
            num_heads=self.gat_heads,
            self_loop_type=getattr(args, "magic_self_loop_type1", 2),
            average=False,
            normalize=getattr(args, "magic_first_gat_normalize", False),
        )
        self.sub_processor2 = MagicGraphAttention(
            self.gat_hidden_dim * self.gat_heads,
            self.hidden_dim,
            num_heads=self.gat_out_heads,
            self_loop_type=getattr(args, "magic_self_loop_type2", 2),
            average=True,
            normalize=getattr(args, "magic_second_gat_normalize", False),
        )

        if self.use_gat_encoder:
            self.gat_encoder = MagicGraphAttention(
                self.hidden_dim,
                self.encoder_out_dim,
                num_heads=getattr(args, "magic_gat_encoder_heads", 2),
                self_loop_type=1,
                average=True,
                normalize=getattr(args, "magic_gat_encoder_normalize", False),
            )
            scheduler_input_dim = self.encoder_out_dim
        else:
            self.gat_encoder = None
            scheduler_input_dim = self.hidden_dim

        if not self.first_graph_complete:
            self.sub_scheduler_mlp1 = self._scheduler_mlp(scheduler_input_dim)
        else:
            self.sub_scheduler_mlp1 = None

        if self.learn_second_graph and not self.second_graph_complete:
            self.sub_scheduler_mlp2 = self._scheduler_mlp(scheduler_input_dim)
        else:
            self.sub_scheduler_mlp2 = None

        self.message_encoder = (
            nn.Linear(self.hidden_dim, self.hidden_dim)
            if getattr(args, "magic_message_encoder", False)
            else None
        )
        self.message_decoder = (
            nn.Linear(self.hidden_dim, self.hidden_dim)
            if getattr(args, "magic_message_decoder", False)
            else None
        )

        self.action_head = nn.Linear(2 * self.hidden_dim, self.n_actions)
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        q_values = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action(
            q_values[bs], avail_actions[bs], t_env, test_mode=test_mode
        )

    def forward(self, ep_batch, t, test_mode=False):
        del test_mode
        bs = ep_batch.batch_size
        inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        base_graph = ep_batch["graph"].float()
        if base_graph.dim() == 2:
            base_graph = base_graph.unsqueeze(0).expand(bs, -1, -1)

        if self.hidden_states is None:
            self.init_hidden(bs)
        prev_h, prev_c = self.hidden_states

        encoded_obs = self.obs_encoder(inputs).reshape(bs * self.n_agents, self.hidden_dim)
        h, c = self.lstm_cell(encoded_obs, (prev_h, prev_c))
        hidden = h.view(bs, self.n_agents, self.hidden_dim)
        comm = hidden

        if self.message_encoder is not None:
            comm = self.message_encoder(comm)

        scheduler_state = comm
        if self.use_gat_encoder:
            scheduler_state = self.gat_encoder(comm, base_graph)

        if self.first_graph_complete:
            adj1 = base_graph
        else:
            adj1 = self._sub_scheduler(self.sub_scheduler_mlp1, scheduler_state, base_graph)

        comm1 = F.elu(self.sub_processor1(comm, adj1))

        if self.learn_second_graph and not self.second_graph_complete:
            adj2 = self._sub_scheduler(
                self.sub_scheduler_mlp2, scheduler_state, base_graph
            )
        elif (not self.learn_second_graph) and (not self.second_graph_complete):
            adj2 = adj1
        else:
            adj2 = base_graph

        comm2 = self.sub_processor2(comm1, adj2)

        if self.message_decoder is not None:
            comm2 = self.message_decoder(comm2)

        q_inputs = th.cat([hidden, comm2], dim=-1)
        q_values = self.action_head(q_inputs)

        if self.args.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                flat = q_values.reshape(bs * self.n_agents, -1)
                flat_avail = avail_actions.reshape(bs * self.n_agents, -1)
                flat[flat_avail == 0] = -1e10
                q_values = flat.view(bs, self.n_agents, -1)
            q_values = th.softmax(q_values, dim=-1)

        self.hidden_states = (h, c)
        return q_values

    def init_hidden(self, batch_size):
        device = next(self.obs_encoder.parameters()).device
        h = th.zeros(batch_size * self.n_agents, self.hidden_dim, device=device)
        c = th.zeros_like(h)
        self.hidden_states = (h, c)

    def parameters(self):
        modules = [
            self.obs_encoder,
            self.lstm_cell,
            self.sub_processor1,
            self.sub_processor2,
            self.action_head,
        ]
        if self.gat_encoder is not None:
            modules.append(self.gat_encoder)
        if self.sub_scheduler_mlp1 is not None:
            modules.append(self.sub_scheduler_mlp1)
        if self.sub_scheduler_mlp2 is not None:
            modules.append(self.sub_scheduler_mlp2)
        if self.message_encoder is not None:
            modules.append(self.message_encoder)
        if self.message_decoder is not None:
            modules.append(self.message_decoder)
        return itertools.chain.from_iterable(module.parameters() for module in modules)

    def load_state(self, other_mac):
        self.obs_encoder.load_state_dict(other_mac.obs_encoder.state_dict())
        self.lstm_cell.load_state_dict(other_mac.lstm_cell.state_dict())
        self.sub_processor1.load_state_dict(other_mac.sub_processor1.state_dict())
        self.sub_processor2.load_state_dict(other_mac.sub_processor2.state_dict())
        self.action_head.load_state_dict(other_mac.action_head.state_dict())
        if self.gat_encoder is not None and other_mac.gat_encoder is not None:
            self.gat_encoder.load_state_dict(other_mac.gat_encoder.state_dict())
        if self.sub_scheduler_mlp1 is not None and other_mac.sub_scheduler_mlp1 is not None:
            self.sub_scheduler_mlp1.load_state_dict(other_mac.sub_scheduler_mlp1.state_dict())
        if self.sub_scheduler_mlp2 is not None and other_mac.sub_scheduler_mlp2 is not None:
            self.sub_scheduler_mlp2.load_state_dict(other_mac.sub_scheduler_mlp2.state_dict())
        if self.message_encoder is not None and other_mac.message_encoder is not None:
            self.message_encoder.load_state_dict(other_mac.message_encoder.state_dict())
        if self.message_decoder is not None and other_mac.message_decoder is not None:
            self.message_decoder.load_state_dict(other_mac.message_decoder.state_dict())

    def cuda(self):
        self.obs_encoder.cuda()
        self.lstm_cell.cuda()
        self.sub_processor1.cuda()
        self.sub_processor2.cuda()
        self.action_head.cuda()
        if self.gat_encoder is not None:
            self.gat_encoder.cuda()
        if self.sub_scheduler_mlp1 is not None:
            self.sub_scheduler_mlp1.cuda()
        if self.sub_scheduler_mlp2 is not None:
            self.sub_scheduler_mlp2.cuda()
        if self.message_encoder is not None:
            self.message_encoder.cuda()
        if self.message_decoder is not None:
            self.message_decoder.cuda()

    def save_models(self, path):
        th.save(self.obs_encoder.state_dict(), f"{path}/magic_obs_encoder.th")
        th.save(self.lstm_cell.state_dict(), f"{path}/magic_lstm.th")
        th.save(self.sub_processor1.state_dict(), f"{path}/magic_sub_processor1.th")
        th.save(self.sub_processor2.state_dict(), f"{path}/magic_sub_processor2.th")
        th.save(self.action_head.state_dict(), f"{path}/magic_action_head.th")
        if self.gat_encoder is not None:
            th.save(self.gat_encoder.state_dict(), f"{path}/magic_gat_encoder.th")
        if self.sub_scheduler_mlp1 is not None:
            th.save(self.sub_scheduler_mlp1.state_dict(), f"{path}/magic_sub_scheduler1.th")
        if self.sub_scheduler_mlp2 is not None:
            th.save(self.sub_scheduler_mlp2.state_dict(), f"{path}/magic_sub_scheduler2.th")
        if self.message_encoder is not None:
            th.save(self.message_encoder.state_dict(), f"{path}/magic_message_encoder.th")
        if self.message_decoder is not None:
            th.save(self.message_decoder.state_dict(), f"{path}/magic_message_decoder.th")

    def load_models(self, path):
        self.obs_encoder.load_state_dict(
            th.load(f"{path}/magic_obs_encoder.th", map_location=lambda s, l: s)
        )
        self.lstm_cell.load_state_dict(
            th.load(f"{path}/magic_lstm.th", map_location=lambda s, l: s)
        )
        self.sub_processor1.load_state_dict(
            th.load(f"{path}/magic_sub_processor1.th", map_location=lambda s, l: s)
        )
        self.sub_processor2.load_state_dict(
            th.load(f"{path}/magic_sub_processor2.th", map_location=lambda s, l: s)
        )
        self.action_head.load_state_dict(
            th.load(f"{path}/magic_action_head.th", map_location=lambda s, l: s)
        )
        if self.gat_encoder is not None:
            self.gat_encoder.load_state_dict(
                th.load(f"{path}/magic_gat_encoder.th", map_location=lambda s, l: s)
            )
        if self.sub_scheduler_mlp1 is not None:
            self.sub_scheduler_mlp1.load_state_dict(
                th.load(f"{path}/magic_sub_scheduler1.th", map_location=lambda s, l: s)
            )
        if self.sub_scheduler_mlp2 is not None:
            self.sub_scheduler_mlp2.load_state_dict(
                th.load(f"{path}/magic_sub_scheduler2.th", map_location=lambda s, l: s)
            )
        if self.message_encoder is not None:
            self.message_encoder.load_state_dict(
                th.load(f"{path}/magic_message_encoder.th", map_location=lambda s, l: s)
            )
        if self.message_decoder is not None:
            self.message_decoder.load_state_dict(
                th.load(f"{path}/magic_message_decoder.th", map_location=lambda s, l: s)
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
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )
        return th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=-1)

    def _scheduler_mlp(self, state_dim):
        hidden = max(16, state_dim // 2)
        hidden2 = max(8, hidden // 2)
        return nn.Sequential(
            nn.Linear(state_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 2),
        )

    def _sub_scheduler(self, scheduler, hidden_state, base_graph):
        """
        hidden_state: [bs, n, d]
        base_graph: [bs, n, n]
        """
        bs, n, d = hidden_state.shape
        hi = hidden_state.unsqueeze(2).expand(-1, -1, n, -1)
        hj = hidden_state.unsqueeze(1).expand(-1, n, -1, -1)
        pair = th.cat([hi, hj], dim=-1)
        logits = scheduler(pair)
        if not self.directed:
            logits = 0.5 * (logits + logits.transpose(1, 2))
        edge_choice = F.gumbel_softmax(logits, hard=True, dim=-1)[..., 1]
        return edge_choice * base_graph
