import contextlib
import itertools

import torch as th
import torch.nn as nn

from modules.agents import REGISTRY as agent_REGISTRY
from .basic_controller import BasicMAC


class LowRankNSMAC(BasicMAC):
    """Low-rank joint Q approximation with non-shared agents."""

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.hidden_dim = getattr(args, "rnn_hidden_dim", args.hidden_dim)

        input_shape = self._get_input_shape(scheme)
        self.agents = nn.ModuleList(
            [agent_REGISTRY[self.args.agent](input_shape, self.args) for _ in range(self.n_agents)]
        )

        self.factor_fun = nn.ModuleList(
            [
                nn.Linear(self.hidden_dim, args.n_actions * args.low_rank)
                for _ in range(self.n_agents)
            ]
        )

        self.add_utilities = args.add_utilities
        if self.add_utilities:
            self.utility_fun = nn.ModuleList(
                [nn.Linear(self.hidden_dim, args.n_actions) for _ in range(self.n_agents)]
            )

        self.device = None
        self._set_idx(args)

    def _set_idx(self, args):
        self.device = next(self.parameters()).device
        idx = th.arange(1, args.n_agents, device=self.device).unsqueeze(0)
        self.idx = idx.repeat(args.n_agents, 1)
        for i in range(1, args.n_agents):
            self.idx[i, :i] = th.arange(i, device=self.device)

    def forward(self, ep_batch, t, actions=None, policy_mode=True, test_mode=False, compute_grads=False):
        del test_mode  # unused
        with th.no_grad() if not compute_grads or policy_mode else contextlib.suppress():
            agent_inputs = self._build_inputs(ep_batch, t).view(
                ep_batch.batch_size, self.n_agents, -1
            )
            avail_actions = ep_batch["avail_actions"][:, t]

            hidden_list = []
            for i, agent in enumerate(self.agents):
                _, h = agent(agent_inputs[:, i], self.hidden_states[i])
                hidden_list.append(h.view(ep_batch.batch_size, -1))
                self.hidden_states[i] = h.view(ep_batch.batch_size, -1)
            self.hidden_states = hidden_list

            if policy_mode:
                actions = self.greedy(avail_actions)
                policy = avail_actions.new_zeros(
                    ep_batch.batch_size, self.n_agents, self.args.n_actions
                )
                policy.scatter_(dim=-1, index=actions, src=policy.new_ones(1, 1, 1).expand_as(actions))
                return policy
            if actions is None:
                actions = self.greedy(avail_actions)
                return actions
            return self.q_values(actions)

    def q_factors(self, batch_size):
        factors = []
        for h, f in zip(self.hidden_states, self.factor_fun):
            factors.append(f(h).view(batch_size, self.args.low_rank, self.args.n_actions))
        return th.stack(factors, dim=1)

    def utilities(self, batch_size):
        utils = []
        for h, f in zip(self.hidden_states, self.utility_fun):
            utils.append(f(h).view(batch_size, 1, self.args.n_actions))
        return th.cat(utils, dim=1)

    def q_values(self, actions):
        batch_size = actions.shape[0]
        factors = self.q_factors(batch_size)
        factors = factors.gather(
            dim=-1, index=actions.unsqueeze(-1).expand(factors.shape[:-1]).unsqueeze(-1)
        ).squeeze(-1)
        values = factors.prod(dim=-2).sum(dim=-1)
        if self.add_utilities:
            values = values + self.utilities(batch_size).gather(dim=-1, index=actions).squeeze(-1).sum(dim=-1)
        return values

    def greedy(self, available_actions=None, policy_mode=False):
        del policy_mode  # unused
        dims = (
            (available_actions.shape[0], self.n_agents)
            if available_actions is not None
            else (self.hidden_states[0].shape[0], self.n_agents)
        )
        unavailable_actions = None if available_actions is None else available_actions == 0
        actions = th.randint(self.args.n_actions, (*dims, 1), device=self.device)
        max_actions = actions.clone()
        max_values = th.full((*dims, 1), -float("inf"), device=self.device)
        factors = self.q_factors(dims[0])
        if self.add_utilities:
            utilities = self.utilities(dims[0])
        for _ in range(self.args.max_iterations):
            values = factors.gather(
                dim=-1, index=actions.expand(factors.shape[:-1]).unsqueeze(dim=-1)
            )
            idx = self.idx.unsqueeze(dim=0).unsqueeze(dim=3).expand(
                *dims, self.n_agents - 1, self.args.low_rank
            )
            values = values.squeeze(dim=3).unsqueeze(dim=1).expand(
                *dims, self.n_agents, self.args.low_rank
            )
            values = values.gather(dim=2, index=idx).prod(dim=2).unsqueeze(dim=-1)
            values = factors * values
            values = values.sum(dim=-2)
            if self.add_utilities:
                values = values + utilities
            if unavailable_actions is not None:
                values.masked_fill_(unavailable_actions, -float("inf"))
            values, actions = values.max(dim=-1, keepdim=True)
            select = values > max_values
            max_values[select] = values[select]
            max_actions[select] = actions[select]
            if not select.any():
                break
        return max_actions

    def init_hidden(self, batch_size):
        self.hidden_states = [ag.init_hidden().expand(batch_size, -1) for ag in self.agents]
        self._set_idx(self.args)

    def cuda(self):
        for ag in self.agents:
            ag.cuda()
        for f in self.factor_fun:
            f.cuda()
        if self.add_utilities:
            for f in self.utility_fun:
                f.cuda()
        self.device = next(self.parameters()).device
        self.idx = self.idx.to(self.device)

    def parameters(self):
        params = itertools.chain(
            *[ag.parameters() for ag in self.agents],
            *[f.parameters() for f in self.factor_fun],
        )
        if self.add_utilities:
            params = itertools.chain(params, *[f.parameters() for f in self.utility_fun])
        return params

    def load_state(self, other_mac):
        for i, ag in enumerate(self.agents):
            ag.load_state_dict(other_mac.agents[i].state_dict())
        for i, f in enumerate(self.factor_fun):
            f.load_state_dict(other_mac.factor_fun[i].state_dict())
        if self.add_utilities:
            for i, f in enumerate(self.utility_fun):
                f.load_state_dict(other_mac.utility_fun[i].state_dict())

    def save_models(self, path):
        for i, ag in enumerate(self.agents):
            th.save(ag.state_dict(), f"{path}/agent_{i}.th")
        for i, f in enumerate(self.factor_fun):
            th.save(f.state_dict(), f"{path}/factors_{i}.th")
        if self.add_utilities:
            for i, f in enumerate(self.utility_fun):
                th.save(f.state_dict(), f"{path}/utilities_{i}.th")

    def load_models(self, path):
        for i, ag in enumerate(self.agents):
            ag.load_state_dict(
                th.load(f"{path}/agent_{i}.th", map_location=lambda storage, loc: storage)
            )
        for i, f in enumerate(self.factor_fun):
            f.load_state_dict(
                th.load(f"{path}/factors_{i}.th", map_location=lambda storage, loc: storage)
            )
        if self.add_utilities:
            for i, f in enumerate(self.utility_fun):
                f.load_state_dict(
                    th.load(f"{path}/utilities_{i}.th", map_location=lambda storage, loc: storage)
                )
