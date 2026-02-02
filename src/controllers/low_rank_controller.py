import contextlib
import itertools

import torch as th
import torch.nn as nn

from modules.agents import REGISTRY as agent_REGISTRY
from .basic_controller import BasicMAC


class LowRankMAC(BasicMAC):
    """Low-rank joint Q approximation (Boehmer et al., 2020)."""

    def __init__(self, scheme, groups, args):
        BasicMAC.__init__(self, scheme, groups, args)
        self.hidden_dim = getattr(args, "rnn_hidden_dim", args.hidden_dim)

        if self.args.fully_observable:
            input_shape = scheme["obs"]["vshape"]
            if self.args.obs_last_action:
                input_shape += scheme["actions_onehot"]["vshape"][0]
            self.agent = agent_REGISTRY[self.args.agent](input_shape * self.n_agents, self.args)

        output_dim = args.n_actions * args.low_rank * (
            self.n_agents if self.args.fully_observable else 1
        )
        self.factor_fun = nn.Linear(self.hidden_dim, output_dim)

        if args.add_utilities:
            output_dim = args.n_actions * (self.n_agents if self.args.fully_observable else 1)
            self.utility_fun = nn.Linear(self.hidden_dim, output_dim)

        self.device = self.factor_fun.weight.device
        self.idx = th.LongTensor([i for i in range(1, args.n_agents)], device=self.device)
        self.idx = self.idx.unsqueeze(dim=0).repeat(args.n_agents, 1)
        for i in range(1, args.n_agents):
            self.idx[i, :i] = th.LongTensor([j for j in range(i)], device=self.device)

    def forward(self, ep_batch, t, actions=None, policy_mode=True, test_mode=False, compute_grads=False):
        with th.no_grad() if not compute_grads or policy_mode else contextlib.suppress():
            agent_inputs = self._build_inputs(ep_batch, t)
            avail_actions = ep_batch["avail_actions"][:, t]
            _, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
            if policy_mode:
                actions = self.greedy(avail_actions)
                policy = self.hidden_states.new_zeros(ep_batch.batch_size, self.n_agents, self.args.n_actions)
                policy.scatter_(dim=-1, index=actions, src=policy.new_ones(1, 1, 1).expand_as(actions))
                return policy
            if actions is None:
                actions = self.greedy(avail_actions)
                return actions
            return self.q_values(actions)

    def q_factors(self, batch_size):
        factors = self.factor_fun(self.hidden_states)
        factors = factors.view(batch_size, self.n_agents, self.args.low_rank, self.args.n_actions)
        return factors

    def utilities(self, batch_size):
        utilities = self.utility_fun(self.hidden_states)
        utilities = utilities.view(batch_size, self.n_agents, self.args.n_actions)
        return utilities

    def q_values(self, actions):
        factors = self.q_factors(actions.shape[0])
        factors = factors.gather(
            dim=-1, index=actions.expand(factors.shape[:-1]).unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        values = factors.prod(dim=-2).sum(dim=-1)
        if self.args.add_utilities:
            values = (
                values
                + self.utilities(actions.shape[0])
                .gather(dim=-1, index=actions)
                .squeeze(dim=-1)
                .sum(dim=-1)
            )
        return values

    def greedy(self, available_actions=None, policy_mode=False):
        dims = (
            self.hidden_states.shape[:-1]
            if available_actions is None
            else available_actions.shape[:-1]
        )
        unavailable_actions = None if available_actions is None else available_actions == 0
        actions = th.randint(self.args.n_actions, (*dims, 1), device=self.device)
        max_actions = actions
        max_values = self.hidden_states.new_ones(*dims, 1) * (-float("inf"))
        factors = self.q_factors(dims[0])
        if self.args.add_utilities:
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
            if self.args.add_utilities:
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

    def _build_inputs(self, batch, t):
        if not self.args.fully_observable:
            return BasicMAC._build_inputs(self, batch, t)
        bs = batch.batch_size
        inputs = [batch["obs"][:, t].view(bs, -1)]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(
                    batch["actions_onehot"][:, t].new_zeros(batch["actions_onehot"][:, t].shape).view(
                        bs, -1
                    )
                )
            else:
                inputs.append(batch["actions_onehot"][:, t - 1].view(bs, -1))
        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        return inputs

    def init_hidden(self, batch_size):
        if self.args.fully_observable:
            self.hidden_states = self.agent.init_hidden().expand(batch_size, -1)
        else:
            BasicMAC.init_hidden(self, batch_size)

    def cuda(self):
        self.agent.cuda()
        self.factor_fun.cuda()
        if self.args.add_utilities:
            self.utility_fun.cuda()
        self.idx = self.idx.cuda()
        self.device = self.factor_fun.weight.device

    def parameters(self):
        param = itertools.chain(BasicMAC.parameters(self), self.factor_fun.parameters())
        if self.args.add_utilities:
            param = itertools.chain(param, self.utility_fun.parameters())
        return param

    def load_state(self, other_mac):
        BasicMAC.load_state(self, other_mac)
        self.factor_fun.load_state_dict(other_mac.factor_fun.state_dict())
        if self.args.add_utilities:
            self.utility_fun.load_state_dict(other_mac.utility_fun.state_dict())

    def save_models(self, path):
        BasicMAC.save_models(self, path)
        th.save(self.factor_fun.state_dict(), f"{path}/factors.th")
        if self.args.add_utilities:
            th.save(self.utility_fun.state_dict(), f"{path}/utilities.th")

    def load_models(self, path):
        BasicMAC.load_models(self, path)
        self.factor_fun.load_state_dict(
            th.load(f"{path}/factors.th", map_location=lambda storage, loc: storage)
        )
        if self.args.add_utilities:
            self.utility_fun.load_state_dict(
                th.load(f"{path}/utilities.th", map_location=lambda storage, loc: storage)
            )
