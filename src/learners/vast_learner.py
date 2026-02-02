import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from .q_learner import QLearner


class VASTNet(nn.Module):
    def __init__(self, nr_input_features, nr_subteams, nr_hidden_units=128):
        super(VASTNet, self).__init__()
        self.nr_input_features = nr_input_features
        self.nr_hidden_units = nr_hidden_units

        self.fc_net = nn.Sequential(
            nn.Linear(self.nr_input_features, self.nr_hidden_units),
            nn.ELU(),
            nn.Linear(self.nr_hidden_units, self.nr_hidden_units),
            nn.ELU(),
        )

        self.action_head = nn.Linear(self.nr_hidden_units, nr_subteams)

    def forward(self, x):
        batch_size, seq_len, input_features = x.size()
        x = x.view(-1, input_features)
        x = self.fc_net(x)
        x = x.view(batch_size, seq_len, self.nr_hidden_units)
        x = self.action_head(x)
        output = F.softmax(x, dim=-1)
        return output


class VastQLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super(VastQLearner, self).__init__(mac, scheme, logger, args)
        self.args = args

        self.vast_input = int(np.prod(args.state_shape)) + int(np.prod(args.obs_shape))
        self.vast_out = args.group_num
        self.VAST = VASTNet(self.vast_input, self.vast_out).to(self.args.device)

        self.params += list(self.VAST.parameters())
        self.optimiser = Adam(params=self.params, lr=args.lr)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if getattr(self.args, "standardise_rewards", False):
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert rewards.size(2) == 1, "Expected singular agent dimension for common rewards"
            rewards = rewards.expand(-1, -1, self.n_agents)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            outputs = self.mac.forward(batch, t=t)
            if isinstance(outputs, tuple):
                agent_outs = outputs[0]
            else:
                agent_outs = outputs
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            outputs = self.target_mac.forward(batch, t=t)
            if isinstance(outputs, tuple):
                target_agent_outs = outputs[0]
            else:
                target_agent_outs = outputs
            target_mac_out.append(target_agent_outs)

        target_mac_out = th.stack(target_mac_out[1:], dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        obs = batch["obs"]
        vast_state = batch["state"].unsqueeze(2)
        vast_state = vast_state.repeat(1, 1, obs.size(2), 1)
        vast_input = torch.cat(
            [
                vast_state.reshape(-1, self.args.n_agents, vast_state.size(-1)),
                obs.reshape(-1, self.args.n_agents, obs.size(-1)),
            ],
            dim=-1,
        )
        sub_group_index = self.VAST(vast_input)
        assignment_dist = Categorical(sub_group_index)
        subteam_ids = assignment_dist.sample().detach()
        subteam_ids = subteam_ids.reshape(batch.batch_size, -1, self.args.n_agents)

        subg_chosen_action_qvals = torch.zeros(
            subteam_ids.size(0),
            subteam_ids.size(1) - 1,
            self.args.group_num,
            device=subteam_ids.device,
        )
        subg_target_max_qvals = torch.zeros(
            subteam_ids.size(0),
            subteam_ids.size(1) - 1,
            self.args.group_num,
            device=subteam_ids.device,
        )

        for subgroup_index in range(self.args.group_num):
            subgroup_mask = subteam_ids == subgroup_index
            subg_chosen_action_qvals[:, :, subgroup_index] = (
                chosen_action_qvals * subgroup_mask[:, :-1]
            ).sum(dim=-1)
            subg_target_max_qvals[:, :, subgroup_index] = (
                target_max_qvals * subgroup_mask[:, 1:]
            ).sum(dim=-1)

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if getattr(self.args, "standardise_returns", False):
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if getattr(self.args, "standardise_returns", False):
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item()
                / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def cuda(self):
        super().cuda()
        self.VAST.cuda()
