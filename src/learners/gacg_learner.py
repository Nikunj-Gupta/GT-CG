import torch
import torch as th
import torch.nn as nn

from components.episode_buffer import EpisodeBatch
from .q_learner import QLearner


class GACGLearner(QLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.group_start = getattr(args, "obs_group_trunk_size", 0)
        self.group_loss_weight = getattr(args, "group_loss_weight", 0.0)

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
        group_index_list = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            outputs = self.mac.forward(batch, t=t)
            if isinstance(outputs, tuple):
                agent_outs = outputs[0]
                group_index = outputs[2] if len(outputs) > 2 else None
            else:
                agent_outs = outputs
                group_index = None
            mac_out.append(agent_outs)
            group_index_list.append(group_index)

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

        is_train_groupnizer = getattr(self.args, "is_train_groupnizer", False)
        if is_train_groupnizer and len(group_index_list) > self.group_start:
            group_index_slice = group_index_list[self.group_start:]
            if all(g is not None for g in group_index_slice):
                group_index_out = th.stack(group_index_slice, dim=1)
                gdistance = []
                for t_g in range(group_index_out.shape[1]):
                    num_group = group_index_out[:, t_g].max() + 1
                    if num_group > 1:
                        g_temp = self.group_distance_ratio(
                            mac_out[:, t_g + self.group_start], group_index_out[:, t_g]
                        )
                        gdistance.append(g_temp)
                    else:
                        gdistance.append(1)
                gdistance_mean = sum(gdistance) / len(gdistance)
                loss = (masked_td_error ** 2).sum() / mask.sum() - self.group_loss_weight * gdistance_mean
            else:
                loss = (masked_td_error ** 2).sum() / mask.sum()
                gdistance_mean = None
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()
            gdistance_mean = None

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
            if gdistance_mean is not None:
                self.logger.log_stat("Gdistance_mean", gdistance_mean, t_env)
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

    @staticmethod
    def _mlp(input, hidden_dims, output):
        hidden_dims = [] if hidden_dims is None else hidden_dims
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        dim = input
        layers = []
        for d in hidden_dims:
            layers.append(nn.Linear(dim, d))
            layers.append(nn.ReLU())
            dim = d
        layers.append(nn.Linear(dim, output))
        return nn.Sequential(*layers)

    @staticmethod
    def group_distance_ratio(x, y, eps=1e-5):
        num_classes = int(y.max()) + 1

        numerator = 0.0
        for i in range(num_classes):
            mask = y == i
            dist = torch.cdist(x[mask].unsqueeze(0), x[~mask].unsqueeze(0))
            numerator += (1 / (dist.numel() + eps)) * float(dist.sum())
        numerator *= 1 / (num_classes - 1) ** 2

        denominator = 0.0
        for i in range(num_classes):
            mask = y == i
            dist = torch.cdist(x[mask].unsqueeze(0), x[mask].unsqueeze(0))
            denominator += (1 / (dist.numel() + eps)) * float(dist.sum())
        denominator *= 1 / num_classes

        return numerator / (denominator + eps)
