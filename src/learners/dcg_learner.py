import torch as th

from components.episode_buffer import EpisodeBatch
from .q_learner import QLearner


class DCGLearner(QLearner):
    """QLearner variant for Deep Coordination Graph controllers."""

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.target_update_interval = getattr(
            self.args, "target_update_interval", getattr(self.args, "target_update_interval_or_tau", 200)
        )

    def _update_targets(self):
        """Hard update of target MAC (kept separate for DCG-style controllers)."""
        self.target_mac.load_state(self.mac)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        if getattr(self.args, "standardise_rewards", False):
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        if self.args.common_reward:
            assert rewards.size(2) == 1, "Expected singular agent dimension for common rewards"
            rewards = rewards.expand(-1, -1, self.args.n_agents)

        target_out = []
        self.target_mac.init_hidden(batch.batch_size)
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            greedy = self.mac.forward(batch, t=t, policy_mode=False)
            target_out.append(
                self.target_mac.forward(batch, t=t, actions=greedy, policy_mode=False)
            )
        target_out = th.stack(target_out[1:], dim=1).unsqueeze(dim=-1)
        targets = rewards + self.args.gamma * (1 - terminated) * target_out

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            val = self.mac.forward(
                batch, t=t, actions=actions[:, t], policy_mode=False, compute_grads=True
            )
            mac_out.append(val)
        mac_out = th.stack(mac_out, dim=1).unsqueeze(dim=-1)

        if getattr(self.args, "standardise_returns", False):
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        td_error = mac_out - targets.detach()
        mask = mask.expand_as(td_error)
        td_error = td_error * mask
        loss = (td_error**2).sum() / mask.sum()

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (td_error.abs().sum().item() / mask_elems), t_env
            )
            self.logger.log_stat(
                "q_taken_mean",
                (mac_out * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env
