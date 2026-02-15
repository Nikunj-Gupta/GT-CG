import torch as th
import torch.nn as nn
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.attention_module import AttentionModule
from components.gcn_module import GCNModule
from modules.graphlearner import GTSModel
from .q_learner import QLearner


class LTSCGLearner(QLearner):
    """LTSCG learner adapted to the current QLearner training interface."""

    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        self.temperature = 0.5
        self.graph_loss_weight = getattr(args, "graph_loss_weight", 1.0)

        self._model_kwargs = dict(args.gtsmodel)
        self._model_kwargs["input_dim"] = self.args.obs_shape
        self._model_kwargs["output_dim"] = self.args.obs_shape
        self._model_kwargs["num_nodes"] = self.args.n_agents
        episode_limit = getattr(args, "episode_limit", None)
        if episode_limit is None:
            episode_limit = getattr(args, "env_args", {}).get("time_limit", 100)
        self._model_kwargs["dim_fc"] = self.args.obs_shape * (int(episode_limit) + 1)
        self.graph_learner = GTSModel(self.temperature, self.logger, self._model_kwargs)

        input_shape = args.obs_shape + 1  # obs + action index
        self.n_gcn_layers = args.number_gcn_layers
        self.graph_emb_dim = args.state_shape
        self.graph_emb_hid = args.graph_emb_hid
        self.graph_input_obs_mlp = self._mlp(input_shape, self.graph_emb_hid, self.graph_emb_dim)
        self.attention_layer = AttentionModule(self.graph_emb_dim, attention_type="general")
        self.gcn_layers = nn.ModuleList(
            [
                GCNModule(
                    in_features=self.graph_emb_dim,
                    out_features=self.graph_emb_dim,
                    bias=True,
                    id=i,
                )
                for i in range(self.n_gcn_layers)
            ]
        )

        self.mlp_emb_hid = args.mlp_emb_hid
        self.mlp_out = args.mlp_out
        self.graph_output_mlp = self._mlp(self.graph_emb_dim, self.mlp_emb_hid, self.mlp_out)
        self.state_mlp = self._mlp(args.state_shape, self.mlp_emb_hid, self.mlp_out)
        self.graph_obs_mlp = self._mlp(args.obs_shape, self.mlp_emb_hid, self.mlp_out)
        self.next_obs_mlp = self._mlp(args.obs_shape, self.mlp_emb_hid, self.mlp_out)

        self.params += list(self.graph_learner.parameters())
        self.params += list(self.graph_input_obs_mlp.parameters())
        self.params += list(self.attention_layer.parameters())
        self.params += list(self.gcn_layers.parameters())
        self.params += list(self.graph_output_mlp.parameters())
        self.params += list(self.state_mlp.parameters())
        self.params += list(self.graph_obs_mlp.parameters())
        self.params += list(self.next_obs_mlp.parameters())
        self.optimiser = Adam(params=self.params, lr=args.lr)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
        if self.args.common_reward:
            rewards = rewards.expand(-1, -1, self.n_agents)

        graph_loss_state, graph_loss_obs = self._compute_graph_losses(batch)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            outputs = self.mac.forward(batch, t=t)
            agent_outs = outputs[0] if isinstance(outputs, tuple) else outputs
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            outputs = self.target_mac.forward(batch, t=t)
            target_outs = outputs[0] if isinstance(outputs, tuple) else outputs
            target_mac_out.append(target_outs)
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

        if self.args.standardise_returns:
            target_max_qvals = (
                target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean
            )

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        q_loss = (masked_td_error**2).sum() / mask.sum()

        total_loss = q_loss + graph_loss_state + self.graph_loss_weight * graph_loss_obs

        self.optimiser.zero_grad()
        total_loss.backward()
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
            mask_elems = mask.sum().item()
            self.logger.log_stat("loss", q_loss.item(), t_env)
            self.logger.log_stat("ltscg_graph_state_loss", graph_loss_state.item(), t_env)
            self.logger.log_stat("ltscg_graph_obs_loss", graph_loss_obs.item(), t_env)
            self.logger.log_stat("ltscg_total_loss", total_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env
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

    def _compute_graph_losses(self, batch: EpisodeBatch):
        max_ep_t = batch.max_seq_length
        obs = batch["obs"]  # [bs, t, n, obs]
        input_obs = obs[:, :-1].transpose(1, 0)  # [t-1, bs, n, obs]
        input_last_a = batch["actions"][:, :-1].transpose(1, 0).float()  # [t-1, bs, n, 1]
        obs_with_action = th.cat([input_obs, input_last_a], dim=3)

        episode_obs = obs.permute(2, 0, 1, 3).reshape(self.args.n_agents, batch.batch_size, -1)
        encoder_input = obs_with_action.reshape(max_ep_t - 1, batch.batch_size, -1)
        graph = batch["graph"][:, :]

        self.graph_learner.train()
        gumbel_adj, graph_decoder_out = self.graph_learner(
            1, encoder_input, episode_obs, graph, self.temperature, 1, 1, 1
        )

        graph_decoder_out = graph_decoder_out.reshape(
            max_ep_t - 1, batch.batch_size, self.args.n_agents, -1
        )
        learned_graph = gumbel_adj.unsqueeze(0).repeat(batch.batch_size, 1, 1)
        batch["graph"][:, :] = learned_graph

        graph_embeddings = [self.graph_input_obs_mlp(obs_with_action)]
        attention_weights = self.attention_layer(graph_embeddings[0])
        for i_layer, gcn_layer in enumerate(self.gcn_layers):
            graph_embeddings.append(
                gcn_layer(graph_embeddings[i_layer], learned_graph.unsqueeze(0) * attention_weights)
            )

        avg_pool = th.mean(graph_embeddings[-1], dim=2).transpose(1, 0)
        graph_state_emb = self.graph_output_mlp(avg_pool)
        state_emb = self.state_mlp(batch["state"][:, :-1])
        mse = nn.MSELoss()
        graph_loss_state = mse(graph_state_emb, state_emb)

        graph_obs_emb = self.graph_obs_mlp(input_obs + graph_decoder_out)
        next_obs_emb = self.next_obs_mlp(obs[:, 1:].transpose(1, 0))
        graph_loss_obs = mse(graph_obs_emb, next_obs_emb)
        return graph_loss_state, graph_loss_obs

    def cuda(self):
        super().cuda()
        self.graph_learner.cuda()
        self.graph_input_obs_mlp.cuda()
        self.attention_layer.cuda()
        self.gcn_layers.cuda()
        self.graph_output_mlp.cuda()
        self.state_mlp.cuda()
        self.graph_obs_mlp.cuda()
        self.next_obs_mlp.cuda()

    def save_models(self, path):
        super().save_models(path)
        th.save(self.graph_learner.state_dict(), f"{path}/ltscg_graph_learner.th")
        th.save(self.graph_input_obs_mlp.state_dict(), f"{path}/ltscg_graph_input_obs_mlp.th")
        th.save(self.attention_layer.state_dict(), f"{path}/ltscg_graph_attention.th")
        th.save(self.gcn_layers.state_dict(), f"{path}/ltscg_graph_gcn_layers.th")
        th.save(self.graph_output_mlp.state_dict(), f"{path}/ltscg_graph_output_mlp.th")
        th.save(self.state_mlp.state_dict(), f"{path}/ltscg_state_mlp.th")
        th.save(self.graph_obs_mlp.state_dict(), f"{path}/ltscg_graph_obs_mlp.th")
        th.save(self.next_obs_mlp.state_dict(), f"{path}/ltscg_next_obs_mlp.th")

    def load_models(self, path):
        super().load_models(path)
        self.graph_learner.load_state_dict(
            th.load(f"{path}/ltscg_graph_learner.th", map_location=lambda storage, loc: storage)
        )
        self.graph_input_obs_mlp.load_state_dict(
            th.load(
                f"{path}/ltscg_graph_input_obs_mlp.th",
                map_location=lambda storage, loc: storage,
            )
        )
        self.attention_layer.load_state_dict(
            th.load(f"{path}/ltscg_graph_attention.th", map_location=lambda storage, loc: storage)
        )
        self.gcn_layers.load_state_dict(
            th.load(f"{path}/ltscg_graph_gcn_layers.th", map_location=lambda storage, loc: storage)
        )
        self.graph_output_mlp.load_state_dict(
            th.load(f"{path}/ltscg_graph_output_mlp.th", map_location=lambda storage, loc: storage)
        )
        self.state_mlp.load_state_dict(
            th.load(f"{path}/ltscg_state_mlp.th", map_location=lambda storage, loc: storage)
        )
        self.graph_obs_mlp.load_state_dict(
            th.load(f"{path}/ltscg_graph_obs_mlp.th", map_location=lambda storage, loc: storage)
        )
        self.next_obs_mlp.load_state_dict(
            th.load(f"{path}/ltscg_next_obs_mlp.th", map_location=lambda storage, loc: storage)
        )

    @staticmethod
    def _mlp(input_dim, hidden_dims, output_dim):
        hidden_dims = [] if hidden_dims is None else hidden_dims
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        dim = input_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        return nn.Sequential(*layers)
