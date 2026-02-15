import itertools

import torch
import torch as th
import torch.nn as nn
import numpy as np
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

from .basic_controller import BasicMAC
from components.attention_module import AttentionModule
from components.gcn_module import GCNModule
from modules.agents import REGISTRY as agent_REGISTRY


class GroupMessageMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.args = args
        self.n_gcn_layers = args.number_gcn_layers
        self.dicg_layers = []
        self.dicg_emb_hid = args.dicg_emb_hid

        org_input_shape = self._get_input_shape(scheme)
        obs_vshape = scheme["obs"]["vshape"]
        obs_dim = int(np.prod(obs_vshape)) if isinstance(obs_vshape, (list, tuple)) else int(obs_vshape)

        # Support both ratio (<1) and absolute (>=1) settings while avoiding zero-dim layers.
        if args.gcn_message_dim < 1:
            self.gcn_message_dim = max(1, int(round(args.gcn_message_dim * obs_dim)))
        else:
            self.gcn_message_dim = max(1, int(args.gcn_message_dim))
        self.concate_mlp_dim = args.concate_mlp_dim

        agent_input_shape = org_input_shape
        if self.args.concate_gcn:
            agent_input_shape = agent_input_shape + self.gcn_message_dim
        if self.args.concate_gcn and self.args.concate_mlp:
            agent_input_shape = agent_input_shape + self.concate_mlp_dim
        self._build_agents(agent_input_shape)

        self.mlp_emb_dim = org_input_shape
        self.mlp_encoder = self._mlp(org_input_shape, self.mlp_emb_dim, self.concate_mlp_dim)
        self.dicg_layers.append(self.mlp_encoder)
        self.attention_layer = AttentionModule(self.concate_mlp_dim, attention_type="general")
        self.dicg_layers.append(self.attention_layer)
        self.gcn_layers = nn.ModuleList(
            [
                GCNModule(
                    in_features=self.concate_mlp_dim,
                    out_features=self.gcn_message_dim,
                    bias=True,
                    id=0,
                ),
                GCNModule(
                    in_features=self.gcn_message_dim,
                    out_features=self.gcn_message_dim,
                    bias=True,
                    id=1,
                ),
            ]
        )
        self.dicg_layers.extend(self.gcn_layers)

        self.temperature = 1
        self.adj_threshold = args.adj_threshold

        # Grouping
        group_num = args.group_num
        self.trunk_size = args.obs_group_trunk_size
        self.group_in_shape = obs_dim * self.trunk_size
        self.groupnizer = self._mlp(self.group_in_shape, self.mlp_emb_dim, group_num)
        self.small_eye_matrix = 0.001 * torch.eye(self.n_agents).unsqueeze(0)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        batch_size = ep_batch.batch_size
        obs_t = ep_batch["obs"][:, t]
        org_agent_inputs = self.build_agent_inputs(ep_batch, t)
        obs_mlp_emb = self.mlp_encoder.forward(org_agent_inputs)

        avail_actions = ep_batch["avail_actions"][:, t]

        embeddings_collection = [obs_mlp_emb]
        attention_weights = self.attention_layer.forward(obs_mlp_emb)

        group_index = None
        if t >= self.trunk_size:
            obs_trunk = ep_batch["obs"][:, t - self.trunk_size : t].permute(0, 2, 1, 3)
            group_index_temp = self.groupnizer(obs_trunk.reshape(batch_size, self.n_agents, -1))
            group_index = group_index_temp.softmax(dim=-1).argmax(dim=2)

            group_mask = (group_index[:, :, None] == group_index[:, None, :]).float()
            covariance_matrix = torch.bmm(group_mask, group_mask.transpose(1, 2))
            posdef_covariance_matrix = covariance_matrix + self.small_eye_matrix.to(
                covariance_matrix.device
            )
            posdef_min = torch.min(posdef_covariance_matrix)
            posdef_max = torch.max(posdef_covariance_matrix)
            posdef_covariance_matrix = (posdef_covariance_matrix - posdef_min) / (
                posdef_max - posdef_min
            )
            posdef_covariance_matrix = posdef_covariance_matrix.unsqueeze(1).repeat(
                1, self.n_agents, 1, 1
            )
            mvn1 = torch.distributions.MultivariateNormal(
                attention_weights, covariance_matrix=posdef_covariance_matrix
            )
            samples = mvn1.sample((1,))
            final_graph = samples[0]
            final_graph = 0.5 * (final_graph + final_graph.transpose(-1, -2))

            min_value = torch.min(final_graph)
            max_value = torch.max(final_graph)
            final_graph = (final_graph - min_value) / (max_value - min_value)
            if self.args.is_sparse:
                final_graph = (final_graph > self.adj_threshold).float() * final_graph
        else:
            attention_dist = RelaxedBernoulli(
                self.temperature, logits=attention_weights.view(ep_batch.batch_size, -1)
            )
            adj_sample = attention_dist.sample().view(
                ep_batch.batch_size, self.n_agents, self.n_agents
            )
            adj_sample = 0.5 * (adj_sample + adj_sample.transpose(-1, -2))
            final_graph = (adj_sample > self.adj_threshold).float() * adj_sample

        for i_layer, gcn_layer in enumerate(self.gcn_layers):
            embeddings_gcn = gcn_layer.forward(embeddings_collection[i_layer], final_graph)
            embeddings_collection.append(embeddings_gcn)

        if self.args.concate_gcn and self.args.concate_mlp:
            temp_org_input = org_agent_inputs.view(-1, org_agent_inputs.shape[-1])
            temp_mlp_message = embeddings_collection[0].view(-1, self.concate_mlp_dim)
            temp_gcn_message = embeddings_collection[-1].view(-1, self.gcn_message_dim)
            agent_input = th.cat([temp_org_input, temp_mlp_message, temp_gcn_message], dim=1)
        elif self.args.concate_gcn:
            temp_org_input = org_agent_inputs.view(-1, org_agent_inputs.shape[-1])
            temp_gcn_message = embeddings_collection[-1].view(-1, self.gcn_message_dim)
            agent_input = th.cat([temp_org_input, temp_gcn_message], dim=1)
        else:
            agent_input = org_agent_inputs.view(-1, org_agent_inputs.shape[-1])

        agent_outs, self.hidden_states = self.agent(agent_input, self.hidden_states)
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return (
            agent_outs.view(batch_size, self.n_agents, -1),
            torch.cat((attention_weights, final_graph), 2),
            group_index,
            obs_mlp_emb,
        )

    def build_agent_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            )
        inputs = th.cat([x.reshape(bs, self.n_agents, -1) for x in inputs], dim=2)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def parameters(self):
        param = itertools.chain(
            BasicMAC.parameters(self),
            self.mlp_encoder.parameters(),
            self.attention_layer.parameters(),
            self.gcn_layers.parameters(),
            self.groupnizer.parameters(),
        )
        return param

    def load_state(self, other_mac):
        BasicMAC.load_state(self, other_mac)
        self.mlp_encoder.load_state_dict(other_mac.mlp_encoder.state_dict())
        self.attention_layer.load_state_dict(other_mac.attention_layer.state_dict())
        self.gcn_layers.load_state_dict(other_mac.gcn_layers.state_dict())
        self.groupnizer.load_state_dict(other_mac.groupnizer.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.mlp_encoder.cuda()
        self.attention_layer.cuda()
        self.gcn_layers.cuda()
        self.groupnizer.cuda()
        self.small_eye_matrix = self.small_eye_matrix.cuda()

    def save_models(self, path):
        BasicMAC.save_models(self, path)
        th.save(self.mlp_encoder.state_dict(), f"{path}/mlp_encoder.th")
        th.save(self.attention_layer.state_dict(), f"{path}/attention_layer.th")
        th.save(self.gcn_layers.state_dict(), f"{path}/gcn_layers.th")
        th.save(self.groupnizer.state_dict(), f"{path}/groupnizer.th")

    def load_models(self, path):
        BasicMAC.load_models(self, path)
        self.mlp_encoder.load_state_dict(
            th.load(f"{path}/mlp_encoder.th", map_location=lambda storage, loc: storage)
        )
        self.attention_layer.load_state_dict(
            th.load(f"{path}/attention_layer.th", map_location=lambda storage, loc: storage)
        )
        self.gcn_layers.load_state_dict(
            th.load(f"{path}/gcn_layers.th", map_location=lambda storage, loc: storage)
        )
        self.groupnizer.load_state_dict(
            th.load(f"{path}/groupnizer.th", map_location=lambda storage, loc: storage)
        )

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

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
