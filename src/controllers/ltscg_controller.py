import itertools

import torch as th
import torch.nn as nn

from .basic_controller import BasicMAC
from components.attention_module import AttentionModule
from components.gcn_module import GCNModule


class LTSCGGraphMAC(BasicMAC):

    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.args = args
        self.n_gcn_layers = args.number_gcn_layers
        self.dicg_emb_hid = args.dicg_emb_hid
        org_input_shape = self._get_input_shape(scheme)
        self.gcn_message_dim = args.gcn_message_dim
        self.concate_mlp_dim = args.concate_mlp_dim
        agent_input_shape = org_input_shape
        if args.concate_gcn:
            agent_input_shape += self.gcn_message_dim
        if args.concate_gcn and args.concate_mlp:
            agent_input_shape += self.concate_mlp_dim
        self._build_agents(agent_input_shape)

        self.dicg_encoder = self._mlp(org_input_shape, self.dicg_emb_hid, self.concate_mlp_dim)
        self.attention_layer = AttentionModule(self.concate_mlp_dim, attention_type="general")
        self.gcn_layers = nn.ModuleList([
            GCNModule(self.concate_mlp_dim, self.gcn_message_dim, bias=True, id=0),
            GCNModule(self.gcn_message_dim, self.gcn_message_dim, bias=True, id=1),
        ])

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        del test_mode
        org_agent_inputs = self._build_ltscg_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        embeddings_collection = []
        embeddings_0 = self.dicg_encoder.forward(org_agent_inputs)
        embeddings_collection.append(embeddings_0)
        attention_weights = self.attention_layer.forward(embeddings_0)
        for i_layer, gcn_layer in enumerate(self.gcn_layers):
            embeddings_gcn = gcn_layer.forward(embeddings_collection[i_layer], attention_weights)
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
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), attention_weights

    def _build_ltscg_inputs(self, batch, t):
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

    def parameters(self):
        param = itertools.chain(BasicMAC.parameters(self), self.dicg_encoder.parameters(), self.attention_layer.parameters(), self.gcn_layers.parameters())
        return param

    def load_state(self, other_mac):
        BasicMAC.load_state(self, other_mac)
        self.dicg_encoder.load_state_dict(other_mac.dicg_encoder.state_dict())
        self.attention_layer.load_state_dict(other_mac.attention_layer.state_dict())
        self.gcn_layers.load_state_dict(other_mac.gcn_layers.state_dict())
    def cuda(self):
        super().cuda()
        self.dicg_encoder.cuda()
        self.attention_layer.cuda()
        self.gcn_layers.cuda()

    def save_models(self, path):
        BasicMAC.save_models(self, path)
        th.save(self.dicg_encoder.state_dict(), f"{path}/ltscg_encoder.th")
        th.save(self.attention_layer.state_dict(), f"{path}/ltscg_attention.th")
        th.save(self.gcn_layers.state_dict(), f"{path}/ltscg_gcn_layers.th")

    def load_models(self, path):
        BasicMAC.load_models(self, path)
        self.dicg_encoder.load_state_dict(
            th.load(f"{path}/ltscg_encoder.th", map_location=lambda storage, loc: storage)
        )
        self.attention_layer.load_state_dict(
            th.load(f"{path}/ltscg_attention.th", map_location=lambda storage, loc: storage)
        )
        self.gcn_layers.load_state_dict(
            th.load(f"{path}/ltscg_gcn_layers.th", map_location=lambda storage, loc: storage)
        )

    @staticmethod
    def _mlp(input, hidden_dims, output):
        """ Creates an MLP with the specified input and output dimensions and (optional) hidden layers. """
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
