import itertools

import torch as th
import torch.nn as nn

from .basic_controller import BasicMAC
from components.attention_module import AttentionModule
from components.transformer_conv_module import TransformerConvModule


class GTCGMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.residual = args.residual
        self.n_conv_layers = args.number_gcn_layers
        self.use_attention_module = getattr(args, "use_attention_module", True)
        self.dicg_layers = []
        self.dicg_emb_hid = args.dicg_emb_hid
        input_shape = self._get_input_shape(scheme)
        self.dicg_emb_dim = input_shape
        self.dicg_encoder = self._mlp(input_shape, self.dicg_emb_hid, self.dicg_emb_dim)
        self.dicg_layers.append(self.dicg_encoder)
        if self.use_attention_module:
            self.attention_layer = AttentionModule(
                self.dicg_emb_dim, attention_type="general"
            )
            self.dicg_layers.append(self.attention_layer)
        else:
            self.attention_layer = None
        self.conv_layers = nn.ModuleList(
            [
                TransformerConvModule(
                    in_features=self.dicg_emb_dim,
                    out_features=self.dicg_emb_dim,
                    heads=getattr(args, "transformer_heads", 1),
                    concat=False,
                    bias=True,
                    id=i,
                )
                for i in range(self.n_conv_layers)
            ]
        )
        self.dicg_layers.extend(self.conv_layers)
        self.dicg_aggregator = self._mlp(input_shape, self.dicg_emb_hid, self.dicg_emb_dim)

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self.build_dicg_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        embeddings_collection = []
        embeddings_0 = self.dicg_encoder.forward(agent_inputs)
        embeddings_collection.append(embeddings_0)
        if self.use_attention_module:
            attention_weights = self.attention_layer.forward(embeddings_0)
        else:
            attention_weights = th.ones(
                embeddings_0.shape[:-1] + (self.n_agents,), device=embeddings_0.device
            )

        graph = ep_batch["graph"]
        for i_layer, conv_layer in enumerate(self.conv_layers):
            embeddings_conv = conv_layer.forward(
                embeddings_collection[i_layer], graph * attention_weights
            )
            embeddings_collection.append(embeddings_conv)

        if self.residual:
            dicg_agent_inputs = embeddings_collection[0] + embeddings_collection[-1]
        else:
            dicg_agent_inputs = embeddings_collection[-1]
        dicg_agent_inputs = self.dicg_aggregator.forward(dicg_agent_inputs)

        agent_outs, self.hidden_states = self.agent(
            dicg_agent_inputs.view(-1, agent_inputs.shape[-1]), self.hidden_states
        )
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), attention_weights

    def build_dicg_inputs(self, batch, t):
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
        param = itertools.chain(
            BasicMAC.parameters(self),
            self.dicg_encoder.parameters(),
            self.conv_layers.parameters(),
            self.dicg_aggregator.parameters(),
        )
        if self.use_attention_module:
            param = itertools.chain(param, self.attention_layer.parameters())
        return param

    def load_state(self, other_mac):
        BasicMAC.load_state(self, other_mac)
        self.dicg_encoder.load_state_dict(other_mac.dicg_encoder.state_dict())
        if self.use_attention_module and other_mac.attention_layer is not None:
            self.attention_layer.load_state_dict(other_mac.attention_layer.state_dict())
        self.conv_layers.load_state_dict(other_mac.conv_layers.state_dict())
        self.dicg_aggregator.load_state_dict(other_mac.dicg_aggregator.state_dict())

    def cuda(self):
        self.agent.cuda()
        self.dicg_encoder.cuda()
        if self.attention_layer is not None:
            self.attention_layer.cuda()
        self.conv_layers.cuda()
        self.dicg_aggregator.cuda()

    def save_models(self, path):
        BasicMAC.save_models(self, path)
        th.save(self.dicg_encoder.state_dict(), f"{path}/dicg_encoder.th")
        if self.attention_layer is not None:
            th.save(self.attention_layer.state_dict(), f"{path}/attention_layer.th")
        th.save(self.conv_layers.state_dict(), f"{path}/conv_layers.th")
        th.save(self.dicg_aggregator.state_dict(), f"{path}/dicg_aggregator.th")

    def load_models(self, path):
        BasicMAC.load_models(self, path)
        self.dicg_encoder.load_state_dict(
            th.load(f"{path}/dicg_encoder.th", map_location=lambda storage, loc: storage)
        )
        if self.attention_layer is not None:
            self.attention_layer.load_state_dict(
                th.load(
                    f"{path}/attention_layer.th",
                    map_location=lambda storage, loc: storage,
                )
            )
        self.conv_layers.load_state_dict(
            th.load(f"{path}/conv_layers.th", map_location=lambda storage, loc: storage)
        )
        self.dicg_aggregator.load_state_dict(
            th.load(f"{path}/dicg_aggregator.th", map_location=lambda storage, loc: storage)
        )

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
