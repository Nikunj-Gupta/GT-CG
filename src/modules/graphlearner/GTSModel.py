import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.graphlearner.cell import DCGRUCell


def sample_gumbel(shape, device, eps=1e-20):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def gumbel_softmax_sample(logits, temperature, eps=1e-10):
    y = logits + sample_gumbel(logits.size(), device=logits.device, eps=eps)
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if not hard:
        return y_soft
    shape = logits.size()
    _, k = y_soft.data.max(-1)
    y_hard = torch.zeros(*shape, device=logits.device)
    y_hard = y_hard.scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
    return (y_hard - y_soft).detach() + y_soft


class Seq2SeqAttrs:
    def __init__(self, **model_kwargs):
        self.max_diffusion_step = int(model_kwargs.get("max_diffusion_step", 2))
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 1000))
        self.filter_type = model_kwargs.get("filter_type", "laplacian")
        self.num_nodes = int(model_kwargs.get("num_nodes", 1))
        self.num_rnn_layers = int(model_kwargs.get("num_rnn_layers", 1))
        self.rnn_units = int(model_kwargs.get("rnn_units", 64))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.dcgru_layers = nn.ModuleList(
            [
                DCGRUCell(
                    self.rnn_units,
                    self.max_diffusion_step,
                    self.num_nodes,
                    filter_type=self.filter_type,
                )
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(self, inputs, adj, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = inputs.new_zeros(
                self.num_rnn_layers, batch_size, self.hidden_state_size
            )
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.output_dim = int(model_kwargs.get("output_dim", 1))
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [
                DCGRUCell(
                    self.rnn_units,
                    self.max_diffusion_step,
                    self.num_nodes,
                    filter_type=self.filter_type,
                )
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(self, inputs, adj, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)


class GTSModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, temperature, logger, model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, **model_kwargs)
        self.encoder_model = EncoderModel(**model_kwargs)
        self.decoder_model = DecoderModel(**model_kwargs)
        self._logger = logger
        self.temperature = temperature
        self.embedding_dim = 100

        # Uses averaged node temporal features to infer graph edges.
        self.node_encoder = nn.LazyLinear(self.embedding_dim)
        self.edge_hidden = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.edge_logits = nn.Linear(self.embedding_dim, 2)

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps)
        )

    def encoder(self, inputs, adj):
        encoder_hidden_state = None
        for t in range(inputs.shape[0]):
            _, encoder_hidden_state = self.encoder_model(inputs[t], adj, encoder_hidden_state)
        return encoder_hidden_state

    def decoder(self, seq_len, encoder_hidden_state, adj, labels=None, batches_seen=None):
        del labels, batches_seen
        batch_size = encoder_hidden_state.size(1)
        go_symbol = encoder_hidden_state.new_zeros(
            batch_size, self.num_nodes * self.decoder_model.output_dim
        )
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol
        outputs = []
        for _ in range(seq_len):
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, adj, decoder_hidden_state
            )
            decoder_input = decoder_output
            outputs.append(decoder_output)
        return torch.stack(outputs)

    def forward(self, label, inputs, node_feas, batch_graph, temp, gumbel_soft, labels=None, batches_seen=None):
        del label, gumbel_soft
        # node_feas: [n_agents, batch, flattened_episode_obs]
        x = node_feas.mean(dim=1)  # [n_agents, feat]
        x = torch.relu(self.node_encoder(x))

        n_agents = x.size(0)
        senders = x.unsqueeze(1).expand(-1, n_agents, -1)
        receivers = x.unsqueeze(0).expand(n_agents, -1, -1)
        edge_feat = torch.cat([senders, receivers], dim=-1)
        logits = self.edge_logits(torch.relu(self.edge_hidden(edge_feat))).reshape(-1, 2)

        edge_prob = gumbel_softmax(logits, temperature=temp, hard=True)
        adj = edge_prob[:, 0].reshape(n_agents, n_agents)

        # Blend with current replay graph statistics to stabilize early training.
        sum_adj = batch_graph.sum(dim=0) / batch_graph.shape[0]
        adj = ((adj + sum_adj) > 0).float()

        encoder_hidden_state = self.encoder(inputs, adj)
        seq_len = inputs.shape[0]
        outputs = self.decoder(
            seq_len, encoder_hidden_state, adj, labels=labels, batches_seen=batches_seen
        )
        return adj, outputs
