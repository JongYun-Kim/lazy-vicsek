import copy
from typing import Dict, List, Union
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import numpy as np

# PyTorch
import torch
import torch.nn as nn

# Custom modules


class ActorTest(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.embedding_dim = 128
        self.agent_embedding_dim = 128
        self.n_enc_layer = 3
        self.n_head = 8
        self.ff_dim = 512
        self.norm_eps = 1e-5

        self.flock_embedding = nn.Linear(input_dim, self.embedding_dim)

        ## Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim,
                                                        nhead=self.n_head,
                                                        dim_feedforward=self.ff_dim,
                                                        dropout=0.0,
                                                        layer_norm_eps=self.norm_eps,
                                                        norm_first=True,
                                                        batch_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_enc_layer,
                                             enable_nested_tensor=False)

        self.Wq = nn.Parameter(torch.randn(self.embedding_dim * 2, self.embedding_dim))
        self.Wk = nn.Parameter(torch.randn(self.embedding_dim, self.embedding_dim))

        self.tanh = nn.Tanh()

    def forward(self, obs_dict: Dict[str, TensorType]):
        # Get data
        agents_info = obs_dict["centralized_agents_info"]  # (batch_size, num_agents_max, d_subobs)

        # run MJ's forward
        att_scores = self.mj_forward_actor(agents_info)  # [batch_size, n_agent, n_agent]

        return att_scores

    def mj_forward_actor(self, obs):
        """
        Input:
            obs: [batch_size, n_agent, input_dim]
        """

        batch_size = obs.shape[0]
        n_agent = obs.shape[1]

        flock_embed = self.flock_embedding(obs)  # [batch_size, n_agent, embedding_dim]

        # # encoder1
        enc = self.encoder(flock_embed)  # [batch_size, n_agent, embedding_dim]

        # context embedding
        context = torch.mean(enc, dim=1)  # [batch_size, embedding_dim]

        flock_context = context.unsqueeze(1).expand(batch_size, n_agent,
                                                    context.shape[-1])  # [batch_size, n_agent, embedding_dim]
        agent_context = torch.cat((enc, flock_context), dim=-1)  # [batch_size, n_agent, embedding_dim*2]

        queries = torch.matmul(agent_context, self.Wq)  # [batch_size, n_agent, embedding_dim]
        keys = torch.matmul(enc, self.Wk)  # [batch_size, n_agent, embedding_dim]
        D = queries.shape[-1]

        # attention
        att_scores = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(D)  # [batch_size, n_agent, n_agent]
        # att_scores = (self.tanh(att_scores) + 1) / 2
        #
        # # make diagonal elements to 1
        # ones = att_scores.new_ones(att_scores.shape[1])  # [n_agent, n_agent]
        # I_mat = torch.diag_embed(ones).expand_as(att_scores)  # [batch_size, n_agent, n_agent]
        # att_scores = att_scores * (1 - I_mat) + I_mat  # [batch_size, n_agent, n_agent]
        #
        # return att_scores

        #
        # Fill the diagonal with very large positive value (to make the corresponding probability close to 1)
        large_val = 1e9  # may cause NaN if it passes through softmax (or exp)
        # large_val = 512
        att_scores *= 2e-3
        att_scores = att_scores - torch.diag_embed(att_scores.new_full((n_agent,), large_val))  # [batch_size, n_agent, n_agent]

        # large_val = 1e9
        # ones = att_scores.new_ones(att_scores.shape[1])  # [n_agent, n_agent]
        # I_mat = torch.diag_embed(ones).expand_as(att_scores)  # [batch_size, n_agent, n_agent]
        # att_scores = att_scores * (1 - I_mat) + (large_val * I_mat)  # [batch_size, n_agent, n_agent]


        # negate the scores (representing the probability of action being 0 in the softmax)
        neg_att_scores = - att_scores  # [batch_size, n_agent, n_agent]

        z_expanded = att_scores.unsqueeze(-1)  # [batch_size, n_agent, n_agent, 1]
        neg_z_expanded = neg_att_scores.unsqueeze(-1)  # [batch_size, n_agent, n_agent, 1]

        # Concat along the new dim
        z_cat = torch.cat((z_expanded, neg_z_expanded), dim=-1)  # [batch_size, n_agent, n_agent, 2]
        # z_cat = torch.cat((neg_z_expanded, z_expanded), dim=-1)  # [batch_size, n_agent, n_agent, 2]

        # Reshape to 2D (batch_size, 2* n_agent*n_agent
        z_reshaped = z_cat.reshape(batch_size, n_agent * n_agent * 2)  # [batch_size, n_agent*n_agent*2]

        return z_reshaped  # [batch_size, n_agent*n_agent*2]


class CriticTest(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.embedding_dim = 128
        self.agent_embedding_dim = 128
        self.n_enc_layer = 3
        self.n_head = 8
        self.ff_dim = 512
        self.norm_eps = 1e-5

        self.flock_embedding = nn.Linear(input_dim, self.embedding_dim)

        ## Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim,
                                                        nhead=self.n_head,
                                                        dim_feedforward=self.ff_dim,
                                                        dropout=0.0,
                                                        layer_norm_eps=self.norm_eps,
                                                        norm_first=True,
                                                        batch_first=True)

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_enc_layer,
                                             enable_nested_tensor=False)

        self.value_net = nn.Linear(self.embedding_dim, 1)

    def forward(self, obs_dict: Dict[str, TensorType]):
        # Get data
        agents_info = obs_dict["centralized_agents_info"]  # (batch_size, num_agents_max, d_subobs)

        # run MJ's forward
        value_unsqueezed = self.mj_forward_critic(agents_info)  # [batch_size, 1]

        return value_unsqueezed  # [batch_size, 1]

    def mj_forward_critic(self, obs):
        """
        Input:
            obs: [batchsize, n_agent, input_dim]
        """

        flock_embed = self.flock_embedding(obs)  # [batch_size, n_agent, embedding_dim]

        # # encoder1
        enc = self.encoder(flock_embed)  # [batch_size, n_agent, embedding_dim]

        # context embedding
        context = torch.mean(enc, dim=1)  # [batch_size, embedding_dim]

        value = self.value_net(context)  # [batch_size, 1]

        return value  # [batch_size, 1]
