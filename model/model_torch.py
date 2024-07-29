import copy
from typing import Dict, List, Union
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import numpy as np

# PyTorch
import torch
import torch.nn as nn

# Custom modules


class LazyVicsekActor(nn.Module):
    def __init__(self, src_embed, encoder, decoder, generator):

        super().__init__()

        # Define the model components
        self.src_embed = src_embed
        self.d_v = src_embed.in_features
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

        # Custom layers, if needed
        #

    def forward(self, obs_dict: Dict[str, TensorType]):
        # Get data
        agent_infos = obs_dict["local_agent_infos"]  # (batch_size, num_agents_max, num_agents_max, obs_dim)
        network = obs_dict["neighbor_masks"]  # (batch_size, num_agents_max, num_agents_max); (:,i,:): i-th agent's net
        padding_mask = obs_dict["padding_mask"]  # (batch_size, num_agents_max); applies over all agents same
        # Caution: masks are torch FLOAT tensors, not boolean tensors, which I don't like in RLlib (v2.1.0)

        # TODO: Currently, padding agents are not supported in the model (network: src_maskS, pad_mask: tgt_mask ?)

        # Get sub-attention scores
        att_scores = torch.zeros_like(network)  # (batch_size, num_agents_max, num_agents_max)
        num_agents_max = agent_infos.shape[1]
        for i in range(num_agents_max):  # TODO: [MUST] push agent dim onto batch dim to parallelize
            local_agent_info = agent_infos[:, i, :, :]
            local_network = network[:, i, :]
            padding_mask = padding_mask

            # shape: (batch_size, num_agents_max)
            sub_att_scores = self.local_forward(local_agent_info, local_network, padding_mask)

            # Get the i-th row of the attention scores
            att_scores[:, i, :] = sub_att_scores

        return att_scores  # (batch_size, num_agents_max, num_agents_max)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)

    def local_forward(self, local_agent_info, local_network, padding_mask):
        """
        :param local_agent_info: (batch_size, num_agents_max, obs_dim)
        :param local_network:    (batch_size, num_agents_max)
        :param padding_mask:     (batch_size, num_agents_max)
        :return: sub_att_scores: (batch_size, num_agents_max)
        """
        assert local_agent_info.shape[2] == self.d_v  # TODO: remove this line after debugging

        # Get data
        src = local_agent_info  # (batch_size, num_agents_max, d_v)

        # Get masks
        src_mask_tokens = local_network  # (batch_size, num_agents_max==seq_len_src)
        src_mask_idx = 0
        src_mask = self.make_src_mask(src_mask_tokens, mask_idx=src_mask_idx)  # (batch_size, seq_len_src, seq_len_src)
        tgt_mask = None  # No (masked) self-attention layer in the decoder block
        context_mask_token = torch.zeros_like(src_mask_tokens[:, 0:1])  # (batch_size, 1); it's 2D
        # In Cross-Attention, Q=tgt=context, K/V=src=enc_out
        src_tgt_mask = self.make_src_tgt_mask(src_mask_tokens, context_mask_token, mask_idx=src_mask_idx)

        # Embedding: in the encoder method

        # Encoder
        # encoder_out: shape: (batch_size, src_seq_len, d_embed) == (batch_size, num_agents_max, d_embed)
        # unsqueeze(1) has been applied to src_mask to broadcast over head dim in the MHA layer
        encoder_out = self.encode(src, src_mask.unsqueeze(1))

        # Context embedding, if needed
        h_c_N = self.get_context_vector(embeddings=encoder_out, pad_tokens=1-src_mask_tokens,
                                        use_embeddings_mask=True, debug=True)  # (batch_size, 1, d_embed_context)

        # Decoder: Cross-Attention (glimpse); Q: context, K/V: encoder_out


        # Generator: raw attention scores by CA; Q: rich-context (decoder_out), K/V: encoder_out

        # sub_att_scores: (batch_size, num_agents_max)
        # decoder_out: (batch_size, 1, d_embed_context)
        # h_c_N: (batch_size, 1, d_embed_context)
        return sub_att_scores, decoder_out, h_c_N

    def get_context_vector(self, embeddings, pad_tokens, use_embeddings_mask=True, debug=False):
        # embeddings: shape (batch_size, num_agents_max==seq_len_src, data_size==d_embed_input)
        # pad_tokens: shape (batch_size, num_agents_max==seq_len_src)

        # Obtain batch_size, num_agents, data_size from embeddings
        batch_size, num_agents_max, data_size = embeddings.shape

        if use_embeddings_mask:
            # Expand the dimensions of pad_tokens to match the shape of embeddings
            mask = pad_tokens.unsqueeze(-1).expand_as(embeddings)  # (batch_size, num_agents_max, data_size)

            # Replace masked values with zero for the average computation
            # embeddings_masked: (batch_size, num_agents_max, data_size)
            embeddings_masked = torch.where(mask == 1, embeddings, torch.zeros_like(embeddings))

            # Compute the sum and count non-zero elements
            embeddings_sum = torch.sum(embeddings_masked, dim=1, keepdim=True)  # (batch_size, 1, data_size)
            embeddings_count = torch.sum((mask == 0), dim=1, keepdim=True).float()  # (batch_size, 1, data_size)

            # Check if there is any sample where all agents are padded
            if debug:
                if torch.any(embeddings_count == 0):
                    raise ValueError("All agents are padded in at least one sample.")

            # Compute the average embeddings, only for non-masked elements
            embeddings_avg = embeddings_sum / embeddings_count
        else:
            # Compute the average embeddings: shape (batch_size, 1, data_size)
            embeddings_avg = torch.mean(embeddings, dim=1, keepdim=True)  # num_agents_max dim is reduced

        # Construct context embedding: shape (batch_size, 1, d_embed_context)
        # The resulting tensor, h_c, will have shape (batch_size, 1, d_embed_context)
        # Concatenate the additional info to h_c, if you need more info for the context vector.
        h_c = embeddings_avg
        # This represents the graph embeddings.
        # It summarizes the information of all nodes in the graph.

        return h_c  # (batch_size, 1, d_embed_context)

    def make_src_mask(self, src, mask_idx=1):
        pad_mask = self.make_pad_mask(src, src, pad_idx=mask_idx)
        return pad_mask  # (batch_size, seq_len_src, seq_len_src)

    def make_src_tgt_mask(self, src, tgt, mask_idx=1):
        # src: key/value; tgt: query
        pad_mask = self.make_pad_mask(tgt, src, pad_idx=mask_idx)
        return pad_mask  # (batch_size, seq_len_tgt, seq_len_src)

    def make_pad_mask(self, query, key, pad_idx=1, dim_check=False):
        # query: (n_batch, query_seq_len)
        # key: (n_batch, key_seq_len)
        # If input_token==pad_idx, then the mask value is 0, else 1
        # In the MHA layer, (no attention) == (attention_score: -inf) == (mask value is 0) == (input_token==pad_idx)
        # WARNING: Choose pad_idx carefully, particularly about the data type (e.g. float, int, ...)

        # Check if the query and key have the same dimension
        if dim_check:
            assert len(query.shape) == 2, "query must have 2 dimensions: (n_batch, query_seq_len)"
            assert len(key.shape) == 2, "key must have 2 dimensions: (n_batch, key_seq_len)"
            assert query.size(0) == key.size(0), "query and key must have the same batch size"

        query_seq_len, key_seq_len = query.size(1), key.size(1)

        key_mask = key.ne(pad_idx).unsqueeze(1)  # (n_batch, 1, key_seq_len); on the same device as key
        key_mask = key_mask.repeat(1, query_seq_len, 1)  # (n_batch, query_seq_len, key_seq_len)

        query_mask = query.ne(pad_idx).unsqueeze(2)  # (n_batch, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, key_seq_len)  # (n_batch, query_seq_len, key_seq_len)

        mask = key_mask & query_mask
        mask.requires_grad = False
        return mask  # output shape: (n_batch, query_seq_len, key_seq_len)  # Keep in mind: 'NO HEADING DIM' here!!


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
        att_scores *= 5e-3
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
