import copy
from typing import Dict, List, Union
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import numpy as np

# PyTorch
import torch
import torch.nn as nn

# Custom modules


class LazyVicsekListener(nn.Module):
    def __init__(self, src_embed, encoder, decoder, generator, d_embed_context):

        super().__init__()

        # Define the model components
        self.src_embed = src_embed
        self.d_v = src_embed.in_features
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.d_embed_context = d_embed_context

        # Custom layers, if needed
        #

    def forward(self, obs_dict: Dict[str, TensorType]):
        # Get data
        agent_infos = obs_dict["local_agent_infos"]  # (batch_size, num_agents_max, num_agents_max, obs_dim)
        network = obs_dict["neighbor_masks"]  # (batch_size, num_agents_max, num_agents_max); (:,i,:): i-th agent's net
        padding_mask = obs_dict["padding_mask"]  # (batch_size, num_agents_max); applies over all agents same
        is_from_my_env = obs_dict["is_from_my_env"]  # (batch_size,); 1: from my env, 0: under the env_check
        # Caution: masks are torch FLOAT tensors, not boolean tensors, which I don't like in RLlib (v2.1.0)

        batch_size, num_agents_max, _, obs_dim = agent_infos.shape

        # TODO: Currently, padding agents are not supported in the model (network: src_maskS, pad_mask: tgt_mask ?)

        # Get sub-attention scores
        att_scores = torch.zeros_like(network, dtype=torch.float32)  # (batch_size, num_agents_max, num_agents_max)
        num_agents = padding_mask.sum(dim=1).int()  # (batch_size,); number of agents in each sample
        h_c_N_accumulator = torch.zeros(batch_size, 1, self.d_embed_context, device=agent_infos.device)

        for i in range(num_agents_max):  # TODO: [MUST] push agent dim onto batch dim to parallelize
            local_agent_info = agent_infos[:, i, :, :]
            local_network = network[:, i, :]

            # sub_att_scores: shape: (batch_size, num_agents_max)
            # h_c_N: shape: (batch_size, 1, d_embed_context)
            sub_att_scores, _, h_c_N = self.local_forward(local_agent_info, local_network, padding_mask, is_from_my_env)

            # Get the i-th row of the attention scores
            att_scores[:, i, :] = sub_att_scores

            # Accumulate h_c_N values
            h_c_N_accumulator += h_c_N

        # Calculate average_h_c_N
        num_agents = num_agents.view(-1, 1, 1).float()  # (batch_size, 1, 1)
        average_h_c_N = h_c_N_accumulator / num_agents

        # att_scores: (batch_size, num_agents_max, num_agents_max)
        # average_h_c_N: (batch_size, 1, d_embed_context)
        return att_scores, average_h_c_N

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)

    def local_forward(self, local_agent_info, local_network, padding_mask, is_from_my_env):
        """
        :param local_agent_info: (batch_size, num_agents_max, obs_dim)
        :param local_network:    (batch_size, num_agents_max)
        :param padding_mask:     (batch_size, num_agents_max)
        :param is_from_my_env:   (batch_size,)
        :return: sub_att_scores: (batch_size, num_agents_max)
        """
        assert local_agent_info.shape[2] == self.d_v  # TODO: remove this line after debugging

        # Get data
        src = local_agent_info  # (batch_size, num_agents_max, d_v)

        # Get masks
        # local_network:
        # # 0: padding / disconnected, 1: connected
        # # 0: no attention,           1: attention
        # # 0: False in mask,          1: True in mask
        src_mask_tokens = local_network.ne(0)  # (batch_size, num_agents_max==seq_len_src)  bool tensor
        src_mask_idx = 0
        src_mask = self.make_src_mask(src_mask_tokens, mask_idx=src_mask_idx)  # (batch_size, seq_len_src, seq_len_src)
        tgt_mask = None  # No (masked) self-attention layer in the decoder block
        context_mask_token = torch.ones_like(src_mask_tokens[:, 0:1], dtype=torch.bool)  # (batch_size, 1); it's 2D
        # In the Cross-Attention, Q=tgt=context, K/V=src=enc_out
        src_tgt_mask = self.make_src_tgt_mask(src_mask_tokens, context_mask_token, mask_idx=src_mask_idx)

        # Embedding: in the encoder method

        # Encoder
        # encoder_out: shape: (batch_size, src_seq_len, d_embed) == (batch_size, num_agents_max, d_embed)
        # unsqueeze(1) has been applied to src_mask to broadcast over head dim in the MHA layer
        encoder_out = self.encode(src, src_mask.unsqueeze(1))

        # Context embedding, if needed
        h_c_N = self.get_context_vector(embeddings=encoder_out, pad_tokens=~src_mask_tokens,
                                        is_from_my_env=is_from_my_env,
                                        use_embeddings_mask=True, debug=True)  # (batch_size, 1, d_embed_context)

        # Decoder: Cross-Attention (glimpse); Q: context, K/V: encoder_out
        # decoder_out: shape: (batch_size, 1, d_embed_context)
        decoder_out = self.decode(h_c_N, encoder_out, tgt_mask, src_tgt_mask.unsqueeze(1))  # h_c_(N+1)

        # Generator: raw attention scores by CA; Q: rich-context (decoder_out), K/V: encoder_out
        sub_att_scores = self.generator(input_query=decoder_out, input_key=encoder_out, mask=src_tgt_mask).squeeze(1)  # kill q dim

        # sub_att_scores: (batch_size, num_agents_max)
        # decoder_out: (batch_size, 1, d_embed_context)
        # h_c_N: (batch_size, 1, d_embed_context)
        return sub_att_scores, decoder_out, h_c_N

    def get_context_vector(self, embeddings, pad_tokens, is_from_my_env, use_embeddings_mask=True, debug=False):
        # embeddings: shape (batch_size, num_agents_max==seq_len_src, data_size==d_embed_input)
        # pad_tokens: shape (batch_size, num_agents_max==seq_len_src)
        # is_from_my_env: shape (batch_size,)

        # Obtain batch_size, num_agents, data_size from embeddings
        batch_size, num_agents_max, data_size = embeddings.shape

        if use_embeddings_mask:  # TODO: Could be way simpler
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
                    if is_from_my_env.sum() == batch_size:  # all samples are from my env but padded
                        raise ValueError("All agents are padded in at least one sample.")
                    # else:  It's in the env_check mode, so it's okay to have all agents padded.
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
