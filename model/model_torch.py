import copy
from typing import Dict, List, Union
from ray.rllib.utils.typing import ModelConfigDict, TensorType
import numpy as np

# PyTorch
import torch
import torch.nn as nn


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

    def forward(self, obs_dict: Dict[str, torch.Tensor]):
        """
        Vectorized version of the forward pass that produces
        identical output ordering to the per-agent for-loop version.
        """
        # ------------------------------------------------------------
        # 1) Unpack the inputs
        # ------------------------------------------------------------
        agent_infos = obs_dict["local_agent_infos"]  # (batch_size, num_agents_max, num_agents_max, obs_dim)
        network = obs_dict["neighbor_masks"]  # (batch_size, num_agents_max, num_agents_max)
        padding_mask = obs_dict["padding_mask"]  # (batch_size, num_agents_max)
        is_from_my_env = obs_dict["is_from_my_env"]  # (batch_size,)
        batch_size, num_agents_max, _, obs_dim = agent_infos.shape

        # Number of (non-padding) agents in each sample
        num_agents_per_sample = padding_mask.sum(dim=1).int()  # (batch_size,)

        # ------------------------------------------------------------
        # 2) Permute + reshape so that agent dimension i is “merged” with the batch
        #
        #    Original shapes (for each i in [0..num_agents_max]):
        #      local_agent_info      = (batch_size, num_agents_max, obs_dim)
        #      local_network         = (batch_size, num_agents_max)
        #      local_padding_flags   = (batch_size,)
        #
        #    We will create flattened shapes:
        #      flat_agent_infos      = (batch_size * num_agents_max, num_agents_max, obs_dim)
        #      flat_network          = (batch_size * num_agents_max, num_agents_max)
        #      flat_local_pad_flags  = (batch_size * num_agents_max,)
        #
        #    And similarly replicate the global masks for neighbors, etc.
        # ------------------------------------------------------------
        # agent_infos: permute(1,0,2,3) => (i, b, neighbor, obs_dim),
        # then reshape => (i*b, neighbor, obs_dim).
        permuted_agent_infos = agent_infos.permute(1, 0, 2, 3)  # (num_agents_max, batch_size, num_agents_max, obs_dim)
        flat_agent_infos = permuted_agent_infos.reshape(num_agents_max * batch_size, num_agents_max, obs_dim)

        # network: permute(1,0,2) => (i, b, neighbor),
        # then reshape => (i*b, neighbor).
        permuted_network = network.permute(1, 0, 2)  # (num_agents_max, batch_size, num_agents_max)
        flat_network = permuted_network.reshape(num_agents_max * batch_size, num_agents_max)

        # padding_mask used for neighbors:
        # Expand so each agent i sees the same "who is padded" info for neighbors.
        # shape => (i, b, neighbor) => flatten => (i*b, neighbor).
        expanded_padding_mask = padding_mask.unsqueeze(0).expand(num_agents_max, -1, -1)
        flat_padding_mask_for_neighbors = expanded_padding_mask.reshape(num_agents_max * batch_size, num_agents_max)

        # local_padding_flags = padding_mask[:, i], but we want the same flatten order (i,b).
        # So permute(1,0) => (i, b) => flatten => (i*b).
        permuted_local_padding = padding_mask.permute(1, 0)  # (num_agents_max, batch_size)
        flat_local_padding_flags = permuted_local_padding.reshape(num_agents_max * batch_size)

        # is_from_my_env: replicate for each agent i => (i, b) => flatten => (i*b).
        expanded_is_from_my_env = is_from_my_env.unsqueeze(0).expand(num_agents_max, -1)  # (num_agents_max, batch_size)
        flat_is_from_my_env = expanded_is_from_my_env.reshape(num_agents_max * batch_size)

        # ------------------------------------------------------------
        # 3) Call local_forward once on the flattened data
        #
        #    local_forward signature (in your original loop) was:
        #       local_forward(
        #           local_agent_info, local_network, padding_mask, is_from_my_env, local_padding_flags
        #       )
        #
        #    So now we pass the flattened arguments:
        # ------------------------------------------------------------
        sub_att_scores_flat, _, h_c_N_flat = self.local_forward(
            flat_agent_infos,  # (batch_size * num_agents_max, num_agents_max, obs_dim)
            flat_network,  # (batch_size * num_agents_max, num_agents_max)
            flat_padding_mask_for_neighbors,  # (batch_size * num_agents_max, num_agents_max)
            flat_is_from_my_env,  # (batch_size * num_agents_max,)
            flat_local_padding_flags  # (batch_size * num_agents_max,)
        )
        # sub_att_scores_flat => (batch_size * num_agents_max, num_agents_max)
        # h_c_N_flat         => (batch_size * num_agents_max, 1, d_embed_context)

        # ------------------------------------------------------------
        # 4) Reshape back to the original (batch_size, num_agents_max, num_agents_max)
        #    ordering so that it matches the loop-based version exactly.
        # ------------------------------------------------------------
        # sub_att_scores_flat => (i*b, neighbor).  Reshape => (i, b, neighbor). Then
        # permute => (b, i, neighbor).
        sub_att_scores_reshaped = sub_att_scores_flat.view(num_agents_max, batch_size, num_agents_max)
        att_scores = sub_att_scores_reshaped.permute(1, 0, 2)  # => (batch_size, num_agents_max, num_agents_max)

        # ------------------------------------------------------------
        # 5) Sum up h_c_N across all i (like h_c_N_accumulator += h_c_N in the og loop).
        #    Then average across the (non-padded) agents at the end.
        # ------------------------------------------------------------
        # h_c_N_flat => (i*b, 1, d_embed_context).  Reshape => (i, b, 1, d_embed_context).
        # Then sum across the i dimension => (b, 1, d_embed_context).
        _, _, d_embed_context = h_c_N_flat.shape
        h_c_N_reshaped = h_c_N_flat.view(num_agents_max, batch_size, 1, d_embed_context)
        h_c_N_accumulator = h_c_N_reshaped.sum(dim=0)  # => (batch_size, 1, d_embed_context)

        # Now average across the actual number of agents (not counting padding).
        num_agents_per_sample = num_agents_per_sample.view(-1, 1, 1).float()  # (batch_size, 1, 1)
        average_h_c_N = h_c_N_accumulator / num_agents_per_sample  # (batch_size, 1, d_embed_context)

        # ------------------------------------------------------------
        # 6) Return results
        # ------------------------------------------------------------
        # att_scores: (batch_size, num_agents_max, num_agents_max)
        # average_h_c_N: (batch_size, 1, d_embed_context)
        return att_scores, average_h_c_N

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        return self.decoder(tgt, encoder_out, tgt_mask, src_tgt_mask)

    def local_forward(
            self,
            local_agent_info,
            local_network,
            padding_mask,
            is_from_my_env,
            local_padding_flags,
    ):
        """
        :param local_agent_info: (batch_size, num_agents_max, obs_dim)
        :param local_network:    (batch_size, num_agents_max)
        :param padding_mask:     (batch_size, num_agents_max)
        :param is_from_my_env:   (batch_size,)
        :param local_padding_flags: (batch_size,)  # Checks if the agent of each batch is padded or not in local_forward
        :return: sub_att_scores: (batch_size, num_agents_max)
        """
        batch_size, num_agents_max, obs_dim = local_agent_info.shape

        # Initialize outputs for all sequences
        decoder_out = torch.zeros(batch_size, 1, self.d_embed_context, device=local_agent_info.device)
        h_c_N = torch.zeros(batch_size, 1, self.d_embed_context, device=local_agent_info.device)
        sub_att_scores = torch.full((batch_size, num_agents_max), -1e9, device=local_agent_info.device)

        # Get mask for non-padded sequences
        non_padded_mask = local_padding_flags.bool()  # (batch_size,)

        if non_padded_mask.any():  # If there is at least one non-padded sequence
            # Select non-padded sequences
            local_agent_info_np = local_agent_info[non_padded_mask]  # (n_non_padded, num_agents_max, obs_dim)
            local_network_np = local_network[non_padded_mask]  # (n_non_padded, num_agents_max)
            is_from_my_env_np = is_from_my_env[non_padded_mask]  # (n_non_padded,)

            # Masks for non-padded sequences
            src_mask_tokens = local_network_np.ne(0)  # (n_non_padded, num_agents_max)
            src_mask_idx = 0
            src_mask = self.make_src_mask(src_mask_tokens, mask_idx=src_mask_idx)
            tgt_mask = None
            context_mask_token = torch.ones_like(src_mask_tokens[:, 0:1], dtype=torch.bool)
            src_tgt_mask = self.make_src_tgt_mask(src_mask_tokens, context_mask_token, mask_idx=src_mask_idx)

            # Encoding
            encoder_out = self.encode(local_agent_info_np, src_mask.unsqueeze(1))

            # Context embedding
            h_c_N_np = self.get_context_vector(
                embeddings=encoder_out,
                pad_tokens=~src_mask_tokens,
                is_from_my_env=is_from_my_env_np,
                use_embeddings_mask=True,
                debug=True
            )  # (n_non_padded, 1, d_embed_context)

            # Decoding
            decoder_out_np = self.decode(h_c_N_np, encoder_out, tgt_mask, src_tgt_mask.unsqueeze(1))

            # Generator
            sub_att_scores_np = self.generator(input_query=decoder_out_np, input_key=encoder_out,
                                               mask=src_tgt_mask).squeeze(1)  # (n_non_padded, num_agents_max)

            # Update results for non-padded sequences
            decoder_out[non_padded_mask] = decoder_out_np
            h_c_N[non_padded_mask] = h_c_N_np
            sub_att_scores[non_padded_mask] = sub_att_scores_np
        else:
            print("WARNING All samples are padded in the batch. If this is not expected such as "
                  "parallelized forward, check the padding mask.")  # TODO: Also need to check env_check mode

        return sub_att_scores, decoder_out, h_c_N

    def get_context_vector(self, embeddings, pad_tokens, is_from_my_env, use_embeddings_mask=True, debug=False):
        # embeddings: shape (batch_size, num_agents_max==seq_len_src, data_size==d_embed_input)
        # pad_tokens: shape (batch_size, num_agents_max==seq_len_src)
        # is_from_my_env: shape (batch_size,)

        # Obtain batch_size, num_agents, data_size from embeddings
        batch_size, num_agents_max, data_size = embeddings.shape

        if use_embeddings_mask:  # TODO: Could be way simpler
            # Expand the dimensions of pad_tokens to match the shape of embeddings
            # # mask==1: padding, mask==0: non-padding
            mask = pad_tokens.unsqueeze(-1).expand_as(embeddings)  # (batch_size, num_agents_max, data_size)

            # Replace masked values with zero for the average computation
            # embeddings_masked: (batch_size, num_agents_max, data_size)
            embeddings_masked = torch.where(mask == 0, embeddings, torch.zeros_like(embeddings))

            # Compute the sum and count non-zero elements
            embeddings_sum = torch.sum(embeddings_masked, dim=1, keepdim=True)  # (batch_size, 1, data_size)
            embeddings_count = torch.sum((mask == 0), dim=1, keepdim=True).float()  # (batch_size, 1, data_size)

            # Check if there is any sample where all agents are padded
            if debug:
                if is_from_my_env.dtype == torch.bool:  # if it isn't in the env_check mode
                    if torch.any(embeddings_count == 0):  # is supposed to never happen due to self-loops
                        raise ValueError("All agents are padded in at least one sample.")
            # Compute the average embeddings, only for non-masked elements
            embeddings_avg = embeddings_sum / embeddings_count
        else:
            # Compute the average embeddings: shape (batch_size, 1, data_size)
            embeddings_avg = torch.mean(embeddings, dim=1, keepdim=True)  # num_agents_max dim is reduced

        # Construct context embedding: shape (batch_size, 1, d_embed_context)
        # The resulting tensor, h_c, will have shape (batch_size, 1, d_embed_context)
        # Concatenate the additional info to h_c, if you need more info for the context vector.
        h_c = embeddings_avg  # no concatenation in this project
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

    def forward_og(self, obs_dict: Dict[str, TensorType]):
        # Get data
        agent_infos = obs_dict["local_agent_infos"]  # (batch_size, num_agents_max, num_agents_max, obs_dim)
        network = obs_dict["neighbor_masks"]  # (batch_size, num_agents_max, num_agents_max); (:,i,:): i-th agent's net
        padding_mask = obs_dict["padding_mask"]  # (batch_size, num_agents_max); applies over all agents same
        is_from_my_env = obs_dict["is_from_my_env"]  # (batch_size,); 1: from my env, 0: under the env_check
        # Caution: masks are torch FLOAT tensors, not boolean tensors, which I don't like in RLlib (v2.1.0)

        batch_size, num_agents_max, _, obs_dim = agent_infos.shape

        """
        In forward_og, padding agents are not *fully* supported in the model (network: src_maskS, pad_mask: tgt_mask ?)
        """

        # Get sub-attention scores
        att_scores = torch.zeros_like(network, dtype=torch.float32)  # (batch_size, num_agents_max, num_agents_max)
        num_agents = padding_mask.sum(dim=1).int()  # (batch_size,); number of agents in each sample
        h_c_N_accumulator = torch.zeros(batch_size, 1, self.d_embed_context, device=agent_infos.device)

        for i in range(num_agents_max):  # In the latest ver, pushed agent dim onto batch dim to parallelize
            local_agent_info = agent_infos[:, i, :, :]
            local_network = network[:, i, :]
            local_padding_flags = padding_mask[:, i]

            # sub_att_scores: shape: (batch_size, num_agents_max)
            # h_c_N: shape: (batch_size, 1, d_embed_context)
            sub_att_scores, _, h_c_N = self.local_forward_og(local_agent_info, local_network, padding_mask, is_from_my_env, local_padding_flags)

            # Get the i-th row of the attention scores
            att_scores[:, i, :] = sub_att_scores

            # Accumulate h_c_N values
            h_c_N_accumulator += h_c_N  # In latest, 패딩 에이전트 안 더해짐.

        # Calculate average_h_c_N
        num_agents = num_agents.view(-1, 1, 1).float()  # (batch_size, 1, 1)
        average_h_c_N = h_c_N_accumulator / num_agents

        # att_scores: (batch_size, num_agents_max, num_agents_max)
        # average_h_c_N: (batch_size, 1, d_embed_context)
        return att_scores, average_h_c_N

    def local_forward_og(
            self,
            local_agent_info,
            local_network,
            padding_mask,
            is_from_my_env,
            local_padding_flags,
    ):
        """
        :param local_agent_info: (batch_size, num_agents_max, obs_dim)
        :param local_network:    (batch_size, num_agents_max)
        :param padding_mask:     (batch_size, num_agents_max)
        :param is_from_my_env:   (batch_size,)
        :param local_padding_flags: (batch_size,)
        :return: sub_att_scores: (batch_size, num_agents_max)
        """

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
