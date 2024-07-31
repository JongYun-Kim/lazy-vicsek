# Everything is copy
import copy
# Please let me get out of ray rllib
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
#
# From envs
from typing import List, Union, Dict
#
# Pytorch
import torch
import torch.nn as nn
#
# For the custom model
from model.model_torch import LazyVicsekListener
# Custom modules
from model.modules.token_embedding import LinearEmbedding
from model.modules.multi_head_attention_layer import MultiHeadAttentionLayer
from model.modules.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from model.modules.encoder_block import EncoderBlock
from model.modules.decoder_block import CustomDecoderBlock as DecoderBlock
from model.modules.encoder import Encoder
from model.modules.decoder import Decoder, DecoderPlaceholder
from model.modules.pointer_net import RawAttentionScoreGenerator, RawAttentionScoreGeneratorPlaceholder


class LazyVicsekModelPPO(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # Get model config
        if model_config is not None:
            cfg = model_config["custom_model_config"]
            share_layers = cfg["share_layers"] if "share_layers" in cfg else True
            d_subobs = cfg["d_subobs"] if "d_subobs" in cfg else ValueError("d_subobs must be specified")
            d_embed_input = cfg["d_embed_input"] if "d_embed_input" in cfg else 128
            d_embed_context = cfg["d_embed_context"] if "d_embed_context" in cfg else 128
            d_model = cfg["d_model"] if "d_model" in cfg else 128
            d_model_decoder = cfg["d_model_decoder"] if "d_model_decoder" in cfg else 128
            n_layers_encoder = cfg["n_layers_encoder"] if "n_layers_encoder" in cfg else 3
            n_layers_decoder = cfg["n_layers_decoder"] if "n_layers_decoder" in cfg else 1
            h = cfg["num_heads"] if "num_heads" in cfg else 8
            d_ff = cfg["d_ff"] if "d_ff" in cfg else 512
            d_ff_decoder = cfg["d_ff_decoder"] if "d_ff_decoder" in cfg else 512
            dr_rate = cfg["dr_rate"] if "dr_rate" in cfg else 0
            norm_eps = cfg["norm_eps"] if "norm_eps" in cfg else 1e-5
            is_bias = cfg["is_bias"] if "is_bias" in cfg else True  # bias in MHA linear layers (W_q, W_k, W_v)
            use_residual_in_decoder = cfg["use_residual_in_decoder"] if "use_residual_in_decoder" in cfg else True
            use_FNN_in_decoder = cfg["use_FNN_in_decoder"] if "use_FNN_in_decoder" in cfg else True

            if use_residual_in_decoder != use_FNN_in_decoder:
                warning_text = "Warning: use_residual_in_decoder != use_FNN_in_decoder; may cause unexpected behavior"
                for i in range(7):
                    print(("%"*i) + warning_text + ("%"*i))
            if n_layers_decoder >= 2 and not use_residual_in_decoder:
                warning_text = "Warning: multiple decoder blocks often require residual connections"
                for i in range(7):
                    print(("%"*i) + warning_text + ("%"*i))
        else:
            raise ValueError("model_config must be specified")

        # 1. Define layers

        # 1-1. Module Level: Encoder
        # Need an embedding layer for the input; 2->128 in the case of Kool2019
        input_embed = LinearEmbedding(
            d_env=d_subobs,
            d_embed=d_embed_input,
        )
        mha_encoder = MultiHeadAttentionLayer(
            d_model=d_model,
            h=h,
            q_fc=nn.Linear(d_embed_input, d_model, is_bias),
            kv_fc=nn.Linear(d_embed_input, d_model, is_bias),
            out_fc=nn.Linear(d_model, d_embed_input, is_bias),
            dr_rate=dr_rate,
        )
        position_ff_encoder = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed_input, d_ff),
            fc2=nn.Linear(d_ff, d_embed_input),
            dr_rate=dr_rate,
        )
        norm_encoder = nn.LayerNorm(d_embed_input, eps=norm_eps)
        # 1-2. Module Level: Decoder
        mha_decoder = MultiHeadAttentionLayer(
            d_model=d_model_decoder,
            h=h,
            q_fc=nn.Linear(d_embed_context, d_model_decoder, is_bias),
            kv_fc=nn.Linear(d_embed_input, d_model_decoder, is_bias),
            out_fc=nn.Linear(d_model_decoder, d_embed_context, is_bias),
            dr_rate=dr_rate,
        )
        position_ff_decoder = PositionWiseFeedForwardLayer(
            fc1=nn.Linear(d_embed_context, d_ff_decoder),
            fc2=nn.Linear(d_ff_decoder, d_embed_context),
            dr_rate=dr_rate,
        ) if use_FNN_in_decoder else None
        norm_decoder = nn.LayerNorm(d_embed_context, eps=norm_eps)

        # 1-3. Block Level
        encoder_block = EncoderBlock(
            self_attention=copy.deepcopy(mha_encoder),
            position_ff=copy.deepcopy(position_ff_encoder),
            norm=copy.deepcopy(norm_encoder),
            dr_rate=dr_rate,
        )
        decoder_block = DecoderBlock(
            self_attention=None,  # No (masked-)self-attention in the decoder_block in this case!
            cross_attention=copy.deepcopy(mha_decoder),
            position_ff=position_ff_decoder,  # No position-wise FFN in the decoder_block in most cases!
            norm=copy.deepcopy(norm_decoder),
            dr_rate=dr_rate,
            efficient=not use_residual_in_decoder,
        )

        # 1-4. Transformer Level (Encoder + Decoder + Generator)
        encoder = Encoder(
            encoder_block=encoder_block,
            n_layer=n_layers_encoder,
            norm=copy.deepcopy(norm_encoder),
        )
        decoder = Decoder(
            decoder_block=decoder_block,
            n_layer=n_layers_decoder,
            norm=copy.deepcopy(norm_decoder),
            # norm=nn.Identity(),
        )
        generator = RawAttentionScoreGenerator(
            d_model=d_model_decoder,
            q_fc=nn.Linear(d_embed_context, d_model_decoder, is_bias),
            k_fc=nn.Linear(d_embed_input, d_model_decoder, is_bias),
            dr_rate=dr_rate,
        )

        action_size = action_space.shape[0]  # num_agents_max?
        assert num_outputs == 2 * (action_size**2), \
            f"num_outputs != 2 * (action_size^2); num_output = {num_outputs}, action_size = {action_size}"

        # 2. Define policy network
        self.actor = LazyVicsekListener(
            src_embed=input_embed,
            encoder=encoder,
            decoder=decoder,
            generator=generator,
            d_embed_context=d_embed_context,
        )

        # 3. Define value network
        self.values = None
        self.share_layers = share_layers
        if not self.share_layers:
            self.critic = LazyVicsekListener(
                src_embed=copy.deepcopy(input_embed),
                encoder=copy.deepcopy(encoder),
                decoder=DecoderPlaceholder(),
                generator=RawAttentionScoreGeneratorPlaceholder(),
                d_embed_context=d_embed_context,
            )

        self.value_branch = nn.Sequential(
            nn.Linear(in_features=d_embed_context, out_features=d_embed_context),
            nn.ReLU(),
            nn.Linear(in_features=d_embed_context, out_features=1),  # state-value function
        )

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        obs_dict = input_dict["obs"]

        # att: (batch_size, num_agents_max, num_agents_max)
        # h_c_N: (batch_size, 1, d_embed_context)
        att, h_c_N = self.actor(obs_dict)
        x = self.attention_scores_to_logits(att)  # (batch_size, num_agents_max * num_agents_max * 2)

        if self.share_layers:
            self.values = h_c_N.squeeze(1)                      # (batch_size, d_embed_context)
        else:
            self.values = self.critic(obs_dict)[1].squeeze(1)   # (batch_size, d_embed_context)

        return x, state

    def attention_scores_to_logits(self, attention_scores: TensorType) -> TensorType:
        """
        Maps attention scores to logits to follow the action distribution format (binary in multinomial dist)
        :param attention_scores: (batch_size, num_agents_max, num_agents_max)
        :return:
        """
        batch_size = attention_scores.shape[0]
        num_agents_max = attention_scores.shape[1]

        # Attention schore scaling: tune this parameter..!
        # scale_factor = 5e-3
        scale_factor = 1.0
        attention_scores *= scale_factor

        # Self-loops: fill diag with large positive values
        large_val = 1e9
        attention_scores = attention_scores - torch.diag_embed(attention_scores.new_full((num_agents_max,), large_val))

        # Get negated attention scores
        negated_attention_scores = -attention_scores

        # Expand attention scores and negated attention scores
        z_expanded = attention_scores.unsqueeze(-1)  # (batch_size, num_agents_max, num_agents_max, 1)
        z_neg_expanded = negated_attention_scores.unsqueeze(-1)  # (batch_size, num_agents_max, num_agents_max, 1)

        # Concatenate them in the last dimension
        # z_concatenated: (batch_size, num_agents_max, num_agents_max, 2)
        z_concatenated = torch.cat((z_expanded, z_neg_expanded), dim=-1)

        # Reshape the tensor to 2D: (batch_size, num_agents_max * num_agents_max * 2)
        logits = z_concatenated.reshape(batch_size, num_agents_max * num_agents_max * 2)

        return logits  # (batch_size, num_agents_max * num_agents_max * 2)

    def value_function(self) -> TensorType:
        assert self.values is not None, "self.values is None"  # TODO: remove these assertions, once stable
        assert self.values.dim() == 2, "self.values.dim() != 2; NOT 2D"
        value = self.value_branch(self.values).squeeze(-1)  # (batch_size,)
        return value


class LazyListenerModelPPOTest(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)  # Initialize nn.Module first
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        # Get actor
        # assert obs_space["centralized_agents_info"].shape[2] == 4, "d_subobs is not 4"
        # self.actor = ActorTest(obs_space["centralized_agents_info"].shape[2])
        self.actor = ActorTest(4)

        # Get critic
        # self.critic = CriticTest(obs_space["centralized_agents_info"].shape[2])
        self.critic = CriticTest(4)
        self.values = None

    def forward(
            self,
            input_dict,
            state: List[TensorType],
            seq_lens: TensorType
    ) -> (TensorType, List[TensorType]):
        # Get and check the observation
        obs_dict = input_dict["obs"]

        x = self.actor(obs_dict)  # (batch_size, num_agents_max * num_agents_max * 2)
        self.values = self.critic(obs_dict)  # (batch_size, 1)

        return x, state

    def value_function(self) -> TensorType:
        value = self.values.squeeze(-1)  # (batch_size,)
        return value