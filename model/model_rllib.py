# Everything is copy
import copy
# Please let me get out of ray rllib
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2, ModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.annotations import override
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
#
# From envs
import numpy as np
from gym.spaces import Discrete
from typing import List, Union, Dict
#
# Pytorch
import torch
import torch.nn as nn
#
# For the custom model
from model.model_torch import ActorTest, CriticTest
from model.model_torch import LazyVicsekActor, LazyVicsekCritic
# Custom modules
from model.modules.token_embedding import LinearEmbedding
from model.modules.multi_head_attention_layer import MultiHeadAttentionLayer
from model.modules.position_wise_feed_forward_layer import PositionWiseFeedForwardLayer
from model.modules.encoder_block import EncoderBlock
from model.modules.decoder_block import CustomDecoderBlock as DecoderBlock
from model.modules.encoder import Encoder
from model.modules.decoder import Decoder, DecoderPlaceholder
from model.modules.pointer_net import GaussianActionDistGenerator, GaussianActionDistPlaceholder
from model.modules.pointer_net import MeanGenerator, PointerPlaceholder, FakeMeanGenerator


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
            n_layers_decoder = cfg["n_layers_decoder"] if "n_layers_decoder" in cfg else 2
            h = cfg["num_heads"] if "num_heads" in cfg else 8
            d_ff = cfg["d_ff"] if "d_ff" in cfg else 512
            d_ff_decoder = cfg["d_ff_decoder"] if "d_ff_decoder" in cfg else 512
            clip_action_mean = cfg["clip_action_mean"] if "clip_action_mean" in cfg else 1.0
            clip_action_log_std = cfg["clip_action_log_std"] if "clip_action_log_std" in cfg else 10.0
            dr_rate = cfg["dr_rate"] if "dr_rate" in cfg else 0
            norm_eps = cfg["norm_eps"] if "norm_eps" in cfg else 1e-5
            is_bias = cfg["is_bias"] if "is_bias" in cfg else True  # bias in MHA linear layers (W_q, W_k, W_v)
            use_residual_in_decoder = cfg["use_residual_in_decoder"] if "use_residual_in_decoder" in cfg else True
            use_FNN_in_decoder = cfg["use_FNN_in_decoder"] if "use_FNN_in_decoder" in cfg else True
            use_deterministic_action_dist = cfg["use_deterministic_action_dist"] \
                if "use_deterministic_action_dist" in cfg else False

            if "ignore_residual_in_decoder" in cfg:
                use_residual_in_decoder = not cfg["ignore_residual_in_decoder"]
                # DeprecationWarning: ignore_residual_in_decoder is deprecated; use_residual_in_decoder is used instead
                print("DeprecationWarning: ignore_residual_in_decoder is deprecated; use use_residual_in_decoder instead")
                use_FNN_in_decoder = use_residual_in_decoder
                print("use_FNN_in_decoder is set to use_residual_in_decoder as {}".format(use_residual_in_decoder))
            if use_residual_in_decoder != use_FNN_in_decoder:
                # Warning: use_residual_in_decoder != use_FNN_in_decoder; but don't change it
                warning_text = "Warning: use_residual_in_decoder != use_FNN_in_decoder; but don't change it"
                for i in range(7):
                    print(("%"*i) + warning_text + ("%"*i))
            if n_layers_decoder >= 2 and not use_residual_in_decoder:
                # Warning: multiple decoder blocks often require residual connections
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
            self_attention=None,  # No self-attention in the decoder_block in this case!
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
        if use_deterministic_action_dist:
            # generator = MeanGenerator(
            #     d_model=d_model,
            #     q_fc=nn.Linear(d_embed_context, d_model, is_bias),
            #     k_fc=nn.Linear(d_embed_input, d_model, is_bias),
            #     clip_value=clip_action_mean,
            #     dr_rate=dr_rate,
            # )
            generator = FakeMeanGenerator(  # is actually a continuous Gaussian dist with very small std
                d_model=d_model,
                q_fc=nn.Linear(d_embed_context, d_model, is_bias),
                k_fc=nn.Linear(d_embed_input, d_model, is_bias),
                clip_value_mean=clip_action_mean,
                clip_value_std=clip_action_log_std,  # all log_std-s are set to this value
                dr_rate=dr_rate,
            )
        else:  # Gaussian action distribution
            generator = GaussianActionDistGenerator(
                d_model=d_model,
                q_fc=nn.Linear(d_embed_context, d_model, is_bias),
                k_fc=nn.Linear(d_embed_input, d_model, is_bias),
                clip_value_mean=clip_action_mean,
                clip_value_std=clip_action_log_std,
                dr_rate=dr_rate,
            )  # outputs a probability distribution over the input sequence

        action_size = action_space.shape[0]  # it gives d given that action_space is a Box of d dimensions

        assert num_outputs == 2 * action_size, "num_outputs must be action_size; use deterministic action distribution"

        # 2. Define policy network
        self.policy_network = LazyVicsekActor(
            src_embed=input_embed,
            encoder=encoder,
            decoder=decoder,
            generator=generator,
        )

        # 3. Define value network
        self.values = None
        self.share_layers = share_layers
        if not self.share_layers:
            self.value_network = LazinessAllocator(
                src_embed=copy.deepcopy(input_embed),
                encoder=copy.deepcopy(encoder),
                # decoder=copy.deepcopy(decoder),
                decoder=DecoderPlaceholder(),
                # TODO: try this although PPO uses state-value function
                #      self.values should use h_c_N instead of h_c_N1 with a different value branch
                #      If so, the decoder is not used in the value network
                # generator=copy.deepcopy(generator),
                generator=GaussianActionDistPlaceholder() #if not use_deterministic_action_dist else PointerPlaceholder(),
            )

        self.value_branch = nn.Sequential(
            nn.Linear(in_features=d_embed_context, out_features=d_embed_context),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(in_features=d_embed_context, out_features=1),  # state-value function
        )





        input_token_dim = obs_space["local_agent_infos"].shape[2]
        # Actor
        self.actor = LazyVicsekActor(input_token_dim)
        # Critic
        self.critic = LazyVicsekCritic(input_token_dim)
        self.values = None

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):

        obs_dict = input_dict["obs"]

        att = self.actor(obs_dict)  # (batch_size, num_agents_max, num_agents_max)
        x = self.attention_scores_to_logits(att)  # (batch_size, num_agents_max * num_agents_max * 2)

        self.values = self.critic(obs_dict)  # (batch_size, 1)

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
        assert self.values.shape[1] == 1, "self.values.shape[1] != 1;  Must be (batch_size, 1)"
        value = self.values.squeeze(-1)
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