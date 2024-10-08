import math
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, d_embed, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_embed)
        self.d_embed = d_embed

    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_embed)
        return out  # shape: (batch_size, seq_len, d_embed)


class LinearEmbedding(nn.Module):

    def __init__(self, d_env, d_embed):
        super(LinearEmbedding, self).__init__()
        self.embedding = nn.Linear(d_env, d_embed)
        self.d_embed = d_embed
        self.in_features = d_env

    def forward(self,
                x  # shape: (batch_size, seq_len, d_env); d_env: dimension of ray environment observation to process
                ):
        out = self.embedding(x)
        return out  # shape: (batch_size, seq_len, d_embed)
