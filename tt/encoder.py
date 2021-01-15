import torch
import torch.nn as nn
from tt.transformer import RelLearnableDecoderLayer
import numpy as np


class BaseEncoder(nn.Module):
    def __init__(self,
                 k_len,
                 n_head,
                 d_model,
                 d_head,
                 d_inner,
                 dropout,
                 **kwargs):
        super(BaseEncoder, self).__init__()

        self.r_emb = nn.Parameter(torch.randn((k_len, n_head, d_head), dtype=torch.float32))
        self.r_w_bias = nn.Parameter(torch.randn((n_head, d_head), dtype=torch.float32))
        self.r_bias = nn.Parameter(torch.randn((k_len, n_head), dtype=torch.float32))

        self.MultiHeadAttention = RelLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, **kwargs, )

    def forward(self, inputs, enc_attn_mask=None):
        assert inputs.dim() == 3

        outputs = self.MultiHeadAttention(inputs, self.r_emb, self.r_w_bias, self.r_bias, enc_attn_mask)

        return outputs


class BuildEncoder(nn.Module):
    def __init__(self, config):
        super(BuildEncoder, self).__init__()
        self.layers = nn.ModuleList([BaseEncoder(
            k_len=config.enc.max_input_length,
            n_head=config.enc.n_head,
            d_model=config.enc.d_model,
            d_head=config.enc.d_head,
            d_inner=config.enc.d_inner,
            dropout=config.dropout)
            for i in range(config.enc.n_layer)])

    def forward(self, inputs, mask=None):
        inputs = torch.transpose(inputs, 0, 1)

        for layer in self.layers:
            inputs = layer(inputs, mask)

        return torch.transpose(inputs, 0, 1)
