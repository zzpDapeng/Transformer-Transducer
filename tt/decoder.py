import torch
import torch.nn as nn
from tt.transformer import RelLearnableDecoderLayer


class BaseDecoder(nn.Module):

    def __init__(self, vocab_size, n_layer, k_len, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(BaseDecoder, self).__init__()

        self.r_emb = nn.Parameter(torch.randn((k_len, n_head, d_head),dtype=torch.float32))
        self.r_w_bias = nn.Parameter(torch.randn((n_head, d_head), dtype=torch.float32))
        self.r_bias = nn.Parameter(torch.randn((k_len, n_head), dtype=torch.float32))

        self.MultiHeadAttention = RelLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, **kwargs)

    def forward(self, inputs, attn_mask=None):

        outputs = self.MultiHeadAttention(inputs, self.r_emb, self.r_w_bias, self.r_bias, attn_mask)

        return outputs


class BuildDecoder(nn.Module):
    def __init__(self, config):
        super(BuildDecoder, self).__init__()
        self.dec_embedding = nn.Embedding(config.vocab_size, config.dec.d_model, padding_idx=0)  #TODO：是否替换？
        self.layers = nn.ModuleList([BaseDecoder(
            vocab_size=config.vocab_size,
            n_layer=config.dec.n_layer,
            k_len=config.dec.max_target_length,
            n_head=config.dec.n_head,
            d_model=config.dec.d_model,
            d_head=config.dec.d_head,
            d_inner=config.dec.d_inner,
            dropout=config.dropout)
            for i in range(config.dec.n_layer)])

    def forward(self, inputs):
        inputs = self.dec_embedding(inputs)
        inputs = torch.transpose(inputs, 0, 1)

        qlen, dez = inputs.size(0), inputs.size(1)
        attn_mask = torch.triu(inputs.new_ones(qlen, qlen), diagonal=1).bool()[:, :, None]
        # attn_mask = torch.triu(inputs.new_ones(qlen, klen), diagonal=1).bool()[:, :, None]  # todo：包括主对角线 报错nan

        for layer in self.layers:
            inputs = layer(inputs, attn_mask)

        return torch.transpose(inputs, 0, 1)
