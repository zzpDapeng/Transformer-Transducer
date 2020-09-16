import sys
import math
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('utils')


# 位置编码
# class PositionalEmbedding(nn.Module):
#     def __init__(self, dropout, dim, max_len=600):
#         super(PositionalEmbedding, self).__init__()
#         pe = torch.zeros(max_len, dim)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
#                               -(math.log(10000.0) / dim)))
#         pe[:, 0::2] = torch.sin(position.float() * div_term)
#         pe[:, 1::2] = torch.cos(position.float() * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#         self.dropout = nn.Dropout(p=dropout)
#         self.dim = dim
#
#     def forward(self, inputs_length, step=None):
#
#         batch_size = inputs_length.size(0)
#         time_steps = inputs_length.max().item()
#         if step is None:
#             pos_enc = self.pe[:, :time_steps].repeat(batch_size, 1, 1)
#         else:
#             pos_enc = self.pe[:, step].repeat(batch_size, 1, 1)
#         return pos_enc


# 用
class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, layer_norm_epsilon=1e-5):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

    def forward(self, inp):
        core_out = self.CoreNet(self.layer_norm(inp))
        output = self.layer_norm(inp + core_out)

        return output


# 用
class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m, :m] = torch.triu(mask[:m, :m])
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen - 1, x.size(2), x.size(3)),
                                   device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:, :, None, None]) \
            .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

        return x

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None):
        raise NotImplementedError


# 用
class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None):
        # w:  [batch, seq_len, d_model],[2, 18, 512]
        # w‘:  [seq_len, batch_size, d_model],[18, 2, 512]
        # r_emb: [klen, n_head, d_head], used for term B       klen是max_len
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)  # qlen seq_len    bsz batch_size

        w_heads = self.qkv_net(w)  # w_heads:[seq_len, batch_size, 3*n_head*d_model] [18, 2, 1536]
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3,
                                                   dim=-1)  # [seq_len, batch_size, n_head*d_head] [18, 2, 512]

        klen = w_head_k.size(0)  # seq_len 18

        w_head_q = w_head_q.view(qlen, bsz, self.n_head,
                                 self.d_head)  # [seq_len, batch_size, n_head, d_head]  [18, 2, 8, 64]
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):  # seq_len > max_len?
            r_emb_pad = r_emb[0:1].expand(klen - r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen - r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        # compute attention score
        rw_head_q = w_head_q + r_w_bias[None]  # seq_len x batch_size x n_head x d_head  None扩展维度，而不改变数据排列顺序

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))  # seq_len x seq_len x batch_size x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))  # seq_len x seq_len x batch_size x n_head
        D_ = r_bias[None, :, None]  # 1    x seq_len x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [seq_len x seq_len x batch_size x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                # attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))  # look_ahead_mask & context_mask
                attn_score += (attn_mask[None, :, :, None] * -1e9)
            elif attn_mask.dim() == 3:
                # attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))  # padding_mask TODO:检查维度填充是否合理
                attn_score += (attn_mask[:, :, :, None] * -1e9)

        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        attn_vec = attn_vec.contiguous().view(  # contiguous()把tensor变成在内存中连续分布的形式
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        output = self.layer_norm(w + attn_out)

        return output


# 用
class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs)

        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, r_emb, r_w_bias, r_bias, attn_mask=None):
        output = self.dec_attn(
            input, r_emb, r_w_bias, r_bias, attn_mask=attn_mask)
        output = self.pos_ff(output)
        output = self.dropout(output)
        return output
