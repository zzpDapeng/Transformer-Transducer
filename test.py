"""
@File:test.py
@Desciption:
@Author:Dapeng
@Contact:zzp_dapeng@163.com
@Time:2020/7/5 下午3:04 
"""
# import torch
#
# inputs = torch.randn((10,2,512),dtype=torch.float32)
# qlen = inputs.size(0)
# print(inputs.new_ones(qlen, qlen).bool())
# up = torch.triu(inputs.new_ones(qlen, qlen), diagonal=5+1)
# down = torch.tril(inputs.new_ones(qlen, qlen), diagonal=-(2+1))
# attn_mask = ~(up+down).bool()
# # attn_mask = torch.triu(inputs.new_ones(qlen, qlen), diagonal=0).bool()[:, :, None]  # todo 关注当前
# print(attn_mask.shape)
# print(attn_mask)
# print(inputs)

import numpy as np
import torch
import rnnt

# def generate_dictionary():
#     dictionary = {}
#     with open('data/aishell/grapheme_table.txt', 'r') as f:
#         lines = f.readlines()
#     for line in lines:
#         word_index = str(line).strip().split(' ')
#         index = word_index[1]
#         index = int(index)
#         word = word_index[0]
#         print(word)
#         dictionary[index] = word
#     return dictionary
#
# dic = generate_dictionary()
# print(dic[1])