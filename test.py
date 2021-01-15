"""
@File:test.py
@Desciption:
@Author:Dapeng
@Contact:zzp_dapeng@163.com
@Time:2020/7/5 下午3:04 
"""
from audio.streamRec_unlimit_dynamic_window import streaming_asr

streaming_asr()

# import torch
#
# k_len = 3
# r_bias = torch.rand(5, 6)
# print(r_bias)
# # 二维
# eye = torch.eye(k_len, 5)
# eye = torch.flip(eye, dims=[0, 1])
# res = torch.matmul(eye, r_bias)
# print(r_bias[-k_len:])
# print(res)
#
# # 三维
# r_emb = torch.rand((5, 6, 4))
# eye = torch.eye(k_len, 5)
# eye = torch.flip(eye, dims=[0, 1])
# r_emb = r_emb.reshape(5, -1)
# res = torch.matmul(eye, r_emb)
# r_emb = r_emb.reshape(5, 6, 4)
# res = res.reshape(k_len, 6, 4)

