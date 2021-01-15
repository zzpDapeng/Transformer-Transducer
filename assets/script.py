# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : script.py
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/8/5 下午3:11
import os
import yaml
import torch
from tt.model import Transducer
from tt.utils import AttrDict, read_wave_from_file, get_feature, concat_frame, subsampling, context_mask

os.chdir('../')

WAVE_OUTPUT_FILENAME = 'audio/5_1812_20170628135834.wav'

# 加载模型
config_file = open("config/joint_streaming.yaml")
config = AttrDict(yaml.load(config_file, Loader=yaml.FullLoader))
model = Transducer(config.model)

checkpoint = torch.load(config.training.load_model)
model.encoder.load_state_dict(checkpoint['encoder'])
model.decoder.load_state_dict(checkpoint['decoder'])
model.joint.load_state_dict(checkpoint['joint'])
print('加载模型')
model.eval()

# 获取音频特征
audio, fr = read_wave_from_file(WAVE_OUTPUT_FILENAME)
feature = get_feature(audio, fr)
feature = concat_frame(feature, 3, 0)
feature = subsampling(feature, 3)
feature = torch.from_numpy(feature)
feature = torch.unsqueeze(feature, 0)
# feature = feature[:, :5, :]
print(feature.shape)

mask = context_mask(feature)[:, :, None]
print(mask.shape)

# torch.jit.script
# script_module = torch.jit.script(model)
# script_module.save('model.pt')

# TODO：transformer.py 158行左右，为了导出pytorch脚本，不能有if_else语句，因此默认设置为else，即音频长度不超过410标签长度不超过42
# encoder_script = torch.jit.trace(model.encoder, (feature, mask))
# encoder_script.save('encoder.pt')
# print("Generate encoder script")
#
# token_list = [0, 1, 8, 6, 4]
# token = torch.tensor([token_list], dtype=torch.long)
# decoder_script = torch.jit.trace(model.decoder, (token,))
# decoder_script.save('decoder.pt')
# print("Generate decoder script")

joint_example = (feature[:, 1, :].view(-1), torch.rand(1, 1, 512).view(-1))
joint_script = torch.jit.trace(model.joint, joint_example)
joint_script.save('joint.pt')
print("Generate joint script")
