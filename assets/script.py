# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : script.py
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/8/5 下午3:11

import yaml
import torch
from tt.model import Transducer
from tt.utils import AttrDict

config_file = open("../config/joint.yaml")
config = AttrDict(yaml.load(config_file, Loader=yaml.FullLoader))
model = Transducer(config.model)

checkpoint = torch.load(config.training.load_model)
model.encoder.load_state_dict(checkpoint['encoder'])
model.decoder.load_state_dict(checkpoint['decoder'])
model.joint.load_state_dict(checkpoint['joint'])
model.eval()

audio = torch.rand(1, 5, 512)
encoder_script = torch.jit.trace(model.encoder, (audio,))
encoder_script.save('encoder.pt')
print("Generate encoder script")

token_list = [0, 1, 8, 6, 4]
token = torch.tensor([token_list], dtype=torch.long)
decoder_script = torch.jit.trace(model.decoder, (token,))
decoder_script.save('decoder.pt')
print("Generate decoder script")


joint_example = (audio[:, 1, :].view(-1), torch.rand(1, 1, 512).view(-1))
joint_script = torch.jit.trace(model.joint, joint_example)
joint_script.save('joint.pt')
print("Generate joint script")

