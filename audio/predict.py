# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : predict.py
# @Desc    : 预测
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/8/11 上午10:20
import wave
import yaml
import torch
from tt.utils import AttrDict, read_wave_from_file, get_feature, concat_frame, subsampling, dict_map
from tt.model import Transducer

WAVE_OUTPUT_FILENAME = 'audio/output_15s.wav'


def init_model():
    config_file = open("config/joint.yaml", encoding='utf-8')
    config = AttrDict(yaml.load(config_file, Loader=yaml.FullLoader))
    model = Transducer(config.model)
    checkpoint = torch.load(config.training.load_model)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.decoder.load_state_dict(checkpoint['decoder'])
    model.joint.load_state_dict(checkpoint['joint'])
    model = model.cuda()
    model.eval()
    print("已加载模型")

    vocab = {}
    with open(config.data.vocab, "r") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            index = int(parts[1])
            vocab[index] = word
    print("已加载词典")

    return model, vocab


def pred():
    model, vocab = init_model()

    audio, fr = read_wave_from_file(WAVE_OUTPUT_FILENAME)
    feature = get_feature(audio, fr)
    feature = concat_frame(feature, 3, 0)
    feature = subsampling(feature, 3)
    feature = torch.from_numpy(feature)
    feature = torch.unsqueeze(feature, 0)
    feature = feature.cuda()
    len = feature.shape[1]
    len = torch.tensor([len])
    len = len.cuda()
    preds = model.recognize(feature, len)
    preds = dict_map(preds, vocab)
    print(''.join(preds[0]))
