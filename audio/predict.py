# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : predict.py
# @Desc    : 预测
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/8/11 上午10:20
import wave
import os
import yaml
import torch
from tt.utils import AttrDict, read_wave_from_file, get_feature, concat_frame, subsampling, dict_map, context_mask, \
    computer_cer
from tt.model import Transducer

os.chdir('../')

# WAVE_OUTPUT_FILENAME = 'audio/luyin.wav'


WAVE_OUTPUT_FILENAME = 'audio/output_15s.wav'


def init_model():
    config_file = open("config/joint_streaming.yaml", encoding='utf-8')
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
    audio_mask = context_mask(feature)[:, :, None]  # 流式语音识别
    # preds = model.recognize(feature, len, audio_mask)
    preds = model.recognize_beam_search(feature, len, audio_mask)
    preds = dict_map(preds, vocab)
    groundtruth = ["疑点之一美方对境内疫情发展时时间线一直讳莫如深唯恐避之不及这不由令人质疑其疫情爆发的时间起点疑点之二"]
    # groundtruth = ["那叫名人呢干嘛要划类啊一分类就有就有帮派了嘛人不要那么化类就是会有对立面不好所以我说通常有命题的话题都不要提"]
    res = ''.join(preds[0])
    dist, num = computer_cer([res], groundtruth)
    print(dist / num, res)


if __name__ == '__main__':
    pred()
