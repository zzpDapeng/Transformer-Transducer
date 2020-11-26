# -*- coding: utf-8 -*-
# @Author  : Dapeng
# @File    : audio_augment.PY
# @Desc    : 
# @Contact : zzp_dapeng@163.com
# @Time    : 2020/11/23 下午3:44
import numpy as np

from augment.noise_augment import gaussian_white_noise_numpy
from augment.speed_augment import speed_baidu
from augment.time_shift_augment import time_shift_numpy
from augment.volume_augment import volume_baidu


def audio_augment(samples):
    if np.random.random() < 0.4:
        samples = gaussian_white_noise_numpy(samples, min_db=1, max_db=10)
    if np.random.random() < 0.4:
        samples = volume_baidu(samples)
    if np.random.random() < 0.4:
        samples = speed_baidu(samples)
    if np.random.random() < 0.1:
        samples = time_shift_numpy(samples)
    return samples