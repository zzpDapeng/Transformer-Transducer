import wave
import torch
import random
import logging
import librosa
import numpy as np
import editdistance
import matplotlib.pyplot as plt


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]

    def __setattr__(self, item, value):
        self.__dict__[item] = value


def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


def computer_cer(preds, labels):
    """计算一个批次的cer"""
    dist = sum(editdistance.eval(label, pred) for label, pred in zip(labels, preds))
    total = sum(len(l) for l in labels)
    return dist, total


def get_saved_folder_name(config):
    return '_'.join([config.data.name, config.training.save_model])


def count_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
    return n_params, enc, dec


def init_parameters(model, type='xnormal'):
    for p in model.parameters():
        if p.dim() > 1:
            if type == 'xnoraml':
                torch.nn.init.xavier_normal_(p)
            elif type == 'uniform':
                torch.nn.init.uniform_(p, -0.1, 0.1)
        else:
            pass


def save_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    checkpoint = {
        'encoder': model.module.encoder.state_dict() if multi_gpu else model.encoder.state_dict(),
        'decoder': model.module.decoder.state_dict() if multi_gpu else model.decoder.state_dict(),
        'joint': model.module.joint.state_dict() if multi_gpu else model.joint.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': optimizer.current_epoch,
        'step': optimizer.global_step
    }

    torch.save(checkpoint, save_name)


def save_ctc_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    checkpoint = {
        'encoder': model.module.encoder.state_dict() if multi_gpu else model.encoder.state_dict(),
        'project_layer': model.module.project_layer.state_dict() if multi_gpu else model.project_layer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': optimizer.current_epoch,
        'step': optimizer.global_step
    }

    torch.save(checkpoint, save_name)


def save_language_model(model, optimizer, config, save_name):
    multi_gpu = True if config.training.num_gpu > 1 else False
    checkpoint = {
        'decoder': model.module.decoder.state_dict() if multi_gpu else model.decoder.state_dict(),
        'project_layer': model.module.project_layer.state_dict() if multi_gpu else model.project_layer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': optimizer.current_epoch,
        'step': optimizer.global_step
    }

    torch.save(checkpoint, save_name)


def concat_frame(features, left_context_width, right_context_width):
    time_steps, features_dim = features.shape
    concated_features = np.zeros(
        shape=[time_steps, features_dim *
               (1 + left_context_width + right_context_width)],
        dtype=np.float32)
    # middle part is just the uttarnce
    concated_features[:, left_context_width * features_dim:
                         (left_context_width + 1) * features_dim] = features

    for i in range(left_context_width):
        # add left context
        concated_features[i + 1:time_steps,
        (left_context_width - i - 1) * features_dim:
        (left_context_width - i) * features_dim] = features[0:time_steps - i - 1, :]

    for i in range(right_context_width):
        # add right context
        concated_features[0:time_steps - i - 1,
        (right_context_width + i + 1) * features_dim:
        (right_context_width + i + 2) * features_dim] = features[i + 1:time_steps, :]

    return concated_features


def subsampling(features, subsample=3):
    interval = subsample
    temp_mat = [features[i]
                for i in range(0, features.shape[0], interval)]
    subsampled_features = np.row_stack(temp_mat)
    return subsampled_features


def generate_dictionary(path):
    index2word = {}
    word2index = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        word_index = str(line).strip().split(' ')
        index = word_index[1]
        index = int(index)
        word = word_index[0]
        index2word[index] = word
        word2index[word] = index
    return index2word, word2index


def read_wave_from_file(audio_file):
    """
    return 一维numpy数组，如（584,） 采样率"""
    wav = wave.open(audio_file, 'rb')
    num_frames = wav.getnframes()
    framerate = wav.getframerate()
    str_data = wav.readframes(num_frames)
    wav.close()
    wave_data = np.frombuffer(str_data, dtype=np.short)
    return wave_data, framerate


def get_feature(wave_data, framerate, feature_dim=128):
    """
    :param wave_data: 一维numpy,dtype=int16
    :param framerate:
    :param feature_dim:
    :return: specgram [序列长度,特征维度]
    """
    wave_data = wave_data.astype("float32")
    specgram = librosa.feature.melspectrogram(wave_data, sr=framerate, n_fft=512, hop_length=160, n_mels=feature_dim)
    specgram = np.ma.log(specgram).T
    specgram = specgram.filled(0)
    return specgram


def get_feature2(wave_data, framerate, feature_dim=128):
    """
    :param wave_data: 一维numpy,dtype=int16
    :param framerate:
    :param feature_dim:
    :return: specgram [序列长度,特征维度]
    """
    wave_data = wave_data.astype("float32")
    specgram = librosa.feature.melspectrogram(wave_data, sr=framerate, n_fft=512, hop_length=160, n_mels=feature_dim)
    specgram = np.where(specgram == 0, np.finfo(float).eps, specgram)
    specgram = np.log10(specgram).T
    return specgram


def get_final_feature(samples, sample_rate=16000, feature_dim=128, left=3, right=0, subsample=3):
    feature = get_feature(samples, sample_rate, feature_dim)
    feature = concat_frame(feature, left, right)
    feature = subsampling(feature, subsample)
    return feature


def dict_map(preds, vocab):
    res = []
    for batch in range(len(preds)):
        batch_res = []
        for i in range(len(preds[batch])):
            word = vocab[preds[batch][i]]
            batch_res.append(word)
        res.append(batch_res)
    return res


def write_result(preds, transcripts, epoch):
    with open("decode_" + str(epoch) + ".txt", "a") as f:
        for batch in range(len(transcripts)):
            f.writelines("Transcripts:" + "".join(transcripts[batch]) + "\n")
            f.writelines("---Predicts:" + "".join(preds[batch]) + "\n")


def look_ahead_mask(label):
    """
    用于标签的前瞻遮挡
    """
    seq_len = label.size(1)
    mask_look_ahead = torch.triu(label.new_ones([seq_len, seq_len]), diagonal=1).bool()
    return mask_look_ahead


def context_mask(audio, left_context=10, right_context=2):
    """
    用于流式语音识别的上下文遮挡
    """
    seq_len = audio.size(1)
    up = torch.triu(audio.new_ones([seq_len, seq_len]), diagonal=right_context + 1)
    down = torch.tril(audio.new_ones([seq_len, seq_len]), diagonal=-left_context - 1)
    # mask_context = (up + down).bool()
    mask_context = (up + down)  # 不转换成bool，在模型中使用mask时再转换，因为安卓端不支持bool类型tensor
    return mask_context


def padding_mask(inputs):
    """
    标签和音频都可用的填充遮挡
    todo:有问题，训练时输出全是0
    """
    if inputs.dim() == 2:
        zero = torch.zeros_like(inputs)
        one = torch.ones_like(inputs)
        mask_padding = torch.where(inputs == 0, one, zero).bool()
    else:
        zero = torch.zeros_like(inputs[:, :, 0])
        one = torch.ones_like(inputs[:, :, 0])
        mask_padding = torch.sum(inputs, dim=-1)
        mask_padding = torch.where(mask_padding == 0., one, zero).bool()
    return mask_padding.transpose(0, 1)


def create_mask(audio, label, left_context=None, right_context=None):
    """
    padding mask有问题，所以这个也不能用
    """
    if left_context is None and right_context is None:
        # 非流式mask
        audio_padding_mask = padding_mask(audio)
        label_padding_mask = padding_mask(label)
        label_look_ahead_mask = look_ahead_mask(label)
        label_mask = torch.max(label_look_ahead_mask[:, :, None], label_padding_mask[:, None, :])
        return audio_padding_mask, label_mask
    else:
        audio_padding_mask = padding_mask(audio)
        audio_context_mask = context_mask(audio, left_context, right_context)
        audio_mask = torch.max(audio_context_mask[:, :, None], audio_padding_mask[:, None, :])
        label_padding_mask = padding_mask(label)
        label_look_ahead_mask = look_ahead_mask(label)
        label_mask = torch.max(label_look_ahead_mask[:, :, None], label_padding_mask[:, None, :])
        return audio_mask, label_mask


def label_smoothing(inputs, epsilon=0.1):
    K = inputs.shape.as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def time_mask_augment(inputs, max_mask_time=5, mask_num=10):
    """
    时间遮掩，
    :param inputs: 三维numpy或tensor，(batch, time_step,  feature_dim)
    :param max_mask_time:
    :param mask_num:
    :return:
    """
    time_len = inputs.shape[1]
    for i in range(mask_num):
        t = np.random.uniform(low=0.0, high=max_mask_time)
        t = int(t)
        t0 = random.randint(0, time_len - t)
        inputs[:, t0:t0 + t, :] = 0

    return inputs


def frequency_mask_augment(inputs, max_mask_frequency=5, mask_num=10):
    """

    :param inputs: 三维numpy或tensor，(batch, time_step,  feature_dim)
    :param max_mask_frequency:
    :param mask_num:
    :return:
    """
    feature_len = inputs.shape[2]
    for i in range(mask_num):
        f = np.random.uniform(low=0.0, high=max_mask_frequency)
        f = int(f)
        f0 = random.randint(0, feature_len - f)
        inputs[:, :, f0:f0 + f] = 0
    return inputs


def tensor_to_img(spectrogram):
    plt.figure()  # arbitrary, looks good on my screen.
    # plt.imshow(spectrogram[0].T)
    plt.imshow(spectrogram.T)
    plt.show()


def save_wav(file_name, audio_data, channels=1, sample_width=2, rate=16000):
    wf = wave.open(file_name, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(rate)
    wf.writeframes(b''.join(audio_data))
    wf.close()


if __name__ == '__main__':
    audio, sr = read_wave_from_file("../audio.wav")
    feature = get_feature(audio, sr)
    tensor_to_img(feature[:500])
    feature = get_feature2(audio, sr)
    tensor_to_img(feature[:500])
