import torch
import wave
import logging
import librosa
import editdistance
import numpy as np
import python_speech_features


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


def subsampling(features, frame_rate):
    if frame_rate != 10:
        interval = int(frame_rate / 10)
        temp_mat = [features[i]
                    for i in range(0, features.shape[0], interval)]
        subsampled_features = np.row_stack(temp_mat)
        return subsampled_features
    else:
        return features


def generate_dictionary(path):
    dictionary = {}
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        word_index = str(line).strip().split(' ')
        index = word_index[1]
        index = int(index)
        word = word_index[0]
        dictionary[index] = word
    return dictionary


def read_wave_from_file(audio_file):
    """
    return 一维numpy数组，如（584,） 采样率"""
    wav = wave.open(audio_file, 'rb')
    num_frames = wav.getnframes()
    framerate = wav.getframerate()
    str_data = wav.readframes(num_frames)
    wav.close()
    wave_data = np.frombuffer(str_data, dtype=np.short)
    return wave_data,framerate


def get_feature(wave_data, framerate):
    """
    :param wave_data: 一维numpy,dtype=int16
    :param framerate:
    :return: specgram [序列长度,特征维度]
    """
    wave_data = wave_data.astype("float32")
    specgram = librosa.feature.melspectrogram(wave_data, sr=framerate, n_fft=512, hop_length=160, n_mels=64)
    specgram = np.ma.log(specgram).T
    specgram = specgram.filled(0)
    return specgram

def dict_map(preds, vocab):
    res = np.empty(np.array(preds).shape,dtype=np.str)
    for batch in range(len(preds)):
        for i in range(len(preds[batch])):
            word = vocab[preds[batch][i]]
            res[batch][i] = word
    return res.tolist()

def write_result(preds, transcripts):
    with open("decode.txt", "a") as f:
        for batch in range(len(transcripts)):
            f.writelines("Transcripts:"+"".join(transcripts[batch])+"\n")
            f.writelines("---Predicts:"+"".join(preds[batch])+"\n")

if __name__ == '__main__':
    a = np.random.randint(0,100,(1,10,8))
    print(a.shape)
    print(a)
    b = concat_frame(a,3,0)
    c = subsampling(b, 30)
    print(b.shape)
    print(b)
    print(c.shape)
    print(c)
