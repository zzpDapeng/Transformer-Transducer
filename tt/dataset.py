import os

import numpy as np
import pandas

import tt.kaldi_io as kaldi_io
from tt.utils import get_feature, read_wave_from_file, concat_frame, subsampling


class Dataset:
    def __init__(self, config, type, dic):
        self.type = type
        self.word2index = dic
        self.name = config.name
        self.feature_dim = config.feature_dim
        self.left_context_width = config.left_context_width
        self.right_context_width = config.right_context_width
        self.subsample = config.subsample
        self.apply_cmvn = config.apply_cmvn
        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length

        # self.arkscp = os.path.join(config.__getattr__(type), 'feats.scp')

        if self.apply_cmvn:
            self.utt2spk = {}
            with open(os.path.join(config.__getattr__(type), 'utt2spk'), 'r') as fid:
                for line in fid:
                    parts = line.strip().split()
                    self.utt2spk[parts[0]] = parts[1]
            self.cmvnscp = os.path.join(config.__getattr__(type), 'cmvn.scp')
            self.cmvn_stats_dict = {}
            self.get_cmvn_dict()

        # self.feats_list, self.feats_dict = self.get_feats_list()

    def __len__(self):
        raise NotImplementedError

    def pad(self, inputs, max_length=None):
        dim = len(inputs.shape)
        if dim == 1:
            if max_length is None:
                max_length = self.max_target_length
            pad_zeros_mat = np.zeros([1, max_length - inputs.shape[0]], dtype=np.int32)
            padded_inputs = np.column_stack([inputs.reshape(1, -1), pad_zeros_mat])
        elif dim == 2:
            if max_length is None:
                max_length = self.max_input_length
            feature_dim = inputs.shape[1]
            pad_zeros_mat = np.zeros([max_length - inputs.shape[0], feature_dim])
            padded_inputs = np.row_stack([inputs, pad_zeros_mat])
        else:
            raise AssertionError(
                'Features in inputs list must be one vector or two dimension matrix! ')
        return padded_inputs

    def get_cmvn_dict(self):
        cmvn_reader = kaldi_io.read_mat_scp(self.cmvnscp)
        for spkid, stats in cmvn_reader:
            self.cmvn_stats_dict[spkid] = stats

    def cmvn(self, mat, stats):
        mean = stats[0, :-1] / stats[0, -1]
        variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
        return np.divide(np.subtract(mat, mean), np.sqrt(variance))


class AudioDataset(Dataset):
    def __init__(self, config, type, dic):
        super(AudioDataset, self).__init__(config, type, dic)
        self.config = config
        self.short_first = config.short_first
        self.df = pandas.read_csv(config.__getattr__(type), index_col=False)
        # TODO:短优先排序，
        # if self.short_first and type == 'train':
        #     self.sorted_list = sorted(self.targets_dict.items(), key=lambda x: len(x[1]), reverse=False)
        # else:
        #     self.sorted_list = None

    def __getitem__(self, index):

        audio_path = self.df.iloc[index, 0]
        label = self.df.iloc[index, 1]

        targets = np.array(self.encode(label))
        wave_data, frame_rate = read_wave_from_file(audio_path)
        features = get_feature(wave_data, frame_rate, self.feature_dim)
        # features = np.load(feats_scp)
        features = concat_frame(features, self.left_context_width, self.right_context_width)
        features = subsampling(features, self.subsample)

        inputs_length = np.array(features.shape[0]).astype(np.int64)
        targets_length = np.array(targets.shape[0]).astype(np.int64)

        features = self.pad(features).astype(np.float32)
        targets = self.pad(targets).astype(np.int64).reshape(-1)

        return features, inputs_length, targets, targets_length

    def __len__(self):
        return self.df.shape[0]

    # 将sym 序列编码为idx
    def encode(self, seq):
        # encoded_seq = [self.word2index.get('<SOS>')]
        encoded_seq = []
        for unit in seq:
            if unit in self.word2index:
                encoded_seq.append(self.word2index[unit])
            else:
                encoded_seq.append(self.word2index['<unk>'])  # TODO：没有设置该标志
        return encoded_seq
