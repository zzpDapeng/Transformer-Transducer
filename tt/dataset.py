import codecs
import copy
import numpy as np
import os
import tt.kaldi_io as kaldi_io
from tt.utils import get_feature, read_wave_from_file
import speechpy
import python_speech_features

class Dataset:
    def __init__(self, config, type):

        self.type = type
        self.name = config.name
        self.left_context_width = config.left_context_width
        self.right_context_width = config.right_context_width
        self.subsample = config.subsample
        self.apply_cmvn = config.apply_cmvn

        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length
        self.vocab = config.vocab

        self.arkscp = os.path.join(config.__getattr__(type), 'feats.scp')

        if self.apply_cmvn:
            self.utt2spk = {}
            with open(os.path.join(config.__getattr__(type), 'utt2spk'), 'r') as fid:
                for line in fid:
                   parts = line.strip().split()
                   self.utt2spk[parts[0]] = parts[1]
            self.cmvnscp = os.path.join(config.__getattr__(type), 'cmvn.scp')
            self.cmvn_stats_dict = {}
            self.get_cmvn_dict()

        self.feats_list, self.feats_dict = self.get_feats_list()

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

    # from feats.txt extract feats_list which correspond with utt_id
    def get_feats_list(self):
        feats_list = []
        feats_dict = {}
        with open(self.arkscp, 'r') as fid:
            for line in fid:
                key, path = line.strip().split(' ')
                feats_list.append(key)
                feats_dict[key] = path
        return feats_list, feats_dict

    def get_cmvn_dict(self):
        cmvn_reader = kaldi_io.read_mat_scp(self.cmvnscp)
        for spkid, stats in cmvn_reader:
            self.cmvn_stats_dict[spkid] = stats

    def cmvn(self, mat, stats):
        mean = stats[0, :-1] / stats[0, -1]
        variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
        return np.divide(np.subtract(mat, mean), np.sqrt(variance))

    def concat_frame(self, features):
        time_steps, features_dim = features.shape
        concated_features = np.zeros(
            shape=[time_steps, features_dim *
                   (1 + self.left_context_width + self.right_context_width)],
            dtype=np.float32)
        # middle part is just the uttarnce
        concated_features[:, self.left_context_width * features_dim:
                          (self.left_context_width + 1) * features_dim] = features

        for i in range(self.left_context_width):
            # add left context
            concated_features[i + 1:time_steps,
                              (self.left_context_width - i - 1) * features_dim:
                              (self.left_context_width - i) * features_dim] = features[0:time_steps - i - 1, :]

        for i in range(self.right_context_width):
            # add right context
            concated_features[0:time_steps - i - 1,
                              (self.right_context_width + i + 1) * features_dim:
                              (self.right_context_width + i + 2) * features_dim] = features[i + 1:time_steps, :]

        return concated_features

    def subsampling(self, features):
        interval = self.subsample
        temp_mat = [features[i]
                    for i in range(0, features.shape[0], interval)]
        subsampled_features = np.row_stack(temp_mat)
        return subsampled_features



class AudioDataset(Dataset):
    def __init__(self, config, type):
        super(AudioDataset, self).__init__(config, type)

        self.config = config
        self.text = os.path.join(config.__getattr__(type), 'grapheme.txt')

        self.short_first = config.short_first

        # if self.config.encoding:
        self.unit2idx = self.get_vocab_map()
        self.targets_dict = self.get_targets_dict()

        if self.short_first and type == 'train':
            self.sorted_list = sorted(self.targets_dict.items(), key=lambda x: len(x[1]), reverse=False)
        else:
            self.sorted_list = None

        self.check_speech_and_text()
        self.lengths = len(self.feats_list)

    def __getitem__(self, index):
        if self.sorted_list is not None:
            utt_id = self.sorted_list[index][0]
        else:
            utt_id = self.feats_list[index]

        feats_scp = self.feats_dict[utt_id]
        seq = self.targets_dict[utt_id]

        targets = np.array(seq)
        wave_data, frame_rate= read_wave_from_file(feats_scp)
        features = get_feature(wave_data, frame_rate)

        # features = np.load(feats_scp)
        # features = kaldi_io.read_mat(feats_scp)
        # features = speechpy.processing.cmvnw(features,win_size=3*16000+1,variance_normalization=True)

        if self.apply_cmvn:
            spk_id = self.utt2spk[utt_id]
            stats = self.cmvn_stats_dict[spk_id]
            features = self.cmvn(features, stats)

        print('source:',features.shape)
        features = self.concat_frame(features)
        print('concat:',features.shape)
        features = self.subsampling(features)
        print('subsample:',features.shape)

        inputs_length = np.array(features.shape[0]).astype(np.int64)
        targets_length = np.array(targets.shape[0]).astype(np.int64)

        features = self.pad(features).astype(np.float32)
        targets = self.pad(targets).astype(np.int64).reshape(-1)

        return features, inputs_length, targets, targets_length

    def __len__(self):
        return self.lengths

    # 切割词表的每一行，分离idx 与sym，并实现词表的sym 到int 的映射
    def get_vocab_map(self):
        unit2idx = {}
        with codecs.open(self.vocab, 'r', encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split()
                unit = parts[0]
                idx = int(parts[1])
                unit2idx[unit] = idx
        return unit2idx

    # 切割文本表的每一行，分离utt_id，与目标序列，并将序列编码为int
    def get_targets_dict(self):
        targets_dict = {}
        with codecs.open(self.text, 'r', encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split(' ')
                utt_id = parts[0]
                contents = parts[1:]
                if len(contents) < 0 or len(contents) > self.max_target_length:
                    continue
                # if self.config.encoding:
                labels = self.encode(contents)
                # else:
                #     labels = [int(i) for i in contents]
                targets_dict[utt_id] = labels
        return targets_dict

    # 将sym 序列编码为idx
    def encode(self, seq):
        encoded_seq = []
        for unit in seq:
            if unit in self.unit2idx:
                encoded_seq.append(self.unit2idx[unit])
            else:
                encoded_seq.append(self.unit2idx['<unk>'])
        return encoded_seq

    # 移除没有转录的音频特征
    def check_speech_and_text(self):
        # featslist = copy.deepcopy(self.feats_list)
        featslist = []
        for utt_id in self.feats_list:
            # if utt_id not in self.targets_dict:
            #     self.feats_list.remove(utt_id)
            if self.targets_dict.get(utt_id) is not None:
                # self.feats_list.remove(utt_id)
                featslist.append(utt_id)
        self.feats_list = featslist

