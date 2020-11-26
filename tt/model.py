import heapq
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tt.decoder import BuildDecoder
from tt.encoder import BuildEncoder
from tt.utils import create_mask
from tt.utils import look_ahead_mask, context_mask


class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)  # 添加维度1
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)

        return outputs


class Transducer(nn.Module):
    def __init__(self, config):
        super(Transducer, self).__init__()
        self.config = config
        self.encoder = BuildEncoder(config)
        self.decoder = BuildDecoder(config)
        self.joint = JointNet(
            input_size=config.joint.input_size,
            inner_dim=config.joint.inner_size,
            vocab_size=config.vocab_size
        )
        if config.share_embedding:
            assert self.decoder.embedding.weight.size() == self.joint.project_layer.weight.size(), '%d != %d' % (
                self.decoder.embedding.weight.size(1), self.joint.project_layer.weight.size(1))
            self.joint.project_layer.weight = self.decoder.embedding.weight

    def forward(self, inputs, targets):
        targets = F.pad(targets, pad=[1, 0, 0, 0], value=0)
        # audio_mask = context_mask(inputs)[:, :, None]        # 流式语音识别
        audio_mask = None  # 非流式语音识别
        label_mask = look_ahead_mask(targets)[:, :, None]
        enc_state = self.encoder(inputs, audio_mask)
        dec_state = self.decoder(targets, label_mask)

        logits = self.joint(enc_state, dec_state)

        return logits

    def recognize(self, inputs, inputs_length, audio_mask=None, mode='greedy'):
        """
        batch分开，识别关注之前所有的输出
        :param inputs:
        :param inputs_length:
        :param audio_mask:
        :param mode:
        :return:
        """
        batch_size = inputs.size(0)
        enc_states = self.encoder(inputs, audio_mask)
        device = torch.device("cuda" if inputs.is_cuda else "cpu")

        def decode(enc_state, lengths):
            # token_list = []
            token_list = [0]
            token = torch.tensor([token_list], dtype=torch.long).to(device)
            dec_state = self.decoder(token)[:, -1, :]
            for t in range(lengths):
                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                pred = torch.argmax(out, dim=0)
                pred = int(pred.item())

                if pred != 0:
                    token_list.append(pred)
                    token = torch.tensor([token_list], dtype=torch.long)

                    if enc_state.is_cuda:
                        token = token.cuda()
                    dec_state = self.decoder(token)[:, -1, :]  # 历史信息输入，但是只取最后一个输出
            # return token_list
            return token_list[1:]

        results = []
        for batch in range(batch_size):
            decoded_seq = decode(enc_states[batch], inputs_length[batch])
            results.append(decoded_seq)
        return results

    def recognize_beam_search(self, inputs, inputs_length, audio_mask=None, mode='greedy'):
        """
        batch分开，识别关注之前所有的输出
        :param inputs:
        :param inputs_length:
        :param audio_mask:
        :param mode:
        :return:
        """
        batch_size = inputs.size(0)
        enc_states = self.encoder(inputs, audio_mask)

        def beam_search(enc_state, lengths, beam_width=5):
            first = True
            device = torch.device("cuda" if enc_state.is_cuda else "cpu")
            # 初始化集束搜索的符号列表和概率列表
            token_list = []  # len=beam_width
            probability = np.zeros((beam_width,), dtype=float)  # len=beam_width,与token_list顺序对应
            token_child_list = []  # len=beam_width**2
            probability_child = np.zeros((beam_width, beam_width),
                                         dtype=float)  # len=beam_width**2,与token_child_list顺序对应
            for i in range(beam_width):
                token_list.append([0])
            for i in range(beam_width):
                token_child_list.append([])
                for j in range(beam_width):
                    token_child_list[i].append([0])

            # todo:帧预测不同步问题，以什么为标准推进音频帧移动，
            for t in range(lengths):  # 循环直到第一次结束暂定以当前概率最高的dec_state以最高的为准
                # 以当前概率最高的为主轴，遇到预测音频帧则进行束搜索
                max_index = probability.argmax()  # 最大之索引，当有重复时取第一个，返回一个int值
                token = torch.tensor([token_list[max_index]], dtype=torch.long).to(device)
                dec_state = self.decoder(token)[:, -1, :]  # 以当前概率最高的token获得dec_state
                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                pred_max = torch.argmax(out, dim=0)
                pred_max = int(pred_max.item())
                if pred_max != 0:
                    for token_index in range(len(token_list)):
                        token = torch.tensor([token_list[token_index]], dtype=torch.long).to(device)
                        dec_state = self.decoder(token)[:, -1, :]  # 以当前概率最高的token获得dec_state
                        logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                        out = F.softmax(logits, dim=0).detach()
                        values, indices = torch.topk(out, k=beam_width + 1, dim=0)
                        values = values.tolist()
                        indices = indices.tolist()
                        if 0 in indices:  # 有0则去掉零
                            zero_index = indices.index(0)
                            indices.pop(zero_index)
                            values.pop(zero_index)
                        else:  # 没0则去掉最后一个
                            indices.pop(-1)
                            values.pop(-1)
                        if first:
                            for i in range(len(indices)):
                                token_child_list[i][token_index].append(indices[i])
                                probability_child[:, token_index] = np.log(values)
                        else:
                            for i in range(len(indices)):
                                token_child_list[token_index][i].append(indices[i])
                                probability_child[token_index] = probability[token_index] + np.log(values)
                        # 取log后直接加到probability上
                        # print(token.tolist(), np.log(values), indices)
                    if first:
                        first = False
                        for i in range(beam_width):
                            token_list[i] = copy.deepcopy(token_child_list[i][0])
                            probability[i] = copy.deepcopy(probability_child[i, 0])
                    else:
                        # probability_child里面取最大的5个到probability，对应的token_child_list放到probability
                        top_k_index = heapq.nlargest(beam_width, range(beam_width ** 2), probability_child.take)
                        for i in range(len(top_k_index)):
                            index = top_k_index[i]
                            probability[i] = copy.deepcopy(probability_child[index // beam_width, index % beam_width])
                            token_list[i] = copy.deepcopy(token_child_list[index // beam_width][index % beam_width])
            max_index = probability.argmax()  # 最大之索引，当有重复时取第一个，返回一个int值
            token_list = token_list[max_index]
            return token_list[1:]

        results = []
        for batch in range(batch_size):
            decoded_seq = beam_search(enc_states[batch], inputs_length[batch], beam_width=5)
            results.append(decoded_seq)
        return results