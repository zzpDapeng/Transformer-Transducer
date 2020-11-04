import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # 流式语音识别
        # audio_mask = context_mask(inputs)[:, :, None]
        # 非流式语音识别
        label_mask = look_ahead_mask(targets)[:, :, None]
        enc_state = self.encoder(inputs)
        dec_state = self.decoder(targets, label_mask)

        logits = self.joint(enc_state, dec_state)

        return logits

    def recognize(self, inputs, inputs_length):
        """
        batch分开，识别关注之前所有的输出
        :param inputs:
        :param inputs_length:
        :return:
        """
        batch_size = inputs.size(0)
        enc_states = self.encoder(inputs)

        zero_token = torch.tensor([[0]], dtype=torch.long)
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            token_list = []
            dec_state = self.decoder(zero_token)[:, -1, :]
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

            return token_list

        results = []
        for batch in range(batch_size):
            decoded_seq = decode(enc_states[batch], inputs_length[batch])
            results.append(decoded_seq)

        return results
