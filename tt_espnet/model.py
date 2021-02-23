"""
@File:model.py
@Descriptions:
@Author:Dapeng
@Contact:zzp_dapeng@163.com
@Time:2021/2/22 下午6:13 
"""
import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet.nets.pytorch_backend.transducer.joint_network import JointNetwork
from espnet.nets.pytorch_backend.transducer.loss import TransLoss


class TransformerTransducer(nn.Module):
    def __init__(self, config):
        super(TransformerTransducer, self).__init__()
        self.vocab_size = config.joint.vocab_size
        self.sos = self.vocab_size - 1
        self.eos = self.vocab_size - 1
        self.ignore_id = -1
        self.encoder_left_mask = config.mask.encoder_left_mask
        self.encoder_right_mask = config.mask.encoder_right_mask
        self.decoder_left_mask = config.mask.decoder_left_mask

        self.encoder = TransformerEncoder(**config.enc)
        self.decoder = TransformerEncoder(**config.dec)
        self.joint = JointNetwork(**config.joint)
        self.loss = TransLoss(trans_type="warp-transducer",
                              blank_id=0)  # todo: check blank id

    def forward(self,
                speech: torch.Tensor,
                speech_lengths: torch.Tensor,
                text: torch.Tensor,
                text_lengths: torch.Tensor, ):
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_out_lens, _ = self.encoder(speech,
                                                        speech_lengths,
                                                        left_mask=self.encoder_left_mask,
                                                        right_mask=self.encoder_right_mask)  # return xs_pad, olens, None

        # 2. Decoder
        # todo: train right shift
        text_in, text_out = add_sos_eos(text, self.sos, self.eos, self.ignore_id)
        text_in_lens = text_lengths + 1
        decoder_out, decoder_out_lens, _ = self.decoder(text_in,
                                                        text_in_lens,
                                                        left_mask=self.decoder_left_mask,
                                                        right_mask=0)  # return xs_pad, olens, None

        # 3.Joint
        # h_enc: Batch of expanded hidden state (B, T, 1, D_enc)
        # h_dec: Batch of expanded hidden state (B, 1, U, D_dec)
        encoder_out = encoder_out.unsqueeze(2)
        decoder_out = decoder_out.unsqueeze(1)
        joint_out = self.joint(h_enc=encoder_out, h_dec=decoder_out)

        # 4.loss
        # pred_pad (torch.Tensor): Batch of predicted sequences
        loss = self.loss(pred_pad=joint_out,  # (batch, maxlen_in, maxlen_out+1, odim)
                         target=text.int(),  # (batch, maxlen_out)
                         pred_len=speech_lengths.int(),  # (batch)
                         target_len=text_lengths.int())  # (batch)
        return loss

    def recognize(self):
        pass


def init_asr_model(configs):
    pass


if __name__ == '__main__':
    model = TransformerTransducer(vocab_size=1000,
                                  encoder_layer_num=2,
                                  decoder_layer_num=2)

    audio = "../audio.wav"
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi

    waveform, sample_rate = torchaudio.load_wav(audio)
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=512,
        frame_length=32,
        frame_shift=10,
        dither=0.1,
        energy_floor=0.0
    )
    speech = mat.unsqueeze(0).repeat([2, 1, 1])
    speech_lengths = torch.tensor([425, 50]).long()
    text = torch.tensor([[1, 2, 3, 4],
                         [2, 3, 4, -1]]).long()
    text_length = torch.tensor([4, 3]).long()
    loss = model(speech, speech_lengths, text, text_length)
    print(loss)
