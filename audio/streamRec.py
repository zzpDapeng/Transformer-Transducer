"""
@File:stream_rec.py
@Descriptions: 调用TT模型进行流式语音识别
@Author:Dapeng
@Contact:zzp_dapeng@163.com
@Time:2020/7/7 下午3:10
"""
import time
import math
import yaml
import torch
import pyaudio
import numpy as np
import tkinter as tk
import tkinter.font as font
import matplotlib.pyplot as plt
from tt.utils import AttrDict
from tt.model import Transducer
from tt.utils import concat_frame, subsampling, generate_dictionary, get_feature, context_mask


class StreamRec:
    def __init__(self,
                 config=None,
                 pred_frame=18,
                 record_seconds=15,
                 chunk=1024,  # or 1024 = 32 * 32
                 sample_width=2,
                 channels=1,
                 rate=16000,  # 固定，不要改
                 layer_num=18):
        # 模型
        self.config = config
        self.__init_model()
        print("加载字典...")
        self.dictionary, _ = generate_dictionary(self.config.data.vocab)
        print("字典加载完毕")

        # 音频参数
        self.chunk = chunk
        self.sample_width = sample_width
        self.channels = channels
        self.rate = rate
        self.record_seconds = record_seconds
        self.max_frame_num = self.__reset_max_frame_num()

        # 模型参数
        self.left_context = self.config.model.enc.left_context
        self.right_context = self.config.model.enc.right_context
        self.enc_layer_num = self.config.model.enc.n_layer

        # 录音机
        self.pa = pyaudio.PyAudio()

        # 录音数据
        self.audio_data = np.empty((0,), dtype=np.short)  # 使用numpy存储录制的音频数据
        self.frame_num = 0  # 记录当前录制的音频帧总数

        # 音频滑动窗口（单位:采样点），计算得出，过程见iPad
        self.win_audio = 15999
        self.audio_step = 15519
        self.win_audio_position = 0  # 滑动窗口当前的的起始位置（单位：帧）

        # 特征滑动窗口（单位：帧数），窗口长度设为帧速率，则窗口长度为1秒。v1
        self.pred_frame = pred_frame  # 预测的帧数，(堆叠下采样之后的帧），等于窗口移动的步幅
        self.win_feature = self.config.model.enc.n_layer * self.left_context \
                           + pred_frame \
                           + self.config.model.dec.n_layer * self.right_context  # 等腰梯形窗口长度，单位：帧
        self.min_pred_frame = self.pred_frame + self.right_context * self.enc_layer_num  # 最低识别帧数
        self.win_feature_position = 0

        # 记录特征和结果
        self.result = []
        self.feature_log_mel = np.empty((0, 128), dtype=np.float32)
        self.feature_concat = np.empty((0, 512), dtype=np.float32)
        self.feature_subsample = np.empty((0, 512), dtype=np.float32)

        # 可视化
        self.window = tk.Tk()
        self.__init_view()

    def __init_model(self):
        model = Transducer(self.config.model)

        checkpoint = torch.load(self.config.training.load_model)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        model.joint.load_state_dict(checkpoint['joint'])

        model = model.cuda()
        model.eval()
        print("已加载模型")
        self.model = model

    def __reset_max_frame_num(self):
        max_frame_num = self.chunk * math.ceil(self.rate * self.record_seconds / self.chunk)
        return max_frame_num

    def __callback(self, in_data, frame_count, time_info, status):
        chunk_data = np.frombuffer(in_data, dtype=np.short)
        self.audio_data = np.concatenate((self.audio_data, chunk_data), axis=0)
        self.frame_num += frame_count
        # 到达录制时间，停止
        if self.frame_num >= self.rate * self.record_seconds:
            return in_data, pyaudio.paComplete

        return in_data, pyaudio.paContinue

    def start_rec(self):
        self.stream = self.pa.open(format=self.pa.get_format_from_width(self.sample_width),
                                   channels=self.channels,
                                   rate=self.rate,
                                   frames_per_buffer=self.chunk,
                                   input=True,
                                   output=False,
                                   stream_callback=self.__callback)

        self.stream.start_stream()
        print("开始录音并识别")

        zero_token = torch.tensor([[0]], dtype=torch.long)
        zero_token = zero_token.cuda()
        dec_state = self.model.decoder(zero_token)
        # todo:流式识别具体过程
        # todo：方案1 以数据帧为窗口，不考虑窗口前后的上下文特征信息，不考虑mask
        while True:  # 第一层窗口
            if self.win_audio_position + self.win_audio <= self.frame_num \
                    or self.win_audio_position + self.win_audio > self.max_frame_num:  # 有足够的语音使得窗口能够移动 or 音频录制结束，剩余音频不足以移动窗口

                # 特征提取平滑
                if self.win_audio_position + self.win_audio > self.max_frame_num:
                    win_audio = self.audio_data[self.win_audio_position:self.frame_num]
                else:
                    win_audio = self.audio_data[self.win_audio_position:self.win_audio_position + self.win_audio]
                win_audio_feature = get_feature(win_audio, self.rate, 128)
                win_audio_feature = win_audio_feature[:-3, :]  # 舍弃最后3帧，得到97帧，因为这3帧音频数据不完整。通过（窗口移动<窗口）弥补这3帧（音频平滑过渡）
                win_audio_feature_len = win_audio_feature.shape[0]
                self.feature_log_mel = np.concatenate((self.feature_log_mel, win_audio_feature), axis=0)
                # print('1', self.feature_log_mel.shape)
                # print('2', win_audio_feature.shape)

                # 堆叠平滑
                win_audio_feature = self.feature_log_mel[-3 - win_audio_feature_len:, :]  # 往前多拿3帧，保证有前面的历史信息（堆叠平滑过渡）
                # print('3', win_audio_feature.shape)
                win_audio_feature = concat_frame(win_audio_feature, 3, 0)
                win_audio_feature = win_audio_feature[3:, :]  # 去掉往前多拿3帧，以这三帧为中心的帧并不需要（堆叠平滑过渡）
                len_feature_concat_before = self.feature_concat.shape[0]
                self.feature_concat = np.concatenate((self.feature_concat,
                                                      win_audio_feature), axis=0)  # 把当前时间窗口的所有堆叠帧加入到self.feature_concat
                # 下采样平滑
                if len_feature_concat_before % 3 == 0:
                    win_audio_feature = subsampling(self.feature_concat[len_feature_concat_before:, :], 3)
                elif len_feature_concat_before % 3 == 1:
                    win_audio_feature = subsampling(self.feature_concat[len_feature_concat_before + 2:, :], 3)
                else:
                    win_audio_feature = subsampling(self.feature_concat[len_feature_concat_before + 1:, :], 3)
                self.feature_subsample = np.concatenate((self.feature_subsample, win_audio_feature), axis=0)

                # todo:特征滑动窗口
                len_feature_subsample = self.feature_subsample.shape[0]
                while True:
                    if len_feature_subsample - self.win_feature_position >= self.min_pred_frame:  # 满足最低识别帧数
                        pass
                    else:
                        break

                # 扩展批次维度
                win_audio_feature = np.expand_dims(win_audio_feature, axis=0)  # 可以不扩展维度吗
                # print('type:', win_audio_feature.dtype)  # float32
                # GPU
                win_audio_feature = torch.from_numpy(win_audio_feature).cuda()
                # print('win_audio_feature:', win_audio_feature.shape)  # torch.Size([1, 34, 512])

                audio_mask = context_mask(win_audio_feature, left_context=self.left_context,
                                          right_context=self.right_context)[:, :, None].cuda()

                win_enc_states = self.model.encoder(win_audio_feature, audio_mask)

                enc_states_len = win_enc_states.shape[1]
                for t in range(enc_states_len):
                    logits = self.model.joint(win_enc_states[:, t, :].view(-1), dec_state.view(-1))
                    out = torch.nn.functional.softmax(logits, dim=0).detach()
                    pred = torch.argmax(out, dim=0)
                    pred = int(pred.item())
                    if pred != 0:
                        self.result.append(pred)
                        word = self.dictionary[pred]
                        self.text.insert('end', word)
                        self.text.update()
                        token = torch.tensor([self.result], dtype=torch.long)
                        token = token.cuda()
                        dec_state = self.model.decoder(token)[:, -1, :]  # 历史信息输入，但是只取最后一个输出
                # 窗口移动
                self.win_audio_position += self.audio_step
            elif self.win_audio_position + self.win_audio > self.max_frame_num:  # 录制完成，剩余音频不足以滑动音频窗口
                pass

            if self.win_audio_position >= self.max_frame_num:
                print("over")
                break

        # 循环监听音频流是否录制完成 用上面的替换
        # while self.stream.is_active():
        #     time.sleep(1)
        #     self.text.insert('end', '你')
        #     self.text.update()

        self.stream.stop_stream()
        self.stream.close()
        print("结束录音")
        self.start_button.config(state=tk.ACTIVE)
        self.start_button.update()
        # 参数恢复
        self.reset_parameter()
        print("Reset parameters")

    def reset_parameter(self):
        """
        重置动态参数
        :return:
        """
        self.audio_data = np.empty((0,), dtype=np.short)
        self.frame_num = 0
        self.win_audio_position = 0
        self.win_feature_position = 0
        self.result = []
        self.feature_log_mel = np.empty((0, 128), dtype=np.float32)
        self.feature_concat = np.empty((0, 512), dtype=np.float32)
        self.feature_subsample = np.empty((0, 512), dtype=np.float32)

        # button功能

    def __button_action(self):

        self.text.delete(0.0, tk.END)
        # 判断输入时间是否有效
        # try:
        rec_secs = self.seconds_var.get()
        self.record_seconds = rec_secs
        self.__reset_max_frame_num()
        self.start_button.config(state=tk.DISABLED)
        self.start_button.update()
        self.start_rec()

    # except:
    #     self.text.insert('end', 'Error Record!')
    #     self.text.update()

    def __init_view(self):
        self.window.title('TT流式语音识别系统')
        self.window.geometry('600x570')
        self.seconds_var = tk.IntVar()
        self.button_var = tk.StringVar()
        self.text_var = tk.StringVar()

        self.font = font.Font(family='song ti', size=14)

        self.button_var.set('Start Rec')
        self.seconds_var.set(self.record_seconds)
        self.text_var.set('识别内容')

        # 设置界面部件
        # 识别结果文本框
        self.text = tk.Text(self.window,
                            font=self.font,
                            height=20,
                            width=50)
        self.text.place(x=20, y=20, anchor='nw')
        self.text.insert('end', self.text_var.get())

        # 录制时间标签
        self.time_label = tk.Label(self.window,
                                   text='Record time:',
                                   font=('Arial', 12))
        self.time_label.place(x=20, y=520, anchor='nw')

        # 录制时间输入框
        self.time_entry = tk.Entry(self.window,
                                   textvariable=self.seconds_var,
                                   font=('Arial', 12),
                                   bg='white')
        self.time_entry.place(x=150, y=520, anchor='nw')

        # 录制时间单位标签
        self.second_label = tk.Label(self.window,
                                     text='s',
                                     font=('Arial', 12))
        self.second_label.place(x=360, y=520, anchor='nw')

        # 录制按钮
        self.start_button = tk.Button(self.window,
                                      textvariable=self.button_var,
                                      font=('Arial', 12),
                                      bg='white',
                                      width=9,
                                      command=self.__button_action)
        self.start_button.place(x=450, y=515, anchor='nw')

        self.window.mainloop()


def streaming_asr():
    config_file = open("config/joint_streaming.yaml")
    configure = AttrDict(yaml.load(config_file, Loader=yaml.FullLoader))
    stream_rec = StreamRec(config=configure)
    stream_rec.start_rec()
