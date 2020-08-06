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
from tt.utils import concat_frame, subsampling, generate_dictionary, get_feature


class StreamRec:
    def __init__(self,
                 record_seconds=5,
                 chunk=16000,  # or 1024
                 sample_width=2,
                 channels=1,
                 rate=16000,
                 left_mask=10,  # 根据模型设置
                 right_mask=2,
                 layer_num=8):
        # 模型
        self.__init_model()
        print("加载字典...")
        self.dictionary = generate_dictionary('data/aishell/grapheme_table.txt')
        print("字典加载完毕")

        # 音频参数
        self.chunk = chunk
        self.sample_width = sample_width
        self.channels = channels
        self.rate = rate
        self.record_seconds = record_seconds
        self.max_frame_num = self.__reset_max_frame_num()

        # 模型参数
        self.left_mask = left_mask
        self.right_mask = right_mask
        self.layer_num = layer_num

        # 录音机
        self.pa = pyaudio.PyAudio()

        # 录音数据
        self.audio_data = np.empty((0,), dtype=np.short)  # 使用numpy存储录制的音频数据
        self.frame_num = 0  # 记录当前录制的音频帧总数

        self.win_len = math.ceil(self.rate / self.chunk) * self.chunk  # 滑动窗口长度（单位：帧数），窗口长度设为镇速率，则窗口长度为1秒
        self.step_len = self.win_len - self.layer_num * (self.right_mask + self.left_mask)  # 推算出的步长
        self.win_position = 0  # 滑动窗口当前的的起始位置（单位：帧）

        # todo:记录特征和结果
        self.result = []
        self.feature = np.empty((0, 128))  # 数据类型？
        self.effective_feature = np.empty((0, 128))

        # 可视化
        self.window = tk.Tk()
        self.__init_view()

    def __init_model(self):
        config_file = open("config/joint.yaml")
        config = AttrDict(yaml.load(config_file, Loader=yaml.FullLoader))

        model = Transducer(config.model)

        checkpoint = torch.load(config.training.load_model)
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
        print(self.audio_data.shape, chunk_data.shape)
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
        while True:
            if self.win_position + self.win_len <= self.frame_num:  # 有足够的语音使得窗口能够移动
                win_audio = self.audio_data[self.win_position:self.win_position + self.win_len]
                # print(win_audio.max())
                # print(win_audio.min())
                win_audio_feature = get_feature(win_audio, self.rate, 128)
                # plt.imshow(win_audio_feature.T, cmap='Reds')
                # plt.show()
                # 数据堆叠和下采样处理
                win_audio_feature = concat_frame(win_audio_feature, 3, 0)
                win_audio_feature = subsampling(win_audio_feature, 3)
                # 扩展批次维度
                win_audio_feature = np.expand_dims(win_audio_feature, axis=0)  # 可以不扩展维度吗
                # GPU
                win_audio_feature = torch.from_numpy(win_audio_feature).cuda()

                win_enc_states = self.model.encoder(win_audio_feature)

                enc_states_len = win_enc_states.shape[1]
                for t in range(enc_states_len):
                    print("decode_state.shape", dec_state.shape)
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
                self.win_position += self.win_len

            if self.win_position == self.max_frame_num:
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
        self.win_position = 0
        self.result = []

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
        self.seconds_var.set(5)
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


if __name__ == '__main__':
    # 模型初始化
    # config_file = open("config/joint.yaml")
    # config = AttrDict(yaml.load(config_file, Loader=yaml.FullLoader))
    #
    # model = Transducer(config.model)
    #
    # checkpoint = torch.load(config.training.load_model)
    # model.encoder.load_state_dict(checkpoint['encoder'])
    # model.decoder.load_state_dict(checkpoint['decoder'])
    # model.joint.load_state_dict(checkpoint['joint'])
    #
    # model = model.cuda()
    # model.eval()
    # print("已加载模型")

    stream_rec = StreamRec()
    # stream_rec = stream_rec()
    # stream_rec.start_rec()
