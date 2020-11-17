"""
@File:stream_rec.py
@Descriptions: 以有效音频段为中心，在左右增加上下文特征
@Author:Dapeng
@Contact:zzp_dapeng@163.com
@Time:2020/7/7 下午3:10
"""
import wave
import yaml
import torch
import pyaudio
import datetime
import numpy as np
import tkinter as tk
import tkinter.font as font
from tt.utils import AttrDict
from tt.model import Transducer
from tt.utils import concat_frame, subsampling, generate_dictionary, get_feature, context_mask


class StreamRec:
    def __init__(self,
                 config=None,
                 pred_frame=18,
                 chunk=1024,  # or 1024 = 32 * 32
                 sample_width=2,
                 channels=1,
                 rate=16000):  # 固定，不要改
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
        self.recording = False  # 开始，结束按钮控制录制状态

        # 模型参数
        self.left_context = self.config.model.enc.left_context
        self.right_context = self.config.model.enc.right_context
        self.enc_layer_num = self.config.model.enc.n_layer

        # 录音机
        self.pa = pyaudio.PyAudio()

        # 录音数据
        self.audio_data = np.empty((0,), dtype=np.short)  # 使用numpy存储录制的音频数据
        self.frame_num = 0  # 记录当前录制的音频帧总数
        self.max_frame_num = 0  # 音频录制完成时的语音帧总数

        # 音频滑动窗口（单位:采样点），计算得出，过程见iPad
        self.win_audio = 15999
        self.audio_step = 15519
        self.win_audio_position = 0  # 滑动窗口当前的的起始位置（单位：帧）

        # 特征信息（单位：帧数)
        self.left_context_len = self.enc_layer_num * self.left_context
        self.right_context_len = self.enc_layer_num * self.right_context
        self.win_feature_position = 0  # 记录有效音频窗口的位置

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

    def __callback(self, in_data, frame_count, time_info, status):
        chunk_data = np.frombuffer(in_data, dtype=np.short)
        self.audio_data = np.concatenate((self.audio_data, chunk_data), axis=0)
        self.frame_num += frame_count
        # 到达录制时间，停止
        if not self.recording:
            print('pyaudio 录制结束')
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
        last_clip = False
        # todo:流式识别具体过程
        while True:  # 第一层窗口
            # print('win_audio_position, ', self.win_audio_position)
            # print('max_frame_num, ', self.max_frame_num)
            # 有足够的语音使得窗口能够移动 or 音频录制结束，剩余音频不足以移动窗口
            if self.win_audio_position + self.win_audio <= self.frame_num \
                    or self.win_audio_position + self.win_audio >= self.max_frame_num != 0:

                # 特征提取平滑
                if self.win_audio_position + self.win_audio >= self.max_frame_num != 0:
                    print('最后录音片段:', self.win_audio_position, " : ", self.frame_num)
                    last_clip = True
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

                # 特征滑动窗口
                len_feature_subsample = self.feature_subsample.shape[0]
                # print('4', len_feature_subsample)
                # 有足够的未来信息可以处理/最后一部分
                if len_feature_subsample - self.win_feature_position > self.right_context_len or last_clip:
                    # print('len_feature_subsample', len_feature_subsample)
                    # print('win_feature_position', self.win_feature_position)
                    left_frame = self.left_context_len  # 正常左窗口
                    right_frame = self.right_context_len  # 正常右窗口
                    start = self.win_feature_position - left_frame  # 增加历史帧,确定开始位置
                    if start < 0:
                        # print('历史帧不足')
                        left_frame = self.win_feature_position
                        start = 0
                    end = len_feature_subsample  # 当前的最大帧数
                    if last_clip:  # 录制结束，最后一段
                        right_frame = 0
                    win_audio_feature = self.feature_subsample[start:end, :]
                    # 扩展批次维度
                    win_audio_feature = np.expand_dims(win_audio_feature, axis=0)
                    win_audio_feature = torch.from_numpy(win_audio_feature).cuda()
                    # 流式mask
                    audio_mask = context_mask(win_audio_feature, left_context=self.left_context,
                                              right_context=self.right_context)[:, :, None].cuda()
                    win_enc_states = self.model.encoder(win_audio_feature, audio_mask)

                    effect_start = left_frame
                    effect_end = -right_frame
                    effect_win_enc_states = win_enc_states[:, effect_start:effect_end, :]
                    effect_len = effect_win_enc_states.shape[1]

                    enc_states_len = effect_win_enc_states.shape[1]
                    for t in range(enc_states_len):
                        logits = self.model.joint(effect_win_enc_states[:, t, :].view(-1), dec_state.view(-1))
                        out = torch.nn.functional.softmax(logits, dim=0).detach()
                        pred = torch.argmax(out, dim=0)
                        pred = int(pred.item())
                        if pred != 0:
                            self.result.append(pred)
                            word = self.dictionary[pred]
                            self.text.insert('end', word)
                            self.text.update()
                            result_len = len(self.result)
                            if result_len > 40:
                                effect_token = self.result[-40:]
                            else:
                                effect_token = self.result
                            token = torch.tensor([effect_token], dtype=torch.long)
                            token = token.cuda()
                            dec_state = self.model.decoder(token)[:, -1, :]  # 历史信息输入，但是只取最后一个输出
                    # print('effect_start:', effect_start)
                    # print('effect_end:', effect_end)
                    # print('effect_len:', effect_len)
                    self.win_feature_position += effect_len
                # 移动音频窗口
                self.win_audio_position += self.audio_step

            if not self.recording:
                self.max_frame_num = self.frame_num

            if last_clip:
                break

        print('识别结束：', ''.join([self.dictionary[x] for x in self.result]))
        # 参数恢复
        self.reset_parameter()
        print("重置参数")

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

    def save_audio(self, file_name=None):
        if file_name is None:
            file_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.wav'
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.sample_width)
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(self.audio_data))
        wf.close()
        return file_name

    # button功能
    def __start_button_action(self):
        # 清除识别结果
        self.text.delete(0.0, tk.END)
        # 更新参数
        self.recording = True
        # 设置按钮选中状态
        self.stop_button.config(state=tk.ACTIVE)
        self.start_button.config(state=tk.DISABLED)
        self.start_button.update()
        self.stop_button.update()
        # 开始识别
        self.start_rec()

    def __stop_button_action(self):
        self.recording = False
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.ACTIVE)
        self.start_button.update()
        self.stop_button.update()
        print('录音结束')
        self.stream.stop_stream()
        self.stream.close()
        file_name = self.save_audio()
        print('保存音频到：', file_name)

    def __init_view(self):
        self.window.title('TT流式语音识别系统')
        self.window.geometry('600x570')
        self.start_button_var = tk.StringVar()
        self.stop_button_var = tk.StringVar()
        self.text_var = tk.StringVar()

        self.font = font.Font(family='song ti', size=14)

        self.start_button_var.set('Start Rec')
        self.stop_button_var.set('Stop Rec')
        self.text_var.set('识别内容')

        # 设置界面部件
        # 识别结果文本框
        self.text = tk.Text(self.window,
                            font=self.font,
                            height=20,
                            width=50)
        self.text.place(x=20, y=20, anchor='nw')
        self.text.insert('end', self.text_var.get())

        # 录制按钮
        self.start_button = tk.Button(self.window,
                                      textvariable=self.start_button_var,
                                      font=('Arial', 12),
                                      bg='white',
                                      width=9,
                                      command=self.__start_button_action)
        self.start_button.place(x=100, y=515, anchor='nw')

        # 结束录制按钮
        self.stop_button = tk.Button(self.window,
                                     textvariable=self.stop_button_var,
                                     font=('Arial', 12),
                                     bg='white',
                                     width=9,
                                     command=self.__stop_button_action)
        self.stop_button.place(x=400, y=515, anchor='nw')
        self.stop_button.config(state=tk.DISABLED)

        self.window.mainloop()


def streaming_asr():
    config_file = open("config/joint_streaming.yaml")
    configure = AttrDict(yaml.load(config_file, Loader=yaml.FullLoader))
    stream_rec = StreamRec(config=configure)
