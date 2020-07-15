"""
@File:data_process.py
@Desciption:
@Author:Dapeng
@Contact:zzp_dapeng@163.com
@Time:2020/7/3 上午8:40 
"""
import os
import wave
import time
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import python_speech_features
import matplotlib.pyplot as plt
from specAugment import spec_augment_pytorch
import librosa
import speechpy


def read_wav_data(filename):
    wav = wave.open(filename, 'rb')
    num_frames = wav.getnframes()
    num_channels = wav.getnchannels()
    framerate = wav.getframerate()
    str_data = wav.readframes(num_frames)
    wav.close()
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data.shape = -1, num_channels
    return wave_data, framerate



def fbank_feature(wave_data,framerate,nfilt):
    feature = python_speech_features.logfbank(
        signal=wave_data,
        samplerate=framerate,
        winlen=0.032,
        winstep=0.010,
        nfilt=nfilt)
    # 降低数据类型，减少存储空间
    feature = feature.astype(np.float32)
    return feature


def generate_aishell_feature(wav_path):
    """
    处理分好类的aishell
    :param wav_path:
    :return:
    """
    save_path = os.path.join(wav_path, "logfbank128")
    subdirs = ["train", "dev", "test"]
    for subdir in subdirs:
        audio_path = os.path.join(wav_path, subdir)
        print("正在处理", subdir)
        with open(os.path.join(save_path, subdir+".scp"), "w") as wf:
            for root, dirs, files in tqdm(os.walk(audio_path)):
                for file in files:
                    file_path = os.path.join(root, file)
                    wave_data, framerate = read_wav_data(file_path)
                    if wave_data.size != 0 :
                        feature = fbank_feature(wave_data, framerate, 128)
                        file_pure_name = str(os.path.basename(file)).split(".")[0]
                        npy_name = file_pure_name+".npy"
                        np_save_path = os.path.join(save_path, subdir, npy_name)
                        np.save(np_save_path, feature)
                        wf.writelines(file_pure_name+" "+np_save_path+"\n")



def generate_aishell_feature2(wav_path):
    """
    处理未分类的aishell
    :param wav_path:
    :return:
    """
    save_path = os.path.join(wav_path, "logfbank128")
    wav_path = os.path.join(wav_path, "wav")

    subdirs = ["train", "dev", "test"]
    for subdir in subdirs:
        print("正在处理", subdir)
        start = time.time()
        with open(os.path.join(save_path, subdir+".scp"), "w") as wf:
            for root, dirs, files in os.walk(wav_path):
                if root.find(subdir) != -1:
                    for file in files:
                        file_path = os.path.join(root, file)
                        wave_data, framerate = read_wav_data(file_path)
                        if wave_data.size != 0:
                            feature = fbank_feature(wave_data, framerate, 128)
                            file_pure_name = str(os.path.basename(file)).split(".")[0]
                            npy_name = file_pure_name + ".npy"
                            np_save_path = os.path.join(save_path, subdir, npy_name)
                            np.save(np_save_path, feature)
                            wf.writelines(file_pure_name + " " + np_save_path + "\n")
        end = time.time()
        print(subdir, "耗时：", end - start)
    return None


def test_different_feature():
    magic_audio = "/media/dapeng/文档/Dataset/中文语音数据集/magicdata/dev/5_1812/5_1812_20170628135834.wav"
    record_audio = 'output.wav'
    record_audio = 'output.wav'


    test_audio = magic_audio

    # torchaudio方式
    waveform, sample_rate = torchaudio.load(test_audio)
    specgram = torchaudio.transforms.MelSpectrogram(n_fft=512, hop_length=160)(waveform)
    specgram1 = specgram[0, 1::].detach().numpy()
    print("Shape of specgram:{}".format(specgram1.shape))
    print("Max of specgram:{}".format(specgram1.max()))
    print("Min of specgram:{}".format(specgram1.min()))

    plt.subplot(3, 2, 1)
    plt.imshow(specgram1, cmap='Reds')

    # logmel
    specgram = specgram.log()[0, 1::].detach().numpy()
    print("Shape of specgram:{}".format(specgram.shape))
    print("Max of specgram:{}".format(specgram.max()))
    print("Min of specgram:{}".format(specgram.min()))

    plt.subplot(3, 2, 2)
    plt.imshow(specgram, cmap='Reds')

    # 　wave方式归一化
    waveform2, sample_rate2 = read_wav_data(test_audio)
    waveform2 = waveform2 / 2 ** 15
    specgram2 = fbank_feature(waveform2, sample_rate2, 128)
    print("Shape of specgram2:{}".format(specgram2.shape))
    print("Max of specgram2:{}".format(specgram2.max()))
    print("Min of specgram2:{}".format(specgram2.min()))
    #
    plt.subplot(3, 2, 3)
    plt.imshow(specgram2.T, cmap='Reds')

    # 　wave方式不归一化
    waveform3, sample_rate3 = read_wav_data(test_audio)
    specgram3 = fbank_feature(waveform3, sample_rate3, 128)
    print("Shape of specgram3:{}".format(specgram3.shape))
    print("Max of specgram3:{}".format(specgram3.max()))
    print("Min of specgram3:{}".format(specgram3.min()))

    plt.subplot(3, 2, 4)
    plt.imshow(specgram3.T, cmap='Reds')

    # 　wave+logmel方式
    waveform4, sample_rate4 = read_wav_data(test_audio)
    waveform4 = waveform4 / 2 ** 15
    waveform4 = torch.tensor(waveform4.T, dtype=torch.float32)
    specgram4 = torchaudio.transforms.MelSpectrogram(n_fft=512, hop_length=160)(waveform4)
    specgram4 = specgram4.log()[0, 1::].detach().numpy()
    print("Shape of specgram:{}".format(specgram4.shape))
    print("Max of specgram:{}".format(specgram4.max()))
    print("Min of specgram:{}".format(specgram4.min()))
    plt.subplot(3, 2, 5)
    plt.imshow(specgram4, cmap='Reds')

    # wave+librosa
    waveform5, sample_rate5 = read_wav_data(test_audio)
    waveform5 = waveform5 / 2 ** 15
    waveform5 = np.reshape(waveform5, (-1,))
    specgram5 = librosa.feature.melspectrogram(waveform5, sr=sample_rate5, n_fft=512, hop_length=160)
    specgram5 = np.log(specgram5)
    print("Shape of specgram:{}".format(specgram5.shape))
    print("Max of specgram:{}".format(specgram5.max()))
    print("Min of specgram:{}".format(specgram5.min()))
    plt.subplot(3, 2, 6)
    plt.imshow(specgram5, cmap='Reds')

    plt.show()


def wave_librosa_logmel(audio_file):
    # 读音频
    wav = wave.open(audio_file, 'rb')
    num_frames = wav.getnframes()
    framerate = wav.getframerate()
    str_data = wav.readframes(num_frames)
    wav.close()
    wave_data = np.frombuffer(str_data, dtype=np.short)

    # 提取特征
    wave_data = wave_data.astype("float32")
    specgram = librosa.feature.melspectrogram(wave_data, sr=framerate, n_fft=512, hop_length=160)
    specgram = np.log(specgram).T
    return specgram


def test_different_dataset():
    magic_audio = "/media/dapeng/文档/Dataset/中文语音数据集/magicdata/dev/5_1812/5_1812_20170628135834.wav"
    aidatatang = "/media/dapeng/文档/Dataset/中文语音数据集/aidatatang_200zh/corpus/dev/G0002/T0055G0002S0001.wav"
    aihshell = "/media/dapeng/文档/Dataset/中文语音数据集/data_aishell/wav/dev/S0724/BAC009S0724W0121.wav"
    thchs30 = "/media/dapeng/文档/Dataset/中文语音数据集/data_thchs30/data/A11_0.wav"
    primewords = "/media/dapeng/文档/Dataset/中文语音数据集/primewords_md_2018_set1/audio_files/0/00/00a1ca32-3ad8-4f62-bb16-7346fca03444.wav"
    st_cmds = "/media/dapeng/文档/Dataset/中文语音数据集/ST-CMDS-20170001_1-OS/20170001P00200I0089.wav"
    record_audio = 'output.wav'

    specgram1 = wave_librosa_logmel(magic_audio)
    plt.subplot(3, 2, 1)
    plt.imshow(specgram1, cmap='Reds')

    specgram2 = wave_librosa_logmel(aidatatang)
    plt.subplot(3, 2, 2)
    plt.imshow(specgram2, cmap='Reds')

    specgram3 = wave_librosa_logmel(aihshell)
    plt.subplot(3, 2, 3)
    plt.imshow(specgram3, cmap='Reds')

    specgram4 = wave_librosa_logmel(thchs30)
    plt.subplot(3, 2, 4)
    plt.imshow(specgram4, cmap='Reds')

    specgram5 = wave_librosa_logmel(primewords)
    plt.subplot(3, 2, 5)
    plt.imshow(specgram5, cmap='Reds')


    specgram6 = wave_librosa_logmel(st_cmds)
    plt.subplot(3, 2, 6)
    plt.imshow(specgram6, cmap='Reds')

    plt.show()


def test_specAugment():
    magic_audio = "/media/dapeng/文档/Dataset/中文语音数据集/magicdata/dev/5_1812/5_1812_20170628135834.wav"
    spec = wave_librosa_logmel(magic_audio)
    masked_spec = spec_augment_pytorch.spec_augment(spec,
                                                    frequency_masking_para=50,
                                                    frequency_mask_num=2,
                                                    time_masking_para=30,
                                                    time_mask_num=10)
    print(spec.shape)
    print(masked_spec.shape)
    plt.subplot(211)
    plt.plot(spec)
    plt.subplot(212)
    plt.plot(masked_spec)
    plt.show()

if __name__ == '__main__':
    test_different_dataset()