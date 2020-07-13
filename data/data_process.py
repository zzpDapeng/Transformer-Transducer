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
from tqdm import tqdm
import numpy as np
import python_speech_features
import librosa
import speechpy


def read_wav_data(filename):
    wav = wave.open(filename, 'rb')
    num_frames = wav.getnframes()
    num_channels = wav.getnchannels()
    framerate = wav.getframerate()
    str_data = wav.readframes(num_frames)
    wav.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
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

if __name__ == '__main__':
    audo = "/nfs/public/materials/data/aishell/data_aishell/wav/S0002/train/S0002/BAC009S0002W0122.wav"
    aishell_path = "/nfs/public/materials/data/aishell/data_aishell/"

    # fbank_feature("/nfs/public/materials/data/aishell/data_aishell/wav/train/S0217/BAC009S0217W0325.wav", 128, 40)
    # wav_data, rate = read_wav_data(audo)
    # fbank_feature(wav_data,rate,128)
    generate_aishell_feature2(aishell_path)
    # get_feature(wav_data,1,rate,128)

