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
import json
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import python_speech_features
import matplotlib.pyplot as plt
from specAugment import spec_augment_pytorch
import librosa
import speechpy

ROOT = "/mnt/32da6dad-b2d9-45a9-8959-49fff09a3aa3/speech_datasets/chinese"
MY_ROOT = "/media/dapeng/文档/Dataset/中文语音数据集"

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

def delete_final_line(file):
    with open(file, "rb+") as f:
        lines = f.readlines()  # 读取所有行
        last_line = lines[-1]  # 取最后一行
        f.seek(-1, os.SEEK_END)
        f.truncate()


def aishell(aishell_root):
    # 生成3个feats.scp
    wav_dir_path = os.path.join(aishell_root, "wav")
    subdirs = ["train","dev","test"]
    for subdir in subdirs:
        save_file = os.path.join(aishell_root,subdir+".scp")
        sub_root = os.path.join(wav_dir_path,subdir)
        print(sub_root)
        with open(save_file, "w") as f:
            for root, dirs, files in os.walk(sub_root):
                for file in files:
                    file_name = file.split(".")[0]
                    file_path = os.path.join(root,file)
                    file_path = file_path.replace(MY_ROOT,ROOT)
                    f.writelines(file_name + " " + file_path + "\n")
        delete_final_line(save_file)


def pad_name(filename):
    if len(filename) == 7:
        return filename
    else:
        parts = filename.split("_")
        a = ""
        b = ""
        if len(parts[0]) == 3:
            a = parts[0]
        else:
            a = parts[0][0]+"0"+parts[0][1]
        if len(parts[1]) == 3:
            b = parts[1]
        elif len(parts[1]) == 2:
            b = "0"+parts[1]
        else:
            b = "00"+parts[1]
    return a+"_"+b

def thchs30(thchs30_root):
    subdirs = ["train","dev","test"]
    for subdir in subdirs:
        save_file = os.path.join(thchs30_root,subdir+".scp")
        sub_root = os.path.join(thchs30_root,subdir)
        print(sub_root)
        with open(save_file, "w") as f:
            for root, dirs, files in os.walk(sub_root):
                for file in files:
                    file_format = file.split(".")[-1]
                    if file_format=="wav":
                        file_name = file.split(".")[0]
                        file_name = pad_name(file_name)
                        file_path = os.path.join(root,file)
                        file_path = file_path.replace(MY_ROOT,ROOT)
                        f.writelines(file_name + " " + file_path + "\n")
        delete_final_line(save_file)


def aidatatang(aidatatang_root):
    # 生成3个feats.scp
    wav_dir_path = os.path.join(aidatatang_root, "corpus")
    subdirs = ["train", "dev", "test"]
    for subdir in subdirs:
        save_file = os.path.join(aidatatang_root, subdir + ".scp")
        save_grapheme_file = os.path.join(aidatatang_root, "grapheme_"+ subdir + ".txt")
        sub_root = os.path.join(wav_dir_path, subdir)
        print(sub_root)
        with open(save_file, "w") as f:
            with open(save_grapheme_file,"w") as fg:
                for root, dirs, files in os.walk(sub_root):
                    for file in files:
                        file_format = file.split(".")[-1]
                        if file_format=="wav":
                            file_name = file.split(".")[0]
                            file_path = os.path.join(root, file)
                            file_path = file_path.replace(MY_ROOT, ROOT)
                            f.writelines(file_name + " " + file_path + "\n")
                        elif file_format=="txt":
                            file_name = file.split(".")[0]
                            with open(os.path.join(root,file),"r") as rf:
                                lines = rf.readlines()
                            content = str(lines[0])
                            content = " ".join(content).strip()
                            fg.writelines(file_name+" "+content+"\n")
        delete_final_line(save_file)
        delete_final_line(save_grapheme_file)


def primeword(primeword_root):
    # 不划分
    wav_dir_path = os.path.join(primeword_root, "audio_files")
    save_file = os.path.join(primeword_root,"feats.scp")
    with open(save_file, "w") as f:
        for root, dirs, files in os.walk(wav_dir_path):
            for file in files:
                file_name = file.split(".")[0]
                file_path = os.path.join(root,file)
                file_path = file_path.replace(MY_ROOT,ROOT)
                f.writelines(file_name + " " + file_path + "\n")
    delete_final_line(save_file)
    json_file = os.path.join(primeword_root,"set1_transcript.json")
    save_grapheme_file = os.path.join(primeword_root,"grapheme_all.txt")
    with open(json_file,"r",encoding="utf-8") as f:
        json_list = json.load(f)
    with open(save_grapheme_file,"w") as wf:
        for j in json_list:
            file_name = j["file"]
            file_name = str(file_name).split(".")[0]
            content = j["text"]
            content = str(content).strip().replace(" ","")
            content = " ".join(content)
            wf.writelines(file_name+" "+content+"\n")
    delete_final_line(save_grapheme_file)


def stcmds(stcmds_root):
    wave_root = os.path.join(stcmds_root,"ST-CMDS-20170001_1-OS")

    save_feats_file = os.path.join(stcmds_root,"feats.scp")
    save_grapheme_file = os.path.join(stcmds_root,"grapheme_all.txt")
    with open(save_feats_file,"w") as f:
        with open(save_grapheme_file,"w") as fg:
            for root, dirs, files in os.walk(wave_root):
                for file in files:
                    file_format = file.split(".")[-1]
                    if file_format == "wav":
                        file_name = file.split(".")[0]
                        file_path = os.path.join(root, file)
                        file_path = file_path.replace(MY_ROOT, ROOT)
                        f.writelines(file_name + " " + file_path + "\n")
                    elif file_format == "txt":
                        file_name = file.split(".")[0]
                        with open(os.path.join(root, file), "r") as rf:
                            lines = rf.readlines()
                        content = str(lines[0]).strip()
                        content = " ".join(content).strip()
                        fg.writelines(file_name + " " + content + "\n")
    delete_final_line(save_feats_file)
    delete_final_line(save_grapheme_file)

def magicdata(magicdatda_root):
    subdirs = ["train","dev","test"]
    for subdir in subdirs:
        save_file = os.path.join(magicdatda_root,subdir+".scp")
        sub_root = os.path.join(magicdatda_root,subdir)
        with open(save_file,"w") as f:
            for root,dirs,files in os.walk(sub_root):
                for file in files:
                    file_format = file.split(".")[-1]
                    if file_format == "wav":
                        file_name = file.split(".")[0]
                        file_path = os.path.join(root,file)
                        f.writelines(file_name+" "+file_path+"\n")
        delete_final_line(save_file)

def magic_grapheme(magic_data_root):
    subfiles = ["train","dev","test"]
    for subfile in subfiles:
        r_file = os.path.join(magic_data_root, subfile+".txt")
        save_file = os.path.join(magic_data_root,"grapheme_"+subfile+".txt")
        with open(r_file,"r") as rf:
            lines = rf.readlines()
        with open(save_file,"w") as wf:
            for i in range(1,len(lines)):
                l = lines[i].split("	")
                file_name = l[0].split(".")[0]
                content = l[-1].split(" ")[-1].strip()
                content = " ".join(content)
                wf.writelines(file_name+" "+content+"\n")
        delete_final_line(save_file)

def merge(joint_root):
    subdirs = ["train","dev","test"]
    for subdir in subdirs:
        sub_root = os.path.join(joint_root,subdir)
        # save_feats = os.path.join(joint_root,subdir+"_feats.scp")
        save_grapheme = os.path.join(joint_root,subdir+"_grapheme.txt")
        # with open(save_feats,"w",encoding="utf-8") as ff:
        with open(save_grapheme,"w",encoding="utf-8") as fg:
            for root, dirs, files in os.walk(sub_root):
                for file in files:
                    # file_format = file.split(".")[-1]
                    # if file_format == "scp":
                    #     with open(os.path.join(root,file),"r") as rf:
                    #         lines = rf.readlines()
                    #     # for line in lines:
                    #     #     ff.writelines(line)
                    #     ff.writelines(lines)
                    #     ff.writelines("\n")
                    # elif file_format == "txt":
                    with open(os.path.join(root,file),"r") as rf:
                        lines = rf.readlines()
                    # for line in lines:
                    #     fg.writelines(line)
                    fg.writelines(lines)
                    fg.writelines("\n")
        # delete_final_line(save_feats)
        delete_final_line(save_grapheme)


def grapheme_table(joint_root):
    subroots = ["train","dev","test"]
    save_file = os.path.join(joint_root,"grapheme_table.txt")
    max = 0
    grapheme = {"<b>":0}
    index = 1
    for subroot in subroots:
        grapheme_text = os.path.join(joint_root,subroot,"grapheme_all.txt")
        with open(grapheme_text,"r") as rf:
            print(grapheme_text)
            for line in rf:
                line = line.strip()
                parts = line.split()
                content = parts[1:]
                length = len(content)
                max =  length if length > max else max
                for word in content:
                    if grapheme.get(word) is None:
                        print(word,index)
                        if word == "▽":
                            print(content,line)
                        if word == "菈":
                            print(content,line)
                        if word == "屮":
                            print(content, line)
                        if word == "録":
                            print(content, line)
                        grapheme[word]=index
                        index+=1

    with open(save_file, "w") as wf:
        for key in grapheme.keys():
            wf.writelines(key+" "+str(grapheme[key])+"\n")
    delete_final_line(save_file)

    print(max)

import string
from zhon.hanzi import punctuation
ct = punctuation
et = string.punctuation
token = ct + et
def is_ok(ch):
    """判断一个unicode是否是汉字"""
    global token
    if ch in token:
        return False
    elif '\u4e00' <= ch <= '\u9fff':
        return True
    return False

def remove_token_and_generate_table(joint_root):
    subroots = ["train","dev","test"]
    save_table = os.path.join(joint_root,"grapheme_table.txt")
    dic = {"<b>":0}
    index = 1
    for subroot in subroots:
        grapheme_txt = os.path.join(joint_root,subroot,"grapheme_all.txt")
        save_grapheme_file = os.path.join(joint_root,subroot,"grapheme.txt")
        with open(grapheme_txt,"r",encoding='utf-8') as rf:
            lines = rf.readlines()
        sub_lines = []
        delete_num = 0
        for line in lines:
            parts = line.strip().split(" ")
            name = parts[0]
            content = parts[1:]
            ok=True
            for word in content:
                if is_ok(word) == False:
                    ok = False
                    delete_num+=1
                    break
            if ok ==True:
                sub_lines.append(line)
                for word in content:
                    if dic.get(word) is None:
                        dic[word]=index
                        index+=1
        print(subroot+"删除:"+str(delete_num))
        with open(save_grapheme_file,"w",encoding="utf-8") as wf:
            wf.writelines(sub_lines)
    with open(save_table,"w",encoding="utf-8") as f:
        for key in dic.keys():
            f.writelines(key+" "+str(dic[key])+"\n")
    delete_final_line(save_table)

if __name__ == '__main__':
    # my_aishell_root = "/media/dapeng/文档/Dataset/中文语音数据集/data_aishell"
    # my_thchs30_root = "/media/dapeng/文档/Dataset/中文语音数据集/data_thchs30"
    # my_aidatatang_root = "/media/dapeng/文档/Dataset/中文语音数据集/aidatatang_200zh"
    # my_magicdata_root = "/media/dapeng/文档/Dataset/中文语音数据集/magicdata"
    # my_primewords_root = "/media/dapeng/文档/Dataset/中文语音数据集/primewords_md_2018_set1"
    # my_stcmdss_root = "/media/dapeng/文档/Dataset/中文语音数据集/ST-CMDS"
    joint_root = "/home/dapeng/Code/Transformer-Transducer/data/joint"
    #
    # merge(joint_root)
    remove_token_and_generate_table(joint_root)

    # feats = "/media/dapeng/文档/Dataset/中文语音数据集/Joint/feats.scp"
    # grapheme = "/media/dapeng/文档/Dataset/中文语音数据集/Joint/grapheme_all.txt"
    # with open(feats,"r") as f:
    #     lines1 = f.readlines()
    # with open(grapheme,"r") as f:
    #     lines2 = f.readlines()
    # print(len(lines1),len(lines2))

    # joint_root = "/home/dapeng/Code/Transformer-Transducer/data/joint"
    # grapheme_table(joint_root)