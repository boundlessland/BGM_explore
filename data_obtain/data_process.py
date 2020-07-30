# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 00:45:18 2020

@author: jsx
"""

import os
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
Const_Image_Format = [".wav"]
class FileFilt:
    fileList = []
    counter = 0
    def __init__(self):
        pass
    def FindFile(self,dirr,filtrate = 1):
        global Const_Image_Format
        for s in os.listdir(dirr):
            newDir = os.path.join(dirr,s)
            if os.path.isfile(newDir):
                if filtrate:
                    if newDir and(os.path.splitext(newDir)[1] in Const_Image_Format):
                        self.fileList.append(s)
                        self.counter+=1
                else:
                    self.fileList.append(s)
                    self.counter+=1

if __name__ == "__main__":
    '''b = FileFilt()
    baseDir = r"D:\jsx-20180929\大二下\数据科学基础\抖音带货视频分析\2020年7月18日采集的样本数据"
    b.FindFile(dirr = baseDir)
    newDir = os.path.join(baseDir,b.fileList[0])'''
    newDir = 'C:\\Users\\jsx\\Desktop\\抖音下载\\1\\6852516216171400455.wav'
    x, sr = librosa.load(newDir,sr = 44100,mono = True,duration = 9.0)

    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(x, sr=sr)

    onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=512, aggregate=np.median)

    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)#平均频率

    chroma_stft = librosa.feature.chroma_stft(y=x, sr=sr)#色度频率

    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)#光谱质心

    spec_bw = librosa.feature.spectral_bandwidth(y=x, sr=sr)

    rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr)#光谱衰减

    zcr = librosa.feature.zero_crossing_rate(x)#过零率

    mfcc = librosa.feature.mfcc(y=x, sr=sr)#梅尔普倒频系数