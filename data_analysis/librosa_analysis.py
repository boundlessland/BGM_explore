import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import csv
import os, sys
import json

# 打开文件
path = "/Users/jameswu/Documents/1/"
dirs = os.listdir(path)

id_list = []
follower_list = []
digg_inc = []

with open('1.json', 'r', encoding='utf8') as fp:
   json_data = json.load(fp)

for i in json_data['data']:
   id = str(i['aweme']['aweme_id']) + '.wav'
   id_list.append(id)
   follower_list.append(i['author']['follower_count'])
   digg_inc.append(i['aweme']['digg_incr'])



def getDict(filepath):
    att_list = ['tempo',  'spec_cent', 'zcr', 'chroma_stft', 'rolloff']
    x, sr = librosa.load(filepath, sr=44100)
    onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=512, aggregate=np.median)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    y, sr = librosa.load(filepath, mono=True, duration=30)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    num_list = [filepath, tempo, np.mean(spec_cent),np.mean(zcr),np.mean(chroma_stft), np.mean(rolloff)]
    # dic = dict(zip(att_list, num_list))

    return num_list

filepath = '.wav'

rows = []
# 输出所有文件和文件夹
for file in dirs:
    tmp_list = []
    name = str(file)
    if file in id_list:
        n = id_list.index(name)
        tmp_list.append(file)
        tmp_list.append(follower_list[n])
        tmp_list.append(digg_inc[n])


        tmp_filepath = path + file
        dict = getDict(tmp_filepath)


        rows.append(tmp_list+dict)

# for i in range(13,15):
#     tmp_filepath = str(i) + filepath
#     print(getDict(tmp_filepath))
#     rows.append(getDict(tmp_filepath))

headers = ['file', 'follower','inc','filepath', 'tempo',  'spec_cent', 'zcr', 'chroma_stft', 'rolloff']


with open('test.csv','w')as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)