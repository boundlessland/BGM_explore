
import matplotlib.pyplot as plt
# basic handling
import json
import os
import glob
import pickle
import numpy as np
# import pydub
import ffmpeg
# audio
import wave
import pyaudio
import librosa
import librosa.display
# normalization
import sklearn
# nn
import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.utils import to_categorical

os.environ['KERAS_BACKEND']='tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_clip(filepath):
    x, sr = librosa.load(filepath)
    # print(x.shape)
    if x.shape[0]>1323000 :
        x=x[0:1323000]
    else:
        x = np.pad(x,(0,1323000-x.shape[0]),'constant')#补足方法
    return x, sr
#定义一个函数用于读取音频片段，如果时间小于1分钟，将它们补零。采样率22050，60秒一共1323000个采样点。
#如果时间大于一分钟则截取前1分钟

# load_clip('/Users/liuruiqi/Desktop/大作业/video/7.20 上/1/6851141523547950343.wav')

def extract_feature_mfccs(filename):
    try:
        x, sr = load_clip(filename)
    except:
        print(filename)
    else:
        mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
        #print(mfccs.shape)
        norm_mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        return norm_mfccs

#平均响度
def mean(a):
    return np.longlong(sum(a)) / len(a)
def abslist(a):
    return list(map(abs,a))
def extract_feature_recvoice(filepath):
    # 取文件名
    name = os.path.splitext(filepath)[0]
    # 打开WAV文档，文件路径根据需要做修改
    wf = wave.open(name + ".wav", "rb")
    # 创建PyAudio对象
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)
    nframes = wf.getnframes()
    framerate = wf.getframerate()
    # 读取完整的帧数据到str_data中，这是一个string类型的数据
    str_data = wf.readframes(nframes)
    wf.close()
    # 将波形数据转换为数组
    wave_data = np.frombuffer(str_data, dtype=np.short)
    # wave_data=list(map(abs,wave_data))
    M = []
    # 求60秒取样的平均值，若为双声道，再多乘2
    n = framerate * 60
    for i in range(0, len(wave_data), n):
        M.append(wave_data[i:i + n] / 10)  # 传化成分贝
    M = map(abslist, M)
    sound = list(map(mean, M))
    # # 时间数组，与sound配对形成系列点坐标
    # time = numpy.arange(0, nframes / (60 * framerate))
    # time = time.astype(int) * 60
    #
    # dataframe = pd.DataFrame({'Time(s)': time, 'Sound': sound})
    return sound

def extract_otherfeature(filepath):
    x, sr = librosa.load(filepath, sr=44100)
    #色度频率
    hop_length = 512
    chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
    #声谱衰减
    spectral_rolloff = librosa.feature.spectral_rolloff(x + 0.01, sr=sr)[0]
    #光谱质心
    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
    #过零率
    zero_crossings = librosa.zero_crossings(x[0:1323000], pad=False)
    onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=512, aggregate=np.median)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    print(filepath)
    # recvoice=extract_feature_recvoice(filepath)
    print(chromagram.shape)
    print(spectral_rolloff.shape)
    print(spectral_centroids.shape)
    print(zero_crossings.shape)
    print(onset_env.shape)
    print(tempo.shape)
    # print(recvoice)

# extract_otherfeature('/Users/liuruiqi/Desktop/大作业/video/7.20 上/2/6851240364699847949.wav')

def sql(data,aweme_id):
    for item in data['data']:
        # print(item['aweme']['aweme_id'])
        # print(aweme_id)
        # print(item['aweme']['aweme_id'])
        if item['aweme']['aweme_id']==aweme_id:
            return item['aweme']['digg_incr']
    return None

def load_dataset(file_dir):
    # print(features.shape)
    cnt = 0
    file1=()
    file2=()
    file3=()
    video_files=file_dir+'/video'
    record_files=file_dir+'/record'
    for files in os.walk(video_files):
        file1=files
        break
    for day in file1[1]:
        if(int(day)!=20):
            for files in os.walk(video_files+'/'+day):
                file2 = files
                break
            print("day:"+day)
            for order in file2[1]:
                features, incr = np.empty((0, 40, 2584)), np.empty(0)
                print("order:"+order)
                with open(record_files+ '/' + str(day)+ '/' +str(order)+ '.json' , 'rb')as fp:
                    buffer = fp.read()
            # print(buffer)
                data = json.loads(buffer)
            # print(data)
                for files in os.walk(video_files + '/' + str(day)+ '/' +order):
                    file3=files
                for filename in file3[2]:
                    cnt += 1
                    print(cnt)
                    mfccs = extract_feature_mfccs(str(file3[0]) + '/' + str(filename))
                # print(cnt)
                    result = sql(data, filename.split('.')[0])
                    a = [result]
                # print(a)
                    if result != None and result != '-' and int(result)>0:
                        features = np.append(features, mfccs[None], axis=0)
                        incr = np.append(incr, a, axis=0)
                    # labels = np.append(labels, filename.split('.')[0])
                        if cnt%100==0:
                            print(features.shape)
                            print(incr.shape)
                            print("success")
                pickle.dump(np.array(features),open('./project/data_x_'+day,'ab'))
                pickle.dump(np.array(incr), open('./project/data_y_' + day, 'ab'))
        # , np.array(labels, dtype=np.int)
#读取整个数据集，从整个数据集提取特征与标签


#如果还没有将音频转换为features，则进行转化并保存
# load_dataset('/Users/liuruiqi/Desktop/大作业')
#
#数据预处理

def load1(filepath):
    data=np.empty((0, 40, 2584))
    with open(filepath,"rb") as f:
        while True:
            try:
                data=np.append(data,pickle.load(f),axis=0)
                # print(data.shape)
            except:
                # print("over")
                return data

def load2(filepath):
    data=np.empty(0)
    with open(filepath,"rb") as f:
        while True:
            try:
                data=np.append(data,pickle.load(f),axis=0)
                # print(data.shape)
            except:
                # print("over")
                return data
def change(values):
    for i in range(len(values)):
        values[i]/=1000
    return values

def settle():
    train_x = np.append(load1('./data_x_20'),load1('./data_x_21'),axis=0)
    train_y = np.append(load2('./data_y_20'),load2('./data_y_21'),axis=0)
    train_x=np.append(train_x,load1('./data_x_24'),axis=0)
    train_y=np.append(train_y,load2('./data_y_24'),axis=0)
    train_y=change(train_y)

    print(train_x.shape)
    print(train_y)
    train_x = train_x.reshape(train_x.shape[0],train_x.shape[1], train_x.shape[2] , 1)#三维
    train_y = to_categorical(train_y)#整数
    # train_x=train_x[0:1000]
    #     # train_y=train_y[0:1000]
    print(train_x.shape)
    print(train_y.shape)

    test_x=load1('./data_x_22')
    test_y=load2('./data_y_22')
# print(test_x.shape[0])
#     n0=20
#     n1=50
#     test_x=test_x[n0:n1]
#     test_y=test_y[n0:n1]
    test_y=change(test_y)
    test_x = test_x.reshape(test_x.shape[0],test_x.shape[1], test_x.shape[2] , 1)#三维
    test_y = to_categorical(test_y)#整数
    print(test_y.shape)

    if(train_y.shape[1]>=test_y.shape[1]):
        k=train_y.shape[1]
        add=np.zeros((test_y.shape[0],train_y.shape[1]-test_y.shape[1]))
        test_y=np.append(test_y, add, axis=1)
    else:
        k = test_y.shape[1]
        add = np.zeros((train_y.shape[0], test_y.shape[1] - train_y.shape[1]))
        train_y = np.append(train_y, add, axis=1)
    #print(text_y.shape)
    return train_x,train_y,test_x,test_y,k
#
def show_history(history):
    print("H.histroy keys：", history.history.keys())
    plt.figure()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')  # 'bo'为画蓝色圆点，不连线
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()  # 绘制图例，默认在右上角

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()
def train():
    train_x,train_y,test_x,test_y,k=settle()
    #模型(新)
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), activation='relu',input_shape = train_x.shape[1:]))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(32, (3, 3),  activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    model.add(Flatten())#扁平参数
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(k, activation='softmax'))
    method=['Adam','sgd']
    model.compile(optimizer=method[1],
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model.summary(line_length=80)
    #模型(旧)
    # model=load_model('my_model5.h5')
    history = LossHistory()
    history=model.fit(train_x, train_y, epochs=20,batch_size=100,validation_data=(test_x, test_y))
                      #callbacks=[history])
    model.save('my_model6.h5')
    #print(history)
    show_history(history)
    history.loss_plot('epoch')
    # score = model.evaluate(test_x, test_y)
    # print('acc', score[1])

train()

def test(test_x,test_y):
    model=load_model('my_model5.h5')
    score = model.evaluate(test_x, test_y)
    print('acc', score[1])

# train_x,train_y,test_x,test_y,k=settle()
# test(test_x,test_y)





