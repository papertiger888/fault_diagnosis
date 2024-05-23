

import tensorflow as tf
import matplotlib
import preprocessing
from scipy.linalg import hankel
import numpy as np
import random
import matplotlib.pyplot as plt
from vmdpy import VMD
from scipy.signal import hilbert
from scipy.fft import fft

alpha = 2000
K = 5
tau = 0
DC = 'true'
init = 1
tol = 1e-7

from vmdpy import VMD
def new_data():
    length = 1024
    number = 200  # 每类样本的数量
    normal = True  # 是否标准化
    rate = [0.5, 0.25, 0.25]  # 测试集验证集划分比例

    path = r'../data/0HP'
    Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y = preprocessing.prepro(
    d_path=path,
    length=length,
    number=number,
    normal=normal,
    rate=rate)


    x_train = np.array(Train_X)
    y_train = np.array(Train_Y)
    x_test = np.array(Test_X)
    y_test = np.array(Test_Y)
    y_test = np.squeeze(y_test)

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = new_data()

# 噪声公式
def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / 1024
    npower = xpower / snr
    random.seed(1)
    noise1 = np.random.randn(1024) * np.sqrt(npower)
    return x + noise1

# 设置matplotlib支持中文的字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 'Microsoft YaHei' 也是一个常用的选项
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


t = np.arange(0, 1024, 1)
fig2 = plt.figure().add_subplot(111)
print("+++++++++++++++++")
print(x_train[500].shape)
# fig2.plot(t, list(x_train[500]), 'b', label='原始数据')
fig2.plot(t, list(x_train[500]), 'b', )
fig2.legend()
fig2.set_xlabel('采样点', size=15)
fig2.set_ylabel('数值', size=15)
plt.tight_layout()
plt.show()

# VMD分解
models, _, _ = VMD(x_train[500],alpha = alpha,tau = tau,K =K,DC = DC,init = init,tol = tol)


for i,mode in enumerate(models):
    plt.figure(figsize=(10,2))
    plt.plot(mode)
    plt.title(f'Mode {i+1}')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

    envelope = np.abs(hilbert(mode))

    # DFT变换得到频谱
    spectrum = np.abs(fft(envelope))
    spectrum = spectrum[1:500]
    # freq = np.fft.fftfreq(len(spectrum), d=1)  # 假设采样频率为1Hz，可以根据实际情况调整
    sample_indices = np.arange(len(spectrum))
    # 绘制包络信号图
    plt.figure(figsize=(10, 2))
    # plt.subplot(2, 1, 1)
    plt.plot(envelope)
    plt.title(f'IMF {i + 1} Envelope Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()

    # 绘制频谱图
    plt.figure(figsize=(10, 2))
    plt.plot(sample_indices, spectrum)
    plt.title(f'IMF {i + 1} Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.tight_layout()
    plt.show()


# 向某一条训练样本添加噪

x_train[500] = wgn(x_train[500], 5)


## 1.待处理信号(1024个采样点)
t = np.arange(0, 1024, 1)

## 2.一维数组转换为二维矩阵
x2array = hankel(x_train[500][0:512], x_train[500][512:1024])

## 3.奇异值分解
U, S, V = np.linalg.svd(x2array)
S_list = list(S)

## 奇异值求和
S_sum = sum(S)

##奇异值序列归一化
S_normalization_list = [x for x in S_list]

E = 0
for i in range(len(S_list)):
    E = S_list[i] * S_list[i] + E

p = []
for i in range(0, len(S_list)):
    if i == len(S_list)-1:
        p.append((S_list[i] * S_list[i]) / E)
    else:
        p.append(((S_list[i] * S_list[i]) - (S_list[i+1] * S_list[i+1])) / E)

X = []
for i in range(len(S_list)):
    X.append(i + 1)

fig1 = plt.figure().add_subplot(111)
fig1.plot(X, p)
fig1.set_xlabel('阶次', size=15)
fig1.set_ylabel('能量差分谱', size=15)
plt.show()

# 4.画图
X = []
for i in range(len(S_list)):
    X.append(i + 1)

fig1 = plt.figure().add_subplot(111)
fig1.plot(X, S_normalization_list)
fig1.set_xlabel('秩', size=15)
fig1.set_ylabel('奇异值归一化', size=15)
plt.show()

## 5.数据重构
K = 20  ## 保留的奇异值阶数
for i in range(len(S_list) - K):
    S_list[i + K] = 0.0

S_new = np.mat(np.diag(S_list))
reduceNoiseMat = np.array(U * S_new * V)

reduceNoiseList = []
for i in range(512):
    reduceNoiseList.append(reduceNoiseMat[i][0])

for i in range(512):
    reduceNoiseList.append((reduceNoiseMat[len(x2array)-1][i]))

## 6.去噪效果展示
fig2 = plt.figure().add_subplot(111)
fig2.plot(t, list(x_train[500]), 'b', label='原始数据')
fig2.plot(t, reduceNoiseList, 'r-', label='处理之后数据')
fig2.legend()
fig2.set_xlabel('采样点', size=15)
fig2.set_ylabel('数值', size=15)
plt.show()


# 对训练数据进行标准化处理
train_X_normalized = tf.keras.utils.normalize(reduceNoiseList, axis=-1, order=2)
print(train_X_normalized.shape)
train_X_normalized = np.squeeze(train_X_normalized)
print(train_X_normalized.shape)
train_X_normalized = list(train_X_normalized)

fig2 = plt.figure().add_subplot(111)
fig2.plot(t, reduceNoiseList, 'b', label='降噪后原始数据')
fig2.plot(t, train_X_normalized, 'r-', label='标准化后数据')
fig2.legend()
fig2.set_xlabel('采样点', size=15)
fig2.set_ylabel('数值', size=15)
plt.show()