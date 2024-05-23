
import time

import matplotlib
from scipy.linalg import hankel
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import preprocessing
from sklearn.metrics import classification_report
import warnings
warnings .filterwarnings("ignore")

#如果是GPU，需要去掉注释，如果是CPU，则注释
# gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# assert len(gpu) == 1
# tf.config.experimental.set_memory_growth(gpu[0], True)

# 保存最佳模型自定义类
class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
            print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, val_loss))
            self.model.save_weights(self.path, overwrite=True)
            self.best_loss = val_loss

# t-sne初始可视化函数
def start_tsne(x_train):
    print("正在进行初始输入数据的可视化...")
    print(x_train.shape)
    # x_train = x_train[:len(x_train) // 3]

    x_train1 = tf.reshape(x_train, (len(x_train), 784))
    X_tsne = TSNE().fit_transform(x_train1)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train1)
    plt.colorbar()
    plt.show()

# t-sne结束可视化函数
def end_tsne(x_test):
    model.load_weights(filepath='../model/1DCNN_SVD.h5')
    hidden_features = model.predict(x_test)
    X_tsne = TSNE().fit_transform(hidden_features)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test)
    plt.colorbar()
    plt.show()

# 噪声函数
def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / 784
    npower = xpower / snr
    random.seed(1)
    noise1 = np.random.randn(784) * np.sqrt(npower)
    return x + noise1
def add_gaussian_noise(features, snr_db):
    # 计算信噪比对应的标准差
    snr_linear = 10 ** (snr_db / 10.0)
    std_dev = np.std(features) / snr_linear

    # 生成高斯白噪声
    noise = np.random.randn(*features.shape) * std_dev

    # 将噪声添加到特征中
    features_noisy = features + noise
    return features_noisy
# 模型定义
def mymodel(x_train):
    inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2]))
    h1 = layers.Conv1D(filters=8, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)

    h1 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(h1)
    h1 = layers.MaxPool1D(pool_size=2, strides=2, padding='same')(h1)
    h1 = layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='same', activation='relu')(h1)

    h1 = layers.Dropout(0.7)(h1)
    h1 = layers.Flatten()(h1)
    h1 = layers.Dense(64, activation='relu')(h1)
    h1 = layers.Dense(10, activation='softmax')(h1)

    deep_model = keras.Model(inputs, h1, name="cnn")
    return deep_model

def macnnModel(x_train):
    print(x_train.shape)


# 绘制acc和loss曲线
def acc_line():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    # 绘制accuracy曲线
    plt.plot(epochs, acc, 'r', linestyle='-.')
    plt.plot(epochs, val_acc, 'b', linestyle='dashdot')
    plt.title('Training and validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["Accuracy", "Validation Accuracy"])

    plt.figure()

    # 绘制loss曲线
    plt.plot(epochs, loss, 'r', linestyle='-.')
    plt.plot(epochs, val_loss, 'b', linestyle='dashdot')
    plt.title('Training and validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Loss", "Validation Loss"])
    plt.show()


# 绘制混淆矩阵
def confusion(x_test, y_test):
    y_pred_gailv = model.predict(x_test, verbose=1)
    y_pred_int = np.argmax(y_pred_gailv, axis=1)
    con_mat = confusion_matrix(y_test.astype(str), y_pred_int.astype(str))
    print(con_mat)
    classes = list(set(y_train))
    classes.sort()
    plt.imshow(con_mat, cmap=plt.cm.Blues)
    indices = range(len(con_mat))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('guess')
    plt.ylabel('true')
    for first_index in range(len(con_mat)):
        for second_index in range(len(con_mat[first_index])):
            plt.text(first_index, second_index, con_mat[second_index][first_index], va='center', ha='center')
    plt.show()


def svd_process(data):
    data_all = []

    for i in range(len(data)):
        ## 一维数组转换为二维矩阵
        x2array = hankel(data[i][0:392], data[i][392:784])

        ## 奇异值分解
        U, S, V = np.linalg.svd(x2array)
        S_list = list(S)

        E = 0
        for i in range(len(S_list)):
            E = S_list[i] * S_list[i] + E

        p = []
        for i in range(0, len(S_list)):
            if i == len(S_list) - 1:
                p.append((S_list[i] * S_list[i]) / E)
            else:
                p.append(((S_list[i] * S_list[i]) - (S_list[i + 1] * S_list[i + 1])) / E)

        X = []
        for i in range(len(S_list)):
            X.append(i + 1)

        # 数据重构,保留的奇异值阶数
        K = 20
        for i in range(len(S_list) - K):
            S_list[i + K] = 0.0

        S_new = np.mat(np.diag(S_list))
        reduceNoiseMat = np.array(U * S_new * V)

        reduceNoiseList = []
        for i in range(392):
            reduceNoiseList.append(reduceNoiseMat[i][0])

        for i in range(392):
            reduceNoiseList.append((reduceNoiseMat[len(x2array) - 1][i]))

        data_all.append(reduceNoiseList)

    data_all = np.array(data_all)
    return data_all




# 对输入到模型中的数据进一步处理
def data_pre():
    length = 784  # 样本长度
    number = 100  # 每类样本的数量
    normal = True  # 是否标准化
    rate = [0.5, 0.25, 0.25]  # 训练集、测试集、验证集的划分比例
    path = r'../data/0HP'  # 数据集路径

    # 得到训练集、验证集、测试集
    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocessing.prepro(d_path=path, length=length, number=number,
                                                                              normal=normal, rate=rate)
    # 转为数组array
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_valid = np.array(x_valid)
    y_valid = np.array(y_valid)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # 标签转为int
    y_train = [int(i) for i in y_train]
    y_valid = [int(i) for i in y_valid]
    y_test = [int(i) for i in y_test]

    # 打乱顺序
    index = [i for i in range(len(x_train))]
    random.seed(1)
    random.shuffle(index)
    x_train = np.array(x_train)[index]
    y_train = np.array(y_train)[index]

    index1 = [i for i in range(len(x_valid))]
    random.shuffle(index1)
    x_valid = np.array(x_valid)[index1]
    y_valid = np.array(y_valid)[index1]

    index2 = [i for i in range(len(x_test))]
    random.shuffle(index2)
    x_test = np.array(x_test)[index2]
    y_test = np.array(y_test)[index2]

    # 加噪声
    x_valid = tf.squeeze(x_valid)
    x_valid = np.array(x_valid)
    # for i in range(0, len(x_valid)):
    #     x_valid[i] = wgn(x_valid[i], 20)

    x_test = tf.squeeze(x_test)
    x_test = np.array(x_test)
    # for i in range(0, len(x_test)):
    #     x_test[i] = wgn(x_test[i], 20)


    # svd去噪声
# 2024.03.04  训练集不去噪7575.2
    x_train = svd_process(x_train)
    print("x_train finished")
    x_valid = svd_process(x_valid)
    print("x_valid finished")
    x_test = svd_process(x_test)
    print("x_test finished")

    print(x_train.shape)
    x_train = tf.keras.utils.normalize(x_train, axis=1, order=2)
    x_valid = tf.keras.utils.normalize(x_valid, axis=1, order=2)
    x_test = tf.keras.utils.normalize(x_test, axis=1, order=2)

    # x_train = np.squeeze(x_train)
    #
    # train_X_normalized = list(train_X_normalized)

    x_train = tf.reshape(x_train, (len(x_train), 784, 1))
    x_valid = tf.reshape(x_valid, (len(x_valid), 784, 1))
    x_test = tf.reshape(x_test, (len(x_test), 784, 1))

    return x_train, y_train, x_valid, y_valid, x_test, y_test


# main函数
if __name__ == '__main__':

    # 获取数据
    x_train, y_train, x_valid, y_valid, x_test, y_test = data_pre()

    print("x_train.shape: ", x_train.shape)
    print("y_train.shape: ", y_train.shape)
    print("x_valid.shape: ", x_valid.shape)
    print("y_valid.shape: ", y_valid.shape)
    print("x_test.shape: ", x_test.shape)
    print("y_test.shape: ", y_test.shape)

    x_train1  = x_train[:len(x_train) // 3]

    y_train1 = y_train[:len(x_train) // 3]

    # t-sne初始可视化
    start_tsne(x_train1)

    # 获取定义模型
    model = mymodel(x_train)

    # 打印模型参数
    model.summary()

    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    # 模型训练
    history = model.fit(x_train, y_train,
                        batch_size=128, epochs=400, verbose=1,
                        validation_data=(x_valid, y_valid),
                        callbacks=[CustomModelCheckpoint(
      model, r'../model/1DCNN_SVD.h5')])

    # 加载模型
    model.load_weights(filepath='../model/1DCNN_SVD.h5')

    # 编译模型
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # 评估模型
    scores = model.evaluate(x_test, y_test, verbose=1)

    print("=========模型训练结束==========")
    print("测试集结果： ", '%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

    y_predict = model.predict(x_test)
    y_pred_int = np.argmax(y_predict, axis=1)

    print("混淆矩阵输出结果：")
    print(classification_report(y_test, y_pred_int, digits=4))

    # 绘制acc和loss曲线
    print("绘制acc和loss曲线")
    acc_line()

    # 训练结束的t-sne降维可视化
    print("训练结束的t-sne降维可视化")
    end_tsne(x_test)

    # 绘制混淆矩阵
    print("绘制混淆矩阵")
    confusion(x_test, y_test)