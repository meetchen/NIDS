# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 13:33
# @Author  : 奥利波德
# @FileName: NuSvm.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_44265507
from sklearn.svm import SVC
import pandas as pd
from remind import remind_over
from struct import unpack
from socket import inet_aton
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

W = 10000
inLens = int(W * 70)
testLens = int(W * 0.2)
path = './NB15/UNSW-NB15_3.csv'


def getData(path):
    print("Loading Data ...")
    # 读取全部数据
    data = pd.read_csv(path, usecols=[0, 1, 2, 3, 48], low_memory=False,
                       names=['srcip', 'sport', 'dstip', 'dsport', 'label'],
                       nrows=inLens)
    data = data.sample(frac=1)
    # ip地址转网络字节序
    data.iloc[:, 0] = data.iloc[:, 0].map(lambda x: float(unpack("!L", inet_aton(x))[0]))
    data.iloc[:, 2] = data.iloc[:, 2].map(lambda x: float(unpack("!L", inet_aton(x))[0]))
    # 剔除脏数据
    data = data[data['dsport'].str.contains('^[1-9]\d*$')]
    # 类型转换
    data['dsport'] = data['dsport'].astype("float32")
    data['sport'] = data['sport'].astype("float32")
    data['label'] = data['label'].astype("float32")
    # 打乱数据
    data = data.sample(frac=1)
    # 获取训练集与测试集
    trainData = data[0:testLens]
    predictData = data[testLens:]
    return trainData, predictData


def process_data(data):
    print("process_data ing ...")
    y = data['label']
    x = data[['srcip', 'sport', 'dstip', 'dsport']]
    return x, y


def train(data):
    x, y = process_data(data)
    print("train ing ...")
    SVCClf = SVC(kernel='rbf', probability=True)
    SVCClf.fit(x, y)
    print(SVCClf)
    return SVCClf


def predict(model, predict_data):
    print("predict ing ...")
    x, y = process_data(predict_data)
    predicts = model.predict(x)
    print(classification_report(y, predicts))
    count = 0
    for i in range(len(predict_data)):
        if y.iloc[i] == predicts[i]:
            count = count + 1
    precision = count / len(predict_data)
    return precision, count


if __name__ == '__main__':
    train_data, predict_data = getData(path)
    model = train(train_data)
    precision, count = predict(model, predict_data)
    print("测试集总量：{}  分类准确次数：{}  准确率：{:.2f}%".format(len(predict_data), count, precision * 100))
