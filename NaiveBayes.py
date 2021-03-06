# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 10:33
# @Author  : 奥利波德
# @FileName: NaiveBayes.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_44265507
import pandas as pd
from struct import unpack
from socket import inet_aton
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

inLens = 10000 * 70
testLens = 10000 * 50
predLens = 10000 * 20
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


def train(train_data):
    print("Training Model ... ")
    train_data_y = train_data['label']
    train_data_X = train_data[['srcip', 'sport', 'dstip', 'dsport']]
    print("GaussianNB Fitting")
    GNBclf = GaussianNB()
    GNBclf.fit(train_data_X, train_data_y)
    return GNBclf


def pred(predict_data, GNBclf):
    print("Predicting ...")
    predict_data_x = predict_data[['srcip', 'sport', 'dstip', 'dsport']]
    predict_data_y = predict_data['label']
    preds = GNBclf.predict(predict_data_x)
    count = 0
    for i in range(len(predict_data_y)):
        if predict_data_y.iloc[i] == preds[i]:
            count = count + 1
    precision = count / len(predict_data_y)
    print(classification_report(predict_data_y,preds))
    return precision,count


if __name__ == '__main__':
    trainData, predictData = getData(path)
    GNBclf = train(trainData)
    precision,count = pred(predictData, GNBclf)

    print("测试集总量：{}  分类准确次数：{}  准确率：{:.2f}%".format(len(predictData), count, precision * 100))
