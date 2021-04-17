# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 10:33
# @Author  : 奥利波德
# @FileName: NaiveBayes.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_44265507
import pandas as pd
from struct import unpack
from socket import inet_aton
import sklearn
from sklearn.naive_bayes import GaussianNB

inLens = 10000 * 70
testLens = 10000 * 50
predLens = 10000 * 20
path = './NB15/UNSW-NB15_3.csv'


def getData(path):
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
    # 获取训练集与测试集
    trainData = data[0:testLens]
    predictData = data[predLens:]
    return trainData, predictData


if __name__ == '__main__':
    trainData, predictData = getData(path)
    # getProb(trainData)
    trainData_y = trainData['label']
    trainData_X = trainData[['srcip', 'sport', 'dstip', 'dsport']]
    GNBclf = GaussianNB()
    model = GNBclf.fit(trainData_X, trainData_y)
    predictData_x = predictData[['srcip', 'sport', 'dstip', 'dsport']]
    predictData_y = predictData['label']
    preds = GNBclf.predict(predictData_x)
    count = 0
    for i in range(len(predictData_y)):
        if predictData_y.iloc[i] == preds[i]:
            count = count + 1
    precision = count / len(predictData_y)
    print(precision)
