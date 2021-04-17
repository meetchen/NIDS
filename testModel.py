# -*- coding: utf-8 -*-
# @Time    : 2021/4/8 21:57
# @Author  : 奥利波德
# @FileName: testModel.py.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_44265507
from struct import  unpack
from socket import inet_aton
import torch
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class Model(nn.Module):
    # 搭建神经网络
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def getTestData(path):
    inLens = 700000
    # 读取全部数据
    data = pd.read_csv(path, usecols=[0, 1, 2, 3, 48], low_memory=False,
                       names=['srcip', 'sport', 'dstip', 'dsport', 'label'],
                       nrows=inLens)
    data = data.sample(frac=1)
    test = data[500000:-1].copy()
    test.iloc[:, 0] = test.iloc[:, 0].map(lambda x: float(unpack("!L", inet_aton(x))[0]))
    test.iloc[:, 2] = test.iloc[:, 2].map(lambda x: float(unpack("!L", inet_aton(x))[0]))
    test = test[test['dsport'].str.contains('^[1-9]\d*$')]
    test['dsport'] = test['dsport'].astype("float32")
    test['sport'] = test['sport'].astype("float32")
    test['label'] = test['label'].astype("float32")
    label = torch.tensor(test['label'].values)
    test = torch.tensor(test.iloc[:, [0, 1, 2, 3]].values)
    test = test.to(torch.float32)
    label = label.to(torch.float32)
    test = test.cuda()
    label = label.cuda()
    return test, label


def getMode(name):
    # 加载模型
    test_model = torch.load(name)
    test_model.eval()
    test_model.cuda()
    return test_model


def getPrediction(model, test, label, plt):
    # 模型预测 计算正确率
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.ion()
    prediction = model(test)
    count = 0
    for i in range(len(prediction)):
        if prediction[i].item() > 0.5:
            prediction[i] = 1
            pred = 1
        else:
            prediction[i] = 0
            pred = 0
        if pred == label[i].item():
            count = count + 1
        if i % 500 == 0:
            plt.clf()
            plt.title('预测准确度', fontdict={'weight': 'normal', 'size': 20})
            mngr = plt.get_current_fig_manager()
            mngr.window.wm_geometry("+850+20")
            # explode 强调某元素
            plt.pie(x=[count, len(prediction) - count], colors=['#9999ff', '#ff9999', '#7777aa'], autopct='%3.1f %%',
                    labels=['预测准确值', '未完成预测值'], explode=(0.1, 0))
            plt.legend(loc="best", fontsize=10, bbox_to_anchor=(1.1, 1.05), borderaxespad=0.3)
            plt.pause(0.2)
    plt.ioff()
    y_t = label.cpu().detach()
    y_p = prediction.cpu().detach().squeeze()
    print(classification_report(y_t, y_p))
    return count


if __name__ == '__main__':
    model = getMode('./saveModel/model.pth')
    test, label = getTestData('./NB15/UNSW-NB15_3.csv')
    plt.figure()
    count = getPrediction(model, test, label, plt)
    print("测试集总量：{}  分类准确次数：{}  准确率：{:.2f}%".format(len(test), count, count / len(test) * 100))
    plt.show()
