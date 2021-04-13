import socket
import struct
import torch
import pandas as pd
from torch import nn, optim
import matplotlib.pyplot as plt

inLens = 700000
# csv文件存储的位置
path = './NB15/UNSW-NB15_3.csv'


def getDataSet(path):
    # 读取全部数据
    data = pd.read_csv(path, usecols=[0, 1, 2, 3, 48], low_memory=False,
                       names=['srcip', 'sport', 'dstip', 'dsport', 'label'],
                       nrows=inLens)
    data = data.sample(frac=1)
    # ip地址转网络字节序
    data.iloc[:, 0] = data.iloc[:, 0].map(lambda x: float(struct.unpack("!L", socket.inet_aton(x))[0]))
    data.iloc[:, 2] = data.iloc[:, 2].map(lambda x: float(struct.unpack("!L", socket.inet_aton(x))[0]))
    # 剔除脏数据
    data = data[data['dsport'].str.contains('^[1-9]\d*$')]
    # 类型转换
    data['dsport'] = data['dsport'].astype("float32")
    data['sport'] = data['sport'].astype("float32")
    data['label'] = data['label'].astype("float32")
    # 获取训练集与测试集
    data = data[0:500000]
    return data


def getTrainSet(data):
    # 转换为torch向量
    # 训练集向量转换处理
    train = torch.tensor(data.iloc[:, [0, 1, 2, 3]].values)
    label = torch.tensor(data['label'].values)
    # 再次转换数据格式
    train = train.to(torch.float32)
    label = label.to(torch.float32)
    # 向量升维
    label = label.unsqueeze(-1)
    label = label.cuda()
    train = train.cuda()
    return train, label


class Model(nn.Module):
    # 搭建神经网络
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 1)
        # sigmoid 激活函数 或者0-1中的 光滑曲线
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 两次均引入非线性系数
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def trainModel(model, train, label,plt):
    # 模型使用GPU加速
    model = model.cuda()
    # 配置损失函数
    criterion = nn.BCELoss(reduction='mean')
    # 配置优化器
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # plt xy坐标数据准备
    x = []
    y = []
    plt.ion()
    # 模型训练
    for epoch in range(10000):
        # 获取模型预测值
        y_pred = model(train)
        # 计算损失函数
        loss = criterion(y_pred, label)
        if epoch % 100 == 0:
            x.append(epoch)
            y.append(loss.item())
            # 清空上一张plt
            plt.clf()
            # 获取当前figure位置
            mngr = plt.get_current_fig_manager()
            # 调整位置
            mngr.window.wm_geometry("+850+20")
            plt.plot(x, y)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("loss & epoch",fontdict={'weight':'normal','size': 20})
            plt.pause(0.1)
        if epoch % 200 == 0:
            # 打印损失率
            print("epoch = {}        loss = {:.3f}%".format(epoch, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    plt.ioff()
    return model


if __name__ == '__main__':
    model = Model()
    data = getDataSet(path)
    train, label = getTrainSet(data)
    plt.figure()
    model = trainModel(model, train, label,plt)
    torch.save(model, 'saveModel/model_.pth')
    plt.show()
