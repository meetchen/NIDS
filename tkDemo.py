# -*- coding: utf-8 -*-
# @Time    : 2021/4/12 13:30
# @Author  : 奥利波德
# @FileName: tkDemo.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_44265507
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import ttk
import pandas as pd
from torch import nn
import torch
from warnings import filterwarnings
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from socket import inet_aton
from struct import unpack

windows = tk.Tk()
windows.title("实时入侵检测系统")
count = 0
attack = 0
figure = plt.figure(figsize=(3, 2))
draw_set = FigureCanvasTkAgg(figure=figure, master=windows)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.ion()
lens = 700000


def getMode(name):
    # 加载模型
    test_model = torch.load(name, map_location=lambda storage, loc: storage)
    return test_model


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


def getWindows(data, model):
    windows.geometry("1200x600")
    # plt
    draw_set.get_tk_widget().place(x=600, y=60, height=500, width=550)
    tree = ttk.Treeview(windows, columns=['srcip', 'sport', 'dstip', 'dsport', 'isAttack'], show='headings', height=25)
    tree.column('srcip', width=100, anchor='center')
    tree.column('sport', width=100, anchor='center')
    tree.column('dstip', width=100, anchor='center')
    tree.column('dsport', width=100, anchor='center')
    tree.column('isAttack', width=100, anchor='center')
    tree.heading('srcip', text='srcip')
    tree.heading('sport', text='sport')
    tree.heading('dstip', text='dstip')
    tree.heading('dsport', text='dsport')
    tree.heading('isAttack', text='isAttack')
    tree.tag_configure('evenColor', background='lightblue')
    tree.tag_configure('False', background='pink')
    tree.place(x=20, y=40)
    style = ttk.Style()
    style.configure("Treeview", font=(None, 10), rowheight=int(20))
    ttk.Button(windows, text='获取流量', command=lambda: go(tree, data)).place(x=500, y=2)
    ttk.Button(windows, text='流量分类', command=lambda: getLabel(tree, data, model)).place(x=600, y=2)
    return windows


def go(tree, data):
    for i in range(lens-1):
        if i % 2 == 1:
            tags = 'evenColor'
        else:
            tags = ''
        item = data.loc[i].tolist()
        item[4] = ''
        tree.insert('', 'end', values=item, tags=tags)


def getLabel(tree, data, model):
    leaf = tree.get_children()
    for i in leaf:
        tree.delete(i)
    for i in range(lens-1):
        item = data.loc[i].tolist()
        item[4] = getClass(dataForModel(data.loc[i]), model)
        tags = ''
        if item[4]:
            tags = 'False'
        tree.insert('', values=item, tags=tags,index=1)


def getData(path):
    data = pd.read_csv(path, usecols=[0, 1, 2, 3, 48], low_memory=False,
                       names=['srcip', 'sport', 'dstip', 'dsport', 'label'],
                       nrows=lens)
    data = data[data['dsport'].str.contains('^[1-9]\d*$')]
    data = data.sample(frac=1)
    return data


def getIPnumber(ip):
    return float(unpack("!L", inet_aton(ip))[0])


def getClass(dataItem, model):
    prediction = model(dataItem)
    global count
    global attack
    count = count + 1
    if prediction.item() > 0.5:
        pred = True
        attack = attack + 1
    else:
        pred = False
    plt.clf()
    plt.title('当前处理流量总量:'+str(count+attack), fontdict={'weight': 'normal', 'size': 20})
    # explode 强调某元素
    plt.pie(x=[attack, count], colors=['#9999ff', '#ff9999', '#7777aa'], autopct='%3.1f %%',
            labels=['异常流量总量:'+str(attack), '正常流量总量:'+str(count)], explode=(0.1, 0))
    draw_set.draw()
    return pred


def dataForModel(dataItem):
    dataItem['srcip'] = getIPnumber(dataItem['srcip'])
    dataItem['dstip'] = getIPnumber(dataItem['dstip'])
    # 类型转换
    dataItem['dsport'] = float(dataItem['dsport'])
    dataItem['dsport'] = float(dataItem['sport'])
    data = dataItem[['srcip', 'srcip', 'dstip', 'dsport']].values
    dataItem = torch.from_numpy(data.astype('float32'))
    dataItem = dataItem.to(torch.float32)
    return dataItem


if __name__ == '__main__':
    filterwarnings("ignore")
    dataPath = './NB15/UNSW-NB15_3.csv'
    data = getData(dataPath)
    model = getMode('./saveModel/model.pth')
    windows = getWindows(data, model)
    windows.mainloop()
