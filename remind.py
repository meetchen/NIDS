# -*- coding: utf-8 -*-
# @Time    : 2021/4/17 14:02
# @Author  : 奥利波德
# @FileName: remind.py
# @Software: PyCharm
# @Blog    ：https://blog.csdn.net/qq_44265507
import winsound


def remind_over(freq=440, duration=2000):
    winsound.Beep(freq, duration)

