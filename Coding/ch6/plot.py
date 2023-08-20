# -*- coding:utf-8 -*-
"""
@File    ：plot.py
@Author  ：ZYJ
@Date    ：2023/7/24 16:20 
"""

# 画出待分类数据的二维分布
import numpy as np
import matplotlib.pyplot as plt

def plotBestFit(dataMat, labelMat):
    # 加载数据
    dataArr = np.array(dataMat)

    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    # print(labelMat)
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i][0])
            ycord1.append(dataArr[i][1])
        else:
            xcord2.append(dataArr[i][0])
            ycord2.append(dataArr[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # print(xcord1)
    # print(ycord1)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    # arange()函数用于创造等差数组
    # 起始点 终止点 步长
    # x=np.arange(-3.0,3.0,0.1)
    # y=(-weights[0]-weights[1]*x)/weights[2]
    # ax.plot(x,y)
    # 添加xy轴标签
    plt.xlabel('X1')
    plt.ylabel('X2')
    # 显示图像
    plt.show()