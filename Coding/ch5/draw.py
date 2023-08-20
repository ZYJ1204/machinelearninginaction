# -*- coding:utf-8 -*-
"""
@File    ：draw.py
@Author  ：ZYJ
@Date    ：2023/7/23 19:39 
"""

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random

def loadDataSet():
    dataMat = []  # 创建数据列表
    labelMat = []  # 创建标签列表
    fr = open('testSet.txt')  # 打开文件
    for line in fr.readlines():  # 逐行读取
        lineArr = line.strip().split()  # 去回车，放入列表
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(int(lineArr[2]))  # 添加标签
    fr.close()  # 关闭文件
    return dataMat, labelMat  # 返回

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # 转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()  # 转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01  # 移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500  # 最大迭代次数
    weights = np.ones((n, 1))
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    return weights.getA(), weights_array  # 将矩阵转换为数组，并返回

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)  # 参数初始化
    weights_array = np.array([])  # 存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取样本
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]] * weights))  # 选择随机选取的一个样本，计算h
            error = classLabels[dataIndex[randIndex]] - h  # 计算误差
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]]  # 更新回归系数
            weights_array = np.append(weights_array, weights, axis=0)  # 添加回归系数到数组中
            del (dataIndex[randIndex])  # 删除已经使用的样本
    weights_array = weights_array.reshape(numIter * m, n)  # 改变维度
    return weights, weights_array  # 返回

def plotWeights(weights_array1, weights_array2):
    # 设置汉字格式
    # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])  # fontproperties='SimHei',fontsize=14
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系', fontproperties='SimHei', fontsize=14)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', fontproperties='SimHei', fontsize=14)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', fontproperties='SimHei', fontsize=14)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', fontproperties='SimHei', fontsize=14)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2', fontproperties='SimHei', fontsize=14)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系', fontproperties='SimHei', fontsize=14)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', fontproperties='SimHei', fontsize=14)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', fontproperties='SimHei', fontsize=14)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', fontproperties='SimHei', fontsize=14)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1', fontproperties='SimHei', fontsize=14)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights1, weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)

    weights2, weights_array2 = gradAscent(dataMat, labelMat)
    plotWeights(weights_array1, weights_array2)