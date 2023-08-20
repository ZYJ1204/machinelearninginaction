# -*- coding:utf-8 -*-
"""
@File    ：logRegres.py
@Author  ：ZYJ
@Date    ：2023/7/22 18:59 
"""
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # 转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()  # 转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001  # 移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500  # 最大迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights  # 将矩阵转换为数组，返回权重数组



def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()  # 加载数据集
    dataArr = np.array(dataMat)  # 转换成numpy的array数组
    n = np.shape(dataMat)[0]  # 数据个数
    xcord1 = [];
    ycord1 = []  # 正样本
    xcord2 = [];
    ycord2 = []  # 负样本
    for i in range(n):  # 根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])  # 1为正样本
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])  # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)  # 绘制正样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)  # 绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')  # 绘制title
    plt.xlabel('X1');
    plt.ylabel('X2')  # 绘制label
    plt.show()


def stocGradAscent0(dataMatrix,classLabels):
    """随机梯度上升法"""
    m,n = shape(dataMatrix)   # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01              # 移动步长,也就是学习速率,控制更新的幅度。
    weights = np.ones(n)      # [1. 1. 1.]
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)         #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)               #参数初始化
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01          #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))           #随机选取样本
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*weights))  #选择随机选取的一个样本，计算h
            error = classLabels[dataIndex[randIndex]] - h               #计算误差
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]] #更新回归系数
            del(dataIndex[randIndex])          #删除已经使用的样本
    return weights


if __name__ == '__main__':
    dataArr, labelMat = loadDataSet()
    # matrix = gradAscent(dataArr, labelMat)
    matrix = stocGradAscent0(np.array(dataArr),labelMat)
    # matrix = stocGradAscent1(dataArr, labelMat)
    plotBestFit(matrix)
