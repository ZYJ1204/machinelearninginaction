# -*- coding:utf-8 -*-
"""
@File    ：regression.py
@Author  ：ZYJ
@Date    ：2023/8/9 11:59 
"""

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1 #获取特征X的数量
    dataMat = [] # 存储数据的列表
    labelMat = [] # 存储标签列的表
    fr = open(fileName)
    for line in fr.readlines(): # 逐行读取
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = mat(xArr); yMat = mat(yArr).T # 转为矩阵
    xTx = xMat.T*xMat #计算x的转置点乘x
    if linalg.det(xTx) == 0.0: # 如果行列式为0，不存在逆矩阵
        print ("不存在逆矩阵")
        return
    ws = xTx.I * (xMat.T*yMat) # 计算w
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = xMat.shape[0] # 样本数据大小
    weights = mat(eye((m))) # 生成对角矩阵
    for j in range(m):  #遍历每个样本
        diffMat = testPoint - xMat[j,:]     # 计算测试点与样本点的坐标差
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2)) # 计算当前样本点的权重 高斯核
    # 下面与普通线性回归一样，计算每个样本的系数
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    testArr = mat(testArr)
    m = testArr.shape[0]
    yHat = zeros(m)
    for i in range(m): # 遍历每个样本点
        yHat[i] = lwlr(testArr[i],xArr,yArr,k) # 计算每个样本点在不同样本权重下的预测值
    return yHat


if __name__ == '__main__':
    # xArr, yArr = loadDataSet('ex0.txt')
    # ws = standRegres(xArr, yArr)
    # xmat = mat(xArr)
    # ymat = mat(yArr)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(xmat[:, 1].flatten().A[0], ymat.T[:, 0].flatten().A[0])  # 把常量x0去掉了，x1是我们的横坐标
    # # flatten()函数返回一个折叠成一维的数组。
    # # 矩阵.A（等效于矩阵.getA()）变成了数组。
    # # 把点按照升序排列，防止图线乱
    # xcopy = xmat.copy()
    # xcopy.sort(0)
    # yhat = xcopy * ws
    # ax.plot(xcopy[:, 1], yhat)
    # plt.show()
    xArr, yArr = loadDataSet('ex0.txt')
    yhat1 = lwlrTest(xArr, xArr, yArr, 1)
    yhat2 = lwlrTest(xArr, xArr, yArr, 0.01)
    yhat3 = lwlrTest(xArr, xArr, yArr, 0.003)

    xmat = mat(xArr)
    ymat = mat(yArr)
    srtInd = xmat[:, 1].argsort(0)  # 排序索引
    xsort = xmat[srtInd][:, 0, :]
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(311)
    ax.plot(xsort[:, 1], yhat1[srtInd])
    ax.scatter(xmat[:, 1].flatten().A[0], ymat.T.flatten().A[0], s=2, c='red')
    ax = fig.add_subplot(312)
    ax.plot(xsort[:, 1], yhat2[srtInd])
    ax.scatter(xmat[:, 1].flatten().A[0], ymat.T.flatten().A[0], s=2, c='red')
    ax = fig.add_subplot(313)
    ax.plot(xsort[:, 1], yhat3[srtInd])
    ax.scatter(xmat[:, 1].flatten().A[0], ymat.T.flatten().A[0], s=2, c='red')
    plt.tight_layout()
    plt.show()

