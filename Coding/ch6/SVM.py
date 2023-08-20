# -*- coding:utf-8 -*-
"""
@File    ：SVM.py
@Author  ：ZYJ
@Date    ：2023/7/24 16:08 
"""

from numpy import *
import  numpy as np
from time import sleep
import random

from Coding.ch6.plot import plotBestFit


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    # 读取文件
    # 按行读取
    for line in fr.readlines():
        # 将每一行的数据进行分割
        lineArr = line.strip().split('\t')
        # 前两列为x，y值
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        # 第三列为标签值
        labelMat.append(float(lineArr[2]))
    # 返回数据集和标签集
    return dataMat, labelMat

#         i输入下标，m为总数
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

# 用于调整大于H或是小于L的alpha的值
def clipAlpha(aj, H, L):
    # 大于H的值取H
    if aj > H:
        aj = H
    # 小于L的值取L
    if L > aj:
        aj = L
    return aj

# 简化版的smo算法
# 构建函数时，采用通用的接口，这样就可以对算法和数据源进行组合和配对处理
# 输入参数：数据集，类别标签，常数C，容错率，取消最大的循环次数
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=mat(dataMatIn)   #输入数据集变为矩阵
    labelMat=mat(classLabels).transpose()
    # 把输入标签变为矩阵并且转置，得到一个列向量，标签的每行对应每行的数据
    b=0 # 初始化
    m,n=shape(dataMatrix)# 得到矩阵的行列数
    alphas=mat(zeros((m,1)))# 定义一个m*1的零矩阵
    iter=0  # 初始化迭代次数
    #只有在所有数据集上遍历maxIter次，且不再发生任何alpha修改之后，程序才会停止并退出while循环
    while (iter<maxIter):   # 迭代次数大于最大迭代次数时，退出迭代
        alphaPairsChanged=0 # 用来记录alpha是否已经进行优化
        for i in range(m):  # 循环每个数据
            # 步骤1：计算误差Ei
            fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            # 预测第i个数据的类别  multiply对应的位置相乘
            Ei=fXi-float(labelMat[i])
            # 实际结果与真实结果的误差，如果误差很大，那么就要对该数据实例所对应的alpha值进行优化
            # 一旦alphas等于0或C，那么它们就巳经在“边界”上了，因而不再能够减小或增大，因此也就不值得再对它们进行优化了
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler)and (alphas[i]>0)):#将违反KKT条件的找出来，具体公式来源见https://blog.csdn.net/youhuakongzhi/article/details/86660281
                j=selectJrand(i,m)  # 在m个中随机选择一个和i不同的j
                fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                # 预测第j个数据的类别
                Ej=fXj-float(labelMat[j])   # 计算第j个的误差
                alphaIold=alphas[i].copy()  # 保存更新前的aplpha值，使用深拷贝
                alphaJold=alphas[j].copy()
                if (labelMat[i]!=labelMat[j]):      # 如果yi和yj的标签不一样
                    L=max(0,alphas[j]-alphas[i])    # alphas[j]new的取值范围
                    H=min(C,C+alphas[j]-alphas[i])
                else:   # 如果yi和yj的标签一样
                    L=max(0,alphas[j]+alphas[i]-C)  # alphas[j]new的取值范围
                    H=min(C,alphas[j]+alphas[i])
                if L==H:print("L==H");continue  # 退出本次循环，直接进行下一次for循环
                # 步骤3：计算eta=-2*Kij+Kii+Kjj，而这儿eta=2*Kij-Kii-Kjj,所以下面公式中用的是减号
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0: print("eta>=0"); continue    # 退出本次循环，直接进行下一次for循环
                # 目标函数是求最小值，这儿eta是负的二阶导数，因此二阶导数不能小于等于0
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta  # 更新alphas[j]
                alphas[j]=clipAlpha(alphas[j],H,L)  # 用于调整aj值，让aj在H和L的范围内
                if (abs(alphas[j]-alphaJold)<0.00001):  # alphas[j]变化太小
                    print("j not moving enough"); continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])    # 更新alphas[i]
                # 更新b1
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*\
                dataMatrix[i,:]*dataMatrix[j,:].T
                # 更新b2
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*\
                dataMatrix[j,:]*dataMatrix[j,:].T
                # 更新b
                if (0<alphas[i]) and (C>alphas[i]): b=b1
                elif (0<alphas[j]) and (C>alphas[j]): b=b2
                else: b=(b1+b2)/2.0
                alphaPairsChanged +=1
                print("iter : %d i:%d, pairs changed %d" %(iter,i,alphaPairsChanged))
        if (alphaPairsChanged==0):iter+=1
        # 检査alpha值是否做了更新，如果有更新则将iter为0后继续运行程序
        else:iter=0
        print("iteration number: %d" %iter)
    return b,alphas

from Coding.ch6.Platt_SMO import calcWs

if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    # plotBestFit(dataArr,labelArr)
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    # print(b, alphas[alphas>0])
    # shape(alphas[alphas > 0])
    # for i in range(100):
    #     if alphas[i] > 0.0:
    #         print(dataArr[i], labelArr[i])
    ws = calcWs(alphas, dataArr, labelArr)
    print(b,ws)