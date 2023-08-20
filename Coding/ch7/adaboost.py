# -*- coding:utf-8 -*-
"""
@File    ：adaboost.py
@Author  ：ZYJ
@Date    ：2023/7/29 16:43 
"""

from numpy import *
import matplotlib.pyplot as plt

def loadSimpData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


"""测试是否有某个值小于或大于我们正在测试的阈值"""
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=ones((shape(dataMatrix)[0],1))     #构建一个全1向量
    if threshIneq=='lt':   #小于阈值的分类为gt，大于阈值的分类为lt
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0   #<=阈值的置为-1
    else:
        retArray[dataMatrix[:, dimen] >threshVal] = -1.0  #>阈值的置为-1
    return retArray

"""遍历stumpClassify的所有可能输入值，并找到数据集上最佳的单层决策树"""
def buildStump(dataArr,classLabels,D):
    dataMatrix=mat(dataArr)
    labelMat=mat(classLabels).T
    m,n=shape(dataMatrix)  #m个样本，n个特征
    numSteps=10.0   #用于在特征的所有可能值上进行遍历
    bestStump={}    #定义一个字典，用于存储给定权重向量D时所得到的的最佳单层决策树的相关信息
    bestClasEst=mat(zeros((m,1)))
    minError=inf    #最小错误率，初始化为无穷大
    for i in range(n):  #在所有特征上遍历（第一次循环），即按照第i个特征来划分类别
        rangeMin=dataMatrix[:,i].min()   #找出所有样本中在特征i上的最小值
        rangeMax=dataMatrix[:,i].max()   #找出所有样本在特征i上的最大值
        stepSize=(rangeMax-rangeMin)/numSteps  #确定步长（步长是用于比较特征值和阈值的）
        for j in range(-1,int(numSteps)+1):   #在特征i的所有可能取值上遍历，每次讲特征值加j*stepSize的大小（第二层循环）
            for inequal in ['lt','gt']:
                threshVal=(rangeMin+float(j)*stepSize)   #根据步长设定阈值
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)  #对样本进行预测分类
                errArr=mat(ones((m,1)))   #构建一个错误统计列向量
                errArr[predictedVals==labelMat]=0  #如果预测和真实标签一致置为0，不一致置为1
                weightedError=D.T*errArr    #根据错误向量*权重向量得到数值weightedError
                if weightedError<minError:  #更新最小错误率
                    minError=weightedError
                    bestClasEst=predictedVals.copy()  #当前最好的分类预测向量
                    bestStump['dim']=i   #当前最好的分类特征
                    bestStump['thresh']=threshVal   #当前的分类阈值
                    bestStump['ineq']=inequal   #分类结果
    return bestStump,minError,bestClasEst

"""基于单层决策树的AdaBoost训练过程"""
def adaBoostTrainDS(dataArr,classLabels,numIt=40):   #数据集，类别标签以及迭代次数（若小于迭代次数时错误率已经为0则直接退出）
    weakClassArr=[]  #弱分类器列表
    m=shape(dataArr)[0]  #m为样本数
    D=mat(ones((m,1))/m)  #初始化每个样本的权值，初始化所有权重均为1/m
    aggClassEst=mat(zeros((m,1)))   #向量aggClassEst，记录每个数据点的类别估计累计值
    for i in range(numIt):   #迭代
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)  #建立一个当前性能最优的单层决策树
        weakClassArr.append(bestStump)     #将当前选出的决策树信息存入列表
        print("各样本权值向量D为:{}".format(D.T))   #打印此时的各样本权重
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))    #计算alpha
        bestStump['alpha']=alpha   #将测试的alpha加入单层决策树信息中
        weakClassArr.append(bestStump)
        print("分类预测结果向量classEst:{}".format(classEst.T))
        #以下三步均为更新D的步骤
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)
        D=multiply(D,exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst
        print("aggClassEst:{}".format(aggClassEst.T))
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))  #通过aggClassEst来计算总的错误率
        errorRate=aggErrors.sum()/m
        print("总错误率:{}".format(errorRate))
        if errorRate==0.0:   #当总的错误率为0.0时退出迭代
            break
    return weakClassArr

"""基于AdaBoost的分类函数"""
def adaClassify(datToClass,classifierArr):   #datToClass为待分类样本，classifierArr是训练出来的弱分类器集合
    dataMatrix=mat(datToClass)
    m=shape(dataMatrix)[0]   #要分类的样本数
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):   #遍历所有的弱分类器，通过stumpClassify对每个分类器得到一个类别的估计值
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst  #然后得到各弱分类器的加权和，就是预测值
        print(aggClassEst)
    return sign(aggClassEst)

if __name__=='__main__':
    dataMatrix, classLabels = loadSimpData()
#     D=mat(ones((5,1))/5)
#     bestStump, minError, bestClasEst=buildStump(dataMatrix,classLabels,D)
#     print("当前最好的分类方法（dim:选取的特征，thresh：阈值，ineq:分类名称）:{}".format(bestStump))
#     print("最小错误率：{}".format(minError))
#     print("当前最好分类的预测值:{}".format(bestClasEst))
    classifierArr=adaBoostTrainDS(dataMatrix,classLabels,30)
    result=adaClassify([[0,0],[5,5]],classifierArr)