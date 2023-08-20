# -*- coding:utf-8 -*-
"""
@File    ：KNN.py
@Author  ：ZYJ
@Date    ：2023/7/20 18:45 
"""

from numpy import *
import operator
from os import listdir
import numpy as np

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                  #读取矩阵第一维度的长度
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #将原矩阵纵向地复制
    sqDiffMat = diffMat**2   #矩阵计算平方
    sqDistances = sqDiffMat.sum(axis=1)   #沿横轴求和 dataSetSize*1
    distances = sqDistances**0.5 #开方
    sortedDistIndicies = distances.argsort() #返回对数组进行排序的索引
    classCount={}   #定义一个字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  #索引对应标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #返回指定键的值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)#降序
    # sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # python3 已经移除iteritems，使用items替代
    return sortedClassCount[0][0]

    # x=np.array([[3,3],[2,2]])
    # print(np.argsort(x, axis=0))  [[1 1] [0 0]]


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #获取文件行数
    returnMat = zeros((numberOfLines,3))        #创建行数*3维的矩阵，初始化为0
    classLabelVector = []                       #标签
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()                     #strip函数会删除头和尾的字符，中间的不会删除
        listFromLine = line.split('\t')         #依照制表符分割
        returnMat[index,:] = listFromLine[0:3]  #得到前三列数据，即飞行时间，游戏，冰激凌
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):                          #归一化
    minVals = dataSet.min(0)                    #每一列的最小值 1*3
    maxVals = dataSet.max(0)                    #每一列的最大值 1*3
    ranges = maxVals - minVals                  #计算每一列的取值范围
    normDataSet = zeros(shape(dataSet))         #初始化矩阵
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))    #当前值减去最小值
    normDataSet = normDataSet/tile(ranges, (m,1))   #除以取值范围
    return normDataSet, ranges, minVals


def datingClassTest():
    hoRatio = 0.50  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 读取
    normMat, ranges, minVals = autoNorm(datingDataMat)               # 归一化
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)   #确定测试向量的数量
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  #加载训练集
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]       #从文件名解析分类数字
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)              #将数字加入标签
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # 加载测试集
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 从文件名解析分类数字
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))

if __name__ == "__main__":
    # group, labels = createDataSet()
    # print(classify0([1,0.4],group,labels,3))
    # datingClassTest()
    handwritingClassTest()

'''
    group = [[1.  1.1]
             [1.  1. ]
             [0.  0. ]
             [0.  0.1]]
    labels = ['A', 'A', 'B', 'B']
    dataSetSize = 4                 #矩阵维度
    inX = [0,0]                     #inX拓展至4行
    diffMat = [[-1.  -1.1]
               [-1.  -1. ]
               [ 0.   0. ]
               [ 0.  -0.1]]
    sqDiffMat = [[1.   1.21]        #平方
                 [1.   1.  ]
                 [0.   0.  ]
                 [0.   0.01]]
    sqDistances = [2.21 2.   0.   0.01]         #横向求和
    distances = [1.48660687 1.41421356 0.         0.1       ] #开方
    sortedDistIndicies = [2 3 1 0]              #排序  由小到大 对应数组下标
    voteIlabel = B                  #labels[2]
    classCount = {'B': 1}
    voteIlabel = B                  #labels[3]
    classCount = {'B': 2}
    voteIlabel = A                  #labels[1]
    classCount = {'B': 2, 'A': 1}
    sortedClassCount = [('B', 2), ('A', 1)]     #排序
    B       #结果
'''