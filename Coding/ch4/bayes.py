# -*- coding:utf-8 -*-
"""
@File    ：bayes.py
@Author  ：ZYJ
@Date    ：2023/7/21 15:00 
"""

import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 代表侮辱性文字，0代表正常言论
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空集，利用集合属性构建不重复词表
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # |表示集合的合并
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    #输入词表、文档，输出文档词向量
    returnVec = [0]*len(vocabList)  # 向量长度等于词表长度
    for word in inputSet:
        if word in vocabList:  # 判断词汇是否在词表中，0：否，1：是
            # 例如 [1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1]
            returnVec[vocabList.index(word)] = 1
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):  #朴素贝叶斯词袋模型
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix, trainCategory):  #朴素贝叶斯分类器训练函数
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0]) # 文档词向量长度
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # 计算class=1的概率 trainCategory=classVec
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)  # 记录出现次数的向量
    p0Denom = 2.0; p1Denom = 2.0  # np.ones()和2.0而不是np.zeros是为了避免某一概率为0，影响乘积也为0
    for i in range(numTrainDocs):  # 逐条访问训练文档矩阵
        if trainCategory[i] == 1:  # 根据class分别对词汇计数
            p1Num += trainMatrix[i]  #向量相加
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)  # 概率计算，其中log防止数值过小下溢
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 将词汇表 中所有词的对应值相加，然后将该值加到类别的对数概率上
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  # 计算不同类别的概率，返回较大概率作为预测
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts, listClasses = loadDataSet()  # 加载数据
    myVocabList = createVocabList(listOPosts)  # 创建词表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  # 训练分类器
    testEntry = ['love', 'my', 'dalmation']  # 测试数据1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']  # 测试数据2
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))




if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    returnVec = setOfWords2Vec(myVocabList,listOPosts[0])
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0v,p1v,pab=trainNB0(trainMat,listClasses)
    print(p0v,p1v,pab)
    testingNB()
    '''
    ['garbage', 'please', 'is', 'dalmation', 'problems', 'food', 'love', 'ate', 'help', 'him', 'stop', 'cute', 'take', 'worthless', 'mr', 'posting', 'licks', 'not', 'buying', 'park', 'dog', 'quit', 
     'flea', 'my', 'so', 'has', 'I', 'maybe', 'steak', 'stupid', 'how', 'to']
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    [[0 1 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 0 1 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 0 1 1 0 0 0 0 0 0 1 0 1 0 1]
     [0 0 1 1 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0]
     [0 0 0 0 0 0 0 1 0 1 1 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 1 0 1 1]
     [0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 1 0 0]]
    [-3.25809654 -2.56494936 -2.56494936 -2.56494936 -2.56494936 -3.25809654
     -2.56494936 -2.56494936 -2.56494936 -2.15948425 -2.56494936 -2.56494936
     -3.25809654 -3.25809654 -2.56494936 -3.25809654 -2.56494936 -3.25809654
     -3.25809654 -3.25809654 -2.56494936 -3.25809654 -2.56494936 -1.87180218
     -2.56494936 -2.56494936 -2.56494936 -3.25809654 -2.56494936 -3.25809654
     -2.56494936 -2.56494936] 
     [-2.35137526 -3.04452244 -3.04452244 -3.04452244 -3.04452244 -2.35137526
     -3.04452244 -3.04452244 -3.04452244 -2.35137526 -2.35137526 -3.04452244
     -2.35137526 -1.94591015 -3.04452244 -2.35137526 -3.04452244 -2.35137526
     -2.35137526 -2.35137526 -1.94591015 -2.35137526 -3.04452244 -3.04452244
     -3.04452244 -3.04452244 -3.04452244 -2.35137526 -3.04452244 -1.65822808
     -3.04452244 -2.35137526] 
     0.5
    '''


