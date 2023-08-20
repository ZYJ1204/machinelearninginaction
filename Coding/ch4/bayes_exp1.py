# -*- coding:utf-8 -*-
"""
@File    ：bayes_exp1.py
@Author  ：ZYJ
@Date    ：2023/7/22 15:51 
"""
from Coding.ch4.bayes import createVocabList, bagOfWords2VecMN, trainNB0, classifyNB
import numpy as np

def textParse(bigString):
    '''输入一个大字符串，将其解析为字符串列表'''
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 去掉少于两个字符的字符串，统一为小写

def spamTest():
    '''对垃圾邮件分类的测试函数'''
    docList = []; classList = []; fullText = []
    for i in range(1, 26):  # 常规邮件与垃圾邮件各25篇，分别读出转化字符串列表
        wordList = textParse(open('email/spam/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)  # 注意append与extend作用的区别
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # 创建词表
    trainingSet = range(50); testSet = []
    for i in range(10):  # 随机生成测试集、训练集的Index值
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))  # 构建训练集
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  # 调用函数对分类器进行训练
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])  #构建测试集
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:  # 判断分类结果
            errorCount += 1
            print("classification error", docList[docIndex])  # 输出分类错误的邮件
    print('the error rate is: ', float(errorCount)/len(testSet))  # 错误率

if __name__ == '__main__':
    emailText = open('email/ham/6.txt').read()
    listOfTokens = textParse(emailText)
    print(listOfTokens)
    spamTest()