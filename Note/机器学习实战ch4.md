---
title: 朴素贝叶斯
tags:
  - 机器学习
cover: https://pic.imgdb.cn/item/649ce61e1ddac507ccb10aca.png
toc: True
date: 2023/7/22
---

# 朴素贝叶斯

### 本章内容

- 使用概率分布进行分类
- 学习朴素贝叶斯分类器
- 解析RSS源数据
- 使用朴素贝叶斯来分析不同地区的态度

---

贝叶斯定理由英国数学家贝叶斯发展，用来描述两个条件概率之间的关系，比如 有P(A|B) 和 P(B|A)，则有公式为：$P(A∩B) = P(A)*P(B|A)=P(B)*P(A|B)$，可变形为：$P(A|B)=P(B|A)*P(A)/P(B)$。比如你发现天空乌云密布，刮起大风，那可能是要下雨了。这就是说，当你不能准确知悉一个事物的本质时，你可以依靠与事物特定本质相关的事件出现的多少去判断其本质属性的概率。 用数学语言表达就是：支持某项属性的事件发生得愈多，则该属性成立的可能性就愈大。

朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法。所谓独立假设也就是说属性变量对于决策结果占有比重相等，且属性之间相互独立。虽然这个简化方式在一定程度上降低了贝叶斯分类算法的分类效果，但是在实际的应用场景中，极大地简化了贝叶斯方法的复杂性。

朴素贝叶斯的一般过程

1. 收集数据：可以使用任何方法。
2. 准备数据：需要数值型或者布尔型数据。
3. 分析数据：有大量特征时，绘制特征作用不大，此时使用直方图效果更好。
4. 训练算法：计算不同的独立特征的条件概率。
5. 测试算法：计算错误率。
6. 使用算法：一个常见的朴素贝叶斯应用是文档分类。可以在任意的分类场景中使用朴素贝叶斯分类器，不一定非要是文本。

---

### 使用Python进行文本分类

从文本中构建词向量

```python
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

if __name__ == '__main__':
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    returnVec = setOfWords2Vec(myVocabList,listOPosts[0])
```

运行结果：

```python
 postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
              ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
              ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
              ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
              ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
              ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
myVocabList = ['ate', 'licks', 'cute', 'take', 'park', 'problems', 'is', 'my', 'dog', 'please', 'not', 'worthless', 'steak', 'food', 'dalmation', 'buying', 'flea', 'mr', 'how', 'stop', 'help', 'has', 'love', 'garbage', 'posting', 'stupid', 'him', 'so', 'to', 'maybe', 'I', 'quit']  #32
listOPosts[0] = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
returnVec = [0 0 0 0 0 1 0 1 1 1 0 0 0 0 0 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0]
```

从词向量计算概率

```python
'''
伪代码如下：
计算每个类别中的文档数目
对每篇训练文档：
	对每个类别：
		如果词条出现在文档中→增加该词条的计数值
		增加所有词条的计数值
	对每个类别：
		对每个词条：
			将该词条的数目除以总词条数目得到条件概率
	返回每个类别的条件概率
'''
```

朴素贝叶斯分类器训练函数：

```python
def trainNB0(trainMatrix, trainCategory):  #朴素贝叶斯分类器训练函数
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0]) # 文档词向量长度
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # 计算class=1的概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)  # 记录出现次数的向量
     # np.ones()和2.0而不是np.zeros是为了避免某一概率为0，影响乘积也为0
    p0Denom = 2.0; p1Denom = 2.0  
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

#对应class = 0
trainMatrix[0] = [0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0 1]
		 p1Num = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
         p1Num = [1 1 2 1 1 1 1 1 1 1 1 2 1 2 1 2 1 1 1 2 1 1 1 1 1 2 1 1 1 1 1 2] 
p0Vect = [-3.25809654 -2.56494936 -2.56494936 -2.56494936 -2.56494936 -3.25809654
          -2.56494936 -2.56494936 -2.56494936 -2.15948425 -2.56494936 -2.56494936
          -3.25809654 -3.25809654 -2.56494936 -3.25809654 -2.56494936 -3.25809654
          -3.25809654 -3.25809654 -2.56494936 -3.25809654 -2.56494936 -1.87180218
          -2.56494936 -2.56494936 -2.56494936 -3.25809654 -2.56494936 -3.25809654
          -2.56494936 -2.56494936] 
p1Vect = [-2.35137526 -3.04452244 -3.04452244 -3.04452244 -3.04452244 -2.35137526
          -3.04452244 -3.04452244 -3.04452244 -2.35137526 -2.35137526 -3.04452244
          -2.35137526 -1.94591015 -3.04452244 -2.35137526 -3.04452244 -2.35137526
          -2.35137526 -2.35137526 -1.94591015 -2.35137526 -3.04452244 -3.04452244
          -3.04452244 -3.04452244 -3.04452244 -2.35137526 -3.04452244 -1.65822808
          -3.04452244 -2.35137526] 
pAbusive = 0.5

```

算法中采用log的原因：防止数值过小下溢

```python
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

t = arange(0.0, 0.5, 0.01)
s = sin(2*pi*t)
logS = log(s)

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(t,s)
ax.set_ylabel('f(x)')
ax.set_xlabel('x')

ax = fig.add_subplot(212)
ax.plot(t,logS)
ax.set_ylabel('ln(f(x))')
ax.set_xlabel('x')
plt.show()
```

<img src="https://pic.imgdb.cn/item/64bbb17f1ddac507cce02277.jpg" style="zoom:67%;" />

朴素贝叶斯分类函数

```python
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 将词汇表 中所有词的对应值相加，然后将该值加到类别的对数概率上
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)  
    # 计算不同类别的概率，返回较大概率作为预测 向量相乘
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
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))  # 创建训练矩阵
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  # 训练分类器
    testEntry = ['love', 'my', 'dalmation']  # 测试数据1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']  # 测试数据2
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
```

测试结果：

```python
testEntry = ['love', 'my', 'dalmation'] 
vec2Classify = [1 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] p1Vec = [-3.04452244 -3.04452244 -2.35137526 -2.35137526 -3.04452244 -3.04452244
         -1.94591015 -3.04452244 -2.35137526 -2.35137526 -3.04452244 -2.35137526
         -3.04452244 -2.35137526 -3.04452244 -1.94591015 -2.35137526 -2.35137526
         -2.35137526 -2.35137526 -1.65822808 -3.04452244 -2.35137526 -3.04452244
         -3.04452244 -3.04452244 -3.04452244 -3.04452244 -3.04452244 -3.04452244
         -3.04452244 -2.35137526]
p1 = -9.826714493730215
p0Vec = [-1.87180218 -2.56494936 -3.25809654 -2.56494936 -2.56494936 -2.56494936
         -2.56494936 -2.56494936 -2.15948425 -2.56494936 -2.56494936 -3.25809654
         -2.56494936 -3.25809654 -2.56494936 -3.25809654 -3.25809654 -3.25809654
         -3.25809654 -3.25809654 -3.25809654 -2.56494936 -3.25809654 -2.56494936
         -2.56494936 -2.56494936 -2.56494936 -2.56494936 -2.56494936 -2.56494936
         -2.56494936 -3.25809654]
p0 = -7.694848072384611
classified as:  0
```

---

### 使用朴素贝叶斯过滤垃圾邮件

```python
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
    # emailText = open('email/ham/6.txt').read()
    # listOfTokens = textParse(emailText)
    spamTest()
    # the error rate is:  0.0
```

---

### 使用朴素贝叶斯分类器从个人广告中获取区域倾向（RSS无效）

```python
import feedparser
from Coding.ch4.bayes import createVocabList, trainNB0, classifyNB, bagOfWords2VecMN
from Coding.ch4.bayes_exp1 import textParse
import numpy as np


def calcMostFreq(vocabList, fullText):
    '''计算高频词'''
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    '''RSS源分类器'''
    import feedparser
    docList = []; classList = []; fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in list(range(minLen)):
        #每次访问一条RSS源
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    #去掉出现次数最高的那些词
    top30Words = calcMostFreq(docList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen)); testSet = []
    for i in list(range(20)):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is :", float(errorCount) / len(testSet))
    return vocabList, p0V, p1V

#最具表征性词汇显示函数
def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])

if __name__ == '__main__':
    # localWords()
    ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
    vocabList, pSF, pNY = localWords(ny, sf)
```
