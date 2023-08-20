# -*- coding:utf-8 -*-
"""
@File    ：tree.py
@Author  ：ZYJ
@Date    ：2023/7/21 12:07 
"""

from math import log
import operator
import matplotlib.pyplot as plt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # 为所有可能的分类创建字典
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  #计算熵
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:   #依照特征进行分割
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    # 求第一行有多少列的Feature
    numFeatures = len(dataSet[0]) - 1
    # 计算没有经过划分的数据的香农熵
    baseEntroy = calcShannonEnt(dataSet)
    # 最优的信息增益，最优的Feature编号
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        ## 创建唯一的分类标签列表，获取第i个的所有特征(信息元素纵列)

        # 将dataSet 中的数据先按行依次放入example中，然后取得example[i]元素，放入列表featList
        featList = [example[i] for example in dataSet]
        # 使用set集，排除featList中重复的标签，得到唯一分类的集合
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 遍历档次uniqueVals中所有的标签value
        for value in uniqueVals:
            # 对第i个数据划分数据集，返回所有包含i的数据（去掉第i个特征)
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算包含i的数据占总数据的百分比
            prob = len(subDataSet) / float(len(dataSet))
            # 计算新的香农熵，不断进行迭代，这个计算过程仅在包含指定特征标签子集中进行
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算信息增益
        infoGain = baseEntroy - newEntropy
        if (infoGain > bestInfoGain):
            # 更新信息增益
            bestInfoGain = infoGain
            # 确定最优的增益的特征索引
            bestFeature = i
    # 返回最优增益的索引
    return bestFeature

#遍历完所有的特征时，仍然不能将数据集划分成仅包含唯一类别的分组，采用多数表决法
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):  #创建树
    # 返回当前数据集下标签列所有值
    classList = [example[-1] for example in dataSet]
    # 当类别完全相同时则停止继续划分，直接返回该类的标签（决策树构造完成)
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时，仍然不能将数据集划分成仅包含唯一类别的分组，返回出现次数最多的类别标签作为返回值
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 获取最好分类特征索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取该特征的名字
    bestFeatLabel = labels[bestFeat]
    print(bestFeatLabel)
    # 这里直接使用字典变量来存储树信息，用于绘制树形图
    myTree = {bestFeatLabel: {}}
    # 删除已经在选取的特征
    del (labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 复制所有的标签
        # 递归调用自身
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# 决策点的属性,boxstyle是文本框类型，sawtooth是锯齿形，fc是边框粗细
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
# 决策树叶子节点的属性
leafNode = dict(boxstyle="round4", fc="0.8")
# 箭头的属性
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)
    # nodeTxt为要显示的文本，centerPt为文本的中心点，parentPt为箭头指向文本的点，xy是箭头尖的坐标，xytext设置注释内容显示的中心位置
    # xycoords和textcoords是坐标xy 与 xytext 的说明（按轴坐标), 若textcoords=None, 则默认textcoords与xycoords相同，若都未设置，则默认为data
    # va/ha设置节点框中文字的位置，va为纵向取值为(u'top',u'bottom',u'center',u'baseline'),ha为纵向取值为(u'center',u'right',u'left')


# def createPlot():
#     # 创建一个画布，背景为白色
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()  # 画布清空
#     # ax1是函数createPlot的一个属性，这个可以在函数里面定义也可以在函数定义后加入也可以
#     createPlot.ax1 = plt.subplot(111, frameon=True) # frameon表示是否绘制坐标轴矩形
#     plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode('a leaf Node', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()


# createPlot()

def getNumLeafs(myTree):
    # 初始化节点数
    numLeafs = 0
    # python3替换注释的两行代码
    firstside = list(myTree.keys())
    firstStr = firstside[0]  # 找到输入的第一个元素，第一个关键词划分数据类别的标签
    secondDict = myTree[firstStr]
    # firstStr = myTree.keys()[]
    # secondDict = myTree[firstStr]
    for key in secondDict.keys():  # 测试数据是否为字典形式
        # type判断子节点是否为字典类型
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
            # 若子节点也为字典，则也是判断节点，需要递归获取num
        else:
            numLeafs += 1
    # 返回整棵树的节点数
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstside = list(myTree.keys())
    firstStr = firstside[0]
    secondDict = myTree[firstStr]
    # firstStr = myTree.keys()[0]
    # secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


# 输出预先存储的树信息，避免每次测试都需要重新创建树
def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    # 计算树的宽度 totalW
    numLeafs = getNumLeafs(myTree)
    # 计算树的高度 totalD
    depth = getTreeDepth(myTree)
    firstside = list(myTree.keys())
    firstStr = firstside[0]  # 找到输入的第一个元素
    # 按照叶子节点个数划分x轴
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    # 标注节点属性
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    # y方向上的摆放位置自下而上绘制，因此递减y值
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # 判断是否为字典 不是则为叶子节点
        if type(secondDict[key]).__name__ == 'dict':
            # 递归继续向下找
            plotTree(secondDict[key], cntrPt, str(key))
        else:  # 为叶子节点
            # x方向计算节点坐标
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            # 绘制
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            # 添加文本信息
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    # 下次重新调用时回恢复y
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 主函数
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def classify(inputTree, featLabels, testVec):  # 使用决策树inputTree对testVec进行分类，featLabels为testVec的特征集
    # firstStr = inputTree.keys()[0]
    firstSides = list(inputTree.keys())  # 先转换成list，再把需要的索引提取出来
    firstStr = firstSides[0]
    secondDict = inputTree[firstStr]  # 键（根）的值（所有支树）
    featIndex = featLabels.index(firstStr)  # 找到根特征在testVec的位置
    key = testVec[featIndex]  # testVec在根特征下的值
    valueOfFeat = secondDict[key]  # 树的特征值（也是键）的值
    if isinstance(valueOfFeat, dict):  # 如果存在支树
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):  # 存储树到文件
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):  # 从文件中读取树
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    dataSet, labels=createDataSet()
    # print(calcShannonEnt(dataSet))
    # print(chooseBestFeatureToSplit(dataSet))
    myTree = retrieveTree(0)
    # myTree['no surfacing'][3]='maybe'
    # createPlot(myTree)
    # print(classify(myTree,labels,[1,1]))
    storeTree(myTree,'classifierStorage.txt')
    print(grabTree('classifierStorage.txt'))