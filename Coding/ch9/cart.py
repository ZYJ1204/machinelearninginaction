# -*- coding:utf-8 -*-
"""
@File    ：cart.py
@Author  ：ZYJ
@Date    ：2023/8/9 13:57 
"""
from numpy import *
import matplotlib.pyplot as plt
import matplotlib
import re
from Coding.ch9.jianzhi import creatPlot

def loadDataSet(fileName):  # 加载数据集
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # 加上list就是完整功能了，不然只用map会生成内存地址
        dataMat.append(fltLine)
    dataMat = array(dataMat)  # 转化为array数组，便于计算
    return dataMat


def binSplitDataSet(dataSet, feature, value):  # 把数据集根据变量(特征)feature按照切分点value分为两类
    mat0 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):  # 计算平均值，负责生成叶子节点，叶子节点根据公式不就是求所在子区域的平均值
    return mean(dataSet[:, -1])


def regErr(dataSet):  # 方差×个数 = 每个点与估计值的差的平方和，这是根据损失函数公式来的
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    tols = ops[0]  # 最小下降损失是1
    tolN = ops[1]  # 最小样本数是4
    if len(set(dataSet[:, -1])) == 1:  # 停止切分的条件之一
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf  # 最优切分点的误差
    bestIndex = 0  # 最优切分变量的下标
    bestValue = 0  # 最优切分点
    for featIndex in range(n - 1):  # 遍历除最后一列的其他特征，我们的数据集是（x,y），因此这里只能遍历x
        for splitVal in set(dataSet[:, featIndex]):  # splitVal是x可能去到的每一个值作为切分点
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)  # 根据切分点划分为两个子区域
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 如果两个子区域的叶子节点小于4个，跳过此切分点
                # print("内层判别小于4___________________")
                # print(shape(mat0)[0],shape(mat1)[0])
                continue
            newS = errType(mat0) + errType(mat1)  # 计算两部分的损失函数
            if newS < bestS:  # 如果损失函数小于最优损失函数
                bestIndex = featIndex  # 最优切分变量保存起来
                bestValue = splitVal  # 最优切分点保存起来
                bestS = newS  # 最优损失函数保存起来
    if (S - bestS) < tols:  # 如果切分之前的损失减切分之后的损失小于1,那么就是切分前后损失减小的太慢，停止.它是主要的停止条件
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)  # 按照最优切分点分成两个区域
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 如果任何一个子区域的样本数小于4个，停止切分
        # print("外层判断切分样本数小于4******************")       # 个人感觉此判断没用，因为上面选择切分点的时候就把这个情况删除了
        return None, leafType(dataSet)

    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)  # 最优切分点切分
    if feat == None:  # 如果是叶子节点，那么val是那个数据集的平均值，即c
        # print("NOne/执行了",val)
        return val
    # print("没执行",val)
    retTree = {}
    retTree['spInd'] = feat  # 最优切分变量(特征)的下标值
    retTree['spVal'] = val  # 最优切分点s，假如执行到这里，说明找到了最优切分点，将继续切分
    lSet, rSet = binSplitDataSet(dataSet, feat, val)  # 切分成两个子区域，lSet是大于切分点的，相当于我们图形的右边
    retTree['left'] = createTree(lSet, leafType, errType, ops)  # 左子树继续切分
    retTree['right'] = createTree(rSet, leafType, errType, ops)  # 右子树继续切分
    return retTree


def drawPicture(dataSet, x_division, y_estimate):  # x_division是切分点，y_estimate是估计值
    matplotlib.rcParams["font.sans-serif"] = ["simhei"]  # 显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()  # 创建画布
    ax = fig.add_subplot(111)
    points_x = dataSet[:, 0]  # 因为咱们的数据是(x,y)二维图形，x是切分变量，y是估计变量，可参照博客中的具体实例加以理解
    points_y = dataSet[:, 1]

    x_min = min(points_x)  # 创造切分区域，所以需要x的最小值和最大值，从而构造区域
    x_max = max(points_x)
    y_estimate.append(y_estimate[-1])

    ax.step([x_min] + list(x_division) + [x_max], y_estimate, where='post', c='green', linewidth=4,
            label="最优估计值")  # 画最优估计值
    ax.scatter(points_x, points_y, s=30, c='red', marker='s', label="样本点")  # 画样本点
    ax.legend(loc=4)  # 添加图例
    # ax.grid()               # 添加网格
    ax.set_yticks(y_estimate)  # 设置总坐标刻度
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


if __name__ == '__main__':
    myData = loadDataSet('cart-tree.txt')
    retTree = createTree(myData)  # 创建树
    retTree_str = str(retTree)  # 转化为字符串好用正则
    print(retTree_str)

    x_division_str = "'spVal': (\d+\.?\d*)"  # 正则提取出切分节点
    y_estimate_str = "'left': (-?\d+\.?\d*)|'right': (-?\d+\.?\d*)"  # 正则表达式取出来叶子节点数据，即最优估计值
    x_division = re.compile(x_division_str).findall(retTree_str)
    y_estimate = re.compile(y_estimate_str).findall(retTree_str)

    x_division = sort(list(map(float, x_division)))  # 切分点排序，因为我们画图都是从左往右画，正好对应下面的最优估计值c
    y_estimate = [float(x) for y in y_estimate for x in y if x != '']  # 估计值的顺序不能乱，树中的是什么顺序就是什么顺序

    print(x_division)
    print(y_estimate)
    drawPicture(myData, x_division, y_estimate)  # 画图，画出样本点和最优估计值
