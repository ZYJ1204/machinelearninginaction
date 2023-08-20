# -*- coding:utf-8 -*-
"""
@File    ：cart_all.py
@Author  ：ZYJ
@Date    ：2023/8/9 14:34 
"""

from numpy import *
import matplotlib.pyplot as plt
import matplotlib
import re
import copy

matplotlib.rcParams["font.sans-serif"] = ["simhei"]  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False

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
    mat0 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]  # 大于成为左子树，实际图形却在右边,为了画图方便我把它改了
    mat1 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):  # 计算平均值，负责生成叶子节点，叶子节点根据公式不就是求所在子区域的平均值
    return mean(dataSet[:, -1])


def regErr(dataSet):  # 方差×个数 = 每个点与估计值的差的平方和，这是根据损失函数公式来的
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 2)):
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


# ------------------------------------ 剪枝处理 ----------------------------------------

def isTree(obj):
    return (type(obj).__name__ == 'dict')  # 判断是叶子还是树


def getMean(tree):
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):  # 剪枝
    if shape(testData)[0] == 0:
        print("判断测试集为空,执行过吗？")
        return getMean(tree)
    if (isTree(tree['left']) or isTree(tree['right'])):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def reExtract(retTree_str):  # 正则提取
    x_division_str = "'spVal': (\d+\.?\d*)"  # 正则提取出切分节点
    y_estimate_str = "'left': (-?\d+\.?\d*)|'right': (-?\d+\.?\d*)"  # 正则表达式取出来叶子节点数据，即最优估计值
    x_division = re.compile(x_division_str).findall(retTree_str)
    y_estimate = re.compile(y_estimate_str).findall(retTree_str)

    x_division = sort(list(map(float, x_division)))  # 切分点排序，因为我们画图都是从左往右画，正好对应下面的最优估计值c
    y_estimate = [float(x) for y in y_estimate for x in y if x != '']  # 估计值的顺序不能乱，树中的是什么顺序就是什么顺序
    return x_division, y_estimate


def drawPicture(dataSet, x_division, y_estimate, title_name):  # x_division是切分点，y_estimate是估计值
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
    ax.set_title(title_name)

    plt.show()


# ------------------------------------------画树状图---------------------------------------

def getTreeLeafNum(tree):  # 得到叶子数
    global numLeafs
    key_list = ['left', 'right']
    for key in key_list:
        if isTree(tree[key]):
            getTreeLeafNum(tree[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(tree):  # 得到树的深度
    max_depth = 0
    key_list = ['left', 'right']
    for key in key_list:
        if isTree(tree[key]):
            depth = 1 + getTreeDepth(tree[key])
        else:
            depth = 1
        if depth > max_depth:
            max_depth = depth
    return max_depth


def plotNode(ax, str, xy_point, xytext_points):  # 画节点
    ax.annotate(str, xy=xy_point, xytext=xytext_points, va="center", ha="center", arrowprops=dict(arrowstyle="<-"),
                bbox=dict(boxstyle="square", color="red", alpha=0.3))


def plotLeaf(ax, str, xy_point, xytext_points):  # 画叶子
    ax.annotate(str, xy=xy_point, xytext=xytext_points, va="center", ha="center", arrowprops=dict(arrowstyle="<-"),
                bbox=dict(boxstyle="round", color="green", alpha=0.6))


def midText(ax, annotation_location, xy_location, content_str):  # 画中间注释
    x = (xy_location[0] - annotation_location[0]) / 2 + annotation_location[0]
    y = (xy_location[1] - annotation_location[1]) / 2 + annotation_location[1]
    ax.text(x, y, content_str)


def plotTree(ax, tree, xy_location, treeWidth, treeDepth, midtext):  # 递归画树
    global x_offset  # x的偏移全局变量，举个例子如果叶子共3个，那么从左到右，第一个叶子x坐标就是1/6,第二个叶子x坐标是3/6,第三个是5/6
    global y_offset  # 画一次这个总坐标就降1/总深度

    leaf_num = getTreeLeafNum(tree)  # 叶子数
    global numLeafs
    numLeafs = 0
    depth = getTreeDepth(tree)  # 深度
    annotation = round(tree['spVal'], 2)
    annotation_location = (x_offset + (1.0 + float(leaf_num)) / 2.0 / treeWidth, y_offset)  # 它是节点的注释位置，却是叶子的箭头位置
    # midText(ax,annotation_location,xy_location,midtext)
    plotNode(ax, annotation, xy_location, annotation_location)  # 画节点
    y_offset = y_offset - 1.0 / treeDepth
    key_list = ['left', 'right']
    for key in key_list:
        if type(tree[key]).__name__ == 'dict':
            # print("x_off:{0}\ny_off:{1}".format(x_offset,y_offset))
            plotTree(ax, tree[key], annotation_location, treeWidth, treeDepth, str(key))  # 递归
        else:
            x_offset = x_offset + 1.0 / treeWidth  # 画一个叶子x_offset往右移动1/叶子总数
            # print("x_off:{0}\ny_off:{1}-----------".format(x_offset,y_offset))
            plotLeaf(ax, round(tree[key], 2), annotation_location, (x_offset, y_offset))  # 画叶子
            # midText(ax,(x_offset,y_offset),annotation_location,str(key))
    y_offset = y_offset + 1.0 / treeDepth  # 递归完一次，总坐标y_offset需要增加一个1/总深度，即这时s形画，再回去


def createPlot(tree, title_name):  # 画决策树
    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False)  # 边框去掉

    tree_width = getTreeLeafNum(tree)  # 树的叶子数
    global numLeafs
    numLeafs = 0
    tree_depth = getTreeDepth(tree)  # 树的深度
    global x_offset
    x_offset = -0.5 / tree_width  # 起始x偏移为-1/(2*叶子数)
    global y_offset
    y_offset = 1.0  # 起始y偏移为1

    plotTree(ax, tree, (0.5, 1.0), tree_width, tree_depth, "")

    ax.set_xticks([])
    ax.set_yticks([])  # 坐标刻度清除
    ax.set_title(title_name)
    plt.show()


if __name__ == '__main__':
    global numLeafs  # 定义全局变量，便于计算一棵树的总叶子数
    numLeafs = 0

    global x_offset  # 用于画注解树
    x_offset = 0
    global y_offset
    y_offset = 0
    # dataset = loadDataSet('/home/zhangqingfeng/test/cart_tree/cart_tree.txt')  # 加载训练集
    myData = loadDataSet('train.txt')  # 加载测试集
    mytestData = loadDataSet('test.txt')
    # first_tree = createTree(dataset,ops=(1,4))
    # createPlot(first_tree,"")
    InialTree = createTree(myData, ops=(100, 4))  # 创建训练集的树
    print("裁剪前的回归树为：", InialTree)
    createPlot(InialTree, "剪之前的决策树")  # 画出剪之前的决策树
    InialTree_str = str(InialTree)
    x_d, y_e = reExtract(InialTree_str)
    drawPicture(mytestData, x_d, y_e, "剪之前的拟合")  # 画出剪枝之前的拟合阶梯函数

    prune_tree = prune(InialTree, mytestData)  # 通过测试集对树进行剪枝

    createPlot(prune_tree, "剪之后的决策树")  # 画出剪枝后的决策树
    retTree_str = str(prune_tree)  # 转化为字符串好用正则
    print("裁剪后的回归树为: ", retTree_str)
    x_division, y_estimate = reExtract(retTree_str)
    drawPicture(mytestData, x_division, y_estimate, "剪之后的拟合")  # 画出剪枝后的拟合阶梯函数