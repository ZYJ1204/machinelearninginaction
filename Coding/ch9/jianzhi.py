# -*- coding:utf-8 -*-
"""
@File    ：jianzhi.py
@Author  ：ZYJ
@Date    ：2023/8/9 14:30 
"""

from numpy import *
import matplotlib.pyplot as plt


def isTree(obj):  # 判断是否为叶子
    return (type(obj).__name__ == 'dict')


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
    global x_offset
    global y_offset

    leaf_num = getTreeLeafNum(tree)  # 叶子数
    global numLeafs
    numLeafs = 0
    depth = getTreeDepth(tree)  # 深度
    annotation = round(tree['spVal'], 2)
    print('tree=', tree)
    print('leaf_num=', leaf_num)

    annotation_location = (x_offset + (1.0 + float(leaf_num)) / 2.0 / treeWidth, y_offset)
    midText(ax, annotation_location, xy_location, midtext)
    plotNode(ax, annotation, xy_location, annotation_location)  # 画节点
    y_offset = y_offset - 1.0 / treeDepth
    key_list = ['left', 'right']
    for key in key_list:
        if type(tree[key]).__name__ == 'dict':
            print("x_off:{0}\ny_off:{1}".format(x_offset, y_offset))
            plotTree(ax, tree[key], annotation_location, treeWidth, treeDepth, str(key))
        else:
            x_offset = x_offset + 1.0 / treeWidth
            print("x_off:{0}\ny_off:{1}-----------".format(x_offset, y_offset))
            plotLeaf(ax, round(tree[key], 2), annotation_location, (x_offset, y_offset))
            midText(ax, (x_offset, y_offset), annotation_location, str(key))
    y_offset = y_offset + 1.0 / treeDepth


def creatPlot(tree):  # 创建画布
    fig = plt.figure()
    ax = fig.add_subplot(111, frameon=False)

    tree_width = getTreeLeafNum(tree)
    global numLeafs
    numLeafs = 0
    tree_depth = getTreeDepth(tree)
    print("宽:", tree_width)
    print("深：", tree_depth)
    global x_offset
    x_offset = -0.5 / tree_width
    global y_offset
    y_offset = 1.0

    plotTree(ax, tree, (0.5, 1.0), tree_width, tree_depth, "")

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


if __name__ == "__main__":
    global numLeafs
    numLeafs = 0

    global x_offset
    x_offset = 0
    global y_offset
    y_offset = 0

    tree = {'spInd': 0, 'spVal': 0.499171,
            'left': {'spInd': 0, 'spVal': 0.339397, 'left': -2.637719329787234, 'right': -0.05},
            'right': {'spInd': 0, 'spVal': 0.729397, 'left': 107.68699163829788,
                      'right': {'spInd': 0, 'spVal': 0.952833,
                                'left': {'spInd': 0, 'spVal': 0.759504, 'left': 78.08564325, 'right': 95.7366680212766},
                                'right': 108.838789625}}}
    print('tree=', tree)
    creatPlot(tree)