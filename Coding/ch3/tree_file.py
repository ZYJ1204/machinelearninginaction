# -*- coding:utf-8 -*-
"""
@File    ：tree_file.py
@Author  ：ZYJ
@Date    ：2023/7/21 14:41 
"""
from Coding.ch3.tree import createTree, storeTree, grabTree, classify, createPlot

if __name__ == '__main__':
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]  # 训练数据
    lenseslabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 数据特征
    lensestree = createTree(lenses, lenseslabels)  # 构造树
    storeTree(lensestree, 'glass.txt')  # 存储树到文件
    mytree = grabTree('glass.txt')  # 再从文件中读取树
    print("训练出来的树是：")
    print(mytree)
    #####################开始测试
    mytest = ['presbyopic', 'hyper', 'no', 'normal']  # 用于测试
    print("测试案例:")
    print(mytest)
    lenseslabels = ['age', 'prescript', 'astigmatic', 'tearRate']  # 在createTree中leb1的值会被改
    myend = classify(mytree, lenseslabels, mytest)  # 使用决策树对mytest进行分类
    print("结果为：" + myend)
    createPlot(mytree)

