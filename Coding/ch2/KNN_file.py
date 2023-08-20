# -*- coding:utf-8 -*-
"""
@File    ：KNN_file.py
@Author  ：ZYJ
@Date    ：2023/7/20 21:33 
"""

from numpy import *
from Coding.ch2.KNN import file2matrix,autoNorm
import matplotlib
import matplotlib.pyplot as plt


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    ax.axis([-2, 25, -0.2, 2.0])
    plt.xlabel('Percentage of Time Spent Playing Video Games')
    plt.ylabel('Liters of Ice Cream Consumed Per Week')
    plt.show()
    print(autoNorm(datingDataMat))