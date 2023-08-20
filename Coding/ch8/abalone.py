# -*- coding:utf-8 -*-
"""
@File    ：abalone.py
@Author  ：ZYJ
@Date    ：2023/8/9 12:19 
"""
from Coding.ch8.regression import loadDataSet, lwlrTest, standRegres
from numpy import *

# 定义计算误差的函数
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

if __name__ == '__main__':
    abX, abY = loadDataSet('abalone.txt')
    # yhat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    # yhat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    # yhat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)

    # # 分别打印不同核的预测误差
    # print(rssError(abY[0:99], yhat01.T))  # 56.78420911837208
    # print(rssError(abY[0:99], yhat1.T))  # 429.89056187030394
    # print(rssError(abY[0:99], yhat10.T))  # 549.1181708826065
    yhat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yhat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yhat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)

    print(rssError(abY[100:199], yhat01.T))
    # K为0.1时的误差：25119.459111157415
    print(rssError(abY[100:199], yhat1.T))
    # K为1时的误差：573.5261441895706
    print(rssError(abY[100:199], yhat10.T))
    # K为1时的误差：573.5261441895706

    ws = standRegres(abX[0:99], abY[0:99])
    yhat = mat(abX[100:199]) * ws
    print(rssError(abY[100:199], yhat.T.A))  # 518.6363153249081

