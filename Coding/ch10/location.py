# -*- coding:utf-8 -*-
"""
@File    ：location.py
@Author  ：ZYJ
@Date    ：2023/8/2 16:34 
"""
from numpy import *

from Coding.ch10.kMeans import biKmeans, loadDataSet

"""
函数distSLC()返回地球表面两点间的距离，单位是英里。给定两个点的经纬度，可以使用
球面余弦定理来计算两点的距离。这里的纬度和经度用角度作为单位，但是sin()以及cos()以
弧度为输入。可以将角度除以180然后再乘以圆周率pi转换为弧度。导入NumPy的时候就会导
入pi
"""
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
"""
第二个函数clusterClubs()只有一个参数，即所希望得到的簇数目。该函数将文本文件的
解析、聚类以及画图都封装在一起，首先创建一个空列表，然后打开places.txt文件获取第4列和
第5列，这两列分别对应纬度和经度。基于这些经纬度对的列表创建一个矩阵。接下来在这些数
据点上运行biKmeans()并使用distSLC()函数作为聚类中使用的距离计算方法。最后将簇以及
簇质心画在图上
"""
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

if __name__ == '__main__':
    datMat = mat(loadDataSet('melon.txt'))
    clusterClubs(5)