# -*- coding:utf-8 -*-
"""
@File    ：kMeans.py
@Author  ：ZYJ
@Date    ：2023/8/1 16:09 
"""

from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

"""
1.选择聚类的个数，k。例如k=3
2.生成k个聚类中心点
3.计算所有样本点到聚类中心点的距离，根据远近聚类。
4.更新质心，迭代聚类。
5.重复第4步骤直到满足收敛要求。（通常就是确定的质心点不再改变）

创建 k 个点作为起始质心（随机选择）
当任意一个点的簇分配结果发生改变时（不改变时算法结束）
    对数据集中的每个数据点
        对每个质心
            计算质心与数据点之间的距离
        将数据点分配到距其最近的簇
    对每一个簇, 计算簇中所有点的均值并将均值作为质心
直到簇不再发生变化或者达到最大迭代次数
"""
#函数功能：计算两个数据集之间的欧式距离
#输入：两个数据集。
#返回：两个数据集之间的欧式距离（此处用距离平方和代替距离）
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)

#创建簇中心矩阵,初始化为k个在数据集的边界内随机分布的簇中心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        #求出数据集中第j列的最小值(即第j个特征)
        minJ = min(dataSet[:,j])
        #用第j个特征最大值减去最小值得出特征值范围
        rangeJ = float(max(dataSet[:,j]) - minJ)
        #创建簇矩阵的第J列,random.rand(k,1)表示产生(10,1)维的矩阵，其中每行值都为0-1中的随机值
        #可以这样理解,每个centroid矩阵每列的值都在数据集对应特征的范围内,那么k个簇中心自然也都在数据集范围内
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

#distMeas为距离计算函数
#createCent为初始化随机质心函数
"""
函数功能：k均值聚类算法
参数说明：
dataset：数据集
k：簇的个数
distMeas：距离计算函数
createCent：随机质心生成函数
返回：
centroids：质心
clusterAssment：所有数据划分结果
"""
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    #创建一个(m,2)维矩阵，第一列存储每个样本对应的质心，第二列存储样本到质心的距离
    clusterAssment = mat(zeros((m,2)))
    #用createCent()函数初始化质心矩阵
    centroids = createCent(dataSet, k)
    #保存迭代中clusterAssment是否更新的状态,如果未更新，那么退出迭代，表示收敛
    #如果更新，那么继续迭代，直到收敛
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        #程序中可以创建一个标志变量clusterChanged，如果该值为True，则继续迭代。
        #上述迭代使用while循环来实现。接下来遍历所有数据找到距离每个点最近的质心，
        #这可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
        #对每个样本找出离样本最近的质心
        for i in range(m):
            #minDist保存最小距离
            #minIndex保存最小距离对应的质心
            minDist = inf
            minIndex = -1
            #遍历质心，找出离i样本最近的质心
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            #如果clusterAssment更新，表示对应样本的质心发生变化，那么继续迭代
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            #更新clusterAssment,样本到质心的距离
            clusterAssment[i,:] = minIndex,minDist**2
        # print(centroids)
        #遍历质心,更新质心为对应簇中所有样本的均值
        for cent in range(k):
            #利用数组过滤找出质心对应的簇。获取某个簇类的所有点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            #对簇求均值,赋给对应的centroids质心
            centroids[cent,:] = mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment

import matplotlib.pyplot as plt

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    #取数据集特征均值作为初始簇中心
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    #centList保存簇中心数组,初始化为一个簇中心
    #create a list with one centroid
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    #迭代，直到簇中心集合长度达到k
    while (len(centList) < k):
        #初始化最小误差
        lowestSSE = inf
        #迭代簇中心集合，找出找出分簇后总误差最小的那个簇进行分解
        for i in range(len(centList)):
            #获取属于i簇的数据集样本
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            #对该簇进行k均值聚类
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            #获取该簇分类后的误差和
            sseSplit = sum(splitClustAss[:,1])
            #获取不属于该簇的样本集合的误差和，注意矩阵过滤中用的是!=i
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            #打印该簇分类后的误差和和不属于该簇的样本集合的误差和
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            #两误差和相加即为分簇后整个样本集合的误差和,找出簇中心集合中能让分簇后误差和最小的簇中心，保存最佳簇中心(bestCentToSplit),
            #最佳分簇中心集合(bestNewCents),以及分簇数据集中样本对应簇中心及距离集合(bestClustAss),最小误差(lowestSSE)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        #更新用K-means获取的簇中心集合，将簇中心换为len(centList)和bestCentToSplit,
        #以便之后调整clusterAssment(总样本集对应簇中心与和簇中心距离的矩阵)时一一对应
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        #更新簇中心集合,注意与bestClustAss矩阵是一一对应的
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment


if __name__ == '__main__':
    datMat = mat(loadDataSet('melon.txt'))
    myCentroids , clusterAssing = biKmeans(datMat,3)
    # 画图，画出聚类结果
    fig = plt.figure()
    # 将dataMat中的点画出来
    ax = fig.add_subplot(111)
    ax.set_xlim(0.1,0.9)
    ax.set_ylim(0,0.8)
    ax.scatter(datMat[:, 0].flatten().A[0], datMat[:, 1].flatten().A[0], s=50, c='blue') # flatten().A[0]降维成一维数组
    # 将聚类中心画出来
    ax.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], s=300, c='red', marker='+')
    plt.show()