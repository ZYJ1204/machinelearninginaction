---
title: 支持向量机
tags:
  - 机器学习
cover: https://pic.imgdb.cn/item/649ce61e1ddac507ccb10aca.png
toc: True
date: 2023/7/24
---

# 支持向量机

### 本章内容

- 简单介绍支持向量机

- 利用SMO进行优化

- 利用核函数对数据进行空间转换

- 将SVM和其他分类器进行对比


---

**1.基于最大间隔分隔数据**

- 优点：泛化错误率低，计算开销不大，结果易解释。  

- 缺点：对参数调节和核函数的选择敏感，原始分类器不加修饰仅适用于处理二类问题。

- 适用数据类型：数值型数据与标称型数据。  

​		支持向量就是离分割超平面最近的那些点，然后找到最大化支持向量到分割面的距离，需要找到此问题的优化求解方法

**2.寻找最大间隔**

**2.1 分类器求解的优化问题**
		使用海维赛德阶跃函数（赫维赛德函数）来进行分类的计算。

<img src="https://pic.imgdb.cn/item/64be30611ddac507cc10ed24.jpg" style="zoom: 80%;" />

​		为了找到分割最好的那一条线，我们在这里选择得到最大化支持向量的最大化间隔。

**3.SMO高效优化算法**

1. 最小化的目标函数

2. 在优化过程中必须遵循约束条件

**3.1 Platt的SMO算法**
		SMO表示序列最小化。PLatt的SMO算法是将大优化问题分解为多个小优化问题来求解。这些小优化问题往往很容易求解，并且对他们进行顺序求解的结果与将他们整体来求解的结果是完全一致的。在结果完全相同的条件下，SMO算法的求解时间就会短很多。  

​		SMO算法的目标是求出一系列alpha和b，一旦求出了这些alpha，就很容易计算出权重向量w，并得到分割超平面。  

​		**SMO算法的工作原理为：每次循环中选择两个alpha进行优化处理。一旦找到一对合适的alpha，那么就增大其中一个同时减小另一个。这里所谓的合适就是指两个alpha必须要符合一定的条件，条件之一就是这两个alpha必须要在间隔边界之外，而这第二个条件则是这两个alpha还没有进行区间化处理或者不在边界上。**

```python
# 伪代码
创建一个alpha向量并将其初始化为0向量
当迭代次数小于最大迭代次数时（外循环）
	对数据集中的每个数据向量（内循环）：
		如果该数据向量可以被优化：
			随机选择另外一个数据向量
			同时优化这两个向量
			如果两个向量都不能被优化，退出内循环
		如果所有向量都没被优化，增加迭代数目，继续下一次循环
```

**SMO算法中的辅助函数**

```python
# 打开文件并对其进行逐行分析，从而得到每行的类标签和整个数据矩阵
def loadDataSet(fileName):
    dataMat=[]
    labelMat=[]
    fr=open(fileName) # 读取文件
    for line in fr.readlines():
        # 将每一行的数据进行分割
        lineArr=line.strip().split('\t')
        # 前两列为x，y值
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        # 第三列为标签值
        labelMat.append(float(lineArr[2]))
    # 返回数据集和标签集
    return dataMat,labelMat

if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    plotBestFit(dataArr,labelArr)
```

绘制数据点：

<img src="https://pic.imgdb.cn/item/64be34761ddac507cc1f1df3.jpg" style="zoom:67%;" />

 简化版SMO的代码如下：

```python
#  i输入下标，m为总数
def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j

# 用于调整大于H或是小于L的alpha的值
def clipAlpha(aj, H, L):
    # 大于H的值取H
    if aj > H:
        aj = H
    # 小于L的值取L
    if L > aj:
        aj = L
    return aj

# 简化版的smo算法
# 构建函数时，采用通用的接口，这样就可以对算法和数据源进行组合和配对处理
# 输入参数：数据集，类别标签，常数C，容错率，取消最大的循环次数
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix=mat(dataMatIn)   #输入数据集变为矩阵
    labelMat=mat(classLabels).transpose()
    # 把输入标签变为矩阵并且转置，得到一个列向量，标签的每行对应每行的数据
    b=0 # 初始化
    m,n=shape(dataMatrix)# 得到矩阵的行列数
    alphas=mat(zeros((m,1)))# 定义一个m*1的零矩阵
    iter=0  # 初始化迭代次数
    #只有在所有数据集上遍历maxIter次，且不再发生任何alpha修改之后，程序才会停止并退出while循环
    while (iter<maxIter):   # 迭代次数大于最大迭代次数时，退出迭代
        alphaPairsChanged=0 # 用来记录alpha是否已经进行优化
        for i in range(m):  # 循环每个数据
            # 步骤1：计算误差Ei
            fXi=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            # 预测第i个数据的类别  multiply对应的位置相乘
            Ei=fXi-float(labelMat[i])
            # 实际结果与真实结果的误差，如果误差很大，那么就要对该数据实例所对应的alpha值进行优化
            # 一旦alphas等于0或C，那么它们就巳经在“边界”上了，因而不再能够减小或增大，因此也就不值得再对它们进行优化了
            if ((labelMat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelMat[i]*Ei>toler)and (alphas[i]>0)):#将违反KKT条件的找出来，具体公式来源见https://blog.csdn.net/youhuakongzhi/article/details/86660281
                j=selectJrand(i,m)  # 在m个中随机选择一个和i不同的j
                fXj=float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                # 预测第j个数据的类别
                Ej=fXj-float(labelMat[j])   # 计算第j个的误差
                alphaIold=alphas[i].copy()  # 保存更新前的aplpha值，使用深拷贝
                alphaJold=alphas[j].copy()
                if (labelMat[i]!=labelMat[j]):      # 如果yi和yj的标签不一样
                    L=max(0,alphas[j]-alphas[i])    # alphas[j]new的取值范围
                    H=min(C,C+alphas[j]-alphas[i])
                else:   # 如果yi和yj的标签一样
                    L=max(0,alphas[j]+alphas[i]-C)  # alphas[j]new的取值范围
                    H=min(C,alphas[j]+alphas[i])
                if L==H:print("L==H");continue  # 退出本次循环，直接进行下一次for循环
                # 步骤3：计算eta=-2*Kij+Kii+Kjj，而这儿eta=2*Kij-Kii-Kjj,所以下面公式中用的是减号
                eta=2.0*dataMatrix[i,:]*dataMatrix[j,:].T-dataMatrix[i,:]*dataMatrix[i,:].T-dataMatrix[j,:]*dataMatrix[j,:].T
                if eta>=0: print("eta>=0"); continue    # 退出本次循环，直接进行下一次for循环
                # 目标函数是求最小值，这儿eta是负的二阶导数，因此二阶导数不能小于等于0
                alphas[j]-=labelMat[j]*(Ei-Ej)/eta  # 更新alphas[j]
                alphas[j]=clipAlpha(alphas[j],H,L)  # 用于调整aj值，让aj在H和L的范围内
                if (abs(alphas[j]-alphaJold)<0.00001):  # alphas[j]变化太小
                    print("j not moving enough"); continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJold-alphas[j])    # 更新alphas[i]
                # 更新b1
                b1=b-Ei-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*\
                dataMatrix[i,:]*dataMatrix[j,:].T
                # 更新b2
                b2=b-Ej-labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*\
                dataMatrix[j,:]*dataMatrix[j,:].T
                # 更新b
                if (0<alphas[i]) and (C>alphas[i]): b=b1
                elif (0<alphas[j]) and (C>alphas[j]): b=b2
                else: b=(b1+b2)/2.0
                alphaPairsChanged +=1
                print("iter : %d i:%d, pairs changed %d" %(iter,i,alphaPairsChanged))
        if (alphaPairsChanged==0):iter+=1
        # 检査alpha值是否做了更新，如果有更新则将iter为0后继续运行程序
        else:iter=0
        print("iteration number: %d" %iter)
    return b,alphas

if __name__ == '__main__':
    dataArr, labelArr = loadDataSet('testSet.txt')
    # plotBestFit(dataArr,labelArr)
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print(b, alphas[alphas>0])
    # b = [[-3.85911353]] 
    # alphas = [[0.10405027 0.25880749 0.01487231 0.34798545]]
```

筛选出得到的支持向量：

```python
shape(alphas[alphas>0])
for i in range(100):
    if alphas[i]>0.0:
        print(dataArr[i],labelArr[i])
        
[4.658191, 3.507396] -1.0
[3.457096, -0.082216] -1.0
[2.893743, -1.643468] -1.0
[6.080573, 0.418886] 1.0
```

向量在数据集中的位置：

<img src="https://pic.imgdb.cn/item/64be395a1ddac507cc30deeb.jpg" style="zoom:67%;" />

**完整版Platt SMO的支持函数**

```python
import numpy
from numpy import *
import random

def loadDataSet(fileName):  # 加载数据
    dataMat = [];
    labelMat = []
    fr = open(fileName)  # 打开文件
    for line in fr.readlines():  # 逐步读取每一行
        lineArr = line.strip().split('\t')  # 以控制符进行分割
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 将前两个特征以列表的形式保存起来,
        labelMat.append(float(lineArr[2]))  # 把标签保存起来
    return dataMat, labelMat


def selectJrand(i, m):  # 随机选择一个和i不同的j
    j = i
    while (j == i):
        j = int(numpy.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):  # 用于调整aj值，让aj在H和L的范围内
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):  # 输入分别为数据集，类标签，常数C,容许错误率，退出前最大的循环次数
    dataMatrix = numpy.mat(dataMatIn)  # 输入数据集变为矩阵
    labelMat = numpy.mat(classLabels).transpose()  # 把输入标签变为矩阵并且转置，得到一个列向量，标签的每行对应每行的数据
    b = 0  # 初始化
    m, n = numpy.shape(dataMatrix)  # 得到矩阵的行列数
    alphas = numpy.mat(numpy.zeros((m, 1)))  # 定义一个m*1的零矩阵
    iter = 0  # 初始化迭代次数
    ##只有在所有数据集上遍历maxIter次，且不再发生任何alpha修改之后，程序才会停止并退出while循环
    while (iter < maxIter):  # 迭代次数大于最大迭代次数时，退出迭代
        alphaPairsChanged = 0  # 用来记录alpha是否已经进行优化
        for i in range(m):  # 循环每个数据
            # 步骤1：计算误差Ei
            fXi = float(numpy.multiply(alphas, labelMat).T * (
                        dataMatrix * dataMatrix[i, :].T)) + b  # 预测第i个数据的类别  numpy.multiply对应的位置相乘
            Ei = fXi - float(labelMat[i])  # 实际结果与真实结果的误差，如果误差很大，那么就要对该数据实例所对应的alpha值进行优化
            # 一旦alphas等于0或C，那么它们就巳经在“边界”上了，因而不再能够减小或增大，因此也就不值得再对它们进行优化了
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (
                    alphas[i] > 0)):  # 将违反KKT条件的找出来，具体公式来源见https://blog.csdn.net/youhuakongzhi/article/details/86660281
                j = selectJrand(i, m)  # 在m个中随机选择一个和i不同的j
                fXj = float(numpy.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b  # 预测第j个数据的类别
                Ej = fXj - float(labelMat[j])  # 计算第j个的误差
                alphaIold = alphas[i].copy()  # 保存更新前的aplpha值，使用深拷贝
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):  # 如果yi和yj的标签不一样
                    L = max(0, alphas[j] - alphas[i])  # alphas[j]new的取值范围
                    H = min(C, C + alphas[j] - alphas[i])
                else:  # 如果yi和yj的标签一样
                    L = max(0, alphas[j] + alphas[i] - C)  # alphas[j]new的取值范围
                    H = min(C, alphas[j] + alphas[i])
                if L == H: print("L==H");continue  # 退出本次循环，直接进行下一次for循环
                # 步骤3：计算eta=-2*Kij+Kii+Kjj，而这儿eta=2*Kij-Kii-Kjj,所以下面公式中用的是减号
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[
                                                                                                            j,
                                                                                                            :] * dataMatrix[
                                                                                                                 j, :].T
                if eta >= 0: print("eta>=0"); continue  # 退出本次循环，直接进行下一次for循环
                # 目标函数是求最小值，这儿eta是负的二阶导数，因此二阶导数不能小于等于0
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta  # 更新alphas[j]
                alphas[j] = clipAlpha(alphas[j], H, L)  # 用于调整aj值，让aj在H和L的范围内
                if (abs(alphas[j] - alphaJold) < 0.00001):  # alphas[j]变化太小
                    print("j not moving enough");
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # 更新alphas[i]
                # 更新b1
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * \
                     dataMatrix[i, :] * dataMatrix[j, :].T
                # 更新b2
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * \
                     dataMatrix[j, :] * dataMatrix[j, :].T
                # 更新b
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter : %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1  # 检査alpha值是否做了更新，如果有更新则将iter为0后继续运行程序
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


# 数据结构保存所有的重要值
# 参数 dataMatIn 数据矩阵，classLabels数据标签 c松弛变量 toler容错率
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn  # X:训练的数据集
        self.labelMat = classLabels  # labelMat:X对应的类别标签
        self.C = C  # C:松弛变量系数
        self.tol = toler  # tol:容错率
        self.m = numpy.shape(dataMatIn)[0]  # m:样本的个数
        self.alphas = numpy.mat(numpy.zeros((self.m, 1)))  # alphas:拉格朗日系数，需要优化项
        self.b = 0  # b:阈值
        self.eCache = numpy.mat(numpy.zeros((self.m, 2)))
        self.K = numpy.mat(numpy.zeros((self.m, self.m)))
        # 第一列 标志位,标志Ek是否有效，1为有效，0为无效 第二列 错误率Ek
        for i in range(self.m):  # 计算所有数据的核K
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)  # 两个数值型变量和一个元组


def calcEk(oS, k):  # 计算第K个数据误差Ek
    # fXk=float(numpy.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T))+oS.b
    # Ek=fXk-float(oS.labelMat[k])
    fXk = float(numpy.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):  # 选择第j个数据
    # 启发式算法选择j，选择具有最大步长的j
    # 1.定义步长maxDeltaE (Ei-Ek)  取得最大步长时的K值maxK
    # 需要返回的Ej （具有最大步长 ，即|Ei-Ej|值最大）
    maxK = -1;
    maxDeltaE = 0
    Ej = 0
    # 2.将Ei保存到数据结构的eCache中去
    oS.eCache[i] = [1, Ei]  # Ei - 标号为i的数据误差，oS - 数据结构
    validEcacheList = numpy.nonzero(oS.eCache[:, 0].A)[0]  # 返回误差不为0的数据的索引值
    # 3.判断 如果len(validEcacheList)>1 遍历validEcacheList，找到最大的|Ei-Ej|
    if (len(validEcacheList)) > 1:  # 有不为0的误差
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)  # 计算第k个误差
            deltaE = abs(Ei - Ek)  # 求取绝对值
            if (deltaE > maxDeltaE):  # 选出最大的deltaE和下标k
                maxK = k;
                maxDeltaE = deltaE
                Ej = Ek
            return maxK, Ej
    else:
        j = selectJrand(i, oS.m)  # 随机选择一个和i不同的j
        Ej = calcEk(oS, j)  # 计算误差
    return j, Ej


def updateEk(oS, k):  # 更新Ek
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]
    '''优化的SMO算法
    Parameters：
        i - 标号为i的数据的索引值
        oS - 数据结构
    Returns:
        1 - 有任意一对alpha值发生变化
        0 - 没有任意一对alpha值发生变化或变化太小'''


def innerL(i, oS):
    Ei = calcEk(oS, i)  # 计算第i数据的误差
    ##将违反KKT条件的找出来，具体公式来源见https://blog.csdn.net/youhuakongzhi/article/details/86660281
    if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
            (oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)  # 根据步长选择第J个数据
        alphaIold = oS.alphas[i].copy()  # 保存更新前的aplpha值，使用深拷贝
        alphaJold = oS.alphas[j].copy()  # 保存更新前的aplpha值，使用深拷贝
        if (oS.labelMat[i] != oS.labelMat[j]):  # 如果yi和yj的标签不一样，alphas[j]new的取值范围
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H");
            return 0
        # 计算eta=-2*Kij+Kii+Kjj，而这儿eta=2*Kij-Kii-Kjj,所以下面公式中用的是减号,也可以看做关于oS.alphas[j]的二阶导数
        # eta=2.0*oS.X[i,:]*oS.X[j,:].T-oS.X[i,:]*oS.X[i,:].T-oS.X[j,:]*oS.X[j,:].T
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0: print("eta>=0"); return 0  # 目标函数是求最小值，这儿eta是负的二阶导数，因此二阶导数不能小于等于0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta  # 更新oS.alphas[j]
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)  # 消减到规定范围内
        updateEk(oS, j)  # 更新Ej至误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving engough");
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)  # 更新Ei至误差缓存
        # 步骤7：更新b_1和b_2
        # b1=oS.b-Ei-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        # b2=oS.b-Ej-oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T-oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[j, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        # 步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
    """
        完整的线性SMO算法
        Parameters：
            dataMatIn - 数据矩阵
            classLabels - 数据标签
            C - 松弛变量
            toler - 容错率
            maxIter - 最大迭代次数
        Returns:
            oS.b - SMO算法计算的b
            oS.alp"""
    oS = optStruct(numpy.mat(dataMatIn), numpy.mat(classLabels).transpose(), C, toler, kTup)  # 初始化数据结构
    iter = 0  # 初始化当前迭代次数
    entireSet = True
    alphaPairsChanged = 0
    # 遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:  # 遍历整个数据集#首先进行完整遍历，过程和简化版的SMO一样
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)  # 使用优化的SMO算法，如果oS.alphas[j]和oS.alphas[i]更新，则返回1
                print("fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1  # 循环完一次，迭代次数加1
        else:  # 非边界遍历，挑选其中alpha值在0和C之间非边界alpha进行优化
            # 遍历不在边界0和C的alpha
            nonBoundIs = numpy.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]  # numpy.nonzeros返回非零元素的下标位置，分别从
            for i in nonBoundIs:  # 遍历非边界元素
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True  # 如果alpha没有更新,计算全样本遍历
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def calcWs(alphas, dataArr, classLabels):
    X = numpy.mat(dataArr)
    labelMat = numpy.mat(classLabels).transpose()
    m, n = numpy.shape(X)
    w = numpy.zeros((n, 1))
    for i in range(m):  # 循环每个数据
        w += numpy.multiply(alphas[i] * labelMat[i], X[i, :].T)  # 计算权值w
    return w


def kernelTrans(X, A, kTup):
    # kTup元组中的第一个参数是描述所用核函数类型的一个字符串，其他2个参数是核函数可能需要的可选参数
    """
       通过核函数将数据转换更高维的空间
       Parameters：
           X - 数据矩阵
           A - 单个数据的向量，新的高维向量
           kTup - 包含核函数信息的元组
       Returns:
           K - 计算的核K
       """
    m, n = numpy.shape(X)
    K = numpy.mat(numpy.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T  # 线性核函数,只进行内积。
    elif kTup[0] == 'rbf':  # 径向基核函数
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = numpy.exp(K / (-2 * kTup[1] ** 2))
    else:
        raise NameError('Houston we have a problem--that kernel is not recognized')  # 核函数不是我们这儿已有的
    return K


def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')  # 加载数据
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    datMat = numpy.mat(dataArr);
    labelMat = numpy.mat(labelArr).transpose()
    svInd = numpy.nonzero(alphas.A > 0)[0]  # 获得支持向量
    sVs = datMat[svInd]  # get matrix of only support vectors#得到仅仅有支持向量的数据矩阵
    labelSV = labelMat[svInd];  # 得到支持向量数据的标签
    print("there are %d Support Vectors" % numpy.shape(sVs)[0])  # 输出支持响亮的个数
    m, n = numpy.shape(datMat)  # 得到原始数据的行列数
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * numpy.multiply(labelSV, alphas[svInd]) + b  # 根据支持向量的点，计算超平面，返回预测结果
        if numpy.sign(predict) != numpy.sign(labelArr[i]): errorCount += 1  # 如果分类预测的和实际的标签不相符，误差数量加1
    print("the training error rate is: %f" % (float(errorCount) / m))  # 打印错误率
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')  # 加载测试集
    errorCount = 0
    datMat = numpy.mat(dataArr);
    labelMat = numpy.mat(labelArr).transpose()
    m, n = numpy.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))  # 计算各个点的核
        predict = kernelEval.T * numpy.multiply(labelSV, alphas[svInd]) + b  # 根据支持向量的点，计算超平面，返回预测结果
        if numpy.sign(predict) != numpy.sign(labelArr[i]): errorCount += 1  # 返回数组中各元素的正负符号，用1和-1表示，并统计错误个数
    print("the test error rate is: %f" % (float(errorCount) / m))  # 打印错误率

if __name__ == '__main__':
    # 打开文件，读取数据
    dataArr, labelArr = loadDataSet('testSet.txt')
    # 将数据进行支持向量机训练
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    ws=calcWs(alphas,dataArr,labelArr)
    dataMat = mat(dataArr)
    
'''
iteration number: 8
b = [[-2.93254384]] 
alphas = [[ 0.64769236] [-0.22845215]]
ws = [[2.31947274]]
label = 1.0
'''
```



---

### 基于SVM的数字识别

```python
# 回顾手写字
# 读取数据到矩阵
def img2vector(filename):
    # 创建向量
    returnVect = numpy.zeros((1, 1024))
    # 打开数据文件，读取每行内容
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        # 将每行前32字符转成int存入向量
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 将文件夹下面的数字文件 转换成数据特征（每个数字1/0就是一个特征，每个文件有1024个特征）和类别标签
def loadImages(dirName):
    from os import listdir  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    hwLabels = []
    trainingFileList = listdir(dirName)  # load the training set
    m = len(trainingFileList)
    trainingMat = numpy.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  ##对文件名进行分割。例如2_45.txt，从.那个地方开始分割文件名，就得到2_45和txt两部分，
        classNumStr = int(fileStr.split('_')[0])  # 对文件名进行分割。例如2_45，从_那个地方开始分割文件名，就得到2和45两部分，相当于数字的标签值
        if classNumStr == 9:
            hwLabels.append(-1)  # 二分类，数字9标签为-1 其他的标签为+1
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))  # 读取数据到矩阵
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('../ch2/trainingDigits')  # 加载训练数据
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = numpy.mat(dataArr);
    labelMat = numpy.mat(labelArr).transpose()
    svInd = numpy.nonzero(alphas.A > 0)[0]  # 得到支持向量
    sVs = datMat[svInd]  # 支持向量数据
    labelSV = labelMat[svInd];  # 支持向量标签
    print("there are %d Support Vectors" % numpy.shape(sVs)[0])
    m, n = numpy.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)  ##计算各个点的核
        predict = kernelEval.T * numpy.multiply(labelSV, alphas[svInd]) + b  # 预测分类平面
        if numpy.sign(predict) != numpy.sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadImages('../ch2/testDigits')  # 加载测试数据
    errorCount = 0
    datMat = numpy.mat(dataArr);
    labelMat = numpy.mat(labelArr).transpose()
    m, n = numpy.shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * numpy.multiply(labelSV, alphas[svInd]) + b
        if numpy.sign(predict) != numpy.sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))
 
if __name__ == '__main__':
    testDigits(('rbf',20))
```

<img src="https://pic.imgdb.cn/item/64bf899d1ddac507cceeca14.jpg" style="zoom:67%;" />
