---
title: Logistic回归
tags:
  - 机器学习
cover: https://pic.imgdb.cn/item/649ce61e1ddac507ccb10aca.png
toc: True
date: 2023/7/23
---

# Logistic回归

### 本章内容

- Sigmoid函数和Logistic回归分类器

- 最优化理论初步

- 梯度下降最优化算法

- 数据中的缺失项处理

---

​		Logistic回归（logistic regression）是统计学习中的经典分类方法，属于对数线性模型，所以也被称为对数几率回归。这里要注意，虽然带有回归的字眼，但是该模型是一种分类算法，Logistic回归是一种线性分类器，针对的是线性可分问题。利用logistic回归进行分类的主要思想是：根据现有的数据对分类边界线建立回归公式，以此进行分类。

**1. Sigmoid函数**

​		我们想要的函数应该是能接受所有的输入，然后预测出类别。例如，在二类情况下，输出0和1。像单位阶跃函数就可以实现，但该函数在跳跃点上从0->1是瞬间跳跃，这个瞬间跳跃过程有时很难处理。于是另一个有类似性质且数学上更易处理的函数出现了--sigmoid函数。

  sigmoid函数表达式如下：

$$σ ( z ) = \frac{1}{1+e^{-z}}$$

  下图给出了Sigmoid函数在不同坐标尺度下的两条曲线图，当x为0时，Sigmoid函数值为0.5。随着x的增大，对应的Sigmoid值将逼近于1；随着x的减小则逼近于0。当横坐标尺度较小时，曲线变化较为平滑，当尺度扩大到一定程度，Sigmoid看起来像一个阶跃函数。

<img src="https://pic.imgdb.cn/item/64bbb8151ddac507ccf6974f.jpg" style="zoom: 67%;" />

```python
# Sigmoid函数实现
def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))
```

**2.Logistic回归的优缺点**

- 优点：计算代价不高，易于理解和实现。
- 缺点：容易欠拟合，分类精度可能不高。
- 使用数据类型：数值型和标称型数据。

**3.logistic回归的一般过程**

1. 收集数据：任何方式 
2. 准备数据：由于要计算距离，因此要求数据都是数值型的，另外结构化数据格式最佳。 
3. 分析数据：采用任一方是对数据进行分析 
4. 训练算法：大部分时间将用于训练，训练的目的为了找到最佳的分类回归系数 
5. 测试算法：一旦训练步骤完成，分类将会很快 
6. 使用算法：首先，我们需要输入一些数据，并将其转化成对应的结构化数值；接着基于训练好的回归系数就可以对这些数值进行简单的回归计算，判定它们属于哪一类别；在这之后，我们就可以在输出的类别上做一些其他的分析工作。

---

### 基于最优化方法的最佳回归系数确定

**1.梯度上升法**

<img src="https://pic.imgdb.cn/item/64bd0ee81ddac507cc832e98.jpg" style="zoom:67%;" />

​		梯度算子总是指向函数值增长最快的方向。这里所说的是移动方向，而未提到移动量的大小。该量值称为步长，记做$α$。用向量来表示的话，梯度上升算法的迭代公式如下：

<img src="https://pic.imgdb.cn/item/64bd0f771ddac507cc84ec7d.jpg" style="zoom:67%;" />

​		该公式将一直被迭代执行，直至达到某个停止条件为止，比如迭代次数达到某个指定值或算 法达到某个可以允许的误差范围。

```python
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # 转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()  # 转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001  # 移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500  # 最大迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights  # 将矩阵转换为数组，返回权重数组

matrix = [[ 4.12414349]
 		  [ 0.48007329]
 		  [-0.6168482 ]]
```

```python
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()  # 加载数据集
    dataArr = np.array(dataMat)  # 转换成numpy的array数组
    n = np.shape(dataMat)[0]  # 数据个数
    xcord1 = [];
    ycord1 = []  # 正样本
    xcord2 = [];
    ycord2 = []  # 负样本
    for i in range(n):  # 根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])  # 1为正样本
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])  # 0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)  # 绘制正样本
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)  # 绘制负样本
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.title('BestFit')  # 绘制title
    plt.xlabel('X1');
    plt.ylabel('X2')  # 绘制label
    plt.show()
```

<img src="https://pic.imgdb.cn/item/64bd10431ddac507cc874abf.jpg" style="zoom:67%;" />

**2.随机梯度上升**

```python
def stocGradAscent0(dataMatrix,classLabels):
    """随机梯度上升法"""
    m,n = shape(dataMatrix)   # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.01              # 移动步长,也就是学习速率,控制更新的幅度。
    weights = np.ones(n)      # [1. 1. 1.]
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)         #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)               #参数初始化
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01          #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))           #随机选取样本
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*weights))  #选择随机选取的一个样本，计算h
            error = classLabels[dataIndex[randIndex]] - h               #计算误差
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]] #更新回归系数
            del(dataIndex[randIndex])          #删除已经使用的样本
    return weights
```

<img src="https://pic.imgdb.cn/item/64bd10ab1ddac507cc88799d.jpg" style="zoom:67%;" />

---

### 回归系数与迭代次数的关系

```python
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import random

def plotWeights(weights_array1, weights_array2):
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])  # fontproperties='SimHei',fontsize=14
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系', fontproperties='SimHei', fontsize=14)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0', fontproperties='SimHei', fontsize=14)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1', fontproperties='SimHei', fontsize=14)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数', fontproperties='SimHei', fontsize=14)
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2', fontproperties='SimHei', fontsize=14)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系', fontproperties='SimHei', fontsize=14)
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0', fontproperties='SimHei', fontsize=14)
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1', fontproperties='SimHei', fontsize=14)
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数', fontproperties='SimHei', fontsize=14)
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1', fontproperties='SimHei', fontsize=14)
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights1, weights_array1 = stocGradAscent1(np.array(dataMat), labelMat)
    weights2, weights_array2 = gradAscent(dataMat, labelMat)
    plotWeights(weights_array1, weights_array2)
```

![](https://pic.imgdb.cn/item/64bd281d1ddac507cccc369e.jpg)

---

### 从疝气病症预测病马的死亡率

```python
import numpy as np
import random

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)         #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)               #参数初始化
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01          #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))           #随机选取样本
            h = sigmoid(sum(dataMatrix[dataIndex[randIndex]]*weights))  #选择随机选取的一个样本，计算h
            error = classLabels[dataIndex[randIndex]] - h               #计算误差
            weights = weights + alpha * error * dataMatrix[dataIndex[randIndex]] #更新回归系数
            del(dataIndex[randIndex])          #删除已经使用的样本
    return weights



def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)  # 转换成numpy的mat
    labelMat = np.mat(classLabels).transpose()  # 转换成numpy的mat,并进行转置
    m, n = np.shape(dataMatrix)  # 返回dataMatrix的大小。m为行数,n为列数。
    alpha = 0.001  # 移动步长,也就是学习速率,控制更新的幅度。
    maxCycles = 500  # 最大迭代次数
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # 梯度上升矢量化公式
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights  # 将矩阵转换为数组，返回权重数组

def colicTest():
    frTrain = open('horseColicTraining.txt')  # 打开训练集
    frTest = open('horseColicTest.txt')  # 打开测试集
    trainingSet = [];
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)  # 使用改进的随即上升梯度训练
    errorCount = 0;
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec) * 100  # 错误率计算
    print("测试集错误率为: %.2f%%" % errorRate)

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

if __name__ == '__main__':
    colicTest()
    # 测试集错误率为: 34.33%
```
