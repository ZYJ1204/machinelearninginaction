---
title: K-近邻算法
tags:
  - 机器学习
cover: https://pic.imgdb.cn/item/649ce61e1ddac507ccb10aca.png
toc: True
date: 2023/7/15
---

# K-近邻算法

### 本章内容？

+ K-近邻分类算法
+ 从文本文件中解析和导入数据
+ 使用Matplotlib创建扩散图
+ 归一化数值

---

### K-近邻分类算法(k-Nearest Neighbor)

k-近邻算法是分类数据最简单最有效的算法，k采用测量不同特征值之间的距离方法进行分类。

**算法原理：存在一个样本数据集合，也称作训练样本集，并且样本集中每个数据都存在标签，即我们知道样本集中每一数据与所属分类的对应关系。输入没有标签的新数据后，将新数据的每个特征与样本集中数据对应的特征进行比较，然后算法提取样本集中特征最相似数据（最近邻）的分类标签。**

最后选择k个最相似数据中出现次数最多的分类，作为新数据的分类。

**K-近邻算法的一般流程**

1. 收集数据
2. 准备数据：距离计算所需要的数值，最好是结构化的数据格式。
3. 分析数据
4. 训练算法：此步骤不适用于k-近邻算法。
5. 测试算法：计算错误率。
6. 使用算法：首先需要输入样本数据和结构化的输出结果，然后运行K-近邻算法判定输入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理。

```python
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                  #读取矩阵第一维度的长度
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #将原矩阵纵向地复制
    sqDiffMat = diffMat**2   #矩阵计算平方
    sqDistances = sqDiffMat.sum(axis=1)   #沿横轴求和 dataSetSize*1
    distances = sqDistances**0.5 #开方
    sortedDistIndicies = distances.argsort() #返回对数组进行排序的索引
    classCount={}   #定义一个字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  #索引对应标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #返回指定键的值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)#降序
    # sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # python3 已经移除iteritems，使用items替代
    return sortedClassCount[0][0]

    # x=np.array([[3,3],[2,2]])
    # print(np.argsort(x, axis=0))  [[1 1] [0 0]]


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

if __name__ == "__main__":
    group, labels = createDataSet()
    print(classify0([1,0.4],group,labels,3))
```

运行过程及结果说明：

```python
group = [[1.  1.1]
         [1.  1. ]
         [0.  0. ]
         [0.  0.1]]
labels = ['A', 'A', 'B', 'B']
dataSetSize = 4                 #矩阵维度
inX = [0,0]                     #inX拓展至4行
diffMat = [[-1.  -1.1]
           [-1.  -1. ]
           [ 0.   0. ]
           [ 0.  -0.1]]
sqDiffMat = [[1.   1.21]        #平方
             [1.   1.  ]
             [0.   0.  ]
             [0.   0.01]]
sqDistances = [2.21 2.   0.   0.01]         #横向求和
distances = [1.48660687 1.41421356 0.         0.1       ] #开方
sortedDistIndicies = [2 3 1 0]              #排序  由小到大 对应数组下标
voteIlabel = B                  #labels[2]
classCount = {'B': 1}
voteIlabel = B                  #labels[3]
classCount = {'B': 2}
voteIlabel = A                  #labels[1]
classCount = {'B': 2, 'A': 1}
sortedClassCount = [('B', 2), ('A', 1)]     #排序
B       #结果
```

---

从文件中读出数据并处理：

```python
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #获取文件行数
    returnMat = zeros((numberOfLines,3))        #创建行数*3维的矩阵，初始化为0
    classLabelVector = []                       #标签
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()                     #strip函数会删除头和尾的字符，中间的不会删除
        listFromLine = line.split('\t')         #依照制表符分割
        returnMat[index,:] = listFromLine[0:3]  #得到前三列数据，即飞行时间，游戏，冰激凌
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):                          #归一化
    minVals = dataSet.min(0)                    #每一列的最小值 1*3
    maxVals = dataSet.max(0)                    #每一列的最大值 1*3
    ranges = maxVals - minVals                  #计算每一列的取值范围
    normDataSet = zeros(shape(dataSet))         #初始化矩阵
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))    #当前值减去最小值
    normDataSet = normDataSet/tile(ranges, (m,1))   #除以取值范围
    return normDataSet, ranges, minVals

from numpy import *
from Coding.ch2.KNN import file2matrix,autoNorm
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
ax.axis([-2,25,-0.2,2.0])
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
plt.show()

print(autoNorm(datingDataMat))
```

运行结果：

```python
(array([[0.44832535, 0.39805139, 0.56233353],
       [0.15873259, 0.34195467, 0.98724416],
       [0.28542943, 0.06892523, 0.47449629],
       ...,
       [0.29115949, 0.50910294, 0.51079493],
       [0.52711097, 0.43665451, 0.4290048 ],
       [0.47940793, 0.3768091 , 0.78571804]]), array([9.1273000e+04, 2.0919349e+01, 1.6943610e+00]), array([0.      , 0.      , 0.001156]))
```

<img src="https://pic.imgdb.cn/item/64b9f9d31ddac507cc71a735.jpg" style="zoom:67%;" />

<img src="https://pic.imgdb.cn/item/64b9f9791ddac507cc705de5.jpg" style="zoom: 67%;" />

---

### 分类器验证：

```python
def datingClassTest():
    hoRatio = 0.50  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 读取
    normMat, ranges, minVals = autoNorm(datingDataMat)               # 归一化
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)   #确定测试向量的数量
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)
```

部分运行结果如下：

<img src="https://pic.imgdb.cn/item/64b9fc1b1ddac507cc7aa8bc.jpg" style="zoom:67%;" />

---

### 使用k-近邻算法的手写识别系统

算法流程：

1. 收集数据：提供文本文件。
2. 准备数据：编写函数classify0()，将图像格式转换为分类器使用的list格式。
3. 分析数据：在Python命令提示符中检查数据，确保它符合要求。
4. 训练算法：此步骤不适用于k-近邻算法。
5. 测试算法：编写函数使用提供的部分数据集作为测试样本，测试样本与非测试样本的区别在于测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。

​		为了使用前面的分类器，我们必须将图像格式化处理为一个向量。我们将把一个32×32的二进制图像矩阵转换为1×1024的向量，这样前两节使用的分类器就可以处理数字图像信息了。

​		我们首先编写一段函数img2vector，将图像转换为向量：该函数创建1×1024的NumPy数组，然后打开给定的文件，循环读出文件的前32行，并将每行的头32个字符值存储在NumPy数组 中，最后返回数组。

```python
def img2vector(filename):
    returnVect = zeros((1, 1024)) #创建数组
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()   #读取32行
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j]) #赋值
    return returnVect
```

分类器代码：

```python
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  #加载训练集
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]       #从文件名解析分类数字
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)              #将数字加入标签
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # 加载测试集
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # 从文件名解析分类数字
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))
```

测试结果如下：

<img src="https://pic.imgdb.cn/item/64b9ff981ddac507cc88f473.jpg" style="zoom:67%;" />
