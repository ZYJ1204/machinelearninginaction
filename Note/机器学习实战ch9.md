---
title: 聚类
tags:
  - 机器学习
cover: https://pic.imgdb.cn/item/649ce61e1ddac507ccb10aca.png
toc: True
date: 2023/8/1
---

# 聚类

### 本章内容

- K-均值聚类算法

- 对聚类得到的簇进行后处理

- 二分K-均值聚类算法

- 对地理位置进行聚类


---

### 1 聚类的定义

​		聚类：是机器学习中的无监督学习，目标是通过对无标记训练样本的学习来解释数据的内在性质以及规律，为进一步的数据分析提供基础。

​		聚类试图将数据集中的样本划分为若干个通常是不相交的子集，每个子集称为一个“簇”。

<img src="https://pic.imgdb.cn/item/64c8be8c1ddac507ccd40c03.jpg" style="zoom: 67%;" />

---

### 2 原型聚类

**2.1 k均值聚类（k-means）算法**

​		给定样本集$D=(x_1,x_2,...,x_m)$，共m个样本集，k均值算法针对聚类所得簇划分$C=(C_1,C_2,... .C_k)$，即需要将样本集中的所有样本根据规则，将其分别划分至合适的簇中，最小化平方误差：

$$E=\sum\limits^k_{i=1}\sum\limits_{x∈C_i}||x-μ_i||^2_2$$

​		最小化平方误差，找到它的最优解需要考察样本集D中的所有可能的簇划分，这是一个NP难问题。因此，k 均值算法采用了贪心策略，通过迭代优化来近似求解上式，算法的具体流程如下：

> 1. 从样本集D中随机选择k个样本作为初始的均值向量（质心），每个样本表示一个簇；
> 2. 计算样本集中的其他样本与初始的均值向量的距离，根据距离最近的均值向量确定该样本所属的簇；
> 3. 对D中的所有样本均划分完成后，根据划分的簇中的样本，重新计算均值向量，更新当前均值向量后，再重复计算样本与均值向量的距离，重新划分；
> 4. 不断重复2、3的步骤，直到迭代的均值向量不再发生变换或达到最大迭代次数，算法终止，得到最终的簇划分。

**2.1.1 计算过程**

<img src="https://pic.imgdb.cn/item/64c8c9a31ddac507cce7d2e0.jpg" style="zoom:67%;" />

>假定 聚类簇数 k=3，归属类分别为$C_1,C_2,C_3$，质心分别为$μ_1,μ_2,μ_3$
>
>随机选取三个样本$x_6,x_{12},_{24}$作为初始质心
>
>$μ_1=(0.403,0.237),μ_2=(0.343,0.099),μ_3=(0.478,0.437)$
>
>计算$\overline{x_1}=(0.697,0.460)$，计算它与上面三个质心的距离
>
>$d_{11}=\sqrt{(0.697-0.403)^2+(0.460-0.237)^2}=0.369$
>
>同理$d_{12}=0.506,d_{13}=0.220$
>
>$∵d_{13}<d_{11}<d_{12}$
>
>$∴x_1归属于C_3$
>
>同理计算D中所有样本的归属类，可得：
>
>$C_1=\{x_3,x_5,x_6,x_7,x_8,x_9,x_{10},x_{13},x_{14},x_{17},x_{18},x_{19},x_{20},x_{23}\}$
>
>$C_2=\{x_{11},x_{12},x_{16}\}$
>
>$C_3=\{x_1,x_2,x_4,x_{15},x_{21},x_{22},x_{24},x_{25},x_{26},x_{27},x_{28},x_{29},x_{30}\}$
>
>于是，根据各个簇，分别再重新计算新的质心：
>
>$μ_1^,=(0.493,0.207),μ_2^,=(0.394,0.066),μ_3^,=(0.602,0.396)$
>
>更新当前质心后，不断重复上述过程。不断迭代，当质心不再变化时，算法终止。

<img src="https://pic.imgdb.cn/item/64c8cdd41ddac507ccf04a44.jpg" style="zoom:67%;" />

**2.1.2 算法代码**

```python
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):  #加载数据集
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

# 计算两个数据集之间的欧式距离
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
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            #对簇求均值,赋给对应的centroids质心
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

if __name__ == '__main__':
    datMat = mat(loadDataSet('melon.txt'))
    myCentroids , clusterAssing = kMeans(datMat,3)
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
```

<img src="https://pic.imgdb.cn/item/64c9fd201ddac507ccf3a45c.jpg" style="zoom:67%;" />

**补充：**

```cmd
>>> a=mat([[1,2,3],[4,5,6]])    # a是一个矩阵
>>> a
matrix([[1, 2, 3],
        [4, 5, 6]])

>>> a.flatten()                 # 降维后还是一个矩阵
matrix([[1, 2, 3, 4, 5, 6]])
 
>>> a.flatten().A            # 二维数组（两个中括号[]）
array([[1,2,3,4,5,6]])       # 该数组大小为一行6列（1，6）
 
>>> a.flatten().A[0]         #  一维数组(一个中括号[])
array([1, 2, 3, 4, 5, 6]) 
 
#  matrix.A :    表示将矩阵matrix转换为二维数组（矩阵a是一个二维矩阵）
#  matrix.A[0] : 取二维数组中第一行元素
```

**2.2 二分K-均值算法**

​		为克服K-均值算法收敛于局部最小值的问题，有人提出了另一个称为二分K-均值（bisecting K-means）的算法。该算法首先将所有点作为一个簇，然后将该簇一分为二。之后选择其中一个簇继续进行划分，选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE的值。上述基于SSE的划分过程不断重复，直到得到用户指定的簇数目为止。

> **用于度量聚类效果的指标是SSE（Sum of Squared Error，误差平方和）**
>
> **SSE值越小表示数据点越接近于它们的质心，聚类效果也越好。因为对误差取了平方，因此更加重视那些远离中心的点。**

**2.2.1 计算过程**

二分K-均值算法的伪代码形式如下：

```
将所有点看成一个簇
当簇数目小于k时
	对于每一个簇计算总误差
		在给定的簇上面进行K-均值聚类（k=2）
		计算将该簇一分为二之后的总误差
	选择使得误差最小的那个簇进行划分操作
```

**2.2.2 算法代码**

```python
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
```

<img src="https://pic.imgdb.cn/item/64ca01571ddac507ccfde314.jpg" style="zoom:67%;" />

**2.3 学习向量量化算法（LVQ）**

​		与k均值算法类似，“学习向量量化”（Learning Vector Quantization，简称LVQ）也是试图找到一组原型向量来刻画聚类结构，但与一般聚类算法不同的是，LVQ假设数据样本具有类别标签（类别标签是人为定义的），学习过程中利用样本的这些监督信息来辅助聚类。

​		在样本数据集上会有一些变化，即样本集$D = (x_1,y_1)(x_2,y_2),...,(x_m,y_m)$，其中，每个样本都有一个类别标签$y_j$。

​		LVQ的目标是学得一组n维原型向量${p_1,P_2,...,P_n}$,每个原型向量代表一个聚类簇。

​		那么重点来了，初始阶段我们随机选取n个原型向量，即聚类的簇数，那么我们怎样通过计算,去更新我们的初始值（原型向量），以达到最好的聚类效果呢?
​		针对样本集D中的一个样本$x_j$，计算其与原型向量之间的距离，选择与$x_j$最近的一个原型向量$p_i$,接下来判断该原型向量的类别与样本$x_j$的类别是否相同，若相同，则令该原型向量$p_i$向$x_j$方向靠拢，由此得到新的原型向量，具体的计算公式为:

$$p^,=p_i+η(x_j-p_i)$$

​		若不同，则令该原型向量$p_i$远离$x_j$，由此得到新的原型向量，具体的计算公式为:

$$p^,=p_i-η(x_j-p_i)$$

​		算法的整体流程如下：

> 1. 从样本集D中选择一组原型向量假设选取n个样本；
> 2. 从样本集D中随机选取样本$(x_j,y_j)$，分别计算其与第一步中选取的原型向量的距离$d=||x_j-p_i||$ ；
> 3. 根据计算的距离，选取与样本距离最近的一个原型向量，接着去判断该样本与原型向量的类别标签是是否相同，根据不同的公式去更新该原型向量；
> 4. 接着不断重复2,3步骤，不断迭代更新，直到满足条件停止更新；
> 5. 输出最终的簇分类结果。

**2.3.1 计算过程**

<img src="https://pic.imgdb.cn/item/64c8c9a31ddac507cce7d2e0.jpg" style="zoom:67%;" />

>取5个作为原型向量$p_1,p_2,p_3,p_4,p_5$，即$x_5,x_{12},x_{18},x_{23},x_{29}$
>
>并设定9-21号样本的类别标签为$c_2$，其他样本为$c_1$
>
>$μ_1=(0.403,0.237),μ_2=(0.343,0.099),μ_3=(0.478,0.437)$
>
>则随机选取样本$x_1$，分别计算其与原型向量的距离
>
>$d_{11}=\sqrt{(0.697-0.556)^2+(0.460-215)^2}=0.283$
>
>同理$d_{12}=0.506,d_{13}=0.434,d_{14}=0.260,d_{15}=0.032$
>
>可得$x_1$与$p_5$距离最近，比较类别标签：相同均为$c_1$
>
>假定学习率为0.1，更新$p_5$得到新的原型向量
>
>$p^,=p_5+η(x_1-p_5)=(0.725,0.445)+0.1*((0.697,0.460)-(0.725,0.445))=(0.722,0.444)$
>
>更新当前质心后，不断重复上述过程。不断迭代，当质心不再变化时，算法终止。

<img src="https://pic.imgdb.cn/item/64ca243e1ddac507cc493681.jpg" style="zoom:67%;" />

**2.4 高斯混合聚类算法**

**2.4.1 多元正态分布**

多元正态分布的概率密度为：

$$p(X)=\frac{1}{(2π)^{\frac{n}{2}}|\sum|^{\frac{1}{2}}}e^{-\frac{1}{2}(X-μ)^T∑^{−1}(X−μ)}$$

上面式子中的 $μ$为n维均值向量，形状为(n,1)，$∑$为随机变量X的协方差矩阵。

定义高斯混合分布：

$$P_{M}(x)=\sum\limits_{i=1}^{K}\alpha_{i} \cdot p(x|μ_i,\sum_{i})$$

即添加了混合系数$\alpha_i$且$\sum\limits_{i=1}^k\alpha_i=1$。

限设样本的生成过程由高斯混合分布给出：首先，确定$α_1,α_2,...,α_k$个混合系数，以及其服从的高斯分布。然后，根据选择的混合成分的概率密度函数进行采样，从而生成相应的样本。

$$p(z_j=i|x_j)=\frac{P(z_j=i)P_M(x_j|z_j=i)}{P_M(x_j)}=\frac{α_i \cdot p(x_j|μ_i,∑_i)}{∑\limits_{l=1}^k α_lp(x_j|μ_1,∑_i)}$$

其中，$α_i$为先验概率，$x_j$属于$α_i$的概率分布，$x_j$可能属于$α_1,α_2,...,α_k$中的任意一个。

接下来更新计算模型参数$α_i,μ_i,\sum_i$：

可采用极大似然估计，即最大化对数似然：

$$LL(D)=ln(\prod\limits^{m}_{j=1} P_{M}(x_j))=\sum\limits_{j=1}^{m}ln(\sum\limits_{i=1}^{k}\alpha_{i}p(x_j|\mu_i,\sum_i))$$

对上式分别关于$α_i,μ_i,\sum_i$求偏导，可得：

<img src="https://pic.imgdb.cn/item/64ca116f1ddac507cc235282.jpg" style="zoom: 67%;" />

**2.4.2 计算过程**

<img src="https://pic.imgdb.cn/item/64ca11fb1ddac507cc249488.jpg" style="zoom: 82%;" />

---

### 3 密度聚类

​		密度聚类也称为“基于密度的聚类”，此类算法假设聚类结构能通过样本分布的紧密程度确定。通常情形下，密度聚类算法从样本密度的角度来考察样本间的可连接性，并基于可连接样本不断扩展聚类簇以获得最终的聚类结果。

​		DBSCAN是密度聚类的代表算法。

​		它是基于一组“邻域”的参数$(\epsilon,MinPts)$来刻画样本分布的紧密程度。

​		所谓的“邻域”参数$(ϵ,MinPts)$，即与样本$x_j$ 距离不大于ϵ的最少样本数MinPts。

<img src="https://pic.imgdb.cn/item/64ca14be1ddac507cc29f5a3.jpg" style="zoom:67%;" />

> 了解几个概念: 核心对象；密度直达；密度可达
>
> 核心对象：是指样本$x_j$的邻域内至少包含MinPts个样本，则称$x_j$为核心对象
>
> 密度直达：若$x_i$位于$x_j$的邻域内，且$x_j$是核心对象，则称$x_i$由$x_j$密度直达
>
> 密度可达：对于样本$x_i$、$x_j$ ，中间有一个样本$x_m$，使得$x_i$由$x_m$密度直达，那么称$x_i$由$x_j$密度可达。

---

### 4 层次聚类

​		层次聚类试图在不同的层次对数据集进行划分，从而形成树形的聚类结构。数据集的划分可采用“自底向上”的聚合策略，也可采用“自顶向下”的分拆策略。

​		AGNES是一种采用“自底向上”聚合策略的层次聚类算法。它先将数据集中的每个样本看作是一个初始聚类簇，然后在算法运行的每一步中找出距离最近的两个聚类簇进行合并，该过程不断重复。直至达到预设的聚类簇个数。

---

### 5 对地图上的点进行聚类

```python
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
```

运行结果:

```python
sseSplit, and notSplit:  3339.5543932456144 0.0
the bestCentToSplit is:  0
the len of bestClustAss is:  69
sseSplit, and notSplit:  513.4576367897055 2375.7915215892217
sseSplit, and notSplit:  1403.8012888692772 963.7628716563925
the bestCentToSplit is:  1
the len of bestClustAss is:  54
sseSplit, and notSplit:  363.20334521429214 1403.8012888692772
sseSplit, and notSplit:  585.8232265931201 1300.0563616457327
sseSplit, and notSplit:  125.27922472399202 2031.27067053633
the bestCentToSplit is:  0
the len of bestClustAss is:  15
sseSplit, and notSplit:  72.39336276412833 1532.5047333341886
sseSplit, and notSplit:  810.649977739478 699.4968352036318
sseSplit, and notSplit:  125.27922472399202 1430.7111440942297
sseSplit, and notSplit:  22.585418879811662 1638.301189618658
the bestCentToSplit is:  1
the len of bestClustAss is:  38
```

<img src="https://pic.imgdb.cn/item/64ca164d1ddac507cc2d1c3d.jpg" style="zoom:67%;" />
