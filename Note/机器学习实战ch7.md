---
title: AdaBoost元算法
tags:
  - 机器学习
cover: https://pic.imgdb.cn/item/649ce61e1ddac507ccb10aca.png
toc: True
date: 2023/7/28
---

# 利用AdaBoost元算法提高分类性能

### 本章内容

- 组合相似的分类器来提高分类性能

- 应用AdaBoost算法

- 处理非均衡分类问题

---

### AdaBoost算法介绍

**AdaBoost优缺点：**

- 优点：泛化错误率低，易编码，可以应用在大部分分类器上，无参数调整。

- 缺点：对离群点敏感。

- 适用数据类型：数值型和标称型数据。

**AdaBoost的一般流程**

1. 收集数据：可以使用任意方法。
2. 准备数据：依赖于所使用的弱分类器类型，本章使用的是单层决策树，这种分类器可以处理任何数据类型。当然也可以使用任意分类器作为弱分类器，第2章到第6章中的任一分类器都可以充当弱分类器。作为弱分类器，简单分类器的效果更好。
3. 分析数据：可以使用任意方法。
4. 训练算法：AdaBoost的大部分时间都用在训练上，分类器将多次在同一数据集上训练弱分类器。
5. 测试算法：计算分类的错误率。
6. 使用算法：同SVM一样，AdaBoost预测两个类别中的一个。如果想把它应用到多个类别的场合，那么就要像多类SVM中的做法一样对AdaBoost进行修改。

---

### 基于单层决策树构建弱分类器

1.  **首先构建如下的简单数据集**

   ```python
   """构建简单的数据集"""
   def loadSimpData():
       datMat=matrix(
           [[1.0,2.1],
            [2.0,1.1],
            [1.3,1.0],
            [1.0,1.0],
            [2.0,1.0]]
       )
       classLabels=[1.0,1.0,-1.0,-1.0,1.0]
       return datMat,classLabels
   ```

<img src="https://pic.imgdb.cn/item/64c4d4241ddac507cc641858.jpg" style="zoom:67%;" />

2. **通过以下函数建立单层决策树**

   伪代码如下：

   ```python
   将minError设置为无穷大
   对数据集中每一个特征：
       对每个步长：
           对每个不等号：
               建立一棵单层决策树并利用加权数据集对其进行测试
               if 错误率<minError:
                   当前单层决策树设为最佳单层决策树
   return 最佳单层决策树
   ```

   代码如下：

   ```python
   """测试是否有某个值小于或大于我们正在测试的阈值"""
   def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
       retArray=ones((shape(dataMatrix)[0],1))     #构建一个全1向量
       if threshIneq=='lt':   #小于阈值的分类为gt，大于阈值的分类为lt
           retArray[dataMatrix[:,dimen]<=threshVal]=-1.0   #<=阈值的置为-1
       else:
           retArray[dataMatrix[:, dimen] >threshVal] = -1.0  #>阈值的置为-1
       return retArray
    
   """遍历stumpClassify的所有可能输入值，并找到数据集上最佳的单层决策树"""
   def buildStump(dataArr,classLabels,D):
       dataMatrix=mat(dataArr)
       labelMat=mat(classLabels).T
       m,n=shape(dataMatrix)  #m个样本，n个特征
       numSteps=10.0   #用于在特征的所有可能值上进行遍历
       bestStump={}    #定义一个字典，用于存储给定权重向量D时所得到的的最佳单层决策树的相关信息
       bestClasEst=mat(zeros((m,1)))
       minError=inf    #最小错误率，初始化为无穷大
       for i in range(n):  #在所有特征上遍历（第一次循环），即按照第i个特征来划分类别
           rangeMin=dataMatrix[:,i].min()   #找出所有样本中在特征i上的最小值
           rangeMax=dataMatrix[:,i].max()   #找出所有样本中在特征i上的最大值
           stepSize=(rangeMax-rangeMin)/numSteps  #确定步长（步长是用于比较特征值和阈值的）
           for j in range(-1,int(numSteps)+1):   #在特征i的所有可能取值上遍历，每次讲特征值加j*stepSize的大小（第二层循环）
               for inequal in ['lt','gt']:
                   threshVal=(rangeMin+float(j)*stepSize)   #根据步长设定阈值
                   predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)  #对样本进行预测分类
                   errArr=mat(ones((m,1)))   #构建一个错误统计列向量
                   errArr[predictedVals==labelMat]=0  #如果预测和真实标签一致置为0，不一致置为1
                   weightedError=D.T*errArr    #根据错误向量*权重向量得到数值weightedError
                   if weightedError<minError:  #更新最小错误率
                       minError=weightedError
                       bestClasEst=predictedVals.copy()  #当前最好的分类预测向量
                       bestStump['dim']=i   #当前最好的分类特征
                       bestStump['thresh']=threshVal   #当前的分类阈值
                       bestStump['ineq']=inequal   #分类结果
       return bestStump,minError,bestClasEst
    
    
    
   if __name__=='__main__':
       dataMatrix, classLabels = loadSimpData()
       D=mat(ones((5,1))/5)
       bestStump, minError, bestClasEst=buildStump(dataMatrix,classLabels,D)
       print("当前最好的分类方法（dim:选取的特征，thresh：阈值，ineq:分类名称）:{}".format(bestStump))
       print("最小错误率：{}".format(minError))
       print("当前最好分类的预测值:{}".format(bestClasEst))
   ```

   运行结果：

   ```python
   当前最好的分类方法（dim:选取的特征，thresh：阈值，ineq:分类名称）:{'dim': 0, 'thresh': 1.3, 'ineq': 'lt'}
   最小错误率：[[0.2]]
   当前最好分类的预测值:[[-1.][ 1.][-1.][-1.][ 1.]]
   ```

---

### **完整的AdaBoost学习器构建：**

利用上述代码构建的单层决策树来实现AdaBoost学习器，伪代码如下：

```python
对每次迭代: 
    利用buildStump()找到最佳的单层决策树
    将最佳单层决策树加入到单层决策树数组
    计算alpha
    计算新的权重向量D
    更新累计类别估计值
    如果错误率等于0.0，退出循环
```

其中D：训练数据中的每个样本拥有一个权重，这些权重构成向量D，初始时所有权重相等，分类错的样本下一次分类时权重增加，分类正确的权重在下次分类时会降低。

```python
"""基于单层决策树的AdaBoost训练过程"""
def adaBoostTrainDS(dataArr,classLabels,numIt=40):   #数据集，类别标签以及迭代次数（若小于迭代次数时错误率已经为0则直接退出）
    weakClassArr=[]  #弱分类器列表
    m=shape(dataArr)[0]  #m为样本数
    D=mat(ones((m,1))/m)  #初始化每个样本的权值，初始化所有权重均为1/m
    aggClassEst=mat(zeros((m,1)))   #向量aggClassEst，记录每个数据点的类别估计累计值
    for i in range(numIt):   #迭代
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)  #建立一个当前性能最优的单层决策树
        weakClassArr.append(bestStump)     #将当前选出的决策树信息存入列表
        print("各样本权值向量D为:{}".format(D.T))   #打印此时的各样本权重
        alpha=float(0.5*log((1.0-error)/max(error,1e-16)))    #计算alpha
        bestStump['alpha']=alpha   #将测试的alpha加入单层决策树信息中
        weakClassArr.append(bestStump)
        print("分类预测结果向量classEst:{}".format(classEst.T))
        #以下三步均为更新D的步骤
        expon=multiply(-1*alpha*mat(classLabels).T,classEst)    
        D=multiply(D,exp(expon))
        D=D/D.sum()
        aggClassEst+=alpha*classEst
        print("aggClassEst:{}".format(aggClassEst.T))
        aggErrors=multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))  #通过aggClassEst来计算总的错误率
        errorRate=aggErrors.sum()/m
        print("总错误率:{}".format(errorRate))
        if errorRate==0.0:   #当总的错误率为0.0时退出迭代
            break
    return weakClassArr

"""基于AdaBoost的分类函数"""
def adaClassify(datToClass,classifierArr):   #datToClass为待分类样本，classifierArr是训练出来的弱分类器集合
    dataMatrix=mat(datToClass)
    m=shape(dataMatrix)[0]   #要分类的样本数
    aggClassEst=mat(zeros((m,1)))
    for i in range(len(classifierArr)):   #遍历所有的弱分类器，通过stumpClassify对每个分类器得到一个类别的估计值
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst  #然后得到各弱分类器的加权和，就是预测值
        print(aggClassEst)
    return sign(aggClassEst)

if __name__=='__main__':
    dataMatrix, classLabels = loadSimpData()
    classifierArr=adaBoostTrainDS(dataMatrix,classLabels,30)
    result=adaClassify([[0,0],[5,5]],classifierArr)
```

运行结果：

```python
各样本权值向量D为:[[0.2 0.2 0.2 0.2 0.2]]
分类预测结果向量classEst:[[-1.  1. -1. -1.  1.]]
aggClassEst:[[-0.69314718  0.69314718 -0.69314718 -0.69314718  0.69314718]]
总错误率:0.2

各样本权值向量D为:[[0.5   0.125 0.125 0.125 0.125]]
分类预测结果向量classEst:[[ 1.  1. -1. -1. -1.]]
aggClassEst:[[ 0.27980789  1.66610226 -1.66610226 -1.66610226 -0.27980789]]
总错误率:0.2

各样本权值向量D为:[[0.28571429 0.07142857 0.07142857 0.07142857 0.5       ]]
分类预测结果向量classEst:[[1. 1. 1. 1. 1.]]
aggClassEst:[[ 1.17568763  2.56198199 -0.77022252 -0.77022252  0.61607184]]
总错误率:0.0
[[-0.69314718]
 [ 0.69314718]]
[[-1.38629436]
 [ 1.38629436]]
[[-2.35924944]
 [ 2.35924944]]
[[-3.33220451]
 [ 3.33220451]]
[[-4.22808424]
 [ 4.22808424]]
[[-5.12396398]
 [ 5.12396398]]
```

