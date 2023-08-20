# -*- coding:utf-8 -*-
"""
@File    ：class.py
@Author  ：ZYJ
@Date    ：2023/8/7 15:48 
"""

from numpy import *

def loadDataSet():
    '''
    加载数据集
    :return: dataset
    '''
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

class AssociationRule(object):

    def __init__(self, minSupport=0.5, minConf=0.7):
        self.minSupport = minSupport
        self.minConf = minConf
        self.dataset = None

    def fit(self, dataset):
        self.dataset = dataset
        self.frequentItemsList, self.supportData = self.apriori(dataset)

    def _createC1(self, dataset):
        '''
        创建C1候选项集，C1是所有大小为1的候选项集的列表
        :return: C1
        '''
        # C1是所有大小为1的候选项集的列表
        C1 = []
        # 遍历数据集，逐个添加到C1中
        for record in dataset:
            for item in record:
                if not [item] in C1:
                    C1.append([item])
        C1.sort()
        # 使用不变集合存储C1内部的每个候选项集，那么就可以将其作为字典的Key，如果是list类型不能直接作为字典的Key
        return list(map(frozenset, C1))

    def _scanDataset(self, ck):
        '''
        扫描数据集，判断频繁项集
        :param ck: ck是所有大小为k的候选项集的列表
        :return: 符合条件的项集、每个项集的支持度
        '''
        # 存储项集的出现次数
        selectedSetCount = {}
        for record in self.dataset:  # 遍历每一条记录
            for candidateSet in ck:
                # 判断当前候选项集是不是当前记录的子集
                if candidateSet.issubset(record):
                    if candidateSet not in selectedSetCount:
                        selectedSetCount[candidateSet] = 1
                    else:
                        selectedSetCount[candidateSet] += 1
        # 计算总条目数
        numItems = float(len(self.dataset))
        # 存储符合条件的项集
        retList = []
        # 存储项集的支持度
        supportData = {}
        for key in selectedSetCount:
            # 计算支持度
            support = selectedSetCount[key] / numItems
            if support >= self.minSupport:
                retList.insert(0, key)
            supportData[key] = support
        return retList, supportData

    def _createCk(self, lastFrequentItems, k):
        '''
        根据k-1项的频繁项集列表生成k项的候选项集
        :param lastFrequentItems: k-1项的频繁项集
        :param k: 第k个项集
        :return: ck项集
        '''
        retList = []
        lenLk = len(lastFrequentItems)
        for i in range(lenLk):
            for j in range(i + 1, lenLk):
                # 因为新构建的ck项集，特征是任意一个k项集其中k-1项都必须存在于lastCk中
                # 通过以下判断，能筛选出那些符合要求的k-1项
                L1 = list(lastFrequentItems[i])[:k - 2]
                L2 = list(lastFrequentItems[j])[:k - 2]
                L1.sort()
                L2.sort()
                if L1 == L2:
                    retList.append(lastFrequentItems[i] | lastFrequentItems[j])
        return retList

    def apriori(self, dataset):
        C1 = self._createC1(dataset)
        k1FrequentItems, supportData = self._scanDataset(C1)
        frequentItemsList = [k1FrequentItems]
        # 应为k=1的频繁项集已经找到，因此从k=2继续
        k = 2
        while True:
            # 根据k-1的频繁项集，创建k候选集，
            # k-1-1是因为列表下表从0开始
            ck = self._createCk(frequentItemsList[k - 1 - 1], k)
            # 再次扫描数据集，找出新的k项频繁项集
            newFrequentItems, supK = self._scanDataset(ck)
            # 更新项集的支持度
            supportData.update(supK)
            # 如果无法生成新的频繁项集，那么推出循环
            if len(newFrequentItems) == 0:
                break
            # 存储所有的频繁项集
            frequentItemsList.append(newFrequentItems)
            k += 1
        return frequentItemsList, supportData

    def generateRules(self):
        # 存储关联规则
        ruleList = []
        # 从含有2项item的频繁项集开始遍历，计算两两的置信度
        for i in range(1, len(self.frequentItemsList)):
            # 遍历每一阶段的频繁项集
            for frequentItem in self.frequentItemsList[i]:
                subItems = [frozenset([item]) for item in frequentItem]
                if (i == 1):
                    # 先计算2项item的频繁项集的置信度，并将关联规则存储到ruleList
                    self._calculateConfidence(frequentItem, subItems, self.supportData, ruleList)
                else:
                    # 然后使用递归依次计算3到k项item频繁项集之间两两的置信度，并提取关联规则
                    self._rulesFromRecursive(frequentItem, subItems, self.supportData, ruleList)
        return ruleList

    def _calculateConfidence(self, frequentItem, subItems, supportData, ruleList):
        # 存储符合最小置信度阈值的item
        retList = []
        for subItem in subItems:
            # 计算置信度
            conf = supportData[frequentItem] / supportData[frequentItem - subItem]
            if conf >= self.minConf:
                print("Rule：", frequentItem - subItem, '-->', subItem, 'confidence:', conf)
                ruleList.append((frequentItem - subItem, subItem, conf))
                retList.append(subItem)
        return retList

    def _rulesFromRecursive(self, frequentItem, subItems, supportData, ruleList):
        m = len(subItems[0])  # 判断当前子项集的长度
        if (len(frequentItem) > (m + 1)):
            # 根据子项集得出CK候选集
            ck = self._createCk(subItems, m + 1)
            # 根据候选集再筛选出符合最小置信度的item集合
            newItems = self._calculateConfidence(frequentItem, ck, supportData, ruleList)
            # 如果符合要求的item至少有2个，那么继续递归
            if (len(newItems) > 1):
                self._rulesFromRecursive(frequentItem, newItems, supportData, ruleList)


if __name__ == '__main__':
    from pprint import pprint
    dataset = loadDataSet()
    ar = AssociationRule()
    ar.fit(dataset)
    pprint(ar.generateRules())
