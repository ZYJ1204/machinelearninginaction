---
title: 基于用户的协同过滤算法（UserCF）
tags:
  - 推荐算法
cover: https://pic.imgdb.cn/item/649ce61e1ddac507ccb10aca.png
toc: True
date: 2023/7/26
---

## 基本思想

基于用户的协同过滤（UserCF）：

- 例如，我们要对用户*A* 进行物品推荐，可以先找到和他有相似兴趣的其他用户。

- 然后，将共同兴趣用户喜欢的，但用户 *A* 未交互过的物品推荐给 *A*。

<img src="https://pic.imgdb.cn/item/64c098c61ddac507cc78282b.jpg" style="zoom:67%;" />

## 计算过程

以下图为例，给用户推荐物品的过程可以形象化为一个猜测用户对物品进行打分的任务，表格里面是5个用户对于5件物品的一个打分情况，就可以理解为用户对物品的喜欢程度。

<img src="https://pic.imgdb.cn/item/64c0992c1ddac507cc78d316.jpg" style="zoom:67%;" />

UserCF算法的两个步骤：

- 首先，根据前面的这些打分情况(或者说已有的用户向量）计算一下 Alice 和用户1， 2， 3， 4的相似程度， 找出与 Alice 最相似的 n 个用户。
- 根据这 n 个用户对物品 5 的评分情况和与 Alice 的相似程度会猜测出 Alice 对物品5的评分。如果评分比较高的话， 就把物品5推荐给用户 Alice， 否则不推荐。

**具体过程：**

1. 计算用户之间的相似度

   - 对于用户 Alice，通过相似度计算选取出与其最相近的 *N* 个用户。

2. 计算用户对新物品的评分预测

   - 常用的方式之一：利用目标用户与相似用户之间的相似度以及相似用户对物品的评分，来预测目标用户对候选物品的评分估计：$$R_{u,p}=\frac{∑_{s∈S}(w_{u,s}⋅R_{s,p})}{∑_{s∈S}(w_{u,s})}$$

     - 其中，权重 $w_{u,s}$ 是用户 *u* 和用户 *s* 的相似度，$R_{s,p}$是用户 *s* 对物品 *p* 的评分。

   - 另一种方式：考虑到用户评分的偏置，即有的用户喜欢打高分， 有的用户喜欢打低分的情况。公式如下：

     $$R_{u,p}=\overline{R}_u+\frac{∑_{s∈S}(w_{u,s}⋅(R_{s,p}-\overline{R}_s)}{∑_{s∈S}(w_{u,s})}$$

     - 其中，$\overline{R}_s$表示用户 *s* 对物品的历史平均评分。

3. 对用户进行物品推荐

   - 在获得用户 *u* 对不同物品的评价预测后， 最终的推荐列表根据预测评分进行排序得到。

**手动计算：**

<img src="https://pic.imgdb.cn/item/64c0992c1ddac507cc78d316.jpg" style="zoom:67%;" />

计算 Alice 与其他用户的相似度（基于皮尔逊相关系数）

1. 手动计算 Alice 与用户 1 之间的相似度：

> 用户向量 Alice:(5,3,4,4),user1:(3,1,2,3),user2:(4,3,4,3),user3:(3,3,1,5),user4:(1,5,5,2)
>
> - 计算Alice与user1的余弦相似性：
>
>   $$sim⁡(Alice,user1)=cos⁡(Alice, user1)=\frac{15+3+8+12}{sqrt⁡(25+9+16+16)∗sqrt⁡(9+1+4+9)}=0.975$$
>
> - 计算Alice与user1皮尔逊相关系数:
>
>   Alice\_ave=4	user1\_ave=2.25
>
> - 向量减去均值: Alice:(1,−1,0,0)    user1:(0.75,−1.25,−0.25,0.75)
>
> - 计算这俩新向量的余弦相似度和上面计算过程一致, 结果是 0.852 。
>
> - 可以看出，与 Alice 相似度最高的用户为用户1和用户2。

2. **根据相似度用户计算 Alice对物品5的最终得分** 

   > 用户1对物品5的评分是3，用户2对物品5的打分是5，那么根据上面的计算公式， 可以计算出 Alice 对物品5的最终得分是：
   >
   > $$P_{Alice,物品5}=\overline{R}_{Alice}+\frac{∑^2_{k=1}(w_{Alice,userk}(R_{userk,物品5}-\overline{R}_{userk}))}{∑^2_{k=1}(w_{Alice,userk})}=4+\frac{0.85∗(3−2.4)+0.7∗(5−3.8)}{0.85+0.7}=4.87$$
   >
   > 同样方式，可以计算用户 Alice 对其他物品的评分预测。

3. **根据用户评分对用户进行推荐**

- 根据 Alice 的打分对物品排个序从大到小：

  $$物品1>物品5>物品3=物品4>物品2$$

- 如果要向 Alice 推荐2款产品的话， 我们就可以推荐物品 1 和物品 5 给 Alice。

至此， 基于用户的协同过滤算法原理介绍完毕。

## UserCF编程实现

1. 建立数据表

```python
import numpy as np
import pandas as pd


def loadData():
    users = {'Alice': {'A': 5, 'B': 3, 'C': 4, 'D': 4},
             'user1': {'A': 3, 'B': 1, 'C': 2, 'D': 3, 'E': 3},
             'user2': {'A': 4, 'B': 3, 'C': 4, 'D': 3, 'E': 5},
             'user3': {'A': 3, 'B': 3, 'C': 1, 'D': 5, 'E': 4},
             'user4': {'A': 1, 'B': 5, 'C': 5, 'D': 2, 'E': 1}
             }
    return users
```

- 这里使用字典来建立用户-物品的交互表。
  - 字典`users`的键表示不同用户的名字，值为一个评分字典，评分字典的键值对表示某物品被当前用户的评分。
  - 由于现实场景中，用户对物品的评分比较稀疏。如果直接使用矩阵进行存储，会存在大量空缺值，故此处使用了字典。

2. 计算用户相似性矩阵

   由于训练数据中共包含 5 个用户，所以这里的用户相似度矩阵的维度也为 5×5。

```python
user_data = loadData()
similarity_matrix = pd.DataFrame(
    np.identity(len(user_data)),
    index=user_data.keys(),
    columns=user_data.keys(),
)

# 遍历每条用户-物品评分数据
for u1, items1 in user_data.items():
    for u2, items2 in user_data.items():
        if u1 == u2:
            continue
        vec1, vec2 = [], []
        for item, rating1 in items1.items():
            rating2 = items2.get(item, -1)
            if rating2 == -1:
                continue
            vec1.append(rating1)
            vec2.append(rating2)
        # 计算不同用户之间的皮尔逊相关系数
        similarity_matrix[u1][u2] = np.corrcoef(vec1, vec2)[0][1]

print(similarity_matrix)
```

```python
          Alice     user1     user2     user3     user4
Alice  1.000000  0.852803  0.707107  0.000000 -0.792118
user1  0.852803  1.000000  0.467707  0.489956 -0.900149
user2  0.707107  0.467707  1.000000 -0.161165 -0.466569
user3  0.000000  0.489956 -0.161165  1.000000 -0.641503
user4 -0.792118 -0.900149 -0.466569 -0.641503  1.000000
```

3. 计算与 Alice 最相似的 `num` 个用户

```python
target_user = 'Alice'
num = 2
# 由于最相似的用户为自己，去除本身
sim_users = similarity_matrix[target_user].sort_values(ascending=False)[1:num+1].index.tolist()
print(f'与用户{target_user}最相似的{num}个用户为：{sim_users}')

# 与用户 Alice 最相似的2个用户为：['user1', 'user2']
```

4. 预测用户 Alice 对物品 `E` 的评分

```python
weighted_scores = 0.
corr_values_sum = 0.

target_item = 'E'
# 基于皮尔逊相关系数预测用户评分
for user in sim_users:
    corr_value = similarity_matrix[target_user][user]
    user_mean_rating = np.mean(list(user_data[user].values()))

    weighted_scores += corr_value * (user_data[user][target_item] - user_mean_rating)
    corr_values_sum += corr_value

target_user_mean_rating = np.mean(list(user_data[target_user].values()))
target_item_pred = target_user_mean_rating + weighted_scores / corr_values_sum
print(f'用户{target_user}对物品{target_item}的预测评分为：{target_item_pred}')

# 用户 Alice 对物品E的预测评分为：4.871979899370592
```

## UserCF优缺点

User-based算法存在两个重大问题：

1. 数据稀疏性
   - 一个大型的电子商务推荐系统一般有非常多的物品，用户可能买的其中不到1%的物品，不同用户之间买的物品重叠性较低，导致算法无法找到一个用户的邻居，即偏好相似的用户。
   - 这导致UserCF不适用于那些正反馈获取较困难的应用场景(如酒店预订， 大件物品购买等低频应用)。

2. 算法扩展性
   - 基于用户的协同过滤需要维护用户相似度矩阵以便快速的找出 *TopN* 相似用户， 该矩阵的存储开销非常大，存储空间随着用户数量的增加而增加。
   - 故不适合用户数据量大的情况使用。

---

本文参考：

[UserCF (datawhalechina.github.io)](https://datawhalechina.github.io/fun-rec/#/ch02/ch2.1/ch2.1.1/usercf?id=基本思想)