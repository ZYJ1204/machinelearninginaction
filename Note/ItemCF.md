---
title: 基于物品的协同过滤算法（ItemCF）
tags:
  - 推荐算法
cover: https://pic.imgdb.cn/item/649ce61e1ddac507ccb10aca.png
toc: True
date: 2023/7/26
---

## 基本思想

基于物品的协同过滤（ItemCF）：

- 预先根据所有用户的历史行为数据，计算物品之间的相似性。
- 然后，把与用户喜欢的物品相类似的物品推荐给用户。

举例来说，如果用户 1 喜欢物品 A ，而物品 A 和 C 非常相似，则可以将物品 C 推荐给用户1。ItemCF算法并不利用物品的内容属性计算物品之间的相似度， 主要通过分析用户的行为记录计算物品之间的相似度， 该算法认为， 物品 A 和物品 C 具有很大的相似度是因为喜欢物品 A 的用户极可能喜欢物品 C。

<img src="https://pic.imgdb.cn/item/64c0d8641ddac507ccecaacf.jpg" style="zoom:67%;" />

## 计算过程

<img src="https://pic.imgdb.cn/item/64c0992c1ddac507cc78d316.jpg" style="zoom:67%;" />

如果想知道 Alice 对物品5打多少分， 基于物品的协同过滤算法会这么做：

- 首先计算一下物品5和物品1， 2， 3， 4之间的相似性。
- 在Alice找出与物品 5 最相近的 n 个物品。
- 根据 Alice 对最相近的 n 个物品的打分去计算对物品 5 的打分情况。

**手动计算：**

1. 手动计算物品之间的相似度

   > 物品向量: 物品1(3,4,3,1),物品2(1,3,3,5),物品3(2,4,1,5),物品4(3,3,5,2),物品5(3,5,41)
   >
   > - 下面计算物品 5 和物品 1 之间的余弦相似性:
   >
   >   $$sim⁡( 物品1, 物品5 )=cosine⁡( 物品1, 物品5 )=\frac{9+20+12+1}{sqrt⁡(9+16+9+1)∗sqrt⁡(9+25+16+1)}$$
   >
   > - 皮尔逊相关系数类似。

2. 基于 `sklearn` 计算物品之间的皮尔逊相关系数

<img src="https://pic.imgdb.cn/item/64c0da061ddac507ccf01ff6.jpg" style="zoom:67%;" />

3. 根据皮尔逊相关系数，可以找到与物品5最相似的2个物品是 item1 和 item4，下面基于上面的公式计算最终得分：

   $$P_{Alice,物品5}=\overline{R}_{物品5}+\frac{∑^2_{k=1}(w_{物品5,物品k}(R_{Alice,物品k}-\overline{R}_{物品k}))}{∑^2_{k=1}(w_{物品K,物品5})}=\frac{13}{4}+\frac{0.97∗(5−3.2)+0.58∗(4−3.4)}{0.97+0.58}=4.6$$

## ItemCF编程实现

1. 构建物品-用户的评分矩阵

```python
import numpy as np
import pandas as pd


def loadData():
    items = {'A': {'Alice': 5.0, 'user1': 3.0, 'user2': 4.0, 'user3': 3.0, 'user4': 1.0},
             'B': {'Alice': 3.0, 'user1': 1.0, 'user2': 3.0, 'user3': 3.0, 'user4': 5.0},
             'C': {'Alice': 4.0, 'user1': 2.0, 'user2': 4.0, 'user3': 1.0, 'user4': 5.0},
             'D': {'Alice': 4.0, 'user1': 3.0, 'user2': 3.0, 'user3': 5.0, 'user4': 2.0},
             'E': {'user1': 3.0, 'user2': 5.0, 'user3': 4.0, 'user4': 1.0}
             }
    return items
```

2. 计算物品间的相似度矩阵

```python
item_data = loadData()

similarity_matrix = pd.DataFrame(
    np.identity(len(item_data)),
    index=item_data.keys(),
    columns=item_data.keys(),
)

# 遍历每条物品-用户评分数据
for i1, users1 in item_data.items():
    for i2, users2 in item_data.items():
        if i1 == i2:
            continue
        vec1, vec2 = [], []
        for user, rating1 in users1.items():
            rating2 = users2.get(user, -1)
            if rating2 == -1:
                continue
            vec1.append(rating1)
            vec2.append(rating2)
        similarity_matrix[i1][i2] = np.corrcoef(vec1, vec2)[0][1]

print(similarity_matrix)
```

```
          A         B         C         D         E
A  1.000000 -0.476731 -0.123091  0.532181  0.969458
B -0.476731  1.000000  0.645497 -0.310087 -0.478091
C -0.123091  0.645497  1.000000 -0.720577 -0.427618
D  0.532181 -0.310087 -0.720577  1.000000  0.581675
E  0.969458 -0.478091 -0.427618  0.581675  1.000000
```

3. 从 Alice 购买过的物品中，选出与物品 `E` 最相似的 `num` 件物品。

```python
target_user = 'Alice'
target_item = 'E'
num = 2

sim_items = []
sim_items_list = similarity_matrix[target_item].sort_values(ascending=False).index.tolist()
for item in sim_items_list:
    # 如果target_user对物品item评分过
    if target_user in item_data[item]:
        sim_items.append(item)
    if len(sim_items) == num:
        break
print(f'与物品{target_item}最相似的{num}个物品为：{sim_items}')
# 与物品E最相似的2个物品为：['A', 'D']
```

4. 预测用户 Alice 对物品 `E` 的评分

```python
target_user_mean_rating = np.mean(list(item_data[target_item].values()))
weighted_scores = 0.
corr_values_sum = 0.

target_item = 'E'
for item in sim_items:
    corr_value = similarity_matrix[target_item][item]
    user_mean_rating = np.mean(list(item_data[item].values()))

    weighted_scores += corr_value * (item_data[item][target_user] - user_mean_rating)
    corr_values_sum += corr_value

target_item_pred = target_user_mean_rating + weighted_scores / corr_values_sum
print(f'用户{target_user}对物品{target_item}的预测评分为：{target_item_pred}')
# 用户 Alice 对物品E的预测评分为：4.6
```

## 协同过滤算法的权重改进

- base 公式

  $$w_{ij}=\frac{|N(i)⋂N(j)|}{N(i)}$$

  - 该公式表示同时喜好物品 *i* 和物品*j* 的用户数，占喜爱物品 *i* 的比例。
  - 缺点：若物品 *j* 为热门物品，那么它与任何物品的相似度都很高。

- 对热门物品进行惩罚

  $$w_{ij}=\frac{|N(i)⋂N(j)|}{\sqrt{|N(i)||N(j)|}}$$

  - 根据 base 公式在的问题，对物品 *j* 进行打压。打压的出发点很简单，就是在分母再除以一个物品 *j* 被购买的数量。

  - 此时，若物品 *j* 为热门物品，那么对应的*N*(*j*) 也会很大，受到的惩罚更多。

- 控制对热门物品的惩罚力度

  $$w_{ij}=\frac{|N(i)⋂N(j)|}{|N(i)|^{1-α}|N(j)|^α}$$
  - 除了第二点提到的办法，在计算物品之间相似度时可以对热门物品进行惩罚外。
  - 可以在此基础上，进一步引入参数 *α* ，这样可以通过控制参数 *α*来决定对热门物品的惩罚力度。

- 对活跃用户的惩罚

  - 在计算物品之间的相似度时，可以进一步将用户的活跃度考虑进来。
  
  $$w_{ij}=\frac{\sum_{u∈N(i)⋂N(j)}\frac{1}{log1+|N(u)|}}{|N(i)|^{1-α}|N(j)|^α}$$
  - 对于异常活跃的用户，在计算物品之间的相似度时，他的贡献应该小于非活跃用户。

## 协同过滤算法的问题分析

协同过滤算法存在的问题之一就是泛化能力弱：

- 即协同过滤无法将两个物品相似的信息推广到其他物品的相似性上。
- 导致的问题是**热门物品具有很强的头部效应， 容易跟大量物品产生相似， 而尾部物品由于特征向量稀疏， 导致很少被推荐**。

比如下面这个例子：

<img src="https://pic.imgdb.cn/item/64c0de6c1ddac507ccfa1899.jpg" style="zoom:67%;" />

- 左边矩阵中，A*,*B*,*C*,*D 表示的是物品。
- 可以看出，*D* 是一件热门物品，其与 A*、*B*、*C 的相似度比较大。因此，推荐系统更可能将 *D* 推荐给用过A*、*B*、*C 的用户。
- 但是，推荐系统无法找出A*,*B*,*C* 之间相似性的原因是交互数据太稀疏， 缺乏相似性计算的直接数据。

所以这就是协同过滤的天然缺陷：**推荐系统头部效应明显， 处理稀疏向量的能力弱**。

为了解决这个问题， 同时增加模型的泛化能力。2006年，**矩阵分解技术(Matrix Factorization, MF**)被提出：

- 该方法在协同过滤共现矩阵的基础上， 使用更稠密的隐向量表示用户和物品， 挖掘用户和物品的隐含兴趣和隐含特征。
- 在一定程度上弥补协同过滤模型处理稀疏矩阵能力不足的问题。

## 思考

1. **什么时候使用UserCF，什么时候使用ItemCF？为什么？**

> （1）UserCF
>
> - 由于是基于用户相似度进行推荐， 所以具备更强的社交特性， 这样的特点非常适于**用户少， 物品多， 时效性较强的场合**。
>   - 比如新闻推荐场景， 因为新闻本身兴趣点分散， 相比用户对不同新闻的兴趣偏好， 新闻的及时性，热点性往往更加重要， 所以正好适用于发现热点，跟踪热点的趋势。
>   - 另外还具有推荐新信息的能力， 更有可能发现惊喜, 因为看的是人与人的相似性, 推出来的结果可能更有惊喜，可以发现用户潜在但自己尚未察觉的兴趣爱好。
>
> （2）ItemCF
>
> - 这个更适用于兴趣变化较为稳定的应用， 更接近于个性化的推荐， 适合**物品少，用户多，用户兴趣固定持久， 物品更新速度不是太快的场合**。
> - 比如推荐艺术品， 音乐， 电影。

---

本文参考：

[ItemCF (datawhalechina.github.io)](https://datawhalechina.github.io/fun-rec/#/ch02/ch2.1/ch2.1.1/itemcf?id=协同过滤算法的权重改进)