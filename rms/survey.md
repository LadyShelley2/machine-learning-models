# 推荐系统算法综述

从上个世纪90年代中期协同过滤的文章出现开始，推荐系统成为一个重要的研究领域。至今在工业界和学术界已有大量成果。在信息爆炸的时代，推荐系统能够有效地帮助user选取有效和有趣的内容，因此至今相关研究仍然火热。本篇文章对已有的推荐系统算法做了调研，并针对不同的方法判断其局限性，讨论扩展方法。

## 问题定义

推荐系统领域和认知科学，近似理论、信息检索等都有关系，但在1990s就成为了独立的学科。早期的推荐系统主要是评分，被形式化为根据user对其他item的打分来估算对其从来没有见过的item的打分，然后向user推荐所估计打分比较高的item。


给定一个集合。。。


通过已知的分数推算出未知的分数通常有两种思路，1.启发式算法，定义效用函数，根据经验验证其性能；2. 定义一种性能标准，例如平均平方误差，然后进行优化。

一旦能够预测到未知评分，我们可以选择评分最高的item推荐给user，也可以选择Top-N的item推荐给user或者Top-N的user给一件商品。

推荐系统的可以分成以下三类：

1. 基于内容的推荐（content-based recommendations）
2. 协同过滤
3. 混合策略（Hybrid approaches）:结合协同过滤和基于内容的推荐。

## Content-based Methods

Content-based Methods 是向user推荐与该user喜欢的item相似的item。例如像user推荐电影，Content-based methods 是试图找出user喜欢电影的共同点，例如导演、类型、演员等特征，然后向该user推荐拥有这些特征的电影。Content-based Methods只能向user推荐与user喜欢的item有高度相似性的item。

很多Content-based Methods 是基于文本信息推荐。根据user的个人资料(profile)可以容易地获得user的喜好和需求，user的个人资料可以通过user填写问卷表单等方式直接获取，也可以通过分析user的行为间接获取。对于item的资料(item profile)，一般是从item资料的内容中提取特征，item资料一般通过文本描述，内容特征可以通过关键字来描述。然后通过TF-IDF对关键字构成向量，再通过cosine similarity计算相似度，可以得到item之间的相似度，选取与user喜好相似度高的item进行推荐。


