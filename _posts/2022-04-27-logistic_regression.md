---
title: LR小结
layout: article
tags: 机器学习 LR
---
部分参考[刘建平博客](https://www.cnblogs.com/pinard/p/6029432.html)，此处只为备忘。
<!--more-->

## 名称来源：
名字里面有“回归”两个字，却不是一个回归算法，为什么有“回归”这个误导性的词呢？\
虽然逻辑回归是分类模型，但是它的原理里面却残留着回归模型的影子。
1. 回归模型：求出输出特征向量Y和输入样本矩阵X之间的线性关系系数$\theta$，满足$\mathbf{Y = X\theta}$。此时我们的$Y$是连续的，所以是回归模型。
2. 分类模型：$Y$是离散的，怎么办呢？一个可以想到的办法是，我们对于这个$Y$再做一次函数转换，变为$g(Y)$。如果我们令$g(Y)$的值在某个实数区间的时候是类别$A$，在另一个实数区间的时候是类别$B$，以此类推，就得到了一个分类模型。如果结果的类别只有两种，那么就是一个二元分类模型了。逻辑回归的出发点就是从这来的。

## 二元逻辑回归定义
sigmoid函数定义：$g(z) = \frac{1}{1+e^{-z}}$\
导数性质：$g^{'}(z) = g(z)(1-g(z))$

令${z = x\theta}$，$h_{\theta}(x) = \frac{1}{1+e^{-x\theta}}$
（$z$为回归模型的输出,$g$将回归模型输出转换为分类概率）

矩阵形式：$h_{\theta}(X) = \frac{1}{1+e^{-X\theta}}$

## 损失函数
$P(y=1|x,\theta ) = h_{\theta}(x)$\
$P(y=0|x,\theta ) = 1- h_{\theta}(x)$

得出$y$的概率分布函数表达式为：
$P(y|x,\theta ) = h_{\theta}(x)^y(1-h_{\theta}(x))^{1-y}$

用似然函数最大化来求解$\theta$，加对数方便求解，取反即为损失函数$J(\theta)$:\
$L(\theta) = \prod\limits_{i=1}^{m}(h_{\theta}(x^{(i)}))^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}$\
$J(\theta) = -lnL(\theta) = -\sum\limits_{i=1}^{m}(y^{(i)}log(h_{\theta}(x^{(i)}))+ (1-y^{(i)})log(1-h_{\theta}(x^{(i)})))$\
矩阵表示：\
$J(\theta) = -Y^Tlogh_{\theta}(X) - (E-Y)^T log(E-h_{\theta}(X))$\
其中E为全1向量。

## Loss 优化
常见的梯度下降法，坐标轴下降法，等牛顿法等均可，若有闭式解则更简单。\
$J(\theta)$对$\theta$求偏导：\
$\frac{\partial}{\partial\theta}J(\theta) = X^T[\frac{1}{h_{\theta}(X)}\odot h_{\theta}(X)\odot (E-h_{\theta}(X))\odot (-Y)] + X^T[\frac{1}{E-h_{\theta}(X)}\odot h_{\theta}(X)\odot (E-h_{\theta}(X))\odot (E-Y)]$\
化简得到：\
$\frac{\partial}{\partial\theta}J(\theta) = X^T(h_{\theta}(X) - Y )$\
因此在梯度下降法中每一步向量$\theta$的迭代公式如下：\
$\theta = \theta - \alpha X^T(h_{\theta}(X) - Y )$\
$\alpha$为梯度下降法的步长。


## 正则化-Regularization
为解决过拟合问题，考虑正则化，常见的有L1正则化和L2正则化。
1. L1正则化损失函数表达式：\
$J(\theta) = -Y^T\bullet logh_{\theta}(X) - (E-Y)^T\bullet log(E-h_{\theta}(X)) +\alpha ||\theta||_1$\
$||\theta||_1$为$\theta$的L1范数.\
逻辑回归的L1正则化loss优化方法常用的有坐标轴下降法和最小角回归法。
2. L2正则化loss：\
$J(\theta) = -Y^T\bullet logh_{\theta}(X) - (E-Y)^T\bullet log(E-h_{\theta}(X)) + \frac{1}{2}\alpha||\theta||_2^2$\
$||\theta||_2$为$\theta$的L2范数.\
逻辑回归的L2正则化损失函数的优化方法和普通的逻辑回归类似。

## 二元逻辑回归推广：多元逻辑回归
扩展到多分类，$y$取值扩展为K个，详见[刘建平博客](https://www.cnblogs.com/pinard/p/6029432.html)


## 参考
李航《统计学习方法》\
[逻辑回归原理小结](https://www.cnblogs.com/pinard/p/6029432.html)\
[scikit-learn 逻辑回归类库使用小结](https://www.cnblogs.com/pinard/p/6035872.html)
