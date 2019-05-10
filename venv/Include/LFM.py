# LMF算法是矩阵降维中 求解损失函数最小值的梯度下降法。
# 与之对应的ALS(交替最小二乘法)同样也是求损失函数最小值的算法。较为复杂 但更精确

# 0、引入依赖
import numpy as np
import pandas as pd

# 1、数据准备

# 评分矩阵 R 6个用户对5个商品的评分
R = np.array([[4,0,2,0,1],
             [0,2,3,0,0],
             [1,0,2,4,0],
             [5,0,0,3,1],
             [0,0,1,5,1],
             [0,3,2,4,1]])
R.shape
"""
算法实现 
@输入参数：
R M*N的评分矩阵
K 隐特征向量个数
max_iter 最大迭代次数
alpha 步长
lamda 正则化系数  惩罚项的系数

@输出
将R分解之后的P Q
P 初始化用户特征矩阵M*K
Q 初始化物品特征矩阵N*K
cost 最终的损失函数值
"""

# 给定超参数
K = 3
max_iter = 50000
alpha = 0.0002
lamda = 0.004


# 核心算法
def LFM_gred_desc(lamda, R, max_iter, alpha=0.0001, K=2):
    # 基本维度参数定义
    M = len(R)
    N = len(R[0])

    # P Q 初始值 随机生成
    P = np.random.rand(M, K)
    Q = np.random.rand(N, K)
    Q = Q.T

    # 开始迭代
    for step in range(max_iter):
        # 对所有的用户u 物品i进行遍历 对应的特征向量Pu、Qi 梯度下降
        for u in range(M):
            for i in range(N):
                if R[u, i] > 0:
                    eui = np.dot(P[u, :], Q[:, i]) - R[u][i]
                    # 代入公式 按梯度下降算法更新当前的Pu，Qi
                    for k in range(K):
                        P[u, k] = P[u, k] - alpha * (2 * eui * Q[k, i] + 2 * lamda * P[u, k])
                        Q[k, i] = Q[k, i] - alpha * (2 * eui * P[u, k] + 2 * lamda * Q[k, i])
        # u,i 遍历完成 所有的特征向量更新完成 可以得到P Q 可以计算预测评分矩阵

        # 计算当前损失函数
        cost = 0

        for u in range(M):
            for i in range(N):
                if R[u, i] > 0:
                    eui = np.dot(P[u, :], Q[:, i]) - R[u][i]
                    cost += eui ** 2
                    # 加上正则化项
                    for k in range(K):
                        cost = cost + lamda * (P[u, k] ** 2 + Q[k, i] ** 2)
        if cost < 0.0001:
            break;

    return P, Q.T, cost


# 测试
P,Q,cost = LFM_gred_desc( lamda,R,max_iter ,alpha=0.0002,K=2)
print(P)
print(Q)
print(cost)

predR = P.dot(Q.T)
print(R)
print(predR)


