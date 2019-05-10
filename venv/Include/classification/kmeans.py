# 无监督学习 k-means聚类算法 k均值
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#数据来源
from sklearn.datasets.samples_generator import make_blobs
#导入距离函数 默认欧式距离
from scipy.spatial.distance import cdist

#生成100条数据 分为6类 标准差0.6 标准差代表每类数据的离散情况 如果为0表示每类数据都是同一个点
x,y = make_blobs(n_samples=100,centers=6,random_state = 1234,cluster_std=0.6)
#绘制原始数据 散点图 生成的x的第一列为横坐标值 第二列为纵坐标值 按y分类为不同的颜色
plt.figure(figsize=(6,6))
plt.scatter(x[:,0],x[:,1],c=y)
plt.title("original")

#算法实现
class K_Means(object):
    #初始化 参数n_clusters=(K)  迭代次数max_iter 初始质心 centroids
    def __init__(self,n_clusters=2,max_iter=300,centroids=[]):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = np.array(centroids,dtype=np.float)

    # 训练模型 K_means聚类过程 传入原始数据
    def fit(self,data):
        #如果没有初始质心 随机选取data中的点作为质心
        if(self.centroids.shape==(0,)):
            #从data中随机生成0到data行数的6个整数 作为索引值
            self.centroids = data[np.random.randint(0,len(data),self.n_clusters),:]

        #迭代
        for i in range(self.max_iter):
            #1.计算距离矩阵 100*6的矩阵 每行有六个数据 分别代表这点距离其中一个质心点的距离
            distances = cdist(data,self.centroids)
            #2.对矩阵按距离由近到远排序 选取最近的质心点的类别 作为当前点的类别
            c_index = np.argmin(distances,axis=1) #取每行的最小值的索引

            #3.对每一类数据进行均值计算 更新质心点
            for i in range(self.n_clusters):
                #排除 没有出现在c_index的类别
                if i in c_index:
                    #选出所有类别是i的点  取data里坐标的均值 更新第i个质心
                    self.centroids[i] = np.mean(data[c_index == i],axis=0) #最后得到一行的数据 列数不变 即将所有列求均值

         #实现预测方法
    def predict(self,samples):
        #计算距离矩阵
        distances = cdist(samples,self.centroids)
        #选取距离最近的质心的类别 返回
        c_index = np.argmin(distances,axis=1)
        return c_index

#测试
def plotKMeans(x,y,centroids,subplot,title):
    #分配子图 121表示1行2列子图中的第一个
    plt.subplot(subplot)
    plt.scatter(x[:,0],x[:,1],c="r")
    #画出质心点
    plt.scatter(centroids[:,0],centroids[:,1],c=np.array(range(6)),s=100)
    plt.title(title)

kmeans = K_Means(n_clusters=6,max_iter=300,centroids=[[2,1],[2,2],[2,3],[2,4],[2,5],[2,6]])

#绘制初始图
plt.figure(figsize=(16,6))
plotKMeans(x,y,kmeans.centroids,121,'initial State')

#训练模型 并绘制训练结束后的图
kmeans.fit(x)
plotKMeans(x,y,kmeans.centroids,122,'final State')


# 预测新数据点的类别并绘制
x_new = np.array([[0,0],[10,7]])
y_pred = kmeans.predict(x_new)

print(y_pred)
print(kmeans.centroids)
plt.scatter(x_new[:,0],x_new[:,1],s=100,c='black')
plt.show()
