import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 引入sklearn里的数据集
from sklearn.datasets import load_iris  # 鸢尾花的数据
from sklearn.model_selection import train_test_split  # 切分数据集为训练集和测试集
from sklearn.metrics import accuracy_score  # 计算分类预测的准确率

iris = load_iris()
# df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# df['class'] = iris.target
# df['class'] = df['class'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})

x = iris.data
# 将一维数组转成列向量
y = iris.target.reshape(-1, 1)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=35, stratify=y)  # 按y数据的分布情况等比例分配

# 核心算法实现

# 距离函数定义
# a是个矩阵假设是105 * 4的矩阵  b只能是个向量 且必须是 1 * 4 的矩阵
# 曼哈顿距离
def l1_distance(a, b):
    return np.sum(np.abs(a - b), axis=1)  # axis=1表示将b与a的每一行相减的结果相加  并保存成列  若没有这个参数表示将 所有行加一起
# 欧式距离
def l2_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


# 分类器
class kNN(object):
    # 定义一个初始化方法
    def __init__(self, n_neighbor=1, dist_func=l1_distance):
        self.n_neighbor = n_neighbor
        self.dist_func = dist_func

    # 训练核心分类算法
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    # 模型预测方法
    def predict(self, x):
        # 初始化预测分类数组
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)

        # 预测
        # 遍历输入的数据 取出每一行的数据和其序号 计算每一行数据分类
        enum_x = enumerate(x)
        for i, x_test in enumerate(x):
            # x_test跟所有训练数据的距离
            distances = self.dist_func(self.x_train, x_test)
            # 得到的距离进行排序 由小到大 取出索引值
            nn_index = np.argsort(distances)

            # 选取最近的k个点，保存其对应的分类类别 转换为一维数组
            nn_y = self.y_train[nn_index[:self.n_neighbor]].ravel()

            # 统计类别中出现频率最高的 赋给y_pred[i]
            y_pred[i] = np.argmax(np.bincount(nn_y))
            # 统计类别中出现频率最高的类别 就是当前序号的x[i] 的 y_pred[i]
        return y_pred

#测试
knn = kNN(n_neighbor=3)
#训练数据
knn.fit(x_train,y_train)

#预测
result_list = []
#根据不同的参数选取 做预测
for p in [1,2]:
    #距离函数
    knn.dist_func = l1_distance if p == 1 else l2_distance
    for k in range(1,10,2):
        #k值
        knn.n_neighbor = k
        y_pred = knn.predict(x_test)
        # 评估预测准确率
        accuracy = accuracy_score(y_test, y_pred)
        result_list.append([k,'l1_distance' if p ==1 else 'l2_distance',accuracy])

df = pd.DataFrame(result_list,columns=["k",'距离函数',"预测准确率"])

print(df)