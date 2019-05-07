import numpy as np
import matplotlib.pyplot as plt

# 1.导入数据
points = np.genfromtxt("data.csv", delimiter=",")

# 提取points中的两列数据 记为x，y
x = points[:, 0]
y = points[:, 1]
# 使用plt绘制原始数据的散点图
plt.scatter(x, y)
# plt.show()


# 2.定义损失函数 均平方
def compute_cost(w, b, x, y):
    total_cost = 0
    m = len(x)
    for i in range(m):
        x_i = x[i]
        y_i = y[i]
        total_cost += (y_i - w * x_i - b) ** 2
    return total_cost / m

#导入线性回归的算法库
from sklearn.linear_model import LinearRegression
lr = LinearRegression() #构建模型

#由一维数组转换为二维数组  第一个参数表示行数 -1指的是随意，  第二个参数表示列数 1列
x_new = x.reshape(-1,1)
y_new = y.reshape(-1,1)

lr.fit(x_new,y_new)

#从训练好的模型中提取系数和截距
w = lr.coef_ #系数
b = lr.intercept_ #截距

print("w is: " ,w)
print("b is: " , b)
cost = compute_cost(w,b,x,y)
print("cost is :" , cost)

pred_y = w[0,0] * x + b[0]
plt.plot(x,pred_y,c="r")
plt.show()
