import numpy as np
import matplotlib.pyplot as plt

# 1.导入数据
points = np.genfromtxt("data.csv", delimiter=",")

# 提取points中的两列数据 记为x，y
x = points[:, 0]
y = points[:, 1]
# 使用plt绘制原始数据的散点图
# plt.scatter(x, y)
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


# 3.定义模型的超参数
alpha = 0.0001  # 步长
initial_w = 0
initial_b = 0
# 迭代次数
num_iter = 10


# 4.定义核心梯度下降算法函数
def grad_desc(points, initial_w, initial_b, alpha, num_iter):
    x = points[:, 0]
    y = points[:, 1]
    w = initial_w
    b = initial_b
    cost_list = []
    # 定义一个列表 保存所有的损失函数值 用来显示损失函数根据每次计算出来的w，b得到的损失值下降的过程
    for i in range(num_iter):
        cost_list.append(compute_cost(w, b, x, y))
        w, b = step_gred_desc(w,b,alpha, points)
    return [w, b, cost_list]

#迭代算法 迭代计算更新w和b
def step_gred_desc(w, b, alpha, points):
    sum_gred_w = 0
    sum_gred_b = 0
    m = len(points)

    # 用公式求当前梯度
    for i in range(m):
        x = points[i,0]
        y = points[i,1]
        sum_gred_w += ( w * x + b - y) * x
        sum_gred_b += ( w * x + b - y)
    gred_w = 2/m * sum_gred_w
    gred_b = 2/m * sum_gred_b

    #梯度下降 更新当前的w和b
    w = w - alpha * gred_w
    b = b - alpha * gred_b

    return w,b

# 5.运行梯度下降算法 计算最优的w和b
w,b,cost_list = grad_desc(points,initial_w,initial_b,alpha,num_iter)

print("w is: " ,w)
print("b is: ",b)

# 绘制损失函数下降折线图
plt.plot(cost_list)
plt.show()

# 绘制最终的拟合模型函数
pred_y = w * x + b
plt.scatter(x, y)
plt.plot(x,pred_y,c="r")
plt.show()