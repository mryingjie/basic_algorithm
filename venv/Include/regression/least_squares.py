import numpy as np
import matplotlib.pyplot as plt

# 1.导入数据
points = np.genfromtxt("data.csv",delimiter=",")

#提取points中的两列数据 记为x，y
x = points[:,0]
y = points[:,1]
#使用plt绘制原始数据的散点图
plt.scatter(x,y)
# plt.show()

# 2.定义损失函数 均平方
def compute_cost(w,b,x,y):
    total_cost = 0
    m = len(x)
    for i in range(m):
        x_i = x[i]
        y_i = y[i]
        total_cost += (y_i - w * x_i - b) ** 2
    return total_cost / m

# 3.定义算法拟合函数
# 3.1先定义求平均数的函数
def average(data):
    sum = 0
    for i in range(len(data)):
        sum += data[i]
    return sum / len(data)

#定义核心拟合函数
def fit(x,y):
    m = len(x)
    x_bar = average(x)
    sum_yx = 0
    sum_x2 = 0
    sum_delta = 0

    for i in range(m):
        x_i = x[i]
        y_i = y[i]
        sum_yx += y_i * (x_i - x_bar)
        sum_x2 += x_i ** 2

    # 根据公式计算
    w = sum_yx / (sum_x2 - m * (x_bar ** 2))

    for i in range(m):
        x_i = x[i]
        y_i = y[i]
        sum_delta += (y_i - w * x_i)

    b = sum_delta/m
    return w,b

# 4.测试
w,b = fit(x,y)
print("w is: " ,w)
print("b is : " , b)
cost = compute_cost(w,b,x,y)
print("cost is :" , cost)

#绘制回归曲线
pred_y = w * x + b
plt.plot(x,pred_y,c="r")

plt.show()