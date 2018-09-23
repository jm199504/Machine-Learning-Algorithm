import numpy as np # 快速操作结构数组的工具
import matplotlib.pyplot as plt  # 可视化工具
from sklearn.linear_model import Ridge,RidgeCV   # Ridge岭回归,RidgeCV带有广义交叉验证的岭回归

# 创建样本数据集
data = list()
for i in range(30):
    data.append([i+np.random.rand()*3,3.5*i+np.random.rand()*3])

# 转换成矩阵
dataMat = np.array(data)
X = dataMat[:,0:1]
y = dataMat[:,1]

# 一般Ridge(alpha：正则化强度)
model = Ridge(alpha=0.5)
# 通过RidgeCV可以设置多个参数值，算法使用交叉验证获取最佳参数值
model = RidgeCV(alphas=[0.1, 1.0, 10.0])
model.fit(X, y)   # 线性回归建模
print('系数:\n',model.coef_)
print('线性回归模型详情:\n',model)
# print('交叉验证最佳alpha值',model.alpha_)  # Ridge()无这个方法，只有RidgeCV算法有
# 使用模型预测
predicted = model.predict(X)

# 绘制散点图 参数：x横轴 y纵轴
plt.scatter(X, y, marker='x')
plt.plot(X, predicted,c='r')

# 绘制x轴和y轴坐标
plt.xlabel("x")
plt.ylabel("y")

# 显示图形
plt.show()