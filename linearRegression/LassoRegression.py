import numpy as np # 快速操作结构数组的工具
import matplotlib.pyplot as plt  # 可视化工具
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV   # Lasso回归,LassoCV交叉验证实现alpha的选取，LassoLarsCV基于最小角回归交叉验证实现alpha的选取

# 创建样本数据集
data = list()
for i in range(30):
    data.append([i+np.random.rand()*3,3.5*i+np.random.rand()*3])

#生成矩阵
dataMat = np.array(data)
X = dataMat[:,0:1]
y = dataMat[:,1]

model = Lasso(alpha=0.01)  # 调节alpha可以实现对拟合的程度
model = LassoCV()  # LassoCV自动调节alpha可以实现选择最佳的alpha。
model = LassoLarsCV()  # LassoLarsCV自动调节alpha可以实现选择最佳的alpha
model.fit(X, y)   # 线性回归建模
print('系数:\n',model.coef_)
print('线性回归模型详情:\n',model)
# print('最佳的alpha：',model.alpha_)  # 只有在使用LassoCV、LassoLarsCV时才有效
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