import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

# sklearn自带的iris数据集，包含四个特征，三种类别
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

# 初始化模型，并限制树的最大深度4
clf = DecisionTreeClassifier(max_depth=4)
# 训练拟合模型
clf.fit(X, y)
print(len(y))
print([2]*150)

# 选取前两个特征进行分类绘图
# 获取最值后确定边界范围
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# np.meshgrid进行关联拓展(不官方)，即meshgrid的两个数组互相影响，生成同样shape的array

# np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等，类似于pandas中的concat()。
# np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])# ravel()将多维数组降为一维
Z = Z.reshape(xx.shape)
a = np.c_[xx.ravel(), yy.ravel()]
# 绘制类别间的等高线
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
plt.show()