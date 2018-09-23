# coding=utf-8
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

x_train,y_train = list(),list()
for i in range(30):
    x_train.append([np.random.uniform(55,95),np.random.uniform(2,15)])
    y_train.append('生物1')
for i in range(30):
    x_train.append([np.random.uniform(10,25),np.random.uniform(35,90)])
    y_train.append('生物2')
for i in range(30):
    x_train.append([np.random.uniform(50,80),np.random.uniform(35,90)])
    y_train.append('生物3')

# 设置KMEANS模型簇的数量
clf = KMeans(n_clusters=3)
# 训练模型
y_pred = clf.fit_predict(x_train)

# 输出模型的完整参数
print(clf)
# 输出聚类预测结果
print(y_pred)

# 获取特征1、特征2
x = [n[0] for n in x_train]
y = [n[1] for n in x_train]
# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 绘制散点图
plt.scatter(x, y, c=y_pred, marker='o')
# 绘制标题
plt.title("k-mean栗子")
# 绘制x轴和y轴坐标
plt.xlabel("耳朵")
plt.ylabel("肤色")
# 显示图形
plt.show()