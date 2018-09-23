from sklearn.neighbors import KNeighborsClassifier
import numpy as np

train_x = np.array([[1,101],[5,89],[108,5],[115,8],[117,4]])
train_y = ['爱情片','爱情片','动作片','动作片','动作片']
test_x = np.array([[100,18],[1,99]])
# KNeighborsClassifier参数介绍
# n_neighbors：knn算法中指定以最近的几个最近邻样本具有投票权，默认参数为5
# weights：给特征赋予权重，'uniform'表示等比重投票，'distance'表示按距离反比投票，[callable]表示自己定义的一个函数
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_x, train_y)
predict_y = knn.predict(test_x)
neighborpoint=knn.kneighbors(test_x,n_neighbors=3,return_distance=False)
print(predict_y)
print(neighborpoint)