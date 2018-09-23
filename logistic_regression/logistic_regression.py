import numpy as np
from sklearn.linear_model import LogisticRegression  # 导入逻辑回归模型
clf = LogisticRegression(penalty='l2',solver='liblinear')
# LogisticRegression参数介绍：
# penalty表示L2的正则化
# solver：优化算法选择参数liblinear：开源的liblinear库，使用了坐标轴下降法来迭代优化损失函数(默认)
# solver：优化算法选择参数newton-cg：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数
# solver：优化算法选择参数sag：随机平均梯度下降
# class_weight：类型权重参数
# sample_weight：样本权重参数

print (clf)
train_feature = np.array([[1,1],[2,1],[2,2]])
label = np.array([0,0,1])
clf.fit(train_feature,label)
predict_feature = np.array([[5,5]])
predict_result = clf.predict(predict_feature)
print(predict_result)