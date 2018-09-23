import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
import time

a = time.time()

n_estimators = 400
learning_rate = 1
# 获取数据集 其中X特征数为10，12000个样本量
X, y = datasets.make_hastie_10_2(n_samples=12000, random_state=1)
# 划分训练集和测试集
X_train, y_train = X[:2000], y[:2000]
X_test, y_test = X[2000:], y[2000:]
# 初始化决策树1，深度1
dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
dt_stump.fit(X_train, y_train)
dt_stump_err = 1.0 - dt_stump.score(X_test, y_test)
# 初始化决策树2，深度9
dt = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)
dt.fit(X_train, y_train)
dt_err = 1.0 - dt.score(X_test, y_test)
# 初始化Adaboost，基于决策树模型
# 参数SAMME  ：对样本集分类效果作为弱学习器权重
ada_discrete = AdaBoostClassifier(base_estimator=dt_stump, learning_rate=learning_rate, n_estimators=n_estimators,
                                  algorithm='SAMME')
ada_discrete.fit(X_train, y_train)
# 初始化Adaboost，基于决策树模型
# 参数SAMME.R  ：对样本集分类的预测概率大小作为弱学习器权重
ada_real = AdaBoostClassifier(base_estimator=dt_stump, learning_rate=learning_rate, n_estimators=n_estimators,
                              algorithm='SAMME.R')
ada_real.fit(X_train, y_train)

fig = plt.figure()
ax = fig.add_subplot()
ax.plot([1, n_estimators], [dt_stump_err] * 2, 'k-', label='Decision Stump Error')
ax.plot([1, n_estimators], [dt_err] * 2, 'k--', label='Decision Tree Error')

ada_discrete_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_test)):
    ada_discrete_err[i] = zero_one_loss(y_pred, y_test)  ######zero_one_loss
ada_discrete_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_discrete.staged_predict(X_train)):
    ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)

ada_real_err = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_test)):
    ada_real_err[i] = zero_one_loss(y_pred, y_test)
ada_real_err_train = np.zeros((n_estimators,))
for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
    ada_discrete_err_train[i] = zero_one_loss(y_pred, y_train)

ax.plot(np.arange(n_estimators) + 1, ada_discrete_err, label='Discrete AdaBoost Test Error', color='red')
ax.plot(np.arange(n_estimators) + 1, ada_discrete_err_train, label='Discrete AdaBoost Train Error', color='blue')
ax.plot(np.arange(n_estimators) + 1, ada_real_err, label='Real AdaBoost Test Error', color='orange')
ax.plot(np.arange(n_estimators) + 1, ada_real_err_train, label='Real AdaBoost Train Error', color='green')

ax.set_ylim((0.0, 0.5))
ax.set_xlabel('n_estimators')
ax.set_ylabel('error rate')

leg = ax.legend(loc='upper right', fancybox=True)
leg.get_frame().set_alpha(0.7)
b = time.time()
print('total running time of this example is :', b - a)
plt.show()