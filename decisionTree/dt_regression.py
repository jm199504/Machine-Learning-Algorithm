import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def f(x1, x2):
    y = 0.5 * np.sin(x1) + 0.5 * np.cos(x2)  + 0.1 * x1 + 3
    return y

def load_data():
    x1_train = np.linspace(0,50,500)
    x2_train = np.linspace(-10,10,500)
    data_train = np.array([[x1,x2,f(x1,x2) + (np.random.random(1)-0.5)] for x1,x2 in zip(x1_train, x2_train)])
    x1_test = np.linspace(0,50,100)+ 0.5 * np.random.random(100)
    x2_test = np.linspace(-10,10,100) + 0.02 * np.random.random(100)
    data_test = np.array([[x1,x2,f(x1,x2)] for x1,x2 in zip(x1_test, x2_test)])
    return data_train, data_test

data_train, data_test = load_data()
x_train,y_train = list(),list()
x_test,y_test = list(),list()

for i in data_train:
    # 注意i是一个小list，前2项为x1，x2，第三项为f(x1,x2)即y
    x_train.append(i[0:2])
    y_train.append(i[-1])
for i in data_test:
    x_test.append(i[0:2])
    y_test.append(i[-1])
# 初始化回归树
clf = DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
           max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, presort=False, random_state=None,
           splitter='best')
# 训练拟合树模型
clf.fit(x_train,y_train)
# 测试集进行预测
result = clf.predict(x_test)
# 对预测结果进行评价
score = clf.score(x_test,y_test)
# 绘图
plt.figure()
plt.plot(np.arange(len(result)),y_test,'go-',label='true value')
plt.plot(np.arange(len(result)),result,'ro-',label='predict value')
plt.title('score: %f'%score)
plt.legend()
plt.show()