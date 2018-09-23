import matplotlib.pyplot as plt
import numpy as np

x_train = [[1,101],[5,89],[108,5],[115,8]]
y_train = ['爱情片','爱情片','动作片','动作片']
for i in range(70):
    x_train.append([np.random.uniform(1,45),np.random.uniform(55,100)])
    y_train.append('爱情片')
for i in range(70):
    x_train.append([np.random.uniform(55,120),np.random.uniform(1,55)])
    y_train.append('动作片')

feature_1,feature_2 = list(),list()
label_1,label_2 = list(),list()
for i in range(len(x_train)):
    if y_train[i]=='爱情片':
        feature_1.append(x_train[i][0])
        label_1.append(x_train[i][1])
    else:
        feature_2.append(x_train[i][0])
        label_2.append(x_train[i][1])

test_point = [50,50]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.scatter(test_point[0],test_point[1],c='green',marker='o')
plt.title('KNN可视化举栗')
plt.xlabel('动作戏成份')
plt.ylabel('爱情戏成份')
plt.scatter(feature_1,label_1,c='red',marker='+')
plt.scatter(feature_2,label_2,c='blue',marker='+')
plt.legend(labels=['测试点','爱情片','动作片'])
plt.show()