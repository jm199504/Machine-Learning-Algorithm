import numpy as np
from collections import Counter

def createDataSet():
	x_train = np.array([[1,101],[5,89],[108,5],[115,8]])
	y_train = ['爱情片','爱情片','动作片','动作片']
	return x_train, y_train

def knn_classify(test_x, dataset, labels, k):
	# 计算向量间的距离
	dist = np.sum((test_x - dataset)**2, axis=1)**0.5
	# 选取距离最小的前k个标签
	k_labels = [labels[index] for index in dist.argsort()[0 : k]]
	# 将出现次数最多的标签作为预测结果
	label = Counter(k_labels).most_common(1)[0][0]
	return label

if __name__ == '__main__':
	# 创建数据集
	group, labels = createDataSet()
	# 预测向量test
	test_x = [100,18]
	# kNN分类
	test_y = knn_classify(test_x, group, labels, 3)
	# 打印分类结果
print('%s预测结果为%s'%(test_x,test_y))

# 补充说明1（np.sum）：
# axis=0表示向量纵向相加；axis=1表示横向相加
# np.sum([[0,1,2],[2,1,3]],axis=0)     axis=0  针对多个向量连接累加   [2,2,5]
# np.sum([[0,1,2],[2,1,3]],axis=1)     axis=1  针对单个向量内部累加   [3,6]

# 补充说明2(ndarray.argsort)：
# 默认从小到大排序并返回对应下标

# 补充说明3(np.argsort(ndarry,axis))
# axis=0    对每列元素排序
# axis=1    对每行元素排序