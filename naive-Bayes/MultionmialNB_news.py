from sklearn.datasets import fetch_20newsgroups  # 导入新闻数据抓取器 fetch_20newsgroups
from sklearn.model_selection import  train_test_split   # 导入数据集分割工具
from sklearn.feature_extraction.text import CountVectorizer  # 导入文本特征向量化模块
from sklearn.naive_bayes import MultinomialNB    # 导入多项式朴素贝叶斯模型
from sklearn.metrics import classification_report # 分类问题的评估报告

# 数据获取
news = fetch_20newsgroups(subset='all')
print (len(news.data))  # 输出数据的条数：18846

# 对数据进行预处理：训练集和测试集分割
# 随机采样25%的数据样本作为测试集
X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=11)

# 对数据进行预处理：文本特征向量化
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# 使用多项式朴素贝叶斯进行训练
mnb = MultinomialNB()   # 使用默认配置初始化朴素贝叶斯
# 训练拟合模型
mnb.fit(X_train,y_train)    # 利用训练数据对模型参数进行估计
# 模型预测
y_predict = mnb.predict(X_test)     # 对参数进行预测

# 结果评估：模型分数及分类报告
print ('The Accuracy of MultinomialNB is:', mnb.score(X_test,y_test))
print (classification_report(y_test, y_predict, target_names = news.target_names))