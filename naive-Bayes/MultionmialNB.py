import numpy as np
from sklearn.naive_bayes import MultinomialNB
X = np.array([[1,2,3,4],[1,3,4,4],[2,4,5,5],[2,5,6,5],[3,4,5,6],[3,5,6,6]])
y = np.array([1,1,4,2,3,3])
clf = MultinomialNB(alpha=2.0)
clf.fit(X,y)

print(clf.predict([[-4,4,2,3]]))