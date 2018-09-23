import numpy as np
from sklearn.naive_bayes import BernoulliNB
X = np.array([[1,2,3,4],[1,3,4,4],[2,4,5,5]])
y = np.array([1,1,2])
clf = BernoulliNB(alpha=2.0,binarize = 3.0,fit_prior=True)
clf.fit(X,y)

print(clf.predict([[3,2,3,4]]))