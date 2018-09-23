import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.tree import DecisionTreeClassifier as DTC

iris = datasets.load_iris()
X = iris.data
y = iris.target
dtc=DTC(min_samples_leaf=5)
test_mse,train_mse = list(),list()

numbers=range(1,100)
for d in numbers:
    clf=ABC(base_estimator=dtc,n_estimators=d,algorithm='SAMME')
    clf_dict=cross_validate(clf,X,y,cv=10,scoring='accuracy')
    test_mse.append(clf_dict['test_score'].mean())
    train_mse.append(clf_dict['train_score'].mean())

# sns.set(style='darkgrid')
plt.plot(numbers,train_mse,'b-.',label='Train Accuracy')
plt.plot(numbers,test_mse,'r-.',label='Test Accuracy')
plt.xlabel(' n estimators')
plt.ylabel('Accuracy')
plt.title('DecisionTree for Adaboost')
plt.legend()
plt.show()