from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as mp

# Importing the data
filename = 'car.data'
labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'clas']
data = pd.read_csv(filename, names=labels)

# Pre-processing the data
buying = {'vhigh' : 0, 'high' : 1, 'med' : 2, 'low' : 3}
maint = {'vhigh' : 0, 'high' : 1, 'med' : 2, 'low' : 3}
doors = {'2' : 2, '3' : 3, '4' : 4, '5more' : 5}
persons = {'2' : 2, '4' : 4, 'more' : 5}
lug_boot = {'small' : 0, 'med' : 1, 'big' : 2}
safety = {'high' : 1, 'med' : 2, 'low' : 3}
clas = {'unacc' : 1, 'acc' : 2, 'good' : 3, 'vgood' : 4}

data.buying = [buying[item] for item in data.buying]
data.maint = [maint[item] for item in data.maint]
data.doors = [doors[item] for item in data.doors]
data.persons = [persons[item] for item in data.persons]
data.lug_boot = [lug_boot[item] for item in data.lug_boot]
data.safety = [safety[item] for item in data.safety]
data.clas = [clas[item] for item in data.clas]

# Splitting the data into Training and Testing data
newdata = pd.DataFrame(data, columns = labels)
train_data, test_data = train_test_split(newdata, test_size = 0.25, random_state = 0)

X = train_data[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]
Y = train_data[['clas']]
X1 = test_data[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]
Y1 = test_data[['clas']]

# Applying Decision Tree Classifier using Entropy as estimator
decision = tree.DecisionTreeClassifier(criterion = "entropy")
decision = decision.fit(X,Y)
print(decision.score(X1,Y1)) # Accuracy of Decision Tree Classifier

# Applying Bagging Classifier using Decision Tree and Entropy as the estimator
bagging = BaggingClassifier(tree.DecisionTreeClassifier(criterion = "entropy"), 
    max_samples=0.5, bootstrap = True, bootstrap_features= True)
Y = np.ravel(Y) 
bagging = bagging.fit(X,Y)
print(bagging.score(X1, Y1)) # Accuracy of Bagging Classifier

# Applying Random Forest Classifier with Entropy as estimator
forest = RandomForestClassifier(n_estimators=50, max_depth = 5, criterion = "entropy")
forest = forest.fit(X,Y)
print(forest.score(X1, Y1)) # Accuracy of Random Forest Classifier

# Variable importance graph plot
graph = forest.feature_importances_ * 100
indices = np.argsort(graph)
mp.barh(range(len(indices)), graph[indices])
mp.yticks(range(len(indices)),X.columns[indices])
mp.xlabel("Feature importance")
mp.ylabel("Features")
mp.show()

# Accuracy changes graph plot
ensemble_clfs = [
    ("RandomForestClassifier", RandomForestClassifier(criterion = "entropy", warm_start = True, oob_score = True, random_state = 150)),
    ("Bagging", BaggingClassifier(tree.DecisionTreeClassifier(criterion = "entropy"), warm_start = False, oob_score = True, random_state = 150))
]
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)
# Range of trees to be plotted
min_estimators = 50
max_estimators = 300
for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X, Y)
        oob_error = 1 - clf.oob_score_
        error_rate[label].append((i, oob_error))
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    mp.plot(xs, ys, label=label)
mp.xlim(min_estimators, max_estimators)
mp.xlabel("Number of trees")
mp.ylabel("OOB error rate")
mp.legend(loc="upper right")
mp.show()