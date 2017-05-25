#!/usr/bin/python
'''
Created on 1 Apr 2015
@author: Jamie Hall
'''
import pickle
import xgboost as xgb
import sklearn
import numpy as np
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris, load_digits, load_boston
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import operator
import matplotlib
import matplotlib.pyplot as plt 
import os 

'''
dataset1 = pd.read_csv('data/dataset1.csv')
dataset1.label.replace(-1,0,inplace=True)
dataset2 = pd.read_csv('data/dataset2.csv')
dataset2.label.replace(-1,0,inplace=True)
dataset3 = pd.read_csv('data/dataset3.csv')
'''

rng = np.random.RandomState(31337)

print("Zeros and Ones from the Digits dataset: binary classification")
digits = load_digits(2)
'''
y = digits['target']
X = digits['data']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(confusion_matrix(actuals, predictions))

print("Iris: multiclass classification")
iris = load_iris()
y = iris['target']
X = iris['data']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBClassifier().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(confusion_matrix(actuals, predictions))

'''
'''
print("Boston Housing: regression")
boston = load_boston()
y = boston['target']
X = boston['data']
kf = KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    xgb_model = xgb.XGBRegressor().fit(X[train_index],y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = y[test_index]
    print(mean_squared_error(actuals, predictions))

print("Parameter optimization")
y = boston['target']
X = boston['data']
xgb_model = xgb.XGBRegressor()
clf = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
clf.fit(X,y)
print(clf.best_score_)
print(clf.best_params_)

# The sklearn API models are picklable
print("Pickling sklearn API models")
# must open in binary format to pickle
pickle.dump(clf, open("best_boston.pkl", "wb"))
clf2 = pickle.load(open("best_boston.pkl", "rb"))
print(np.allclose(clf.predict(X), clf2.predict(X)))
'''
# Early-stopping

X = digits['data']
y = digits['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = xgb.XGBClassifier()
print("Parameter optimization")
clf = GridSearchCV(clf,
                   {'gamma':[0.1,0.2],
                    'min_child_weight':[1.1,1.0],
                    'max_depth': [2,4,6]
                    }, verbose=1)
clf.fit(X,y)
print(1-clf.best_score_)
print(clf.best_params_)

clf2 = xgb.XGBClassifier()
clf2.fit(X_train, y_train, early_stopping_rounds=2, 
        eval_set=[(X_test, y_test)])
#result.to_csv('{0}_{1}_{2}.csv'.format(predict_data_path, exec_time, tag), index=False)
'''
print("Iris: multiclass classification")
iris = load_iris()
y = iris['target']
X = iris['data']
# specify parameters via map
param={'booster':'gbtree',
    'eval_metric':'error',
    'gamma':0.1,
    'min_child_weight':1.1,
    'max_depth':5,
    'lambda':10,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'colsample_bylevel':0.7,
    'eta': 0.01,
    'tree_method':'exact',
    'seed':0,
    'nthread':12
    }
num_round = 2
traindata = xgb.DMatrix( X, label=y)
kf = KFold(n_splits=2, shuffle=True, random_state=rng)
for train_index, test_index in kf.split(X):
    bst = xgb.train(param, traindata, num_round)
    predictions = bst.predict(X[test_index])
    actuals = y[test_index]
    print(confusion_matrix(actuals, predictions))
'''