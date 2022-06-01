import os

from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score
from sklearn import linear_model
import platform
file_path = os.path.dirname(os.path.abspath(''))
if platform.system().lower() == 'windows':
    print("windows")
    train_X = np.load(file_path + '\data\\train_X.npy')
    train_Y = np.load(file_path + '\data\\train_Y.npy')
    test_X = np.load(file_path + '\data\\test_X.npy')
    test_Y = np.load(file_path + '\data\\test_Y.npy')
elif platform.system().lower() == 'linux':
    print("linux")
    train_X = np.load(file_path + '/data/train_X.npy')
    train_Y = np.load(file_path + '/data/train_Y.npy')
    test_X = np.load(file_path + '/data/test_X.npy')
    test_Y = np.load(file_path + '/data/test_Y.npy')
clf = SVR(kernel ='rbf',degree = 3,gamma ='auto',coef0 = 0.0,tol = 0.001,C = 1.0,epsilon = 0.1,shrinking = True,cache_size = 200,verbose = False,max_iter = -1 )



clf.fit(train_X, train_Y)
y_hat = clf.predict(test_X)
e=mean_squared_error(test_Y,y_hat)
f=mean_absolute_error(test_Y,y_hat)
g=r2_score(test_Y,y_hat)
print("mean_squared_error: ", e)
print("mean_absolute_error: ",f)
print("r2_score: ",g)
