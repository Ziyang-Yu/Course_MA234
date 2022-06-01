import os

import numpy as np
from sklearn.datasets import make_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差

kng=KNeighborsRegressor(n_neighbors=5)
import platform
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
kng.fit(train_X,train_Y)
prediction=kng.predict(test_X)
kng_test_score=kng.score(test_X,test_Y)
kng_train_score=kng.score(test_X,test_Y)

print('test data score:{:.2f}'.format(kng_test_score))
print('mean squared error: ' + str(mean_squared_error(test_Y, prediction)))
print('mean absolute error: ' + str(mean_absolute_error(test_Y, prediction)))