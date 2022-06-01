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


from classification_results import results
from sklearn.metrics import f1_score
y = prediction
for i in range(0, len(y)):
    if y[i] <= 35:
        y[i] = 0
    elif y[i] <= 150:
        y[i] = 1
    else:
        y[i] = 2
    if test_Y[i] <= 35:
        test_Y[i] = 0
    elif test_Y[i] <= 150:
        test_Y[i] = 1
    else:
        test_Y[i] = 2
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
print(accuracy_score(test_Y, y))
print(f1_score(test_Y, y, average='macro'))