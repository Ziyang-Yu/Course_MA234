import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score
from sklearn.linear_model import ElasticNet

# 初始化弹性网络回归器
reg = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False)
# 拟合线性模型
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
reg.fit(train_X, train_Y)
y=reg.predict(test_X)
e=mean_squared_error(test_Y,y)
f=mean_absolute_error(test_Y,y)
g=r2_score(test_Y,y)
print("mean_squared_error: ", e)
print("mean_absolute_error: ",f)
print("r2_score: ",g)

from classification_results import results
from sklearn.metrics import f1_score

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
