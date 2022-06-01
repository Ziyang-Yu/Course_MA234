import os


import torch
import cv2
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
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
print('开始训练...')
# 直接初始化LGBMRegressor
# 这个LightGBM的Regressor和sklearn中其他Regressor基本是一致的
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        learning_rate=0.1,
                        n_estimators=40)

# 使用fit函数拟合
gbm.fit(train_X, train_Y,
        eval_set=[(test_X, test_Y)],
        eval_metric='l1',
        early_stopping_rounds=5)

# 预测
print('开始预测...')
y = gbm.predict(test_X, num_iteration=gbm.best_iteration_)
estimator = lgb.LGBMRegressor(num_leaves=31)
e=mean_squared_error(test_Y,y)
f=mean_absolute_error(test_Y,y)
g=r2_score(test_Y,y)
print("mean_squared_error: ", e)
print("mean_absolute_error: ",f)
print("r2_score: ",g)

param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}

gbm = GridSearchCV(estimator, param_grid)

gbm.fit(train_X, train_Y)

print('用网格搜索找到的最优超参数为:')
print(gbm.best_params_)

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