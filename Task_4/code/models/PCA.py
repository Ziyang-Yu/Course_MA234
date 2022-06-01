import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from sklearn.decomposition import PCA

file_path_1 = os.path.dirname(os.path.abspath('')) + "/Big_data_project/Task_4/data/train_X.npy"
file_path_2 = os.path.dirname(os.path.abspath('')) + "/Big_data_project/Task_4/data/train_Y.npy"
file_path_3 = os.path.dirname(os.path.abspath('')) + "/Big_data_project/Task_4/data/test_X.npy"
file_path_4 = os.path.dirname(os.path.abspath('')) + "/Big_data_project/Task_4/data/test_Y.npy"
train_X = np.load(file_path_1)
train_Y = np.load(file_path_2)
test_X = np.load(file_path_3)
test_Y = np.load(file_path_4)


from sklearn.decomposition import PCA
model = PCA(n_components=2)
x1 = model.fit_transform(train_X)
plt.scatter(x1[:, 0], x1[:, 1], c=train_Y)
plt.show()
model.fit(train_X)
X_new = model.fit_transform(train_X)
Maxcomponent = model.components_
ratio = model.explained_variance_ratio_
score = model.score(train_X)
print('降维后的数据:', X_new)
print('返回具有最大方差的成分:', Maxcomponent)
print('保留主成分的方差贡献率:', ratio)
print('所有样本的log似然平均值:', score)
print('奇异值:', model.singular_values_)
print('噪声协方差:', model.noise_variance_)
