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

file_path_1 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(''))) + "/Task_4/data/train_X.npy")
file_path_2 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(''))) + "/Task_4/data/train_Y.npy")
file_path_3 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(''))) + "/Task_4/data/test_X.npy")
file_path_4 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(''))) + "/Task_4/data/test_Y.npy")
train_X = np.load(file_path_1)
train_Y = np.load(file_path_2)
test_X = np.load(file_path_3)
test_Y = np.load(file_path_4)
dataset_x = np.load('D:\Big_data_project (1)\Big_data_project\Task_2\data\dataset_X.npy')
dataset_y = np.load('D:\Big_data_project (1)\Big_data_project\Task_2\data\dataset_Y.npy')
# 使用PCA降维
from sklearn.decomposition import PCA

model1 = PCA(n_components=2)
x2 = model1.fit_transform(dataset_x)
plt.scatter(x2[:, 0], x2[:, 1], c=dataset_y)
plt.show()