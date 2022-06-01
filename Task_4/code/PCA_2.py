import os
import matplotlib.pyplot as plt
from boto import sns
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
from sklearn.decomposition import PCA
model1 = PCA(n_components=2)
x2 = model1.fit_transform(dataset_x)
plt.scatter(x2[:, 0], x2[:, 1], c=dataset_y)
plt.show()
df=dataset_x
def meanX(dataX):
    return np.mean(dataX, axis=0)  # axis=0表示依照列来求均值。假设输入list,则axis=1
average = meanX(df)
print(average)
m, n = np.shape(df)
print(m, n)
data_adjust = []
avgs = np.tile(average, (m, 1))
print(avgs)
data_adjust = df - avgs
print(data_adjust)
covX = np.cov(data_adjust.T)  # 计算协方差矩阵
print(covX)

featValue, featVec = np.linalg.eig(covX)  # 求解协方差矩阵的特征值和特征向量
print(featValue, featVec)
# 对特征值进行排序并输出 降序
featValue = sorted(featValue)[::-1]
print(featValue)
# 绘制散点图和折线图
# 同样的数据绘制散点图和折线图
plt.scatter(range(1, df.shape[1] + 1), featValue)
plt.plot(range(1, df.shape[1] + 1), featValue)
# 显示图的标题和xy轴的名字
plt.title("Scree Plot")
plt.xlabel("Factors")
plt.ylabel("Eigenvalue")
plt.grid()  # 显示网格
plt.show()  # 显示图形
# 求特征值的贡献度
gx = featValue / np.sum(featValue)
print(gx)
# 求特征值的累计贡献度
lg = np.cumsum(gx)
lg = np.around(lg , 3)
print(lg)
# 选出主成分
k = [i for i in range(len(lg)) if lg[i] < 0.85]
k = list(k)
print(k)
# 选出主成分对应的特征向量矩阵
selectVec = np.matrix(featVec.T[k]).T
selectVe = selectVec * (-1)
print(selectVec)
finalData = np.dot(data_adjust, selectVec)
print(finalData)




