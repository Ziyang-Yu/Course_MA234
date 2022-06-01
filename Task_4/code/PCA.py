import os
import numpy as np
import matplotlib.pyplot as plt


file_path_1 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(''))) + "/Task_4/data/train_X.npy")
file_path_2 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(''))) + "/Task_4/data/train_Y.npy")
file_path_3 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(''))) + "/Task_4/data/test_X.npy")
file_path_4 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(''))) + "/Task_4/data/test_Y.npy")
train_X = np.load(file_path_1)
train_Y = np.load(file_path_2)
test_X = np.load(file_path_3)
test_Y = np.load(file_path_4)
from sklearn.decomposition import PCA
model = PCA(n_components=5)
x1 = model.fit_transform(train_X)
plt.scatter(x1[:, 0], x1[:, 1], c=train_Y)
plt.show()
model.fit(train_X)
X_new = model.fit_transform(train_X)
X_new= np.around(X_new, 3)
Maxcomponent = model.components_
Maxcomponent= np.around(Maxcomponent, 3)
ratio = model.explained_variance_ratio_
ratio= np.around(ratio, 3)
score = model.score(train_X)
score = np.around(score , 3)
print('降维后的数据:', X_new)
print('返回具有最大方差的成分:', Maxcomponent)
print('保留主成分的方差贡献率:', ratio)
print('所有样本的log似然平均值:', score)
print('奇异值:', model.singular_values_)
print('噪声协方差:', model.noise_variance_)
