import os

from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np
file_path_1 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath('')))+ "/Task_4/data/train_X.npy")
file_path_2 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath('')))+ "/Task_4/data/train_Y.npy")
file_path_3 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath('')))+ "/Task_4/data/test_X.npy")
file_path_4= os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath('')))+ "/Task_4/data/test_Y.npy")
x_train = np.load(file_path_1)
y_train= np.load(file_path_2)
x_test = np.load(file_path_3)
y_test = np.load(file_path_4)
x, y = x_train, y_train
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA

# 获得半月形的数据集
X, y = x_train, y_train

# 建立目标维度为2的RBF模型
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)

# 使用KPCA降低数据维度，直接获得投影后的坐标
X_skernpca = scikit_kpca.fit_transform(X)

# 数据可视化
plt.scatter(X_skernpca[y==0, 0], X_skernpca[y==0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y==1, 0], X_skernpca[y==1, 1], color='blue', marker='o', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()
plt.show()

