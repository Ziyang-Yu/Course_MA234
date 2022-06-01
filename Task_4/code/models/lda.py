from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

file_path_1 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath('')))+ "/Task_4/data/train_X.npy")
file_path_2 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath('')))+ "/Task_4/data/train_Y.npy")
file_path_3 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath('')))+ "/Task_4/data/test_X.npy")
file_path_4= os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath('')))+ "/Task_4/data/test_Y.npy")
x_train = np.load(file_path_1)
y_train= np.load(file_path_2)
x_test = np.load(file_path_3)
y_test = np.load(file_path_4)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model2 = LinearDiscriminantAnalysis(n_components=2)
model2.fit(x_train,y_train.astype(int))
x2=model2.transform(x_train)
plt.scatter(x2[:,0],x2[:,1],c=y_train)
plt.show()
