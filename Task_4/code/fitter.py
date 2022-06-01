import os

from matplotlib import pyplot as plt
from sklearn.utils import column_or_1d
from xgboost import XGBRegressor as XGBR
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsClassifier as KNN, KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC
file_path_1 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath('')))+ "/Task_4/data/train_X.npy")
file_path_2 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath('')))+ "/Task_4/data/train_Y.npy")
file_path_3 = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath('')))+ "/Task_4/data/test_X.npy")
file_path_4= os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath('')))+ "/Task_4/data/test_Y.npy")
x_train = np.load(file_path_1)
y_train= np.load(file_path_2)
x_test = np.load(file_path_3)
y_test = np.load(file_path_4)
x = pd.DataFrame(x_train)
y = pd.DataFrame(y_train)

selector = VarianceThreshold()#实例化，默认方差为0
x_var0 = selector.fit_transform(x)
x_fsvar =VarianceThreshold(np.median(x.var())).fit_transform(x)
#将中位数设置为阈值，砍掉一般的特征
print(x.var())
RFC_ = RFC(n_estimators=10,random_state=0)

from sklearn.feature_selection import RFE
RFC_ = RFC(n_estimators=10,random_state=0)#实例化模型
selector = RFE(RFC_,n_features_to_select=340,step=50).fit(x,y.astype(int))#50代表每迭代一次帮我删除50个特征
print(selector.support_.sum())#返回所有的特征是否被选中的布尔矩阵
print(selector.ranking_)#返回特征的按数次迭代中综合重要性的排名
y = column_or_1d(y, warn=True)

##x_embedded = SelectFromModel(RFC_,threshold=0.005).fit_transform(x,y.astype(int))
##print(x_embedded.shape)
##threshold = np.linspace(0, (RFC_.fit(x, y.astype(int)).feature_importances_).max(), 20)

##score = []
##for i in threshold:
    ##x_embedded = SelectFromModel(RFC_, threshold=i).fit_transform(x, y.astype(int))
    ##once = cross_val_score(RFC_, x_embedded, y.astype(int), cv=5).mean()
    ##score.append(once)

##plt.plot(threshold, score)
##plt.show()

score = []
for i in range(1,751,50):
    x_wrapper = RFE(RFC_,n_features_to_select=i,step=50).fit_transform(x,y.astype(int))
    once = cross_val_score(RFC_,x_wrapper,y.astype(int),cv=5).mean()
    score.append(once)


plt.figure(figsize = [20,5])
plt.plot(range(1,751,50),score)
plt.xticks(range(1,751,50))
plt.show()

