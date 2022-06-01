import os
import numpy as np
dataset_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '\data'
dataset_X = np.load(dataset_path+'\dataset_X.npy')
dataset_Y = np.load(dataset_path+'\dataset_Y.npy')
dataset_X_new = np.copy(dataset_X)
for i in range(1, len(dataset_X)-1):
    dataset_X[i] = (dataset_X_new[i-1] + dataset_X_new[i] + dataset_X_new[i+1])/3
dataset_X = dataset_X[1:len(dataset_X)-1]
dataset_Y = dataset_Y[1:len(dataset_Y)-1]
np.save(dataset_path+'\dataset_X_smooth.npy', dataset_X)
np.save(dataset_path+'\dataset_Y_smooth.npy', dataset_Y)
