import datetime
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import platform
import os

if platform.system().lower() == 'windows':
    print("windows")
    dataset_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '\data\PRSA_data.csv'
elif platform.system().lower() == 'linux':
    print("linux")
    dataset_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/data/PRSA_data.csv'


def dataset_delete():
    with open(dataset_path, 'r') as f:
        data = f.read()
    data = data.split('\n')
    for i in range(0, len(data)):
        data[i] = data[i].split(',')
    data_after = []
    for i in range(0, len(data)):
        if len(data[i]) >= 5 and data[i][5] != 'NA':
            data_after.append(data[i])
    del data_after[0]
    for i in range(0, len(data_after)):
        for j in range(0, len(data_after[i])):
            if j == 9:
                continue
            data_after[i][j] = (float)(data_after[i][j])

    for i in range(0, len(data_after)):
        data_after[i] = [data_after[i][6:], data_after[i][5]]
        temp_1 = data_after[i][0][:3]
        if data_after[i][0][3] == 'NE':
            temp_2 = [1., 0., 0., 0.]
        if data_after[i][0][3] == 'SE':
            temp_2 = [0., 1., 0., 0.]
        if data_after[i][0][3] == 'NW':
            temp_2 = [0., 0., 1., 0.]
        if data_after[i][0][3] == 'cv':
            temp_2 = [0., 0., 0., 1.]
        temp_3 = data_after[i][0][4:]
        data_after[i][0] = temp_1 + temp_2 + temp_3

    dataset_X = []
    dataset_Y = []
    for i in range(0, len(data_after)):
        dataset_X.append(data_after[i][0])
        dataset_Y.append(data_after[i][1])

    dataset_X = np.array(dataset_X)
    dataset_Y = np.array(dataset_Y)

    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/dataset_X.npy", dataset_X)
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/dataset_Y.npy", dataset_Y)


    fill_time()









def preprocess_fill_interpolate():

    

    with open(dataset_path, 'r') as f:
        data = f.read()
    data = data.split('\n')
    for i in range(0, len(data)):
        data[i] = data[i].split(',')
    data_after = []
    for i in range(0, len(data)):
        if len(data[i]) >= 5:
            data_after.append(data[i])
    del data_after[0]
    data_after = np.array(data_after).tolist()
    for i in range(0, len(data_after)):
        if data_after[i][9] == 'NE':
            data_after[i][9] = [1., 0., 0., 0.]
        if data_after[i][9] == 'SE':
            data_after[i][9] = [0., 1., 0., 0.]
        if data_after[i][9] == 'NW':
            data_after[i][9] = [0., 0., 1., 0.]
        if data_after[i][9] == 'cv':
            data_after[i][9] = [0., 0., 0., 1.]
    data_after = np.array(data_after)
    data_after = data_after.T
    temp = data_after[5]
    temp[temp=='NA'] = np.nan
    s = pd.Series(temp, dtype=np.float64)
    temp = s.interpolate(limit_direction='both')
    data_after[5] = temp
    data_after = data_after.T
    data_after = data_after.tolist()
    for i in range(0, len(data_after)):
        data_after[i] = data_after[i][0:9]+data_after[i][9]+data_after[i][10:]
    for i in range(0, len(data_after)):
        for j in range(0, len(data_after[0])):
            data_after[i][j] = (float)(data_after[i][j])

    for i in range(0, len(data_after)):
        data_after[i] = data_after[i][5:]

    

    dataset_X = []
    dataset_Y = []

    for i in range(0, len(data_after)):
        dataset_X.append(data_after[i][1:])
        dataset_Y.append(data_after[i][0])

    dataset_X = np.array(dataset_X)
    dataset_Y = np.array(dataset_Y)

    ### 保存为.npy方便随时快速读取
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/dataset_X.npy", dataset_X)
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/dataset_Y.npy", dataset_Y)


    fill_time()








def preprocess_fill_average():


    with open(dataset_path, 'r') as f:
        data = f.read()
    data = data.split('\n')
    for i in range(0, len(data)):
        data[i] = data[i].split(',')
    data_after = []
    for i in range(0, len(data)):
        if len(data[i]) >= 5 :
            data_after.append(data[i])
    del data_after[0]

    sum = 0.
    count = 0
    for i in range(0, len(data_after)):
        if data_after[i][5] != 'NA':
            sum += (float)(data_after[i][5])
            count += 1

    average = sum/count
    for i in range(0, len(data_after)):
        if data_after[i][5] == 'NA':
            data_after[i][5] = average

    for i in range(0, len(data_after)):
        for j in range(0, len(data_after[i])):
            if j == 9:
                continue
            data_after[i][j] = (float)(data_after[i][j])



    
    ### one-hot嵌入
    data_after = data_after.tolist()

    s = {x[9] for x in data_after}

    for i in range(0, len(data_after)):
        data_after[i] = [data_after[i][6:], data_after[i][5]]
        temp_1 = data_after[i][0][:3]
        if data_after[i][0][3] == 'NE':
            temp_2 = [1., 0., 0., 0.]
        if data_after[i][0][3] == 'SE':
            temp_2 = [0., 1., 0., 0.]
        if data_after[i][0][3] == 'NW':
            temp_2 = [0., 0., 1., 0.]
        if data_after[i][0][3] == 'cv':
            temp_2 = [0., 0., 0., 1.]
        temp_3 = data_after[i][0][4:]
        data_after[i][0] = temp_1 + temp_2 + temp_3



    dataset_X = []
    dataset_Y = []

    for i in range(0, len(data_after)):
        dataset_X.append(data_after[i][0])
        dataset_Y.append(data_after[i][1])
    
    dataset_Y = np.array(dataset_Y)
    dataset_X = np.array(dataset_X)



    ### 保存为.npy方便随时快速读取
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/dataset_X.npy", dataset_X)
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/dataset_Y.npy", dataset_Y)

    fill_time()







def preprocess_fill_KNN():

    with open(dataset_path, 'r') as f:
        data = f.read()
    data = data.split('\n')
    for i in range(0, len(data)):
        data[i] = data[i].split(',')
    data_after = []
    for i in range(0, len(data)):
        if len(data[i]) >= 5:
            data_after.append(data[i])
    del data_after[0]
    for i in range(0, len(data_after)):
        if data_after[i][9] == 'NE':
            data_after[i][9] = [1., 0., 0., 0.]
        if data_after[i][9] == 'SE':
            data_after[i][9] = [0., 1., 0., 0.]
        if data_after[i][9] == 'NW':
            data_after[i][9] = [0., 0., 1., 0.]
        if data_after[i][9] == 'cv':
            data_after[i][9] = [0., 0., 0., 1.]

    for i in range(0, len(data_after)):
        data_after[i] = data_after[i][0:9]+data_after[i][9]+data_after[i][10:]

    for i in range(0, len(data_after)):
        if data_after[i][5] == 'NA':
            data_after[i][5] = np.nan

    # fill with KNN
    knn_imputer = KNNImputer(n_neighbors=2)

    data_after = knn_imputer.fit_transform(data_after)
    for i in range(0, len(data_after)):
        for j in range(0, len(data_after[i])):
            data_after[i][j] = (float)(data_after[i][j])


    data_after = data_after.tolist()
    for i in range(0, len(data_after)):
        data_after[i] = data_after[i][5:]


    dataset_X = []
    dataset_Y = []

    for i in range(0, len(data_after)):
        dataset_X.append(data_after[i][1:])
        dataset_Y.append(data_after[i][0])


    ### 保存为.npy方便随时快速读取
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/dataset_X.npy", dataset_X)
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/dataset_Y.npy", dataset_Y)
    
    fill_time()



def save_as_npy(train_X, train_Y, test_X, test_Y):
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/train_X.npy", train_X)
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/train_Y.npy", train_Y)
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/test_X.npy", test_X)
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/test_Y.npy", test_Y)




def fill_time():
    dataset_X = np.load(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/dataset_X.npy")
    dataset_X = dataset_X.tolist()
    with open(dataset_path, 'r') as f:
        data = f.read()
    data = data.split('\n')
    del data[0]
    del(data[-1])
    for i in range(0, len(data)):
        data[i] = data[i].split(',')
    for i in range(0, len(data)):
        if len(data) < 5:
            continue
        for j in range(1, 5):
            data[i][j] = (float)(data[i][j])
    for i in range(0, len(dataset_X)):
        dataset_X[i] = dataset_X[i] + data[i][1:5]
    dataset_X = np.array(dataset_X)
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/dataset_X.npy", dataset_X)
