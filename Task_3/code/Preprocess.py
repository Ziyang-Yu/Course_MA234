import datetime
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import os
import platform

if platform.system().lower() == 'windows':
    print("windows")
    dataset_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '\data\PRSA_data.csv'
elif platform.system().lower() == 'linux':
    print("linux")
    dataset_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + '/data/PRSA_data.csv'


def getDateList(start_date, end_date):
    date_list = []
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    date_list.append(start_date.strftime('%Y-%m-%d'))
    while start_date < end_date:
        start_date += datetime.timedelta(days=1)
        date_list.append(start_date.strftime('%Y-%m-%d'))
    return date_list

def preprocess_delete():
    ### 导入数据并且转换成float类型

    
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

    ### 截取时间获得数据集

    date_list=getDateList("2010-01-07","2014-12-25")
    for i in range(0, len(date_list)):
        date_list[i] = date_list[i].split('-')
    for i in range(0, len(date_list)):
        for j in range(0, len(date_list[i])):
            date_list[i][j] = (float)(date_list[i][j])

    ### 切分数据集

    test_date = []
    test_data = []
    train_data = []
    for i in range(0, len(date_list), 7):
        test_date.append(date_list[i])

    for i in range(0, len(test_date)):
        for j in range(0, len(data_after)):
            if test_date[i][0] == data_after[j][1] and test_date[i][1] == data_after[j][2] and test_date[i][2] == data_after[j][3]:
                test_data.append(j)

    for i in range(0, len(data_after)):
        if i not in test_data:
            train_data.append(i)

    for i in range(0, len(train_data)):
        train_data[i] = data_after[train_data[i]]

    for i in range(0, len(test_data)):
        test_data[i] = data_after[test_data[i]]

    ### one-hot嵌入

    s = {x[9] for x in data_after}

    for i in range(0, len(train_data)):
        train_data[i] = [train_data[i][6:], train_data[i][5]]
        temp_1 = train_data[i][0][:3]
        if train_data[i][0][3] == 'NE':
            temp_2 = [1., 0., 0., 0.]
        if train_data[i][0][3] == 'SE':
            temp_2 = [0., 1., 0., 0.]
        if train_data[i][0][3] == 'NW':
            temp_2 = [0., 0., 1., 0.]
        if train_data[i][0][3] == 'cv':
            temp_2 = [0., 0., 0., 1.]
        temp_3 = train_data[i][0][4:]
        train_data[i][0] = temp_1 + temp_2 + temp_3

    for i in range(0, len(test_data)):
        test_data[i] = [test_data[i][6:], test_data[i][5]]
        temp_1 = test_data[i][0][:3]
        if test_data[i][0][3] == 'NE':
            temp_2 = [1., 0., 0., 0.]
        if test_data[i][0][3] == 'SE':
            temp_2 = [0., 1., 0., 0.]
        if test_data[i][0][3] == 'NW':
            temp_2 = [0., 0., 1., 0.]
        if test_data[i][0][3] == 'cv':
            temp_2 = [0., 0., 0., 1.]
        temp_3 = test_data[i][0][4:]
        test_data[i][0] = temp_1 + temp_2 + temp_3

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for i in range(0, len(train_data)):
        train_X.append(train_data[i][0])
        train_Y.append(train_data[i][1])
    for i in range(0, len(test_data)):
        test_X.append(test_data[i][0])
        test_Y.append(test_data[i][1])
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    ### 保存为.npy方便随时快速读取
    print(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    if platform.system().lower() == 'windows':
        print("windows")
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\train_X.npy", train_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\train_Y.npy", train_Y)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\test_X.npy", test_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\test_Y.npy", test_Y)
    elif platform.system().lower() == 'linux':
        print("linux")    
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/train_X.npy", train_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/train_Y.npy", train_Y)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/test_X.npy", test_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/test_Y.npy", test_Y)
        













def preprocess_fill_interpolate():

    date_list=getDateList("2010-01-07","2014-12-25")
    for i in range(0, len(date_list)):
        date_list[i] = date_list[i].split('-')
    for i in range(0, len(date_list)):
        for j in range(0, len(date_list[i])):
            date_list[i][j] = (float)(date_list[i][j])


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

    ### 切分数据集

    test_date = []
    test_data = []
    train_data = []
    for i in range(0, len(date_list), 7):
        test_date.append(date_list[i])

    for i in range(0, len(test_date)):
        for j in range(0, len(data_after)):
            if test_date[i][0] == data_after[j][1] and test_date[i][1] == data_after[j][2] and test_date[i][2] == data_after[j][3]:
                test_data.append(j)

    for i in range(0, len(data_after)):
        if i not in test_data:
            train_data.append(i)

    for i in range(0, len(train_data)):
        train_data[i] = data_after[train_data[i]][5:]

    for i in range(0, len(test_data)):
        test_data[i] = data_after[test_data[i]][5:]


    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for i in range(0, len(train_data)):
        train_X.append(train_data[i][1:])
        train_Y.append(train_data[i][0])
    for i in range(0, len(test_data)):
        test_X.append(test_data[i][1:])
        test_Y.append(test_data[i][0])

    ### 保存为.npy方便随时快速读取
    print(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    if platform.system().lower() == 'windows':
        print("windows")
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\train_X.npy", train_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\train_Y.npy", train_Y)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\test_X.npy", test_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\test_Y.npy", test_Y)
    elif platform.system().lower() == 'linux':
        print("linux")    
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/train_X.npy", train_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/train_Y.npy", train_Y)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/test_X.npy", test_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/test_Y.npy", test_Y)







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


    ### 截取时间获得数据集

    date_list=getDateList("2010-01-07","2014-12-25")
    for i in range(0, len(date_list)):
        date_list[i] = date_list[i].split('-')
    for i in range(0, len(date_list)):
        for j in range(0, len(date_list[i])):
            date_list[i][j] = (float)(date_list[i][j])

    ### 切分数据集

    test_date = []
    test_data = []
    train_data = []
    for i in range(0, len(date_list), 7):
        test_date.append(date_list[i])

    for i in range(0, len(test_date)):
        for j in range(0, len(data_after)):
            if test_date[i][0] == data_after[j][1] and test_date[i][1] == data_after[j][2] and test_date[i][2] == data_after[j][3]:
                test_data.append(j)

    for i in range(0, len(data_after)):
        if i not in test_data:
            train_data.append(i)

    for i in range(0, len(train_data)):
        train_data[i] = data_after[train_data[i]]

    for i in range(0, len(test_data)):
        test_data[i] = data_after[test_data[i]]

    ### one-hot嵌入

    s = {x[9] for x in data_after}

    for i in range(0, len(train_data)):
        train_data[i] = [train_data[i][6:], train_data[i][5]]
        temp_1 = train_data[i][0][:3]
        if train_data[i][0][3] == 'NE':
            temp_2 = [1., 0., 0., 0.]
        if train_data[i][0][3] == 'SE':
            temp_2 = [0., 1., 0., 0.]
        if train_data[i][0][3] == 'NW':
            temp_2 = [0., 0., 1., 0.]
        if train_data[i][0][3] == 'cv':
            temp_2 = [0., 0., 0., 1.]
        temp_3 = train_data[i][0][4:]
        train_data[i][0] = temp_1 + temp_2 + temp_3

    for i in range(0, len(test_data)):
        test_data[i] = [test_data[i][6:], test_data[i][5]]
        temp_1 = test_data[i][0][:3]
        if test_data[i][0][3] == 'NE':
            temp_2 = [1., 0., 0., 0.]
        if test_data[i][0][3] == 'SE':
            temp_2 = [0., 1., 0., 0.]
        if test_data[i][0][3] == 'NW':
            temp_2 = [0., 0., 1., 0.]
        if test_data[i][0][3] == 'cv':
            temp_2 = [0., 0., 0., 1.]
        temp_3 = test_data[i][0][4:]
        test_data[i][0] = temp_1 + temp_2 + temp_3

    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for i in range(0, len(train_data)):
        train_X.append(train_data[i][0])
        train_Y.append(train_data[i][1])
    for i in range(0, len(test_data)):
        test_X.append(test_data[i][0])
        test_Y.append(test_data[i][1])
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_Y = np.array(test_Y)

    ### 保存为.npy方便随时快速读取
    print(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    if platform.system().lower() == 'windows':
        print("windows")
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\train_X.npy", train_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\train_Y.npy", train_Y)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\test_X.npy", test_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\test_Y.npy", test_Y)
    elif platform.system().lower() == 'linux':
        print("linux")    
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/train_X.npy", train_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/train_Y.npy", train_Y)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/test_X.npy", test_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/test_Y.npy", test_Y)

    




def preprocess_fill_KNN():

    date_list=getDateList("2010-01-07","2014-12-25")
    for i in range(0, len(date_list)):
        date_list[i] = date_list[i].split('-')
    for i in range(0, len(date_list)):
        for j in range(0, len(date_list[i])):
            date_list[i][j] = (float)(date_list[i][j])


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
        for j in range(0, len(data_after[0])):
            data_after[i][j] = (float)(data_after[i][j])

    ### 切分数据集

    test_date = []
    test_data = []
    train_data = []
    for i in range(0, len(date_list), 7):
        test_date.append(date_list[i])

    for i in range(0, len(test_date)):
        for j in range(0, len(data_after)):
            if test_date[i][0] == data_after[j][1] and test_date[i][1] == data_after[j][2] and test_date[i][2] == data_after[j][3]:
                test_data.append(j)

    for i in range(0, len(data_after)):
        if i not in test_data:
            train_data.append(i)

    for i in range(0, len(train_data)):
        train_data[i] = data_after[train_data[i]][5:]

    for i in range(0, len(test_data)):
        test_data[i] = data_after[test_data[i]][5:]


    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for i in range(0, len(train_data)):
        train_X.append(train_data[i][1:])
        train_Y.append(train_data[i][0])
    for i in range(0, len(test_data)):
        test_X.append(test_data[i][1:])
        test_Y.append(test_data[i][0])

    ### 保存为.npy方便随时快速读取
    print(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    if platform.system().lower() == 'windows':
        print("windows")
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\train_X.npy", train_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\train_Y.npy", train_Y)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\test_X.npy", test_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "\data\\test_Y.npy", test_Y)
    elif platform.system().lower() == 'linux':
        print("linux")    
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/train_X.npy", train_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/train_Y.npy", train_Y)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/test_X.npy", test_X)
        np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/test_Y.npy", test_Y)

def save_as_npy(train_X, train_Y, test_X, test_Y):
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/train_X.npy", train_X)
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/train_Y.npy", train_Y)
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/test_X.npy", test_X)
    np.save(os.path.abspath(os.path.dirname(os.path.dirname(__file__))) + "/data/test_Y.npy", test_Y)