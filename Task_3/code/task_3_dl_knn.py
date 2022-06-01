#!/usr/bin/env python
# coding: utf-8

# ## Experiments of different models

# ### Get Experimental Data

# In[ ]:


import numpy as np
import os
import Preprocess_nosplit
import platform

Preprocess_nosplit.preprocess_fill_KNN()



file_path = os.path.abspath('') 

if platform.system().lower() == 'windows':
    print("windows")
    dataset_X = np.load(file_path + "\Task_3\data\dataset_X.npy")
    dataset_Y = np.load(file_path + "\Task_3\data\dataset_Y.npy")
elif platform.system().lower() == 'linux':
    print("linux")
    dataset_X = np.load(file_path + "/Task_3/data/dataset_X.npy")
    dataset_Y = np.load(file_path + "/Task_3/data/dataset_Y.npy")


data_X = []
data_Y = []

num = 24

for i in range(0, len(dataset_X)-num, 24):
    data_X.append(dataset_X[i:i+num])
    data_Y.append(dataset_Y[i:i+num])
    
data_X = np.array(data_X)
data_Y = np.array(data_Y)

train_X = data_X[:1200]
train_Y = data_Y[:1200]
test_X = data_X[1200:]
test_Y = data_Y[1200:]

Preprocess_nosplit.save_as_npy(train_X, train_Y, test_X, test_Y)


# In[ ]:


import numpy as np
import platform

if platform.system().lower() == 'windows':
    print("windows")
    train_X = np.load(file_path + '\Task_3\data\\train_X.npy')
    train_Y = np.load(file_path + '\Task_3\data\\train_Y.npy')
    test_X = np.load(file_path + '\Task_3\data\\test_X.npy')
    test_Y = np.load(file_path + '\Task_3\data\\test_Y.npy')
elif platform.system().lower() == 'linux':
    print("linux")
    train_X = np.load(file_path + '/Task_3/data/train_X.npy')
    train_Y = np.load(file_path + '/Task_3/data/train_Y.npy')
    test_X = np.load(file_path + '/Task_3/data/test_X.npy')
    test_Y = np.load(file_path + '/Task_3/data/test_Y.npy')



# ### Construct dataset

# In[ ]:


from models.customed_dataset import customed_dataset
import numpy as np
import torch
import os


# dataset_X = np.load(file_path + "\Task_2\data\\dataset_X.npy")
# dataset_Y = np.load(file_path + "\Task_2\data\\dataset_Y.npy")

# dataset = customed_dataset(dataset_X, dataset_Y)

# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_dataset = customed_dataset(train_X, train_Y)
test_dataset = customed_dataset(test_X, test_Y)

training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)


# ### LSTM model

# In[ ]:


from models import lstm

from datetime import datetime
import torch


model = lstm.lstm_model().cuda()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0

EPOCHS = 2000

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = lstm.train_one_epoch(model, epoch_number, training_loader, optimizer, loss_fn)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.float().cuda()
        vlabels = vlabels.float().cuda()
        voutputs = model(vinputs)
        voutputs = torch.squeeze(voutputs)
        #print(voutputs.size())
        #print(vlabels.size())
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))


    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        if platform.system().lower() == 'windows':
            print("windows")
            model_path = file_path + '\Task_3\code\models\LSTM_model_knn'
        elif platform.system().lower() == 'linux':
            print("linux")
            model_path = file_path + '/Task_3/code/models/LSTM_model_knn'
        torch.save(model.state_dict(), model_path)

    epoch_number += 1


# ### GRU

# In[ ]:


from models import gru

from datetime import datetime
import torch

model = gru.gru_model().cuda()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
epoch_number = 0

EPOCHS = 2000

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = gru.train_one_epoch(model, epoch_number, training_loader, optimizer, loss_fn)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    h_0 = torch.zeros(1, 4, 32).cuda()
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.float().cuda()
        vlabels = vlabels.float().cuda()
        if vinputs.size()[0] != 4:
            continue    
        voutputs = model(vinputs)
        voutputs = torch.squeeze(voutputs)
        #print(voutputs.size())
        #print(vlabels.size())
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))



    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        if platform.system().lower() == 'windows':
            print("windows")
            model_path = file_path + '\Task_3\code\models\GRU_model_knn'
        elif platform.system().lower() == 'linux':
            print("linux")
            model_path = file_path + '/Task_3/code/models/GRU_model_knn'
        torch.save(model.state_dict(), model_path)

    epoch_number += 1


# ### Tranformer

# In[4]:


from models import transformer

from datetime import datetime
import torch

model = transformer.transformer_model().cuda()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

epoch_number = 0

EPOCHS = 2000

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = transformer.train_one_epoch(model, epoch_number, training_loader, optimizer, loss_fn)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    # h_0 = torch.zeros(1, 4, 32).cuda()
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        vinputs = vinputs.float().cuda()
        vlabels = vlabels.float().cuda()
        if vinputs.size()[0] != 4:
            continue    
        voutputs = model(vinputs)
        voutputs = torch.squeeze(voutputs)
        #print(voutputs.size())
        #print(vlabels.size())
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))



    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        if platform.system().lower() == 'windows':
            print("windows")
            model_path = file_path + '\Task_3\code\models\Transformer_model_knn'
        elif platform.system().lower() == 'linux':
            print("linux")
            model_path = file_path + '/Task_3/code/models/Transformer_GRU_model_knn'
        torch.save(model.state_dict(), model_path)

    epoch_number += 1


# ### Test of Models

# #### Load Models

# In[5]:


import torch

from models import lstm
from models import gru
from models import transformer



### lstm
model_lstm = lstm.lstm_model()
model_lstm.load_state_dict(torch.load('models/LSTM_model_knn'))
model_lstm.eval()

### gru
model_gru = gru.gru_model()
model_gru.load_state_dict(torch.load('models/GRU_model_knn'))
model_gru.eval()

### transformer
model_transformer = transformer.transformer_model()
model_transformer.load_state_dict(torch.load('models/Transformer_model_knn'))
model_transformer.eval()


# #### lstm test

# In[6]:


import torch
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score

lstm_predictions = []
lstm_targets = []

validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

for batch_idx, (data, target) in enumerate(validation_loader):
    lstm_predictions.append(model_lstm.forward(data.float()).detach().numpy().tolist())
    lstm_targets.append(target.numpy().tolist())
lstm_predictions = torch.squeeze(torch.tensor(lstm_predictions)).reshape(-1).numpy()
lstm_targets = torch.squeeze(torch.tensor(lstm_targets)).reshape(-1).numpy()

print(mean_squared_error(lstm_targets, lstm_predictions))
print(mean_absolute_error(lstm_targets, lstm_predictions))
print(r2_score(lstm_targets, lstm_predictions))


# ### gru test

# In[7]:


import torch
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score

model_gru = model_gru.cuda()

gru_predictions = []
gru_targets = []

validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

for batch_idx, (data, target) in enumerate(validation_loader):
    if len(data) != 4:
        continue
    gru_predictions.append(model_gru.forward(data.float().cuda()).cpu().detach().numpy().tolist())
    gru_targets.append(target.numpy().tolist())
gru_predictions = torch.squeeze(torch.tensor(gru_predictions)).reshape(-1).numpy()
gru_targets = torch.squeeze(torch.tensor(gru_targets)).reshape(-1).numpy()

print(mean_squared_error(gru_targets, gru_predictions))
print(mean_absolute_error(gru_targets, gru_predictions))
print(r2_score(gru_targets, gru_predictions))


# ### Transformer

# In[8]:


import torch
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
from sklearn.metrics import r2_score

model_transformer = model_transformer.cuda()

transformer_predictions = []
transformer_targets = []

validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

for batch_idx, (data, target) in enumerate(validation_loader):
    if len(data) != 4:
        continue
    transformer_predictions.append(model_transformer.forward(data.float().cuda()).cpu().detach().numpy().tolist())
    transformer_targets.append(target.numpy().tolist())
transformer_predictions = torch.squeeze(torch.tensor(transformer_predictions)).reshape(-1).numpy()
transformer_targets = torch.squeeze(torch.tensor(transformer_targets)).reshape(-1).numpy()

print(mean_squared_error(transformer_targets, transformer_predictions))
print(mean_absolute_error(transformer_targets, transformer_predictions))
print(r2_score(transformer_targets, transformer_predictions))

