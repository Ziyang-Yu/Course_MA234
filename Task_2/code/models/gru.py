import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class gru_model(nn.Module):

    def __init__(self, input_size=13, hidden_size=32):
        super(gru_model,self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
 
        self.linear_1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=1)

        self.h_0 = torch.zeros(1, 4, hidden_size).cuda()
 
    def forward(self,x):
        h_0 = self.h_0.cuda()
        # print(x.size())
        output, h_n = self.gru.forward(x, h_0)
        x = self.linear_1.forward(output)
        x = self.leaky_relu.forward(x)
        x = self.linear_2.forward(output)
        return x








def train_one_epoch(model, epoch_index, training_loader, optimizer, loss_fn):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        inputs = inputs.float().cuda()
        labels = labels.float().cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)


        outputs = torch.squeeze(outputs)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        
        if i % 300 == 299:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))

            running_loss = 0.
        
    
    #print(last_loss)
    return last_loss



