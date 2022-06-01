import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class transformer_model(nn.Module):

    def __init__(self, d_model = 13):
        super(transformer_model,self).__init__()
        
        self.transformer_model = nn.Transformer(
            d_model = d_model,
            nhead = 13,
            batch_first = True,
        )
 
        self.out = nn.Linear(in_features=d_model, out_features=1)
 
    def forward(self, src):
        """
        tgt = tgt.unsqueeze(2)
        padding = (
            0, 12,
            0, 0,
            0, 0
        )
        tgt = F.pad(tgt, padding).cuda()       
        """
        tgt = torch.zeros(size=src.size()).cuda()
        # src : (batch_size, time_length, features)
        # tgt : (batch_size, time_length, PM2.5fill)
        output = self.transformer_model.forward(src=src, tgt=tgt) # output: (batch_size, time_length, features)
        output = self.out(output) # output: (batch_size, time_length, 1)
        output = torch.squeeze(output) # # output: (batch_size, time_length)
        return output



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

        # Transform dim of labels

        # Make predictions for this batch
        outputs = model.forward(src=inputs)

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
            print('batch {} loss: {}'.format(1000, last_loss))
            running_loss = 0.
        
    
    #print(last_loss)
    return last_loss
