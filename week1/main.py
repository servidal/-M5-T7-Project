import os
import datetime
from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import dataset
import model_creator


##Instantiate Tensorboard Writer
#Create log folders
root='../week1/experiments/'
train_logdir = os.path.join(root, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'train')
val_logdir = os.path.join(root, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'validation')
#Create summary writer
train_writer = SummaryWriter(log_dir=train_logdir)
val_writer = SummaryWriter(log_dir=val_logdir)


# check for CUDA availability
if torch.cuda.is_available():
    print('CUDA is available, setting device to CUDA')
# set device to  CUDA for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Variables
DATASETDIR = '../MIT_split/'
MODEL_FNAME = 'model.h5'
BATCH_SIZE = 32
EPOCHS = 100
INPUT_SIZE = 64


# get dataloaders
train_loader, test_loader = dataset.get_dataloaders(DATASETDIR, INPUT_SIZE, BATCH_SIZE)

# Create the model
model = model_creator.get_model()
summary(model, (3, INPUT_SIZE, INPUT_SIZE), device='cpu')

#Send model to GPU
model.to(device)

#Define Loss function and optimizer
criterion = nn.CrossEntropyLoss()   #logsoftmax layer + NLLLoss
optimizer = optim.Adam(model.parameters(), lr=0.02)

#Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40) #Learning rate is divided by 2 every 40 epochs


#Accuracy function
def accuracy(output, target):
  pred = output.argmax(dim=1)  # get the index of the max log-probability
  return (pred == target).sum().item() / target.numel()  #return the mean accuracy in the batch


#Train function
def train_epoch(model, train_loader, optimizer, criterion, epoch, scheduler):   #Training function for one epoch
  
    model.train() #train mode
    
    losses = []
    accs = []
    lr = scheduler.get_last_lr()    

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        #send to GPU device
        inputs, labels = inputs.to(device), labels.to(device)
        #set all gradients to zero
        optimizer.zero_grad()  
        #forward pass
        outputs = model(inputs)
        #compute the loss in the batch
        loss = criterion(outputs, labels)
        #backward pass
        loss.backward()
        #Optimize: update the parameters
        optimizer.step()


        #Calculate the mean accuracy in the batch
        acc = 100 * accuracy(outputs, labels)  
        losses.append(loss.item()) #save the loss value in a list of losses
        accs.append(acc) #save the accuracy value in a list of accuracies
        

        if batch_idx >= len(train_loader):
            print('Train Epoch: {} \tLR: {} \tAverage Loss: {:.4f}\tAverage Acc: {:.2f} %'.format(
                epoch, lr , np.mean(losses), np.mean(accs)))
            
            train_writer.add_scalar('Loss', np.mean(losses), epoch) #log training loss for one epoch to Tensorboard
            train_writer.add_scalar('Acc', np.mean(accs), epoch)    #log training accuracy for one epoch to Tensorboard

    return np.mean(losses), np.mean(accs)


#Validation function
def eval_epoch(model, test_loader, criterion, epoch):  #evaluation function after one epoch of training
  
    model.eval()     #eval mode
      
    with torch.no_grad():
        eval_losses = []
        eval_accs = []
        for (inputs, labels) in test_loader:
            #send to GPU device
            inputs, labels = inputs.to(device), labels.to(device)            
            #forward pass
            outputs = model(inputs)  
            #compute the loss in the batch
            eval_loss = criterion(outputs, labels)
            eval_losses.append(eval_loss.item()) #save loss value
            #compute the accuracy in the batch
            eval_acc = 100 * accuracy(outputs, labels) 
            eval_accs.append(eval_acc) #save the accuracy value 
            
        print('Val Epoch: {} \tAverage loss: {:.4f}\tAverage Acc: {:.2f} %'.format(
            epoch, np.mean(eval_losses), np.mean(eval_accs)))
        
        val_writer.add_scalar('Loss', np.mean(eval_losses), epoch)  #log validation loss for one epoch to Tensorboard
        val_writer.add_scalar('Acc',  np.mean(eval_accs), epoch)    #log validation accuracy for one epoch to Tensorboard

    return np.mean(eval_losses), np.mean(eval_accs)    


def train_net(model, train_loader, val_loader, optimizer, criterion, num_epochs, scheduler): 
    """ Function that trains and evals a network for n epochs.
    """
    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, epoch, scheduler)  #train the model
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, epoch)                         #eval the model 
    
        scheduler.step() #Step for LR decay
    
        if best_accuracy < val_acc:
            best_accuracy = val_acc
            torch.save(model.state_dict(), train_logdir + '/best_params.pt')  #save the model state for best val accuracy
  
    return best_accuracy  

best_accuracy = train_net(model, train_loader, test_loader, optimizer, criterion, EPOCHS, scheduler)
print('Best validation accuracy = ', best_accuracy)
