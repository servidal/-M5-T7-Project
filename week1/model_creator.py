import torch
import torch.nn as nn
from torch.nn import functional
import torch.nn.functional as F


"""
def conv_block(in_channels, out_channels, kernel_size=3, padding='same'):
    #Function to ensure He normal initialization and to create a convolutional block: conv + BN + Elu
     
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
    nn.init.kaiming_normal_(conv.weight)  #He normal weigths initialization
    
    return nn.Sequential(conv, nn.ELU(), nn.BatchNorm2d(out_channels))


def get_model():
    model = nn.Sequential(
        conv_block(3, 32),
        conv_block(32, 32),
        nn.MaxPool2d(2),
        nn.Dropout(0,3),

        conv_block(32, 64),
        conv_block(64, 64),
        nn.MaxPool2d(2),
        nn.Dropout(0,5),

        nn.AvgPool2d(16),  #nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64,8)
        )
    
    return model
"""


# define the CNN architecture
class Net(nn.Module):

    def __init__(self, in_channels=3, kernel_size=3, padding="same"):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size, padding=padding)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size, padding=padding)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.bn1_2 = nn.BatchNorm2d(32)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size, padding=padding)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size, padding=padding)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.bn2_2 = nn.BatchNorm2d(64)
        
        self.max_pool = nn.MaxPool2d(2,2)
        
        self.dropout = nn.Dropout(0.3)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
    
        self.fc1 = nn.Linear(64, 8)
        
        #self.logmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        ## Define forward behavior
        x = self.conv1_1(x)
        x = F.elu(self.bn1_1(x))
        x = self.conv1_2(x)
        x = F.elu(self.bn1_2(x))
        x = self.max_pool(x)

        x = self.conv2_1(x)
        x = F.elu(self.bn2_1(x))
        x = self.conv2_2(x)
        x = F.elu(self.bn2_2(x))
        x = self.max_pool(x)
        x = self.dropout(x)

        x = self.global_avg_pool(x)
        x = self.flat(x)


        #x = x.view(-1, 64)

        x = self.fc1(x)
        
        return x