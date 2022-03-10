import torch
import torch.nn as nn
from torch.nn import functional


def conv_block(in_channels, out_channels, kernel_size=3, padding='same'):
    """
    Function to ensure He normal initialization and to create a convolutional block: conv + BN + Elu
    """
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
        nn.Linear(64,8),
        nn.LogSoftmax()
        )
    
    return model