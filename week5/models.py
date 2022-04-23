from torch.nn import Module, Linear, ReLU, init, Sequential, Dropout, LayerNorm
import numpy as np
import torch
from torchvision import models


class ImgEncoder(Module):
    def __init__(self, dim=4096,embedding_size = 1000):
        super(ImgEncoder, self).__init__()

        self.linear1 = Linear(dim, embedding_size)
        self.activation = ReLU()
        self.init_weights()

    def init_weights(self):
        # Linear
        init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        x = self.activation(x)
        x = self.linear1(x)
        x = x / x.pow(2).sum(1, keepdim=True).sqrt()
        return x


class TextEncoder(Module):
    def __init__(self, embedding_size = 1000):
        super(TextEncoder, self).__init__()
        self.linear1 = Linear(300, embedding_size)
        self.activation = ReLU()

        self.init_weights()

    def init_weights(self):
        # Linear
        init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        x = self.activation(x)
        x = self.linear1(x)
        x = x / x.pow(2).sum(1, keepdim=True).sqrt()
        return x


class LinearEncoder(Module):
    def __init__(self, input_size: int, layer_sizes: list):
        super(LinearEncoder, self).__init__()
        layers = [Linear(input_size, layer_sizes[0])]

        for ii in range(len(layer_sizes) - 1):
            layers.append(ReLU())
            # layers.append(Dropout(0.5))
            # layers.append(LayerNorm(layer_sizes[ii]))
            layers.append(Linear(layer_sizes[ii], layer_sizes[ii + 1]))

        self.linear = Sequential(*layers)

    def init_weights(self):
        # Linear
        for layer in self.linear:
            if isinstance(layer, Linear):
                init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.linear(x)
        x = x / x.pow(2).sum(1, keepdim=True).sqrt()
        return x


class FasterRCNN(Module):
    def __init__(self, embed_size=4096):
        super(FasterRCNN, self).__init__()
        original_model = models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
        self.fc = Linear(3840, embed_size)
        self.backbone = Sequential(*list(original_model.backbone.children())[:-1])
        self.features = Sequential(self.fc)

    def forward(self, x):
        x = self.backbone(x)
        outputs = []
        for v in x.values():
            outputs.append(v.reshape(v.shape[0], v.shape[1], -1).max(dim=-1)[0])
        out = torch.cat(outputs, dim=1)
        out = self.fc(out)
        return out
