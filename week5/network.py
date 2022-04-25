from torch import nn 
from torchvision import models

class LinearEncoder(nn.Module):
    def __init__(self, dim=4096,embedding_size = 512):
        super(LinearEncoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(dim, 2048),
                                 nn.ReLU(),
                                 nn.Linear(2048, embedding_size))
        
        self.apply(self.init_weights)


    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)
    
    def forward(self, x):
        out = self.fc1(x)
        return out
