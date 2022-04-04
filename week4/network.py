import torch.nn as nn
import torch.nn.functional as F

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
        
        #self.logmax = nn.LogSoftmax(dim=1)216, 128)

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

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(3, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(10816, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)