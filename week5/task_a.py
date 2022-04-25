import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import datetime
import numpy as np
import torch


import umap
import matplotlib.pyplot as plt

from torch import optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from network import LinearEncoder
from dataset import  get_dataloaders, Img_text_Dataset

from sklearn.neighbors import KNeighborsClassifier
from pytorch_metric_learning import distances, losses, miners, reducers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from scipy import io
    
DATA_PATH = '../flickr30k'
OUT_PATH = "results"
LOGS_DIR = os.path.join(OUT_PATH, 'log')
PLOTS_DIR = os.path.join(OUT_PATH, 'plots')
MODEL_WEIGHT_DIR = os.path.join(OUT_PATH, "model")
BATCH_SIZE = 16
EPOCHS = 100
ANCHOR = "text" # text or image
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(device_str)
os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_WEIGHT_DIR, exist_ok=True)

distance = distances.CosineSimilarity()
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
mining_func = miners.TripletMarginMiner(
    margin=0.2, distance=distance, type_of_triplets="all"
)

BERT_FEATURES_FILENAME = "bert_features.npy"
VGG_FEATURES_FILENAME = "vgg_feats.mat"

vgg_features_path = os.path.join(DATA_PATH, VGG_FEATURES_FILENAME)
bert_features_path = os.path.join("results", BERT_FEATURES_FILENAME)
train_set = Img_text_Dataset(vgg_features_path, bert_features_path,'train',DATA_PATH)


train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# TEXT & IMGS MODELS
image_model = LinearEncoder(dim=4096, embedding_size = 512)
text_model = LinearEncoder(dim=300, embedding_size = 512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_model.to(device)
text_model.to(device)    # init weights    


# optimizer
params = list(image_model.parameters())
params += list(text_model.parameters())

optimizer = optim.Adam(params, lr=0.02)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40) #Learning rate is divided by 2 every 40 epochs

# training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for i, img_triple in enumerate(train_dataloader):

        # execute image_triple
        img_features, pos_text_features, neg_text_features = img_triple
        img_features, pos_text_features, neg_text_features = img_features.to(
            device), pos_text_features.to(device), neg_text_features.to(device)
        image_encoded = image_model(img_features)
        pos_text_encoded = text_model(pos_text_features)
        neg_text_encoded = text_model(neg_text_features)

        loss = loss_func(image_encoded, pos_text_encoded, neg_text_encoded)

        optimizer.zero_grad()
        loss.backward()
        total_loss += loss
        
        optimizer.step()

        print("epoch " +str(epoch))
        print("total_loss " +str(total_loss))
