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
from dataset import  get_dataloaders

from sklearn.neighbors import KNeighborsClassifier
from pytorch_metric_learning import distances, losses, miners, reducers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from scipy import io


DATA_PATH = '../flickr30k'
OUT_PATH = "results"
LOGS_DIR = os.path.join(OUT_PATH, 'log')
PLOTS_DIR = os.path.join(OUT_PATH, 'plots')
MODEL_WEIGHT_DIR = os.path.join(OUT_PATH, "model")
BATCH_SIZE = 2
EPOCHS = 100
ANCHOR = "text" # text or image
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(device_str)
os.makedirs(OUT_PATH, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_WEIGHT_DIR, exist_ok=True)


#Create summary writer
train_logdir = os.path.join(LOGS_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'train')
val_logdir = os.path.join(LOGS_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), 'validation')
train_writer = SummaryWriter(log_dir=train_logdir)

accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=5)

def validate(valid_dataloader, image_model, text_model):
    all_img_features = []
    all_txt_features = []

    image_model.eval()
    text_model.eval()

    with torch.no_grad():
        for _, (img_features, txt_features) in enumerate(valid_dataloader):
            img_features = img_features.to(device)  # (batch, ifeatures)
            txt_features = txt_features.to(device)  # (batch, ncaptions, tfeatures)

            # Reshape textual features so they are all encoded at once

            img_encoded = image_model(img_features)
            txt_encoded = text_model(txt_features)

            all_img_features.append(img_encoded.detach().to("cpu").numpy())
            all_txt_features.append(txt_encoded.detach().to("cpu").numpy())

    all_img_features = np.vstack(all_img_features)
    all_txt_features = np.vstack(all_txt_features)

    all_txt_labels = np.arange(len(all_img_features)).repeat(
        len(all_txt_features) // len(all_img_features)
    )

    knn = KNeighborsClassifier(5, metric="euclidean")

    knn = knn.fit(all_txt_features, all_txt_labels)
    neighbors = knn.kneighbors(all_img_features, return_distance=False)
    predictions = all_txt_labels[neighbors]

    accuracies = accuracy_calculator.get_accuracy(
        all_txt_labels, predictions, False
    )

    image_model.train()
    text_model.train()

    return accuracies["precision_at_1"], accuracies["precision_at_5"]

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


img_features_vgg = io.loadmat(vgg_features_path)['feats']
text_features_bert = np.load(bert_features_path, allow_pickle=True)
train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
    DATA_PATH, BATCH_SIZE, img_features_vgg, text_features_bert
)

image_model = LinearEncoder(dim=4096, embedding_size = 512)
text_model = LinearEncoder(dim=768, embedding_size = 512)

device = torch.device(device_str)
image_model.to(device)
text_model.to(device)

params = list(image_model.parameters())
params += list(text_model.parameters())

optimizer = optim.Adam(params, lr=0.02)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40) #Learning rate is divided by 2 every 40 epochs

image_model.train()
text_model.train()
# training loop
for epoch in range(EPOCHS):
    print("start epoch " +  str(epoch))
    for i, (img_features, txt_features) in enumerate(train_dataloader):
        img_features = img_features.to(device)  # (batch, ifeatures)
        txt_features = txt_features.to(device)  # (batch, ncaptions, tfeatures)

        img_embeding = image_model(img_features)
        txt_embeding = text_model(txt_features)

        img_labels = torch.arange(BATCH_SIZE)
        txt_labels = torch.arange(BATCH_SIZE)


        indices_tuple = mining_func(img_embeding, img_labels, txt_embeding, txt_labels)
        loss = loss_func(img_embeding, img_labels, indices_tuple, ref_emb=txt_embeding, ref_labels=txt_labels)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        train_writer.add_scalar('train loss',loss, epoch) #log training loss for one epoch to Tensorboard


    press1, press5 = validate(val_dataloader, image_model, text_model, ANCHOR, epoch)
    print(press5)
    scheduler.step()

state_dict = [image_model.state_dict(), text_model.state_dict()]
weigths_paths = os.path.join(MODEL_WEIGHT_DIR, ANCHOR + '_weights.pth')
torch.save(state_dict, weigths_paths)

print("done")