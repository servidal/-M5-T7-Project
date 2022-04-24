import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from models import ImgEncoder, TextEncoder
from dataset import Img2TextDataset
import os
dataset_dir = '/home/marcelo/Documents/Master_CV/M5/M5-T7-Project/week5/data/'
img_features_file = '/home/marcelo/Documents/Master_CV/M5/M5-T7-Project/week5/data/vgg_feats.mat'
text_features_file = '/home/marcelo/Documents/Master_CV/M5/M5-T7-Project/week5/data/fasttext_feats.npy'
output_path = "./results/task_a/"


import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

parser = ArgumentParser(
        #description='Torch-based image classification system',
        #formatter_class=ArgumentDefaultsHelpFormatter
    )
parser.add_argument("num_epochs",
                    type=int,
                    default=10,
                    )
parser.add_argument("lr",
                    type=float,
                    )
parser.add_argument("weight_decay",
                    type=float,
                    default=1e-3,
                    )
parser.add_argument("batch_size",
                    type=int,
                    default=128,
                    )
parser.add_argument("margin",
                    type=float,
                    default=1,
                    )


args = parser.parse_args()

if __name__ == '__main__':

    #loss_func = nn.TripletMarginLoss(args.margin, p=2)
    loss_func = TripletLoss(args.margin)

    train_set = Img2TextDataset(img_features_file, text_features_file,'train',dataset_dir)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # TEXT & IMGS MODELS
    image_model = ImgEncoder()
    text_model = TextEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model.to(device)
    text_model.to(device)    # init weights    
    

    # optimizer
    params = list(image_model.parameters())
    params += list(text_model.parameters())

    optimizer = Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    for epoch in range(args.num_epochs):
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

        print(f'epoch: {epoch}\titeration: {i}\tLoss: {total_loss}')
    
    model_folder = str(output_path + "/models")
    os.makedirs(model_folder, exist_ok=True)
    torch.save(image_model.state_dict(), '{0}/Image2Text_image.pth'.format(model_folder))
    torch.save(text_model.state_dict(), '{0}/Image2Text_text.pth'.format(model_folder))
