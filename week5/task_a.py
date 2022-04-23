import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from models import ImgEncoder, TextEncoder
from dataset import Img2TextDataset
import os

img_features_file = '/home/marcelo/Documents/Master_CV/M5/M5-T7-Project/week5/data/vgg_feats.mat'
text_features_file = '/home/marcelo/Documents/Master_CV/M5/M5-T7-Project/week5/data/fasttext_feats.npy'
output_path = "./results/task_a/"

parser = ArgumentParser(
        #description='Torch-based image classification system',
        #formatter_class=ArgumentDefaultsHelpFormatter
    )
parser.add_argument("num_epochs",
                    type=int,
                    default=10,
                    help="Number of epochs")
parser.add_argument("lr",
                    type=float,
                    help="learning rate")
parser.add_argument("weight_decay",
                    type=float,
                    default=1e-3,
                    help="weight decay")
parser.add_argument("batch_size",
                    type=int,
                    default=128,
                    help="batch size")
parser.add_argument("margin",
                    type=float,
                    default=1e-1,
                    help="change margin for triplet loss")
parser.add_argument("grad_clip",
                    type=int,
                    default=0,
                    help="grad_clip")

args = parser.parse_args()

if __name__ == '__main__':

    loss_func = nn.TripletMarginLoss(args.margin, p=2)

    train_set = Img2TextDataset(img_features_file, text_features_file,'train')
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # TEXT & IMGS MODELS
    image_model = ImgEncoder()
    text_model = TextEncoder()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model.to(device)
    text_model.to(device)
    # init weights
    image_model.init_weights()
    text_model.init_weights()

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
            if args.grad_clip > 0:
                clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

        print(f'epoch: {epoch}\titeration: {i}\tLoss: {total_loss}')
    
    state_dict = [image_model.state_dict(), text_model.state_dict()]
    model_folder = str(output_path + "/models")
    os.makedirs(model_folder, exist_ok=True)
    torch.save(state_dict, '{0}/Image2Text_weights.pth'.format(model_folder))
