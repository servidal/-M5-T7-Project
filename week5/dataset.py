import os
import json

import os
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils import data
import numpy as np
import torch
from scipy import io


BERT_FEATURES_FILENAME = "bert_features.npy"
VGG_FEATURES_FILENAME = "vgg_feats.mat"

def get_dataloaders(dataset_dir, batch_size, img_features, text_features):
    """
    Get train and test dataloaders from the dataset
    """

    train_data = FlickrDatasetBert(dataset_dir, "train", img_features, text_features)
    val_data = FlickrDatasetBert(dataset_dir, "val", img_features, text_features)
    test_data = FlickrDatasetBert(dataset_dir, "test", img_features, text_features)

    # prepare dataloaders
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class FlickrDatasetBert(data.Dataset):

    def __init__(self, dataset_path, dataset_split, vgg_features, text_features):
        image_ids = []

        dataset_split_path = os.path.join(dataset_path, dataset_split + '.json' )
        with open(dataset_split_path) as f:
            split_json = json.load(f)
            for img in split_json:
                image_ids.append(img["imgid"])
            image_ids = np.asarray(image_ids)

        self.img_features = vgg_features.T[image_ids]
        text_array = []
        for i in range(len(text_features)):
            sentence_array = []
            for sent in text_features[i]:
                sentence_array.append(np.mean(sent, axis=0))
            text_array.append(sentence_array)
        self.text_features = np.asarray(text_array)

    def __getitem__(self, index):
        img_features = torch.tensor(self.img_features[index]).float()
        txt_features = torch.tensor(self.text_features[index][np.random.randint(5, size=1)[0]]).float()

        return img_features, txt_features

    def __len__(self):
        return len(self.img_features)

class Img_text_Dataset(data.Dataset):
    def __init__(self, img_features_file, text_features_file, mode,dataset_dir):

        #load 
        self.img_features = io.loadmat(img_features_file)['feats']
        self.text_features = np.load(text_features_file, allow_pickle=True)
        text_array = []
        for i in range(len(self.text_features)):
            sentence_array = []
            for sent in self.text_features[i]:
                sentence_array.append(np.mean(sent, axis=0))
            text_array.append(sentence_array)
        self.text_features = np.asarray(text_array)
        
      
        if mode == 'train':
            os.path.join(dataset_dir, 'train.json')
            with open(os.path.join(dataset_dir, 'train.json'), 'rb') as f:
                train_annotation = json.load(f)
                train_ids = [annotation['imgid'] for annotation in train_annotation]
                self.img_features = self.img_features[:,train_ids]
                self.text_features = self.text_features[train_ids]

        elif mode == 'val':
            with open(os.path.join(dataset_dir, 'val.json'), 'rb') as f:
                val_annotation = json.load(f)
                val_ids = [annotation['imgid'] for annotation in val_annotation]
                self.img_features = self.img_features[:,val_ids]
                self.text_features = self.text_features[val_ids]
        else:
            with open(os.path.join(dataset_dir, 'test.json'), 'rb') as f:
                test_annotation = json.load(f)
                test_ids = [annotation['imgid'] for annotation in test_annotation]
                self.img_features = self.img_features[:,test_ids]
                self.text_features = self.text_features[test_ids]


    def __getitem__(self, index):
        if self.mode != 'test':
            image = self.img_features[:, index]
            caption_id = np.random.randint(0, self.text_features.shape[1] - 1)
            pos_caption = self.text_features[index][caption_id]

            while True:
                negative_cap_id = np.random.randint(0, self.text_features.shape[1] - 1)
                if negative_cap_id != index:
                    break

            negative_cap_sub_id = np.random.randint(0, self.text_features.shape[1] - 1)
            negative_caption = self.text_features[negative_cap_id][negative_cap_sub_id]

            return image, pos_caption, negative_caption
        else:
            return  self.img_features[:, index],self.text_features[index]

    def __len__(self):
        return len(self.img_features)