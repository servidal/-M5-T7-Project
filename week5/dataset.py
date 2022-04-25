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