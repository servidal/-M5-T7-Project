import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
import json



from utils import resize_text_embeds
 


class Img2TextDataset(Dataset):
    def __init__(self, img_features_file: str, text_features_file: str, mode: str,dataset_dir):

        assert img_features_file.endswith('.mat') and text_features_file.endswith('npy')
        assert mode == 'train' or mode == 'test' or mode == 'val'
        #load 
        self.img_features = loadmat(img_features_file)['feats']
        self.text_features = np.load(text_features_file, allow_pickle=True)
        self.text_features = resize_text_embeds(self.text_features)
        
        self.train_percetange = 0.75
        self.val_percetange = 0.90
        self.test_percentage = 1
        self.mode = mode

        
        if mode == 'train':
            with open(dataset_dir + 'train.json', 'rb') as f:
                train_ann = json.load(f)
                train_ids = [ann['imgid'] for ann in train_ann]
                self.img_features = self.img_features[:,train_ids]
                self.text_features = self.text_features[train_ids]

        elif mode == 'val':
            with open(dataset_dir + 'val.json', 'rb') as f:
                val_ann = json.load(f)
                val_ids = [ann['imgid'] for ann in val_ann]
                self.img_features = self.img_features[:,val_ids]
                self.text_features = self.text_features[val_ids]
        else:
            with open(dataset_dir + 'test.json', 'rb') as f:
                test_ann = json.load(f)
                test_ids = [ann['imgid'] for ann in test_ann]
                self.img_features = self.img_features[:,test_ids]
                self.text_features = self.text_features[test_ids]


    def __getitem__(self, index):
        if self.mode != 'test':
            image = self.img_features[:, index]
            positive_cap_sub_id = np.random.randint(0, self.text_features.shape[1] - 1)
            pos_caption = self.text_features[index][positive_cap_sub_id]

            while True:
                negative_cap_id = np.random.randint(0, self.text_features.shape[1] - 1)
                if negative_cap_id != index:
                    break

            # neg caption extraction
            negative_cap_sub_id = np.random.randint(0, self.text_features.shape[1] - 1)
            negative_caption = self.text_features[negative_cap_id][negative_cap_sub_id]

            return image, pos_caption, negative_caption
        else:
            return    self.img_features[:, index],self.text_features[index]

    def __len__(self):
        return self.img_features.shape[1]
