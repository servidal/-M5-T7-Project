import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset



from utils import resize_text_embeds
 


class Img2TextDataset(Dataset):
    def __init__(self, img_features_file: str, text_features_file: str, mode: str):

        assert img_features_file.endswith('.mat') and text_features_file.endswith('npy')
        assert mode == 'train' or mode == 'test' or mode == 'val'
        #load 
        self.img_features = loadmat(img_features_file)['feats']
        self.text_features = np.load(text_features_file, allow_pickle=True)
        self.text_features = resize_text_embeds(self.text_features)
        self.train_percetange = 0.75
        self.val_percetange = 0.90
        self.test_percentage = 1

        
        if mode == 'train':
            self.img_features = self.img_features[:, :int(self.train_percetange*len(self.img_features))]
            self.text_features = self.text_features[:int(self.train_percetange*len(self.text_features)), :]
        elif mode == 'val':
            self.img_features = self.img_features[:, int(self.train_percetange*len(self.img_features)):int(self.val_percetange*len(self.img_features))]
            self.text_features = self.text_features[int(self.train_percetange*len(self.text_features)):int(self.val_percetange*len(self.text_features)), :]
        else:
            self.img_features = self.img_features[:, int(self.val_percetange*len(self.img_features)):-1]
            self.text_features = self.text_features[int(self.val_percetange*len(self.text_features)):-1, :]

    def __getitem__(self, index):
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

    def __len__(self):
        return self.img_features.shape[1]
