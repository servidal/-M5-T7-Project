import os
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.io import read_image
from typing import Any, Callable, List, Optional, Tuple
import torch
import cv2



def get_dataloaders(dataset_dir, input_size, batch_size, kwargs):
    """
    Get train and test dataloaders from the dataset
    """
    def read_pil_image(img_path, height, width): 
        with open(img_path, 'rb') as f:
            return np.array(Image.open(f).convert('RGB').resize((width, height)))

    def load_all_images(dataset_path, height, width, img_ext='jpg'):
        return np.array([read_pil_image(str(p), height, width) for p in 
                                        Path(dataset_path).rglob("*."+img_ext)]) 

    # Load the data 
    loaded_imgs_train = load_all_images(dataset_dir+'/train/', input_size, input_size)
    
    #calculate mean and standard deviation per channels
    mean = [(loaded_imgs_train[..., i]/255).mean() for i in range(loaded_imgs_train.shape[-1])]
    #print(mean)
    std = [(loaded_imgs_train[..., i]/255).std() for i in range(loaded_imgs_train.shape[-1])]
    #print(std)
    
    #data transforms for trainig and testing
    train_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=0.5),
                        #transforms.RandomRotation(degrees=20),
                        transforms.RandomAffine(degrees=20, translate=(0, 0.3)),
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = mean, std = std)])

    test_transform = transforms.Compose([
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = mean, std = std)])    


    mit_dataset_train = MitDataset(os.path.join(dataset_dir,'train'),  True, train_transform)
    mit_dataset_test = MitDataset(os.path.join(dataset_dir,'test'),  False, test_transform)


    siamese_train_dataset = SiameseDataset(mit_dataset_train, True) # Returns pairs of images and target same/different
    siamese_test_dataset = SiameseDataset(mit_dataset_test, False)

    # prepare dataloaders
    train_loade_siamese = DataLoader(siamese_train_dataset, batch_size = batch_size, shuffle=True, **kwargs)
    test_loader_siamese = DataLoader(siamese_test_dataset, batch_size = batch_size, shuffle=False, **kwargs)
    
    train_loader = DataLoader(mit_dataset_train, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(mit_dataset_test, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loade_siamese, test_loader_siamese, train_loader, test_loader


class MitDataset(Dataset):
    LABELS_DICT = {"Opencountry":0 ,  "coast":1,   "forest":2, "highway":3, "inside_city":4,  "mountain":5 , "street":6,  "tallbuilding":7}

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super(MitDataset, self).__init__()
        self.train = train  # training set or test set
        self.data_path = root
        self.data, self.targets = self._load_data()
        self.transform = transform
        self.target_transform = target_transform

    def _load_data(self):
        train_imgs = []
        train_label = []
        for class_t in os.listdir(self.data_path):
            for img in os.listdir(os.path.join(self.data_path, class_t)):
                if ".jpg" in img:
                    #print(os.path.join(self.data_path, class_t, img))
                    #train_imgs.append(read_image(os.path.join(self.data_path, class_t, img)))
                    train_imgs.append(os.path.join(self.data_path, class_t, img))
                    train_label.append(self.LABELS_DICT[class_t])

        return train_imgs, torch.LongTensor(train_label)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img)

        #target= torch.tensor(target).float()
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        #img = torch.permute(img, (2,0,1))

        return img, target

    def __len__(self) -> int:
        return len(self.data)



LABELS_DICT = {"Opencountry":0 ,  "coast":1,   "forest":2, "highway":3, "inside_city":4,  "mountain":5 , "street":6,  "tallbuilding":7}
class SiameseDataset(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, dataset, train):
        
        self.dataset = dataset
        self.train = train

        if self.train:
            self.train_labels = self.dataset.targets
            #self.train_data = self.dataset.data
            #self.train_labels = torch.LongTensor(train_label)
            #self.train_data = train_imgs
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.dataset.targets
            #self.test_data = self.dataset.data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_labels), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_labels), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.dataset[index][0], self.dataset[index][1]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.dataset[siamese_index][0]
        else:
            img1 = self.dataset[self.test_pairs[index][0]][0]
            img2 = self.dataset[self.test_pairs[index][1]][0]
            target = self.test_pairs[index][2]

        return (img1.float(), img2.float()), torch.tensor(target).float()

    def __len__(self):
        return len(self.dataset.data)