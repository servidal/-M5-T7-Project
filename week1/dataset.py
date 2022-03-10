import os
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(dataset_dir, input_size, batch_size, num_workers=4):
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
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = mean, std = std)])

    test_transform = transforms.Compose([
                        transforms.Resize((input_size, input_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean = mean, std = std)])    


     # prepare datasets
    train_data = datasets.ImageFolder(dataset_dir+'train', transform = train_transform)
    test_data = datasets.ImageFolder(dataset_dir+'test', transform = test_transform)

    # prepare dataloaders
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


