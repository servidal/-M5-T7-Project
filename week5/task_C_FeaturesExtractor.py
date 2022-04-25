import os
import numpy as np
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

"""
#Code to change image filenames from Flicker30k to keep the order from .json files-> 00000.jps, 00001.jpg, ...
with open('Flickr30k/dataset.json','rb') as f:
    data = json.load(f)

for i in range(len(data["images"])):
    filename = data["images"][i]["filename"]
    img = cv2.imread(os.path.join('Flickr30k/images', filename))
    cv2.imwrite("Flickr30k/imgs/%05d.jpg" %i,  img.astype(int))
"""

class Flickr30kImages(Dataset):
    def __init__(self, dir):
        self.img_path = dir
        self.images = os.listdir(self.img_path)
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_path, self.images[idx]))
        tensor_img = self.transforms(image)
        return tensor_img


img_path = 'Flickr30k/imgs/'
res_path = "Flickr30k/"


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = Flickr30kImages(img_path)
    dataloader = DataLoader(data, batch_size=32, shuffle=False)

    #create the Faster RCNN model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained = True) #models.resnet50(pretrained=True)
    model.to(device)
    print(model)
    #print(model.backbone.body)
    #print(model.backbone.body._modules.keys())

    model.eval()

    #Extracting visual features
    feats= np.zeros((2048,len(os.listdir(img_path))))
    k=0
    for batch_idx, images in enumerate(dataloader):
        
        with torch.no_grad():
            features = model.backbone.body(images.to(device))  # Get feature maps from that batch
            #print(features.keys())
        features = features['3'].cpu().detach().numpy()  #taking features from last resnet block of backbone
        #print("Initial features shape:", features.shape)
        
        # Reduce the size of the features tensor by doing average pooling
        features = np.mean(features, axis=(2, 3))
        #print("After average pooling:", features.shape)

        all_features = np.transpose(features)
        #print("Final features shape:", all_features.shape)

        if batch_idx != 0:
            feats[:,k:k+len(images)] = all_features #store features in an array
        else:
            feats[:,k:k+len(images)] = all_features

        k += len(images)
        

    #Saving visual features     
    with open('{}/FasterRCNN_features.mat'.format(res_path), "wb") as f:
        np.save(f, feats)       