import os
import pickle

import cv2
from detectron2.structures import BoxMode

KITTY_DATASET = '/home/mcv/datasets/KITTI/'
TRAIN_IMAGES_LIST = 'train_kitti.txt'  #'val_kitti.txt'
TRAIN_IMG_PATH = os.path.join(KITTY_DATASET, 'data_object_image_2/training/image_2/')
TRAIN_LABELS_PATH = os.path.join(KITTY_DATASET, 'training/label_2/')

def read_array_file(filename):
    with open(filename) as f:
        data = f.readlines()
        data = [i.replace('\n', '') for i in data]

    return data


def get_dataset_dicts(img_dir):

    classes = {
        'Car': 0,
        'Van': 1,
        'Truck': 2,
        'Pedestrian': 3,
        'Person_sitting': 4,
        'Cyclist': 5,
        'Tram': 6,
        'Misc': 7,
        'DontCare': 8
    }

    train_files = os.path.join(img_dir, TRAIN_IMAGES_LIST)
    labels_files = read_array_file(train_files)

    dataset_dicts = []


    for label_file in labels_files:
        annotations = read_array_file(os.path.join(TRAIN_LABELS_PATH, label_file))
        idx = label_file.split(".")[0]

        record = {}

        filename = os.path.join(TRAIN_IMG_PATH, idx + '.png')
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        for annotation in annotations:
            annot_data = annotation.split(" ")

            obj = {
                "bbox": [
                    int(float(annot_data[4])),
                    int(float(annot_data[5])),
                    int(float(annot_data[6])),
                    int(float(annot_data[7]))
                ],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": classes[annot_data[0]],
            }

            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


if __name__ == '__main__':
    
    train_dict = get_dataset_dicts(KITTY_DATASET)

    train_dict_file = open('train_dict.dat', 'wb')
    pickle.dump(train_dict, train_dict_file)

