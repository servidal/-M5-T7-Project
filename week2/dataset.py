import os
import cv2

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

import mots_utils

DATASET_PATH = '/home/group07/M5-T7-Project/KITTI-MOTS'

TRAINING_SEQ = ["0011","0017","0009","0020","0019","0005","0000","0015","0001", "0004" , "0003" , "0012"]
TESTING_SEQ = ["0002","0006" ,"0007" ,"0008" ,"0010" ,"0013" ,"0014" ,"0016" ,"0018"]
CLASSES = ['Cars', 'Pedestrian']
CLASSES_MAP = {1:2,2:0}

def get_dataset_files(dataset_path, type_seq):
    sequence_map = {
        "train": TRAINING_SEQ,
        "test": TESTING_SEQ
    }
    sequence = sequence_map[type_seq]
    image_paths = []
    instances_path = []

    for seq in sequence:
        image_paths.append(os.path.join(dataset_path, "training/image_02", seq))
        instances_path.append(os.path.join(dataset_path, 'instances_txt', seq + '.txt'))

    image_folders = sorted(image_paths)
    instances_txts = sorted(instances_path)
    return [(folder,txt) for folder,txt in zip(image_folders, instances_txts)]

def get_dataset_dicts(dataset_path, type_seq):
    dataset_dicts = []
    for train_folder, train_txt in get_dataset_files(dataset_path, type_seq):
        # get data folder and its corresponding txt file
        # load the annotations for the folder
        annotations = mots_utils.load_txt(train_txt)
        image_paths = sorted(os.listdir(train_folder))
        for indx, (image_path, (file_id, objects)) in enumerate(zip(image_paths, list(annotations.items()))):
            #check the file is png or jpg
            if image_path.split('.')[1] in ['png','jpg']:
                record = {}

                filename = os.path.join(train_folder, image_path)
                height,width = cv2.imread(filename).shape[:2]

                record["file_name"] = filename
                record["image_id"] = filename
                record["height"] = height
                record["width"] = width

                objs = []
                for obj in objects:
                    if obj.track_id != 10000:
                        category_id = obj.class_id    
                        bbox = mots_utils.rletools.toBbox(obj.mask)

                        obj_dic = {
                            "bbox" : list(bbox),
                            "bbox_mode" : BoxMode.XYWH_ABS,
                            "category_id" : CLASSES_MAP[category_id]
                        }
                        objs.append(obj_dic)

                record["annotations"] = objs
                dataset_dicts.append(record)
    return dataset_dicts


if __name__ == '__main__':
    
    for d in ['train', 'test']:
        DatasetCatalog.register("kitti_" + d, lambda d= d: get_dataset_dicts(DATASET_PATH, d))
        MetadataCatalog.get("kitti_" + d).set(thing_classes=CLASSES)


    kitti_metadata = MetadataCatalog.get("kitti_train")
    dataset_dicts = get_dataset_dicts(DATASET_PATH, "train")

