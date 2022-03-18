import os
import pickle
import random

import cv2
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

KITTI_DATASET = '/home/mcv/datasets/KITTI/'
CLASSES = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting',
           'Cyclist', 'Tram', 'Misc', 'DontCare']


def load_train_dataset():
    train_dict_file = open('train_dict.dat', 'rb')
    train_dict = pickle.load(train_dict_file)
    return train_dict


def load_val_dataset():
    val_dict_file = open('val_dict.dat', 'rb')
    val_dict = pickle.load(val_dict_file)
    return val_dict

#Register dataset
DatasetCatalog.register("kitti_train", load_train_dataset)
DatasetCatalog.register("kitti_val", load_val_dataset)
#Add metadata
MetadataCatalog.get("kitti_train").set(thing_classes=CLASSES)
MetadataCatalog.get("kitti_val").set(thing_classes=CLASSES)
kitti_metadata = MetadataCatalog.get("kitti_train")

#obtain a list of dictionaries(image info) in COCO format
train_dataset_dicts = load_train_dataset()
val_dataset_dicts = load_val_dataset()

#To verify the data loading is correct, let's visualize the annotations of randomly selected samples in the training set:
for d in random.sample(train_dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imwrite(d["image_id"]+".png", out.get_image()[:, :, ::-1])


##Now, let's fine-tune a COCO-pretrained R50-FPN Mask R-CNN model on the KITTI-MOTS dataset
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("kitty_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

print("Start traininig...", flush=True)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("End traininig!", flush=True)

print("Test inference")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
predictor = DefaultPredictor(cfg)

# we randomly select several samples to visualize the prediction results
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], 
                   metadata=kitti_metadata, 
                   scale=1.2,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                  )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])
    #cv2.imwrite(output_path, out.get_image()[:, :, ::-1])


#We can also evaluate its performance using AP metric implemented in COCO API
evaluator = COCOEvaluator("kitti_val", ("bbox", "segm"), False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "kitti_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))