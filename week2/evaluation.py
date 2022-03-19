# Some basic setup:
# Setup detectron2 logger
import imp
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, sys
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from dataset import get_dataset_dicts

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

model_ref = "faster_rcnn_X_101_32x8d_FPN_3x"
model_path = 'COCO-Detection/' + model_ref + '.yaml'
results_dir = '/home/group07/M5-T7-Project/week2/results/'
DATASET_PATH = '/home/group07/M5-T7-Project/KITTI-MOTS'
CLASSES = ['Cars', 'Pedestrian']
print(model_path)

# Run a pre-trained detectron2 model
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(model_path))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_path)
predictor = DefaultPredictor(cfg)

cfg.OUTPUT_DIR = results_dir


for d in ['train', 'test']:
    DatasetCatalog.register("kitti_mots_" + d, lambda d= d: get_dataset_dicts(DATASET_PATH, d))
    MetadataCatalog.get("kitti_mots_" + d).set(thing_classes=CLASSES)


dataset="kitti_mots_"

#metadata = MetadataCatalog.get(dataset + 'train')

cfg.DATASETS.TRAIN = (dataset + 'train',)
cfg.DATASETS.TEST = (dataset + 'test',)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

evaluator = COCOEvaluator(dataset + 'test', cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, dataset + 'test')
print('Evaluation with model ', model_ref)
print(inference_on_dataset(trainer.model, val_loader, evaluator))


