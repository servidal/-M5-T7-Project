# Some basic setup:
# Setup detectron2 logger
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
from detectron2.engine import DefaultTrainer, HookBase

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.utils import comm

from dataset import get_dataset_dicts

import argparse
import torch

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

# ValidationLoss
class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in
                                 comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced,
                                                 **loss_dict_reduced)

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='mask_rcnn_R_50_FPN_3x',
                        help='pre-trained model to run inference on KITTI-MOTS dataset')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--iter', type=int, default=300,
                        help='max iterations (epochs)')

    parser.add_argument('--batch', type=int, default=128,
                        help='batch size')

    return parser.parse_args(args)

if __name__ == "__main__":

    args = parse_args()

    #Model
    model = 'COCO-InstanceSegmentation/' + args.model + '.yaml'
    print('[INFO] Using model: ', model)

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    cfg.OUTPUT_DIR = '/home/group07/M5-T7-Project/week2/results/' + args.model + '/lr_' + str(args.lr).replace('.', '_') + '_iter_' + str(args.iter) + '_batch_' + str(args.batch) + '/'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    DATASET_PATH= '/home/group07/M5-T7-Project/KITTI-MOTS'
    CLASSES = ['Cars', 'Pedestrian']
    
    for d in ['train', 'test']:
        DatasetCatalog.register("kitti_mots_" + d, lambda d= d: get_dataset_dicts(DATASET_PATH, d))
        MetadataCatalog.get("kitti_mots_" + d).set(thing_classes=CLASSES)

    dataset="kitti_mots_"

    cfg.DATASETS.TRAIN = (dataset + 'train',)
    cfg.DATASETS.TEST = (dataset + 'test',)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.TEST.EVAL_PERIOD = 100
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.iter
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch   # faster, and good enough for the tutorial dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

    #Train
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLossl
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    trainer.resume_or_load(resume=False)
    trainer.train()

    ###-------INFERENCE AND EVALUATION---------------------------
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set the testing threshold for this model

     
    ### MAP #####
    #We can also evaluate its performance using AP metric implemented in COCO API.
    evaluator = COCOEvaluator(dataset + 'test', cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, dataset + 'test')
    print('---------------------------------------------------------')
    print(model)
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    print('---------------------------------------------------------')   
