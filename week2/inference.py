import os
import cv2
import torch
from detectron2.utils.logger import setup_logger


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
MODEL_NAME = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
RESULT_DIR = '/home/marcelo/Documents/Master_CV/M5/M5-T7-Project/results'
DATA_PATH = '/home/marcelo/Documents/Master_CV/M5/M5-T7-Project/datasets/training/image_02'
if __name__ == '__main__':
    setup_logger()
    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file(MODEL_NAME))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_NAME)
    predictor = DefaultPredictor(cfg)

    for subdir, dirs, files in os.walk(DATA_PATH):
        results_path = os.path.join(RESULT_DIR, subdir.split('/Data')[1])
        os.makedirs(results_path, exist_ok=True)
        for file in files:
            input_path = os.path.join(subdir, file)
            output_path = os.path.join(results_path, file)
            # Read image
            im = cv2.imread(input_path)
            # Forward pass
            outputs = predictor(im)

            # Save to disk
            v = Visualizer(
                im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
            img = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite(output_path, img.get_image()[:, :, ::-1])
