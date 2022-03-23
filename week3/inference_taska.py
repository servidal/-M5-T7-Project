import os


import cv2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


MODEL_NAME = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
RESULT_DIR = 'results'
DATA_PATH = '../out_of_context'

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

setup_logger()
cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file(MODEL_NAME))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_NAME)
predictor = DefaultPredictor(cfg)

for image in os.listdir(DATA_PATH):
    img_path = os.path.join(DATA_PATH, image)
    print(img_path)
    output_path = os.path.join(RESULT_DIR, image)
    
    # Read image
    im = cv2.imread(img_path)
    
    # Forward pass
    outputs = predictor(im)

    # Save to disk
    v = Visualizer(
        im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    img = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(output_path, img.get_image()[:, :, ::-1])