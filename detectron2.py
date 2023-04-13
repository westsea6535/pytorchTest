import cv2
import numpy as np
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode

# Initialize the Detectron2 configuration and model
setup_logger()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
predictor = DefaultPredictor(cfg)

# Load the image you want to detect k-pop photo cards in
image = cv2.imread("path/to/image.jpg")

# Perform object detection using Mask R-CNN
outputs = predictor(image)

# Extract the bounding boxes and masks for the detected objects
instances = outputs['instances']
masks = instances.pred_masks.cpu().numpy()
boxes = instances.pred_boxes.tensor.cpu().numpy()

# Loop over each detected object and extract the binary mask
for i in range(len(boxes)):
    mask = masks[i]
    box = boxes[i]

    # Convert the binary mask to a contour and find the border
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    border = cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

# Save the resulting image with the borders around the detected k-pop photo cards
cv2.imwrite("path/to/output.jpg", border)