import cv2
import numpy as np
import torch
from torchvision import transforms

# Load the image
image = cv2.imread("./detectionTestImage/test1.jpg")

# Define the target size of the trading card
target_size = (322, 502)

# Define the pre-processing transform
transform = transforms.Compose([
  transforms.ToTensor(),
])

# Load the pre-trained object detection model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Perform object detection
results = model(image)

# Extract the trading cards
trading_cards = []
for result in results.xyxy[0]:
  xmin, ymin, xmax, ymax, conf, cls = result
  width = xmax - xmin
  height = ymax - ymin
  if (width, height) == target_size:
    trading_card = image[int(ymin):int(ymax), int(xmin):int(xmax)]
    trading_card = cv2.cvtColor(trading_card, cv2.COLOR_BGR2RGB)
    trading_card = transform(trading_card).unsqueeze(0)
    trading_cards.append(trading_card)

# Display the extracted trading cards
for trading_card in trading_cards:
  print('exist')
  cv2.imshow("Trading Card", trading_card)
  cv2.waitKey(0)