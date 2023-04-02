# Siamese network with local augmented Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim


import random
from PIL import Image

transform = transforms.Compose([
  transforms.Resize((96, 96)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


transformOrignalImg = transforms.Compose([
  transforms.Grayscale(num_output_channels=3),
  transforms.Resize((96, 96)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

Images = datasets.ImageFolder(root='./trainImage_3', transform=transform)
images_loader = torch.utils.data.DataLoader(Images, batch_size=1)

testImages = datasets.ImageFolder(root='./testImage', transform=transform)
testImages_loader = torch.utils.data.DataLoader(testImages, batch_size=1)

originalImages = datasets.ImageFolder(root='./originalImage', transform=transformOrignalImg)
originalImages_loader = torch.utils.data.DataLoader(originalImages, batch_size=1)

# print(Images.imgs)
img1_tuple = random.choice(Images.imgs) 
print(img1_tuple)
img1 = Image.open(img1_tuple[0])
print(img1.cuda())
# for i, (idx, sample) in enumerate(images_loader):
#   print(sample)