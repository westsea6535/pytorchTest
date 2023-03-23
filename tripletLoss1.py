# Siamese network with local augmented Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

from PIL import Image

from torch.utils.data import DataLoader, Dataset

from time import time
import random

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

class ContrastiveLoss(nn.Module):
  def __init__(self, margin=2.0):
    super(ContrastiveLoss, self).__init__()
    self.margin = margin

  def forward(self, output1, output2, label):
    euclidean_distance = nn.functional.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
    print(f'distance: {euclidean_distance} loss: {loss_contrastive}')
    # 같으면 1 다르면 0
    return loss_contrastive



class SiameseNetworkDataset(Dataset):
  def __init__(self, imageFolderDataset, should_invert=True):
    self.imageFolderDataset = imageFolderDataset
    self.should_invert = should_invert

  def __getitem__(self, index):
    # Get an image
    img0_tuple = self.imageFolderDataset.imgs[index]
    # print(img0_tuple)

    img0 = Image.open(img0_tuple[0])
    img1 = Image.open(img0_tuple[0])

    img0 = augmentationTransform(img0)
    img1 = augmentationTransform(img1)

    if index == 0:
      print(img0)
      print(img1)
      print(img1 == img0)

    return img0, img1

    # while True:
    #   img1_tuple = random.choice(self.imageFolderDataset.imgs) 
    #   if img0_tuple[1]==img1_tuple[1]:
    #     break


  #   # Get an image from a different class
  #   while True:
  #     #keep looping till a different class image is found
          
  #     img2_tuple = random.choice(self.imageFolderDataset.imgs) 
  #     if img0_tuple[1] !=img2_tuple[1]:
  #       break

  #   width,height = (244,244)

  #   pathList = []
  #   pathList.append((img0_tuple[0],img1_tuple[0],img2_tuple[0]))

  #   # anchor, positive image, negative image
  #   return img0, img1 , img2, pathList

  def __len__(self):
    return len(self.imageFolderDataset.imgs)

# Images = datasets.ImageFolder(root='./trainImage_3', transform=transform)
# images_loader = torch.utils.data.DataLoader(Images, batch_size=1)

# testImages = datasets.ImageFolder(root='./testImage', transform=transform)
# testImages_loader = torch.utils.data.DataLoader(testImages, batch_size=1)

originalImages = datasets.ImageFolder(root='./originalImage', transform=transformOrignalImg)
originalImages_loader = torch.utils.data.DataLoader(originalImages, batch_size=1)

siamese_dataset = SiameseNetworkDataset(imageFolderDataset=originalImages, should_invert=False)

train_dataloader = torch.utils.data.DataLoader(siamese_dataset, batch_size=1)

for i, (idx,idx1) in enumerate(train_dataloader):
  print('1')