# Siamese network with local augmented Image
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F

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

class SiameseNetwork(nn.Module):
  def __init__(self, num_classes=1000):
    super(SiameseNetwork, self).__init__()
    self.resnet = models.resnet101(pretrained=True)
  def forward(self, input):
    output = self.resnet(input)
    return output


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

class TripletLoss(nn.Module):
  def __init__(self, margin):
    super(TripletLoss, self).__init__()
    self.margin = margin

  def forward(self, anchor, positive, negative, size_average=True):
    distance_positive = (anchor - positive).pow(2).sum(1)
    distance_negative = (anchor - negative).pow(2).sum(1)
    losses = F.relu(distance_positive - distance_negative + self.margin)

    return losses.mean() if size_average else losses.sum()



Images = datasets.ImageFolder(root='./trainImage_3', transform = transforms.ToTensor())
images_loader = torch.utils.data.DataLoader(Images, batch_size=1)

testImages = datasets.ImageFolder(root='./testImage', transform=transform)
testImages_loader = torch.utils.data.DataLoader(testImages, batch_size=1)

originalImages = datasets.ImageFolder(root='./originalImage', transform=transformOrignalImg)
originalImages_loader = torch.utils.data.DataLoader(originalImages, batch_size=1)



model = SiameseNetwork().cuda()
model = nn.DataParallel(model, device_ids=[0])


optimizer = optim.Adam(model.parameters(), lr = 0.0005)
criterion = TripletLoss(margin=2.);

trainStage = False
testStage = True

if testStage:
  model.load_state_dict(torch.load('trained_model_200.pth'))

imgPerLabel = 3
epoch = 200

if trainStage:
  time_start = time()
  for epoch in range(0, epoch):
    for imgLabel, (idx, _) in enumerate(originalImages_loader):
      # imgLabel is label num
      anchor_img = idx

      # choosing random positive image
      positive_img = Images.imgs[imgPerLabel * imgLabel + random.randrange(0, imgPerLabel)]
      # print(positive_img)

      # choosing random negative image
      while True:
        #keep looping till a different class image is found
        negative_img = random.choice(Images.imgs) 
        if imgLabel != negative_img[1]:
          # print(negative_img)
          break
      
      pathArr = [positive_img[1], negative_img[1]]

      positive_img = Image.open(positive_img[0])
      negative_img = Image.open(negative_img[0])

      positive_img = transform(positive_img)
      negative_img = transform(negative_img)

      positive_img = positive_img.unsqueeze(0)
      negative_img = negative_img.unsqueeze(0)

      optimizer.zero_grad()
      anchor_img = model(anchor_img)
      positive_img = model(positive_img)
      negative_img = model(negative_img)

      loss_contrastive = criterion(anchor_img, positive_img, negative_img)
      loss_contrastive.backward()
      optimizer.step()
      print(pathArr[0], pathArr[1], imgLabel)

  torch.save(model.state_dict(), f'trained_model_{epoch}.pth')
  print(f'{time() - time_start} spent)')


totalDataset = 300
correctCount = 0
incorrectCount = 0

if testStage:
  # pre download img vector to array
  originalImgVector = [];
  for i, (idx, sample) in enumerate(originalImages_loader):
    originalImgVector.insert(i, model(idx))
    # print([originalImages.imgs[i][0]])

  for i, (idx, sample) in enumerate(testImages_loader):
    for j, vector in enumerate(originalImgVector):
      testImgVector = model(idx)
      euclidean_distance = nn.functional.pairwise_distance(testImgVector, vector)
      # print(f'{i} {i // 6} {j} {euclidean_distance}')

      if euclidean_distance < 1.5:
        if i // 6 == j:
          correctCount = correctCount + 1
        else:
          incorrectCount = incorrectCount + 1
        print(f'{i} : {i // 6} {j} {[originalImages.imgs[j][0]]} {euclidean_distance}')
  print(f'correctCount: {correctCount}, incorrectCount: {incorrectCount}, noMatch: {totalDataset - correctCount - incorrectCount}')
        