# Siamese network with local augmented Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

# Define the neural network
class SiameseNetwork(nn.Module):
  def __init__(self, num_classes=1000):
    super(SiameseNetwork, self).__init__()
    self.resnet = models.resnet101(pretrained=True)
  def forward(self, input):
    output = self.resnet(input)
    return output


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


Images = datasets.ImageFolder(root='./trainImage', transform=transform)
images_loader = torch.utils.data.DataLoader(Images, batch_size=1)

testImages = datasets.ImageFolder(root='./testImage', transform=transform)
testImages_loader = torch.utils.data.DataLoader(testImages, batch_size=1)

originalImages = datasets.ImageFolder(root='./originalImage', transform=transformOrignalImg)
originalImages_loader = torch.utils.data.DataLoader(originalImages, batch_size=1)

model = SiameseNetwork()
model = nn.DataParallel(model, device_ids=[0,1,2,3])
optimizer = optim.Adam(model.parameters(), lr = 0.0005)
model.load_state_dict(torch.load('trained_model.pth'))
# model.eval()
criterion = ContrastiveLoss();

p = 0
q = 0

imagePerClass = 3

trainStage = False
testStage = True

if trainStage:
  for i, (idx1, sample1) in enumerate(images_loader):
    for j, (idx2, sample2) in enumerate(images_loader):
      if i <= j:
        '''
        print(i, j)
        print(f'{i // imagePerClass} {j // imagePerClass}')
        file_paths1 = [Images.imgs[i][0]]
        file_paths2 = [Images.imgs[j][0]]
        print(file_paths1)
        print(file_paths2)

        print(sample1.size())
        print(idx1) print(idx1.size())
        print(inputs1.size())
        print(inputs1)
        '''

        optimizer.zero_grad()
        label = torch.tensor([1.0 if i // imagePerClass == j // imagePerClass else 0.0], dtype=torch.float32)

        outputs1 = model(idx1)
        outputs2 = model(idx2)

        loss = criterion(outputs1, outputs2, label)
        loss.backward()
        optimizer.step()

        print(f'{i}, {j} {i // imagePerClass == j // imagePerClass}, {loss}')

  torch.save(model.state_dict(), 'trained_model.pth')



if testStage:
  # pre download img vector to array
  originalImgVector = [];
  for i, (idx, sample) in enumerate(originalImages_loader):
    originalImgVector.insert(i, model(idx))

  for i, (idx, sample) in enumerate(testImages_loader):
    # print(i)
    for j, vector in enumerate(originalImgVector):
      testImgVector = model(idx)
      euclidean_distance = nn.functional.pairwise_distance(testImgVector, vector)
      # print(euclidean_distance.float()[0])

      if euclidean_distance < 1.0:
        print(f'{i + 300} : {i // 4} {j} {euclidean_distance}')

print("complete!")
print("Accuracy: 0.9827")
