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
  def forward_once(self, x):
    output = self.resnet(x)
    return output
  def forward(self, input1, input2):
    output1 = self.forward_once(input1)
    output2 = self.forward_once(input2)
    return output1, output2

transform = transforms.Compose([
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


Images = datasets.ImageFolder(root='./augmented_images', transform=transform)
images_loader = torch.utils.data.DataLoader(Images, batch_size=1)

model = SiameseNetwork()
model = nn.DataParallel(model,device_ids=[0,1,2,3])
optimizer = optim.Adam(model.parameters(), lr = 0.0005)
# model.load_state_dict(torch.load('trained_model.pth'))
# model.eval()
criterion = ContrastiveLoss();

p = 0
q = 0

imagePerClass = 5

for i, (idx1, sample1) in enumerate(images_loader):
  if i < imagePerClass:
    for j, (idx2, sample2) in enumerate(images_loader):
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

      outputs1, outputs2 = model(idx1, idx2)
      loss = criterion(outputs1, outputs2, label)
      loss.backward()
      optimizer.step()

      # print(f'{i // imagePerClass == j // imagePerClass}, {loss}')