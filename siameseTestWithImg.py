# Test of Siamese network using self-made CNN structure and local image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define the neural network
class SiameseNetwork(nn.Module):
  def __init__(self):
    super(SiameseNetwork, self).__init__()
    self.fc1 = nn.Linear(27648, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 1)

  def forward(self, x1, x2):
    x1 = F.relu(self.fc1(x1))
    x2 = F.relu(self.fc1(x2))
    x = torch.abs(x1 - x2)
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


transform = transforms.Compose([
  transforms.Resize((96, 96)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

Images = datasets.ImageFolder(root='./testImage', transform=transform)
images_loader = torch.utils.data.DataLoader(Images)

model = SiameseNetwork()
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()
criterion = nn.BCEWithLogitsLoss()

with torch.no_grad():
  for idx1, sample1 in enumerate(images_loader):
    for idx2, sample2 in enumerate(images_loader):

      inputs1 = sample1[0].reshape(-1)
      inputs2 = sample2[0].reshape(-1)
      label = torch.tensor([1.0 if idx1 == idx2 else 0.0], dtype=torch.float32)

      outputs = model(inputs1, inputs2)
      loss = criterion(outputs, label)
      
      # if loss.item() > 0.7 and idx1 == idx2:
      #   print(f'x: {idx1}, y: {idx2}, same correct, loss: {round(loss.item() * 100, 1)}%')
      # elif loss.item() <= 0.3 and idx1 != idx2:
      #   print(f'x: {idx1}, y: {idx2}, diff correct, loss: {round(loss.item() * 100, 1)}%')
      # else:
      #   print(f'x: {idx1}, y: {idx2}, wrong, loss:  {round(loss.item() * 100, 1)}%')
      if loss.item() > 0.5 and idx1 < idx2:
        # print(f'x: {idx1}, y: {idx2}, loss: {round(loss.item() * 100, 1)}%')
        print(f'x: {idx1}, y: {idx2}, loss: {loss.item()}%')