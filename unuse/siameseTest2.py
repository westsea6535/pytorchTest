# Siamese network test with STL10 library
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

# Instantiate the model
model = SiameseNetwork()

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Load the STL-10 dataset
stl_dataset = datasets.STL10('./data', download=False, transform=transforms.ToTensor())

epoch = 0

for x_idx in range(10):
  for y_idx in range(10):
    inputs1 = stl_dataset[x_idx][0].reshape(-1)
    inputs2 = stl_dataset[y_idx][0].reshape(-1)
    label = torch.tensor([1.0 if x_idx == y_idx else 0.0], dtype=torch.float32)

    # Forward pass
    outputs = model(inputs1, inputs2)
    loss = criterion(outputs, label)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print(f'x: {x_idx}, y: {y_idx}, loss: {loss.item()}')
    if loss.item() > 0.7 and x_idx == y_idx:
      print(f'x: {x_idx}, y: {y_idx}, same correct, loss: {loss.item()}')
    elif loss.item() <= 0.3 and x_idx != y_idx:
      print(f'x: {x_idx}, y: {y_idx}, diff correct, loss: {loss.item()}')
    else:
      print(f'x: {x_idx}, y: {y_idx}, wrong, loss: {loss.item()}')

torch.save(model.state_dict(), 'trained_model.pth')