# Test of Siamese network with self made CNN structure
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random

# Define the neural network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(27648, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, x1, x2):
        x1 = F.relu(self.fc1(x1))
        x2 = F.relu(self.fc1(x2))
        x = torch.abs(x1 - x2)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Instantiate the model
model = SiameseNetwork()

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Load the STL-10 dataset
stl_dataset = datasets.STL10('./data', download=False, transform=transforms.ToTensor())

# Train the model
for i in range(100):
    # Randomly select a pair of images from the STL-10 dataset
    idx1, idx2 = random.randint(0, 30), random.randint(0, 30)
    inputs1, inputs2 = stl_dataset[idx1][0].reshape(-1), stl_dataset[idx2][0].reshape(-1)
    label = torch.tensor([1.0 if idx1 == idx2 else 0.0], dtype=torch.float32)

    # Forward pass
    outputs = model(inputs1, inputs2)
    loss = criterion(outputs, label)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss
    if i % 10 == 0:
        print(f'Epoch {i} Loss: {loss.item()}')