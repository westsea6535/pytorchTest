import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network
class BinaryClassifier(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(BinaryClassifier, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# Instantiate the model
model = BinaryClassifier(input_size=4, hidden_size=8, output_size=1)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Generate dummy input data
inputs = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
labels = torch.tensor([[1.0]], dtype=torch.float32)

# Train the model
for i in range(10000):
  # Forward pass
  outputs = model(inputs)
  loss = criterion(outputs, labels)
  
  # Backward pass and optimization
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # Print loss
  if i % 1000 == 0:
    print(f'Epoch {i} Loss: {loss.item()}')
    print(f'output: {outputs}')


# Generate new test data
test_inputs = torch.tensor([[0.5, 0.6, 0.7, 0.8]], dtype=torch.float32)
test_labels = torch.tensor([[0.0]], dtype=torch.float32)

# Pass the test data through the model
test_outputs = model(test_inputs)

# Calculate the test loss
test_loss = criterion(test_outputs, test_labels)

# Print the test loss
print(f'Test Loss: {test_loss.item()}')

# Calculate the accuracy of the model
predicted = (test_outputs >= 0.0).float()
accuracy = (predicted == test_labels).float().mean()

# Print the accuracy
print(f'Accuracy: {accuracy.item() * 100}%')