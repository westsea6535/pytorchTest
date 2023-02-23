# Linear Regression basic example

import torch
import torch.nn as nn

# Define the model
class LinearRegression(nn.Module):
  def __init__(self):
    super(LinearRegression, self).__init__()
    self.linear = nn.Linear(1, 1)

  def forward(self, x):
    return self.linear(x)

# Prepare the data
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

# Create the model and loss function
model = LinearRegression()
criterion = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
  inputs = x_data
  labels = y_data

  # Forward pass
  outputs = model(inputs)

  loss = criterion(outputs, labels)

  if (epoch + 1) % 100 == 0:
    # print(f'output: {outputs}')
    print(f'loss: {loss}')
    for name, param in model.named_parameters():
      if param.requires_grad:
        print(name, param.data)

  # Backward pass and optimization
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if (epoch + 1) % 100 == 0:
    print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')
    with torch.no_grad():
      predicted = model(torch.Tensor([[4.0]]))
      print(f'Predicted value: {predicted.item()}')

# # Predict using the trained model
# with torch.no_grad():
#   predicted = model(torch.Tensor([[4.0]]))
#   print(f'Predicted value: {predicted.item()}')