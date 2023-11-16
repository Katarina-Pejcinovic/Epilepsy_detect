# -*- coding: utf-8 -*-

import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def run_CNN(data, labels):
# Define the CNN architecture
  class SimpleCNN(nn.Module):
      def __init__(self):
          super(SimpleCNN, self).__init__()
          self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
          self.relu = nn.ReLU()
          self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
          self.fc1 = nn.Linear(16 * 64 * 64, 2)  # Adjust the input size based on your data

      def forward(self, x):
          x = self.conv1(x)
          x = self.relu(x)
          x = self.pool(x)
          x = x.view(-1, 16 * 64 * 64)  # Adjust the size based on your data
          x = self.fc1(x)
          return x

  # Define a custom dataset
  class CustomDataset(Dataset):
      def __init__(self, data, labels):
          self.data = torch.Tensor(data)  # Assuming data is a numpy array
          self.labels = torch.Tensor(labels)  # Assuming labels is a numpy array

      def __len__(self):
          return len(self.data)

      def __getitem__(self, idx):
          return self.data[idx], self.labels[idx]

  # Create the CNN model and set the device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = SimpleCNN().to(device)

  # Create a DataLoader for training
  data = np.random.randn(100, 3, 128, 128)  # Replace this with your actual data
  labels = np.random.randint(0, 2, size=(100,))  # Replace this with your actual labels
  dataset = CustomDataset(data, labels)
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

  # Define loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  # Train the model
  num_epochs = 10
  for epoch in range(num_epochs):
      for inputs, targets in dataloader:
          inputs, targets = inputs.to(device), targets.to(device)

          # Forward pass
          outputs = model(inputs)
          loss = criterion(outputs, targets.long())

          # Backward pass and optimization
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

  # Save the trained model
  torch.save(model.state_dict(), 'simple_cnn_model.pth')

  # Instantiate the model
  model_instance = SimpleCNN()
  return model_instance


#try to run the function 
edf_data = mne.io.read_raw_edf('aaaaaajy_s001_t000.edf', preload=True)
multichannel_data, time = edf_data[:, :]
print("boo")
print(multichannel_data.shape) #33 by 437500
print(time.shape)
#combined_array = np.hstack((multichannel_data, time))
labels = [1]
print(multichannel_data)
model = run_CNN(multichannel_data, labels)
model.load_state_dict(torch.load('simple_cnn_model.pth'))
print(model.eval())

