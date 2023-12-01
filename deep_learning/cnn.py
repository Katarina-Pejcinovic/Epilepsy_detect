import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def run_CNN(train_data, train_labels, test_data, test_labels):

    # CNN architecture
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # Adjust in_channels
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            # Adjust the input size for nn.Linear based on the output size after max pooling
            self.fc1 = nn.Linear(8 * 10 * 32, 2)  # Adjust the input size based on your data

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.pool(x)
            x = x.view(-1, 8 * 10 * 32)  # Adjust the size based on your data
            x = self.fc1(x)
            return x

    # Define a custom dataset
    class CustomDataset(Dataset):
        def __init__(self, data, labels):
            self.data = torch.Tensor(data[:, np.newaxis, :, :])  # Add a channel dimension
            self.labels = torch.Tensor(labels)  # Assuming labels is a numpy array

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    # Create the CNN model and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    # Create a DataLoader for training
    train_dataset = CustomDataset(train_data, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


    # Instantiate the model
    model_instance = SimpleCNN()
    # print("Model successfully created")

    # Create a DataLoader for testing
    test_dataset = CustomDataset(test_data, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Make predictions
    predictions = []
    model.eval()

    with torch.no_grad():
        for inputs, _ in test_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # get probabilities instead of predictions
    class_prob = torch.softmax(outputs, dim=1)

    # get most probable class and its probability
    class_prob, _ = torch.max(class_prob, dim=1)

    # convert to numpy
    prob = class_prob.cpu().numpy()

    return predictions, prob
