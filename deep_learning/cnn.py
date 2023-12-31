import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm



#runs EEGnet
def run_EEGnet(train_data_og, train_labels, test_data_og, test_labels, batch_size):

    print("in EEGnet")
    #transpose to: (samples, time channels)
    train_data = np.transpose(train_data_og, (0, 2, 1))
    test_data = np.transpose(test_data_og, (0, 2, 1))

    #convert 3D numpy array into 4D 
    train_data = train_data[:, np.newaxis, :, :]
    test_data = test_data[:, np.newaxis, :, :]

    class EEGNet(nn.Module):
        def __init__(self):
            super(EEGNet, self).__init__()
            self.T = 60000 #NOTE: change this later 
            
            # Layer 1
            self.conv1 = nn.Conv2d(1, 16, (1, 26), padding = 0)
            self.batchnorm1 = nn.BatchNorm2d(16, False)
            
            # Layer 2
            self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
            self.conv2 = nn.Conv2d(1, 4, (2, 32))
            self.batchnorm2 = nn.BatchNorm2d(4, False)
            self.pooling2 = nn.MaxPool2d(2, 4)
            
            # Layer 3
            self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
            self.conv3 = nn.Conv2d(4, 4, (8, 4))
            self.batchnorm3 = nn.BatchNorm2d(4, False)
            self.pooling3 = nn.MaxPool2d((2, 4))

            # FC Layer
            # NOTE: This dimension will depend on the number of timestamps per sample in your data.
            # I have 120 timepoints. 
            self.fc1 = nn.Linear(4*2*3750, 1)
        
        def forward(self, x):
            # Layer 1
            x = F.elu(self.conv1(x))
            x = self.batchnorm1(x)
            x = F.dropout(x, 0.25)
            x = x.permute(0, 3, 1, 2)
            
            # Layer 2
            x = self.padding1(x)
            x = F.elu(self.conv2(x))
            x = self.batchnorm2(x)
            x = F.dropout(x, 0.25)
            x = self.pooling2(x)
            
            # Layer 3
            x = self.padding2(x)
            x = F.elu(self.conv3(x))
            x = self.batchnorm3(x)
            x = F.dropout(x, 0.25)
            x = self.pooling3(x)
            
            # FC Layer
            x = x.reshape(-1, 4*2*3750) #NOTE: change this later 
            x = F.sigmoid(self.fc1(x))
            return x

    
    # Create the CNN model and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet().to(device)

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    num_epochs = 1
    for epoch in range(num_epochs):
        # Use tqdm to create a progress bar for the outer loop
        for i in tqdm(range(0, len(train_data), batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = torch.from_numpy(train_data[i:i+batch_size])
            labels = torch.FloatTensor(np.array([train_labels[i:i+batch_size]]).T*1.0)

            # Move data to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # Make predictions
    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():  # Disable gradient computation during inference
        for i in range(0, len(test_data), batch_size):
            inputs = torch.from_numpy(test_data[i:i+batch_size])
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    # Convert predictions to binary values based on a threshold (e.g., 0.5 for sigmoid)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)

    return binary_predictions, predictions


