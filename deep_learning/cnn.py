import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, accuracy_score, recall_score

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 75000 #NOTE: change this later 
        
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
        self.fc1 = nn.Linear(4*2*4687, 1)
    
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
        x = x.reshape(-1, 4*2*4687) #NOTE: change this later 
        x = F.sigmoid(self.fc1(x))
        return x

#runs EEGnet trains 
def run_EEGnet(data, labels, batch_size, counter):

    data = np.transpose(data, (0, 2, 1))
    print("in eegnet data type", data.dtype)

    #convert 3D numpy array into 4D 
    data = data[:, np.newaxis, :, :]
    
    # Create the CNN model and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNet().to(device)
    
    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    num_epochs = 1
    for epoch in range(num_epochs):
        # Use tqdm to create a progress bar for the outer loop
        for i in tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = torch.from_numpy(data[i*batch_size:i*batch_size+batch_size])
            labels = torch.FloatTensor(np.array([labels[i*batch_size:i*batch_size+batch_size]]).T*1.0)
            labels = labels.view(-1, 1)

            # Move data to the device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(model.state_dict(), f'trained_model{counter}.pth')
    print( "og", model)

def predictions_cnn(test_data, counter):
    print("data type", test_data.dtype)
    
    load_model = EEGNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the weights and biases into the architecture 
    load_model.load_state_dict(torch.load(f'trained_model{counter}.pth', map_location = device))

    # #transpose to: (samples, time, channels)
    test_data_1 = np.transpose(test_data, (0, 2, 1))

    # #convert 3D numpy array into 4D 
    test_data_2 = test_data_1[:, np.newaxis, :, :]

    # Ensure the model is in evaluation mode
    load_model.eval()

    inputs = torch.from_numpy(test_data_2)
    with torch.no_grad():  # Disable gradient computation during inference
        outputs = load_model(inputs)
    predictions = outputs.cpu().numpy()  # Convert predictions to NumPy array

    # Convert predictions to binary values (0 or 1) based on a threshold (e.g., 0.5 for sigmoid)
    threshold = 0.5
    binary_predictions = (predictions > threshold).astype(int)
    print("predictions", predictions)
    return binary_predictions, predictions

#data must be in shape recordins, channels, time points 

def run_EEGnetCV(strat_kfold, data, batch_size):
    #transpose to: (samples, channels, timedata)
    #data = np.transpose(data, (2, 0, 1))
    
    print("in EEGnetCV")
    #split data into metadata and time-series data 
    train_label = data[:, 0, 0]
    print("train_label", train_label.shape)

    #cut off metadata 
    z_length = data.shape[2]
    train_data = data[:, :, 3:z_length,]
    print("train data shape", train_data.shape)

    counter = 0
    f2 = []
    precision = []
    accuracy = []
    recall = []

    for train_index, val_index in strat_kfold:
        print("counter", counter)
        X_train, x_val = train_data[train_index], train_data[val_index]
        Y_train, y_val = train_label[train_index], train_label[val_index]
        print("X_train shape", X_train.shape)
        print("x_val shape", x_val.shape)
        print("in EEGNet")
        run_EEGnet(X_train, Y_train, batch_size, counter)
        predictions, probas =  predictions_cnn(x_val, counter)
        print("calculating f2 score")
        print("predictions", predictions)
        print("y_val", y_val)
        f2.append(fbeta_score(predictions, y_val, beta = 2))
        precision.append(precision_score(predictions, y_val))
        accuracy.append(accuracy_score(predictions,y_val ))
        recall.append(recall_score(predictions, y_val))
        counter += 1

    print("f2", f2)
    f2_np = np.array([f2])
    maximum = np.max(f2_np)
    print("max", maximum)
    arg_max  = np.argmax(f2)
    return arg_max, f2, precision, recall, accuracy, counter


