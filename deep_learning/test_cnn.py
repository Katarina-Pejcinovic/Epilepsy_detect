import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib as plt
from cnn import *

#create training datset
edf_data = mne.io.read_raw_edf('aaaaaebo_s001_t000.edf', preload=True)
multichannel_data_train, time = edf_data[:, :]

#create valdiation datset
edf_data = mne.io.read_raw_edf('aaaaaebo_s001_t000.edf', preload=True)
multichannel_data_test, time = edf_data[:, :]

print(multichannel_data_train.shape) #33 by 437500
print(multichannel_data_test.shape)
# print(time.shape)
#combined_array = np.hstack((multichannel_data, time))
print("array")
labels = np.array([1])
val_labels = np.array([1])

#labels, predictions = run_CNN(multichannel_data_train, labels, multichannel_data_test, val_labels)
print("something")
# Generate fake data
train_data = np.random.rand(100, 41, 1, 318750).astype(np.float32)
test_data = np.random.rand(50, 41, 1, 318750).astype(np.float32)

# Generate fake labels
train_labels = np.random.randint(0, 2, size=(100,))
test_labels = np.random.randint(0, 2, size=(50,))

print("Starting")
predictions = run_CNN(train_data, train_labels, test_data, test_labels)
print('predictions', predictions)

# Example usage:
# predictions = run_CNN(train_data, train_labels, test_data, test_labels)

# # Run the CNN and get predictions
# all_labels, all_predictions = run_CNN(data, labels, validation_data, validation_labels)

# # Calculate ROC-AUC
# fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
# roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()
