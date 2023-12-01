import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from cnn import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# #create training datset
# edf_data = mne.io.read_raw_edf('aaaaaebo_s001_t000.edf', preload=True)
# train_data, time = edf_data[:, :]
# #create valdiation datset
# edf_data = mne.io.read_raw_edf('aaaaaebo_s001_t000.edf', preload=True)
# test_data, time = edf_data[:, :]
import numpy as np

# Generate random 2D numpy arrays for training and testing
train_data = np.random.randn(2, 16, 21)
train_labels = np.array([0,1])
test_data = np.random.randn(2, 16, 21)
test_labels = np.array([0,1])
print(train_data.shape)
print(train_labels.shape)

# Run the CNN
model_instance, predictions, output = run_CNN(train_data, train_labels, test_data, test_labels)

'''evaluate the effectiveness of the model'''
def evaluate_model(true_labels, predicted_labels):

    # Compute accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")

    # Compute precision, recall, and F1 score
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels, output)

    # Calculate AUC
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend()
    plt.show()
    print("test_predictions", predictions)


# Call the evaluation function
evaluate_model(test_labels, predictions)