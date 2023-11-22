import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib as plt
from cnn import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# #create training datset
# edf_data = mne.io.read_raw_edf('aaaaaebo_s001_t000.edf', preload=True)
# train_data, time = edf_data[:, :]
# #create valdiation datset
# edf_data = mne.io.read_raw_edf('aaaaaebo_s001_t000.edf', preload=True)
# test_data, time = edf_data[:, :]

# Example usage:
train_data = np.random.randn(100, 3, 128, 128)
train_labels = np.random.randint(0, 2, size=(100,))
test_data = np.random.randn(20, 3, 128, 128)
test_labels = np.random.randint(0, 2, size=(20))

model_instance, predictions = run_CNN(train_data, train_labels, test_data)
print("test_predictions", predictions)

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

# Call the evaluation function
print("test labels", test_labels, "\npredictions", len(predictions))
evaluate_model(test_labels, predictions)



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
