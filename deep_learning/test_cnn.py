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

# Example usage:
train_data = np.random.randn(100, 3, 128, 128)
train_labels = np.random.randint(0, 2, size=(100,))
test_data = np.random.randn(20, 3, 128, 128)
test_labels = np.random.randint(0, 2, size=(20))



model_instance, predictions, output = run_CNN(train_data, train_labels, test_data, test_labels)
print(output)


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


