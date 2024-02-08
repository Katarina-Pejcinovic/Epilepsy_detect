import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from cnn import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np



#Test EEGNet modification 
#current dimensions: (samples, channels, time)
X_train = np.random.rand(10, 26, 75000).astype('float32') # np.random.rand generates between [0, 1)
test = np.random.rand(10, 26, 75000).astype('float32')

batch_size = 10
run_EEGnet(X_train, batch_size = batch_size)

predictions, probs = predictions_cnn(test_data=test)
print(predictions)
print("probs", probs)

'''
# Compute accuracy
accuracy = accuracy_score(y_test, bi_predictions)
print(f"Accuracy: {accuracy:.4f}")

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probas)

# Calculate AUC
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend()
#plt.show()

# Compute precision, recall, and F1 score
precision = precision_score(y_test, bi_predictions)
recall = recall_score(y_test, bi_predictions)
f1 = f1_score(y_test, bi_predictions)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Compute confusion matrix
cm = confusion_matrix(y_test, bi_predictions)
print("Confusion Matrix:")
print(cm)
'''