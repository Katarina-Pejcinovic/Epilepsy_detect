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
from sklearn.model_selection import StratifiedKFold




#Test EEGNet modification 
#current dimensions: (samples, channels, time)
# X_train = np.random.rand(10, 26, 75000).astype('float32') # np.random.rand generates between [0, 1)
# test = np.random.rand(10, 26, 75000).astype('float32')

# batch_size = 10
# run_EEGnet(X_train, batch_size = batch_size)

# predictions, probs = predictions_cnn(test_data=test)
# print(predictions)
# print("probs", probs)

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
#Test CNN
X_train = np.random.rand(26, 75000, 126).astype('float32') # np.random.rand generates between [0, 1)
test = np.random.rand(26, 75000, 126).astype('float32')

X_train = np.append(np.ones((26, 1, 126), dtype = 'float32'), X_train, axis = 1)
# X_train.astype('float32')
print("X_train.shape", X_train.dtype)

patient_id = np.append(np.ones(100), np.zeros(26))

print("patient id shape", patient_id.shape)

batch_size = 42
splits = 3

#transpose to samples, channels, time points 
X_train = np.transpose(X_train, (2, 0, 1))

strat_kfold_object = StratifiedKFold(n_splits=splits, shuffle=True, random_state=10)

strat_kfold = strat_kfold_object.split(X_train, patient_id)

print("size right before call", X_train.shape)
# arg_max, f2, precision, accuracy, recall = run_EEGnetCV(strat_kfold, X_train, batch_size)

# print(arg_max)
# print(f2)
# print(precision)
# print(accuracy)
# print(recall)

binary, proba = predictions_cnn(X_train, 1)

print(proba)


# run_EEGnet(X_train, batch_size = batch_size)