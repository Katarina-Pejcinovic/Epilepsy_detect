import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

#generate  data
train = np.random.rand(100, 34, 437500).astype(np.float32)
val = np.random.rand(50, 34, 437500).astype(np.float32)
test = np.random.rand(50, 34, 437500).astype(np.float32)

#generate labels
train_label = np.random.randint(2, size=100)
val_label = np.random.randint(2, size=50)
test_label = np.random.randint(2, size=50)

#test
preds = rnn_model(train, train_label, test)

#confusion matrix
print(confusion_matrix(test_label,preds))
print(accuracy_score(test_label,preds))
print(precision_score(test_label,preds))
print(recall_score(test_label,preds))

