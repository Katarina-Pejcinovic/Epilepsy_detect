# -*- coding: utf-8 -*-
"""validate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1S05I44rIGCgPM5rcir8GxhLO54FeXEbf
"""

def validate(train_data, train_labels, validation_data, validation_labels, parameters):

  import numpy as np
  from sklearn.metrics import fbeta_score

  ## Run models

  svm_pred = svm_model(data, labels, val_data, parameters)

  rf_pred = random_forest_model(data, labels, val_data, parameters)

  hmm_pred = hmm_model(data, labels, val_data, parameters)

  kmeans_pred = kmeans_model(data, labels, val_data, parameters)

  # CNN

  rnn_pred = rnn_model(train_data, train_labels, validation_data)


  ## Compare using F2 scoring (beta > 1 gives more weight to recall)
  svm_f2_score = fbeta_score(y_true, svm_pred, average='weighted', beta=2)
  rf_f2_score = fbeta_score(y_true, rf_pred, average='weighted', beta=2)
  hmm_f2_score = fbeta_score(y_true, hmm_pred, average='weighted', beta=2)
  kmeans_f2_score = fbeta_score(y_true, kmeans_pred, average='weighted', beta=2)
  cnn_f2_score = fbeta_score(y_true, cnn_pred, average='weighted', beta=2)
  rnn_f2_score = fbeta_score(y_true, rnn_pred, average='weighted', beta=2)

  # Compare using confusion matrices
  svm_cm = confusion_matrix(y_true, svm_pred)
  rf_cm = confusion_matrix(y_true, rf_pred)
  hmm_cm = confusion_matrix(y_true, hmm_pred)
  kmeans_cm = confusion_matrix(y_true, kmeans_pred)
  cnn_cm = confusion_matrix(y_true, cnn_pred)
  rnn_cm = confusion_matrix(y_true, rnn_pred)

  #Compare using ROC curves
  #svm_roc = roc_auc(y_true, svm_pred)
  #rf_roc = roc_auc(y_true, rf_pred)
  #hmm_roc = roc_auc(y_true, hmm_pred)
  #kmeans_roc = roc_auc(y_true, kmeans_pred)
  #cnn_roc = roc_auc(y_true, cnn_pred)
  #rnn_roc = roc_auc(y_true, rnn_pred)

  results = []

  print("The model with the highest validation accuracy is XX")

  return results