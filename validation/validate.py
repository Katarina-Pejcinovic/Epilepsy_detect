# -*- coding: utf-8 -*-
"""validate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1S05I44rIGCgPM5rcir8GxhLO54FeXEbf


"""

from classical_ML.classical_ml_models import *
from deep_learning.rnn import *
from deep_learning.cnn import *

def validate(train_data, train_labels, validation_data, validation_labels, train_data_ml, train_labels_ml, 
             validation_data_ml, validation_labels_ml, parameters, cnn_train, cnn_test):

  import numpy as np
  from sklearn.metrics import fbeta_score
  from sklearn.metrics import confusion_matrix

  y_true = validation_labels

  ## Run classical models
  svm_pred, svm_proba = svm_model(train_data_ml, train_labels_ml, validation_data_ml, parameters[0])

  rf_pred, rf_proba = random_forest_model(train_data_ml, train_labels_ml, validation_data_ml, parameters[1])


  xg_pred, xg_proba = xg_boost_model(train_data_ml, train_labels_ml, validation_data_ml, parameters[3])


  gmm_pred, gmm_proba = gmm_model(train_data_ml, train_labels_ml, validation_data_ml, parameters[4])



#run cnn model and obtain the model instance, predictions on test datset (1, 0), and probabilities (decimals)
  cnn_pred, cnn_proba= run_CNN(cnn_train, train_labels, cnn_test, validation_labels)
  print("in validate")

  # RNN
  rnn_pred, rnn_proba = rnn_model(train_data, train_labels, validation_data, epochs=3)


  # Compare using F2 scoring (beta > 1 gives more weight to recall)
  svm_f2_score = fbeta_score(validation_labels_ml, svm_pred, average='weighted', beta=2)
  rf_f2_score = fbeta_score(validation_labels_ml, rf_pred, average='weighted', beta=2)
  xg_f2_score = fbeta_score(validation_labels_ml, xg_pred, average='weighted', beta=2)
  gmm_f2_score = fbeta_score(validation_labels_ml, gmm_pred, average='weighted', beta=2)
  cnn_f2_score = fbeta_score(y_true, cnn_pred, average='weighted', beta=2)
  rnn_f2_score = fbeta_score(y_true, rnn_pred, average='weighted', beta=2)

  # Compare using confusion matrices
  svm_cm = confusion_matrix(validation_labels_ml, svm_pred)
  rf_cm = confusion_matrix(validation_labels_ml, rf_pred)
  xg_cm = confusion_matrix(validation_labels_ml, xg_pred)
  gmm_cm = confusion_matrix(validation_labels_ml, gmm_pred)
  cnn_cm = confusion_matrix(y_true, cnn_pred)
  rnn_cm = confusion_matrix(y_true, rnn_pred)

  # F2 Highest Score
  results_f2_score = [svm_f2_score, rf_f2_score, xg_f2_score, gmm_f2_score, cnn_f2_score, rnn_f2_score]
  print("The model with the highest f2 score is", max(results_f2_score, key=lambda x: x))
  with open('validation_results/figure_list.txt', 'a') as f:
     f.write(f"The model with the highest f2 score is {max(results_f2_score, key=lambda x: x)}")

  # Compare using ROC curves
  # model_names = ['SVM', 'Random Forest', 'HMM', 'KMeans', 'CNN', 'RNN']
  model_names = ['SVM', 'Random Forest', 'XG Boost', 'Gaussian Mixture', 'CNN','RNN']

  # for i, pred in enumerate([svm_pred, rf_pred, hmm_pred, kmeans_pred, cnn_pred, rnn_pred]):
  for i, pred in enumerate([svm_proba, rf_proba, xg_proba, gmm_proba,cnn_proba, rnn_proba]):
    if i < 4:
       print("ostensible 1")
       pred = np.amax(pred, axis =1)
       fpr, tpr, _ = roc_curve(validation_labels_ml, pred)
    else:
      print(i)
      print(pred)
      fpr, tpr, _ = roc_curve(y_true, pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {model_names[i]}')
    plt.legend(loc="lower right")
    plt.savefig("validation_results/{}_roc_auc.jpg".format(model_names[i]))
    plt.show()
    with open('validation_results/figure_list.txt', 'a') as f:
        f.write('validation_results/{}_roc_auc.jpg\n'.format(model_names[i]))

  # Confusion matrices
  # confusion_matrices = [svm_cm,rf_cm,hmm_cm,kmeans_cm,cnn_cm,rnn_cm]
  confusion_matrices = [svm_cm, rf_cm, xg_cm, gmm_cm,cnn_cm, rnn_cm]

  for i, matrix in enumerate(confusion_matrices):
    true_positives = matrix[1][1]
    false_positives = matrix[0][1]
    false_negatives = matrix[1][0]
    true_negatives = matrix[0][0]

    # Calculate precision, accuracy, and recall
    precision = true_positives / (true_positives + false_positives)
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)
    recall = true_positives / (true_positives + false_negatives)

    # Print the results
    print(matrix)
    print(f"Precision: {precision:.2f}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")

    with open('validation_results/figure_list.txt', 'a') as f:
        f.write(model_names[i])
        f.write(str(matrix))
        f.write(f'\n Precision: {precision}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Recall: {recall}\n\n')

