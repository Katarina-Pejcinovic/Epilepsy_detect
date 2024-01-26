# -*- coding: utf-8 -*-
"""validate.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1S05I44rIGCgPM5rcir8GxhLO54FeXEbf


"""

from classical_ML.classical_ml_models import *
from deep_learning.rnn import *
from deep_learning.cnn import *

def validate(train_data, 
             train_labels, 
             validation_data, 
             validation_labels, 
             deep_data_train, 
             deep_data_test, 
             parameters
):

  import numpy as np
  from sklearn.metrics import fbeta_score
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

  y_true = validation_labels


  reduced_data_train = deep_data_train[:, :, :60000]
  reduced_data_test = deep_data_test[:, :, :60000]
  print("reduced data size", reduced_data_train.shape)
  print("running deep learning")
  print("train_data data types", type(reduced_data_train))
  cnn_pred, cnn_proba= run_EEGnet(reduced_data_train, train_labels, reduced_data_test, validation_labels, batch_size = 50)
  rnn_pred, rnn_proba = rnn_model_test(deep_data_train, train_labels, deep_data_test, epochs=3)


  # Run classical models
  svm_pred, svm_proba = svm_model(train_data, train_labels, validation_data, parameters[0])

  rf_pred, rf_proba = random_forest_model(train_data, train_labels, validation_data, parameters[1])

  xg_pred, xg_proba = xg_boost_model(train_data, train_labels, validation_data, parameters[2])

  gmm_pred, gmm_proba = gmm_model(train_data, train_labels, validation_data, parameters[3])


  # run cnn model and obtain the model instance, predictions on test datset (1, 0), and probabilities (decimals)

  
  # Compare using F2 scoring (beta > 1 gives more weight to recall)
  svm_f2_score = fbeta_score(validation_labels, svm_pred, average='weighted', beta=2)
  rf_f2_score = fbeta_score(validation_labels, rf_pred, average='weighted', beta=2)
  xg_f2_score = fbeta_score(validation_labels, xg_pred, average='weighted', beta=2)
  gmm_f2_score = fbeta_score(validation_labels, gmm_pred, average='weighted', beta=2)
  cnn_f2_score = fbeta_score(y_true, cnn_pred, average='weighted', beta=2)
  rnn_f2_score = fbeta_score(y_true, rnn_pred, average='weighted', beta=2)

  # Compare using confusion matrices
  svm_cm = confusion_matrix(validation_labels, svm_pred)
  rf_cm = confusion_matrix(validation_labels, rf_pred)
  xg_cm = confusion_matrix(validation_labels, xg_pred)
  gmm_cm = confusion_matrix(validation_labels, gmm_pred)
  cnn_cm = confusion_matrix(y_true, cnn_pred)
  rnn_cm = confusion_matrix(y_true, rnn_pred)

  # Compare using ROC curves
  model_names = ['SVM', 'Random Forest', 'XG Boost', 'Gaussian Mixture', 'CNN','RNN']

  # F2 Highest Score
  results_f2_score = [svm_f2_score, rf_f2_score, xg_f2_score, gmm_f2_score, cnn_f2_score, rnn_f2_score]
  print("The highest f2 score is ", max(results_f2_score, key=lambda x: x))

  for i,score in enumerate(results_f2_score):
    print("f2 score for ", model_names[i], ": ", score, sep = '')
    with open('validation_results/figure_list.txt', 'a') as f:
     f.write(f"The f2 score for {model_names[i]} is {score}")
  
  with open('validation_results/figure_list.txt', 'a') as f:
     f.write(f"The highest f2 score is {max(results_f2_score, key=lambda x: x)} \n\n")

  # for i, pred in enumerate([svm_pred, rf_pred, hmm_pred, kmeans_pred, cnn_pred, rnn_pred]):
  for i, pred in enumerate([svm_proba, rf_proba, xg_proba, gmm_proba,cnn_proba, rnn_proba]):
    if i < 4:
      #  print("ostensible 1")
       pred = np.amax(pred, axis =1)
       fpr, tpr, _ = roc_curve(validation_labels, pred)
    else:
      # print(i)
      # print(pred)
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
    with open('validation_results/figure_list.txt', 'a') as f:
        f.write('validation_results/{}_roc_auc.jpg\n'.format(model_names[i]))

  # Confusion matrices
  confusion_matrices = [svm_cm, rf_cm, xg_cm, gmm_cm,cnn_cm, rnn_cm]

  for i, matrix in enumerate(confusion_matrices):
    true_positives = matrix[1][1]
    false_positives = matrix[0][1]
    false_negatives = matrix[1][0]
    true_negatives = matrix[0][0]

    # Restructure confusion matrix to match conventional layout
    temp_matrix = matrix
    temp_matrix[1][1] = true_negatives
    temp_matrix[0][0] = true_positives
    matrix = temp_matrix

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
    
  for i, matrix in enumerate(confusion_matrices):
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, 
                                  display_labels=['Non-Epileptic', 'Epileptic'])
    disp.plot()
    plt.title(f'{model_names[i]}')
    plt.savefig("validation_results/{}_cm_heatmap.jpg".format(model_names[i]))