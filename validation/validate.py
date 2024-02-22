
from classical_ML.classical_ml_models import *
from deep_learning.rnn import *
from deep_learning.cnn import *
import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tabulate import tabulate

def validate(train_data, 
             train_labels,
             test_data, 
             test_labels, 
             deep_data_test, 
             parameters, 
             stratCV,
             argmax
):

  y_true = test_labels

##DEEP LEARNING 
  #training
  # run_EEGnet(train_data, batch_size = 50)
  # rnn_model(train_data, learning_rate=0.001, gradient_threshold=1, batch_size=32, epochs=2, n_splits=5, strat_kfold=stratCV)
  
  # #testing
  # cnn_pred, cnn_proba= predictions_cnn(deep_data_test)
  # rnn_pred, rnn_proba = rnn_model_test(deep_data_test)

##CLASSICAL TESTING
  svm_pred, svm_proba = svm_model(train_data, train_labels, test_data, parameters[0])

  rf_pred, rf_proba = random_forest_model(train_data, train_labels, test_data, parameters[1])

  xg_pred, xg_proba = xg_boost_model(train_data, train_labels, test_data, parameters[2])

  gmm_pred, gmm_proba = gmm_model(train_data, train_labels, test_data, parameters[3])

  deep_data_test_cnn = deep_data_test.astype('float32')
  print("data type", deep_data_test_cnn.dtype)
  deep_data_test_cnn = np.transpose(deep_data_test_cnn, (2, 0, 1))
  print("running cnn model in validate")

  # run cnn model and obtain the model instance, predictions on test datset (1, 0), and probabilities (decimals)
  cnn_pred, cnn_proba = predictions_cnn(deep_data_test_cnn, counter = argmax)
  print("cnn_pred", cnn_pred, cnn_proba)
  rnn_pred, rnn_proba = rnn_model_test(deep_data_test)
  
  # Compare using F2 scoring (beta > 1 gives more weight to recall)
  svm_f2_score = fbeta_score(test_labels, svm_pred, average='weighted', beta=2)
  rf_f2_score = fbeta_score(test_labels, rf_pred, average='weighted', beta=2)
  xg_f2_score = fbeta_score(test_labels, xg_pred, average='weighted', beta=2)
  gmm_f2_score = fbeta_score(test_labels, gmm_pred, average='weighted', beta=2)
  cnn_f2_score = fbeta_score(y_true, cnn_pred, average='weighted', beta=2)
  rnn_f2_score = fbeta_score(y_true, rnn_pred, average='weighted', beta=2)

  # Compare using confusion matrices
  svm_cm = confusion_matrix(test_labels, svm_pred)
  rf_cm = confusion_matrix(test_labels, rf_pred)
  xg_cm = confusion_matrix(test_labels, xg_pred)
  gmm_cm = confusion_matrix(test_labels, gmm_pred)
  cnn_cm = confusion_matrix(y_true, cnn_pred)
  rnn_cm = confusion_matrix(y_true, rnn_pred)

  # Compare using ROC curves
  model_names = ['SVM', 'Random Forest', 'XG Boost', 'Gaussian Mixture', 'CNN','RNN']
  #model_names = ['SVM', 'Random Forest', 'XG Boost', 'Gaussian Mixture', 'CNN',]

  # F2 Highest Score
  results_f2_score = [svm_f2_score, rf_f2_score, xg_f2_score, gmm_f2_score, cnn_f2_score, rnn_f2_score]
  #results_f2_score = [svm_f2_score, rf_f2_score, xg_f2_score, gmm_f2_score, cnn_f2_score]

  print("The highest f2 score is ", max(results_f2_score, key=lambda x: x))

  for i,score in enumerate(results_f2_score):
    print("f2 score for ", model_names[i], ": ", score, sep = '')
    with open('validation_results/figure_list.txt', 'a') as f:
     f.write(f"The f2 score for {model_names[i]} is {score}")
  
  with open('validation_results/figure_list.txt', 'a') as f:
     f.write(f"The highest f2 score is {max(results_f2_score, key=lambda x: x)} \n\n")

  # for i, pred in enumerate([svm_pred, rf_pred, hmm_pred, kmeans_pred, cnn_pred, rnn_pred]):
  for i, pred in enumerate([svm_proba, rf_proba, xg_proba, gmm_proba,cnn_proba, rnn_proba]):
  #for i, pred in enumerate([svm_proba, rf_proba, xg_proba, gmm_proba,cnn_proba]):

    if i < 4:
      #  print("ostensible 1")
       pred = np.amax(pred, axis =1)
       fpr, tpr, _ = roc_curve(test_labels, pred)
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
  #confusion_matrices = [svm_cm, rf_cm, xg_cm, gmm_cm,cnn_cm, rnn_cm]
  confusion_matrices = [svm_cm, rf_cm, xg_cm, gmm_cm,cnn_cm]
  metrics = []
  precisions = []
  accuracies = []
  recalls = []


  for i, matrix in enumerate(confusion_matrices):
    true_positives = matrix[1][1]
    false_positives = matrix[0][1]
    false_negatives = matrix[1][0]
    true_negatives = matrix[0][0]

    # Calculate precision, accuracy, and recall
    precision = true_positives / (true_positives + false_positives)
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives)
    recall = true_positives / (true_positives + false_negatives)

    precisions.append(precision)
    accuracies.append(accuracy)
    recalls.append(recall)

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

  for i in range(len(model_names)):
      model_data = [
          model_names[i],
          #confusion_matrices[i],
          precisions[i],
          accuracies[i],
          recalls[i],
          results_f2_score[i]
      ]
      metrics.append(model_data)

  #headers = ["Model Name", "Confusion Matrix", "Precision", "Accuracy", "Recall", 'F2-Score']
  headers = ["Model Name", "Precision", "Accuracy", "Recall", 'F2-Score']
  print("headers" '\n', len(headers))
  print("metrics" '\n', len(metrics))
 
  table = tabulate(metrics, headers, tablefmt='grid')
  with open('validation_results/figure_list.txt', 'a') as f:
    f.write(table)
    
  for i, matrix in enumerate(confusion_matrices):
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, 
                                  display_labels=['Non-Epileptic', 'Epileptic'])
    disp.plot()
    plt.title(f'{model_names[i]}')
    plt.savefig("validation_results/{}_cm_heatmap.jpg".format(model_names[i]))