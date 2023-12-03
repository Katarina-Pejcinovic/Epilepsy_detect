#only import stuff you call in this file 
import os
import numpy as np

#run get_features()
from feature_selection.get_features import * 

# Batch Processing
if __name__ == "__main__":
    print('Running sample file')
    data_file_path = 'data/'
else:
    print('Running batch file(s)')
    from batch_processing import data_file_batch
    data_file_path = data_file_batch

# Get preprocessed data 
preprocessed_data = []
state = ['epilepsy/', 'no_epilepsy/']
train_ep_files = os.listdir(data_file_path + 'training/' + state[0])
train_noep_files = os.listdir(data_file_path + 'training/' + state[1])
test_ep_files = os.listdir(data_file_path + 'testing/' + state[0])
test_noep_files = os.listdir(data_file_path + 'testing/' + state[1])
for i, file in enumerate(train_ep_files):
    train_ep_file = np.load(data_file_path + 'training/' + state[0] + file)
    if i != 0:
        preprocessed_train_ep = np.concatenate((preprocessed_train_ep, train_ep_file))
    else:
        preprocessed_train_ep = train_ep_file
for i, file in enumerate(train_noep_files):
    train_noep_file = np.load(data_file_path + 'training/' + state[1] + file)
    if i != 0:
        preprocessed_train_noep = np.concatenate((preprocessed_train_noep, train_noep_file))
    else:
        preprocessed_train_noep = train_noep_file
for i, file in enumerate(test_ep_files):
    test_ep_file = np.load(data_file_path + 'testing/' + state[0] + file)
    if i != 0:
        preprocessed_test_ep = np.concatenate((preprocessed_test_ep, test_ep_file))
    else:
        preprocessed_test_ep = test_ep_file
for i, file in enumerate(test_noep_files):
    test_noep_file = np.load(data_file_path + 'testing/' + state[1] + file)
    if i != 0:
        preprocessed_test_noep = np.concatenate((preprocessed_test_noep, test_noep_file))
    else:
        preprocessed_test_noep = test_noep_file

# print('ostensibly')

preprocessed_train = [preprocessed_train_ep, preprocessed_train_noep]
preprocessed_test = [preprocessed_test_ep, preprocessed_test_noep]

# Feature extraction
features_one = get_features(preprocessed_train_ep)
features_two = get_features(preprocessed_train_noep)
features_three = get_features(preprocessed_test_ep)
features_four = get_features(preprocessed_test_noep)

# Data and labels sent to the cnn, rnn, and classical  models
training = np.append(features_one, features_two, axis =0)
testing = np.append(features_three, features_four, axis =0)

# Classical labels
labels_train = np.append(np.ones(16), np.zeros(16), axis =0)
labels_test = labels_train

patient_ids_train = np.concatenate((np.ones(8),np.ones(8)*2, np.ones(8)*3, np.ones(8)*4))
patient_ids_test = np.concatenate((np.ones(8)*5, np.ones(8)*6, np.ones(8)*7, np.ones(8)*8))
print(patient_ids_train)

# Deep learning labels
labels_train_dp = np.array([1,0])
labels_test_dp = np.array([1,0])

from classical_ML.train_test_tune import * 
from deep_learning.cnn import *
from validation.validate import *
import feature_selection.get_features_2 as get_features_2

# cnn features
pre_train = get_features_2.get_features('data_copy/training/epilepsy')
pre_test = get_features_2.get_features('data_copy/testing/epilepsy')

# tune parameters for the classical ml model
params_table, best_params = train_test_tune(training, labels_train, patient_ids_train)
# best_params = [{'svc__C': 1, 'svc__kernel': 'linear', 'umap__n_components': 1, 'umap__n_neighbors': 10}, {'randomforestclassifier__max_features': 25, 'randomforestclassifier__min_samples_leaf': 1, 'randomforestclassifier__n_estimators': 10, 'umap__n_components': 1, 'umap__n_neighbors': 10}, {'kmeans__init': 'k-means++', 'kmeans__n_clusters': 3, 'umap__n_components': 1, 'umap__n_neighbors': 5}, {'umap__n_components': 1, 'umap__n_neighbors': 5, 'xgbclassifier__learning_rate': 0.01, 'xgbclassifier__max_depth': 2, 'xgbclassifier__n_estimators': 100}, {'gaussianmixture__init_params': 'k-means++', 'umap__n_components': 1, 'umap__n_neighbors': 5}]

# validate the models
validate(preprocessed_train, labels_train_dp, preprocessed_test, labels_test_dp, training, labels_train, testing, labels_test, best_params, pre_train, pre_test)
