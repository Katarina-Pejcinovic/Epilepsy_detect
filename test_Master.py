#only import stuff you call in this file 
import os
import numpy as np
import pandas as pd
from glob import glob

# Batch Processing
if __name__ == "__main__":
    print('Running sample file')
    data_file_path = 'data/'
else:
    print('Running batch file(s)')
    from batch_processing import data_file_batch
    data_file_path = data_file_batch

# Get preprocessed data 
# Beta subset: ep_train = [aaaaaanr], ep_test = [aaaaalug], noep_train = [aaaaaebo], noep_test = [aaaaappo]
# ep_train_ID = ['aaaaaanr']
# ep_test_ID = ['aaaaalug']
# noep_train_ID = ['aaaaaebo']
# noep_test_ID = ['aaaaappo']

# FIX: Load in the txt files instead as a list of ID names
ep_train_df = pd.read_table('data/subject_ids_epilepsy_training.txt', delimiter="\t")
ep_train_ID = ep_train_df["IDs"].values.tolist()
noep_train_df = pd.read_table('data/subject_ids_no_epilepsy_training.txt', delimiter="\t")
noep_train_ID = noep_train_df["IDs"].values.tolist()

ep_test_df = pd.read_table('data/subject_ids_epilepsy_testing.txt', delimiter="\t")
ep_test_ID = ep_test_df["IDs"].values.tolist()
noep_test_df = pd.read_table('data/subject_ids_no_epilepsy_testing.txt', delimiter="\t")
noep_test_ID = noep_test_df["IDs"].values.tolist()


# Get all patient names
preprocessed_path = data_file_path + 'preprocessed_data/'
state = ['epilepsy_edf/', 'no_epilepsy_edf/']

ep_patients_train = [filename for filename in os.listdir(preprocessed_path + state[0]) 
                     if filename in ep_train_ID]
noep_patients_train = [filename for filename in os.listdir(preprocessed_path + state[1])
                        if filename in noep_train_ID]
ep_patients_test = [filename for filename in os.listdir(preprocessed_path + state[0])
                     if filename in ep_test_ID]
noep_patients_test = [filename for filename in os.listdir(preprocessed_path + state[1])
                       if filename in noep_test_ID]

preprocessed_train_ep = []
preprocessed_train_noep = []
preprocessed_test_ep = []
preprocessed_test_noep = []

# Load in all the pre-processed files
for patient in ep_patients_train:
    patient_folder_path = preprocessed_path + state[0] + "/" + patient
    processed_files = [filename for filename in os.listdir(patient_folder_path) 
                       if filename.startswith("p")]
    for file in processed_files:
        file = np.load(patient_folder_path + "/" + file)
        preprocessed_train_ep.append(file)
    
for patient in noep_patients_train:
    patient_folder_path = preprocessed_path + state[1] + "/" + patient
    processed_files = [filename for filename in os.listdir(patient_folder_path) 
                       if filename.startswith("p")]
    for file in processed_files:
        file = np.load(patient_folder_path + "/" + file)
        preprocessed_train_noep.append(file)

for patient in ep_patients_test:
    patient_folder_path = preprocessed_path + state[0] + "/" + patient
    processed_files = [filename for filename in os.listdir(patient_folder_path) 
                       if filename.startswith("p")]
    for file in processed_files:
        file = np.load(patient_folder_path + "/" + file)
        preprocessed_test_ep.append(file)

for patient in noep_patients_test:
    patient_folder_path = preprocessed_path + state[1] + "/" + patient
    processed_files = [filename for filename in os.listdir(patient_folder_path) 
                       if filename.startswith("p")]
    for file in processed_files:
        file = np.load(patient_folder_path + "/" + file)
        preprocessed_test_noep.append(file)


preprocessed_train = [preprocessed_train_ep, preprocessed_train_noep]
preprocessed_test = [preprocessed_test_ep, preprocessed_test_noep]

# Feature extraction
from feature_selection.get_features import * 

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
# best_params = [{'svc__C': 1, 'svc__kernel': 'linear', 'umap__n_components': 1, 'umap__n_neighbors': 10}, {'randomforestclassifier__max_features': 25, 'randomforestclassifier__min_samples_leaf': 1, 'randomforestclassifier__n_estimators': 10, 'umap__n_components': 1, 'umap__n_neighbors': 10}, {'umap__n_components': 1, 'umap__n_neighbors': 5, 'xgbclassifier__learning_rate': 0.01, 'xgbclassifier__max_depth': 2, 'xgbclassifier__n_estimators': 100}, {'gaussianmixture__init_params': 'k-means++', 'umap__n_components': 1, 'umap__n_neighbors': 5}]

# validate the models
validate(preprocessed_train, labels_train_dp, preprocessed_test, labels_test_dp, training, labels_train, testing, labels_test, best_params, pre_train, pre_test)
