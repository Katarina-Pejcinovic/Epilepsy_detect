
#only import stuff you call in this file 
import os
import numpy as np
import pandas as pd
from load_data import *
from data_organization.new_data_struct import *
from data_organization.patient_id_dict import *
from preprocessing.impute import * 
from classical_ML.train_test_tune_umap import * 
from classical_ML.train_test_tune_selectkbest import * 
from classical_ML.train_test_tune_ica import * 
from classical_ML.find_best_feat_select import * 
from classical_ML.load_best_params import *
from feature_selection.get_features import *
from feature_selection.cut_segments import *
from deep_learning.cnn import *
from deep_learning.rnn import *
from validation.validate import *
from preprocessing.train_test_split import *
from preprocessing.functions_prepro import *

# data/
# Batch Processing
if __name__ == "__main__":
    print('Running sample file')
    # data_file_path = '/radraid/arathi/'
    data_file_path = 'data/'
else:
    print('Running batch file(s)')
    from batch_processing import data_file_batch
    data_file_path = data_file_batch
if __name__ == "__main__":
    print('Running sample file')
    # data_file_path = '/radraid/arathi/'
    data_file_path = 'data/'
else:
    print('Running batch file(s)')
    from batch_processing import data_file_batch
    data_file_path = data_file_batch

# Train-Test Split
# train_data, test_data = split(data, data_file_path, data_file_path)
with open(data_file_path + 'train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open(data_file_path + 'test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Break down train data structure
data_full = train_data[:, 3:, :]
labels = train_data[0, 0, :]
patient_id = train_data[0, 1, :]
num_segments = train_data.shape[2]
num_channels = train_data.shape[0]
num_data = train_data.shape[1] - 3
data_reshape = np.reshape(data_full, (num_segments, num_channels, num_data))
print("Train data reshape ran")
print(data_reshape.shape)

# Break down test data structure
data_full_test = test_data[:, 3:, :]
labels_test = test_data[0, 0, :]
patient_id_test = test_data[0, 1, :]
num_segments_test = test_data.shape[2]
num_channels_test = test_data.shape[0]
num_data_test = test_data.shape[1] - 3
data_reshape_test = np.reshape(data_full_test, (num_segments_test, num_channels_test, num_data_test))
print("Train data reshape ran")
print(data_reshape_test.shape)

# # Load in features after it has been generated locally
with open('data/features_3d_array.pkl', 'rb') as f:
    features_3d_array = pickle.load(f)
with open('data/features_3d_array_test.pkl', 'rb') as f:
    features_3d_array_test = pickle.load(f)

print("Train features array", features_3d_array.shape)
print("Test features array", features_3d_array_test.shape)

# Create Stratified Cross Validation object
splits = 3
strat_kfold_object = StratifiedKFold(n_splits=splits, shuffle=True, random_state=10)
strat_kfold = strat_kfold_object.split(data_reshape, patient_id)


#Load in locally generated
with open('results/best_umap_params_dict.pkl', 'rb') as f:
    umap_params = pickle.load(f)
with open('results/best_ica_params_dict.pkl', 'rb') as f:
    ica_params = pickle.load(f)
with open('results/best_umap_scores.pkl', 'rb') as f:
    umap_scores = pickle.load(f)
with open('results/best_ica_scores.pkl', 'rb') as f:
    ica_scores = pickle.load(f)

#Load in best determined model params
with open('results/best_params_dict.pkl', 'rb') as f:
    best_model_params = pickle.load(f)
with open('results/classical_ml_scores.pkl', 'rb') as f:
    best_model_params_scores = pickle.load(f)

train_data_type = train_data.astype('float32')
train_data_cnn = np.transpose(train_data_type, (2, 0, 1))

with open('results/cnn_results.txt', 'r') as file:
    cnn_arg_max = file.readline().strip()

# print("CNN Arg Max: ", cnn_arg_max)

# Testing
validate(train_data = features_3d_array, 
          train_labels = labels, 
          test_data = features_3d_array_test, 
          test_labels = labels_test, 
          deep_data_test = data_full_test, 
          parameters = best_model_params,
          stratCV = strat_kfold,
          argmax = cnn_arg_max)

print('Full pipeline finished')
