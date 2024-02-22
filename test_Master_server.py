
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


# # PREPROCESSING
# base_dir = data_file_path
# master_prepro(base_dir)

# # Order: Train EP, Train No EP, Test EP, Test No EP
# # data_list = list of 4 lists that contain 2D numpy arrays
# # label_list = list of 4 1D numpy arrays
# # patientID_list = list of 4 1D numpy arrays
# data_file_path = 'data/'
# [data_list, label_list, patientID_list] = load_data(data_file_path)

# # Check load data 
# print(len(data_list))
# print(len(label_list))
# print(len(patientID_list))
# print(len(data_list[0]))
# print(label_list[0].shape)
# print(patientID_list[0].shape)
# print("finished loading data")

# # Cut segments into 5 min slices
# result_4d, label_result, patientID_result = cut_segments(data_list, label_list, patientID_list)

# print(len(result_4d))
# print(result_4d[0].shape)
# print(len(label_result))
# print(label_result[0].shape)
# print(len(patientID_result))
# print(patientID_result[0].shape)
# print("-----------------")
# print(result_4d[1].shape)
# print(label_result[1].shape)
# print(patientID_result[1].shape)
# print("-----------------")

# # Create or load in full data structure
# patient_list_folder = data_file_path
# save_file_path = data_file_path

# # Run once
# full_data_array = new_data_struct(result_4d, label_result, patientID_result, patient_list_folder, save_file_path)

# Load in data after it has been generated locally
# with open(data_file_path + 'full_3d_array.pkl', 'rb') as f:
#      full_data_array = pickle.load(f)
# print("Full data array shape:", full_data_array.shape)

# # Impute function 
# data = run_impute(full_data_array)
# print("impute ran")
# print(data.shape)

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

# # Extract features

# Run once
#features_3d_array = get_features(data_reshape)
# with open('data/features_3d_array.pkl', 'wb') as f:
#     pickle.dump(features_3d_array, f)

# features_3d_array_test = get_features(data_reshape_test)
# with open('data/features_3d_array_test.pkl', 'wb') as f:
#     pickle.dump(features_3d_array_test, f)

# # Load in features after it has been generated locally
with open('data/features_3d_array.pkl', 'rb') as f:
    features_3d_array = pickle.load(f)
with open('data/features_3d_array_test.pkl', 'rb') as f:
    features_3d_array_test = pickle.load(f)

print("Train features array", features_3d_array.shape)
print("Test features array", features_3d_array_test.shape)

# Create Stratified Cross Validation object
splits = 5
strat_kfold_object = StratifiedKFold(n_splits=splits, shuffle=True, random_state=10)
strat_kfold = strat_kfold_object.split(data_reshape, patient_id)

# Tune classical parameters

# Run once when training models
# umap_scores, umap_params = train_test_tune_umap(features_3d_array, labels, patient_id, strat_kfold)
# ica_scores, ica_params = train_test_tune_ica(features_3d_array, labels, patient_id, strat_kfold)

#Load in locally generated
with open('results/best_umap_params_dict.pkl', 'rb') as f:
    umap_params = pickle.load(f)
with open('results/best_ica_params_dict.pkl', 'rb') as f:
    ica_params = pickle.load(f)
with open('results/best_umap_scores.pkl', 'rb') as f:
    umap_scores = pickle.load(f)
with open('results/best_ica_scores.pkl', 'rb') as f:
    ica_scores = pickle.load(f)

# Find best feature selection method and keep those parameters
best_model_params, best_model_params_scores = find_best_feat_select(umap_params, umap_scores, ica_params, ica_scores)

#Load in best determined model params
with open('results/best_params_dict.pkl', 'rb') as f:
    best_model_params = pickle.load(f)
with open('results/classical_ml_scores.pkl', 'rb') as f:
    best_model_params_scores = pickle.load(f)

train_data_type = train_data.astype('float32')
train_data_cnn = np.transpose(train_data_type, (2, 0, 1))

# # Deep Learning
# cnn_arg_max, cnn_f2, cnn_precision, cnn_recall, cnn_accuracy, counter = run_EEGnetCV(strat_kfold, train_data_cnn, batch_size = 42)
# rnn_val_preds_binary, rnn_val_preds, rnn_f2_list, rnn_precision_list, rnn_recall_list, rnn_accuracy_list = rnn_model(train_data, 
#         learning_rate=0.001, gradient_threshold=1, batch_size=32, epochs=1, n_splits=splits, strat_kfold=strat_kfold)

# with open('results/cnn_results', 'w') as f:
#     for item in [cnn_arg_max, cnn_f2, cnn_precision, cnn_recall, cnn_accuracy]:
#         f.write("%s\n" % item)

# with open('results/rnn_results', 'w') as f:
#     for item in [rnn_f2_list, rnn_precision_list, rnn_recall_list, rnn_accuracy_list]:
#         f.write("%s\n" % item)

# with open('results/cnn_results.txt', 'w') as f:
#     for item in [cnn_arg_max, cnn_f2, cnn_precision, cnn_recall, cnn_accuracy]:
#         f.write("%s\n" % item)

# with open('results/rnn_results.txt', 'w') as f:
#     for item in [rnn_f2_list, rnn_precision_list, rnn_recall_list, rnn_accuracy_list]:
#         f.write("%s\n" % item)

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
