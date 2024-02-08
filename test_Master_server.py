
#only import stuff you call in this file 
import os
import numpy as np
import pandas as pd
from load_data import *
from data_organization.new_data_struct import *
from data_organization.patient_id_dict import *
from preprocessing.impute import * 
from classical_ML.train_test_tune_umap import * 
from classical_ML.load_best_params import *
from feature_selection.get_features import *
from feature_selection.cut_segments import *
from deep_learning.cnn import *
from deep_learning.rnn import *
from validation.validate import *
from preprocessing.train_test_split import*

from preprocessing.functions_prepro import *



# Batch Processing
if __name__ == "__main__":
    print('Running sample file')
    data_file_path = 'data/'
else:
    print('Running batch file(s)')
    from batch_processing import data_file_batch
    data_file_path = data_file_batch


###PREPROCESSING

base_dir = data_file_path
master_prepro(base_dir)


# Order: Train EP, Train No EP, Test EP, Test No EP
# data_list = list of 4 lists that contain 2D numpy arrays
# label_list = list of 4 1D numpy arrays
# patientID_list = list of 4 1D numpy arrays
# data_file_path = 'data/'
[data_list, label_list, patientID_list] = load_data(data_file_path)

# Check load data 
print(len(data_list))
print(len(label_list))
print(len(patientID_list))
print(len(data_list[0]))
print(label_list[0].shape)
print(patientID_list[0].shape)
print("finished loading data")

# Cut segments into 5 min slices
result_4d, label_result, patientID_result = cut_segments(data_list, label_list, patientID_list)

print(len(result_4d))
print(result_4d[0].shape)
print(len(label_result))
print(label_result[0].shape)
print(len(patientID_result))
print(patientID_result[0].shape)
print("-----------------")
print(result_4d[1].shape)
print(label_result[1].shape)
print(patientID_result[1].shape)
print("-----------------")

# Create or load in full data structure
patient_list_folder = data_file_path
save_file_path = data_file_path
full_data_array = new_data_struct(result_4d, label_result, patientID_result, patient_list_folder, save_file_path)

with open(data_file_path + 'full_3d_array.pkl', 'rb') as f:
    full_data_array = pickle.load(f)
print("full data array", full_data_array.shape)




#impute function 
data = run_impute(full_data_array)
print("impute ran")
print(data.shape)

#Train-Test Split
train_data, test_data = split(data, data_file_path, data_file_path)

# Break down data structure
data_full = data[:, 3:, :]
labels = data[0, 0, :]
patient_id = data[0, 1, :]
num_segments = data.shape[2]
num_channels = data.shape[0]
num_data = data.shape[1] - 3
data_reshape = np.reshape(data_full, (num_segments, num_channels, num_data))
print("data reshape ran")
print(data_reshape.shape)

# # Extract features
features_3d_array = get_features(data_reshape)
print("features array", features_3d_array.shape)

# # Create Stratified Cross Validation object
splits = 3
strat_kfold_object = StratifiedKFold(n_splits=splits, shuffle=True, random_state=10)
strat_kfold = strat_kfold_object.split(data_reshape, patient_id)

# Tune parameters for classical ML
from classical_ML.train_test_tune_umap import * 
from classical_ML.load_best_params import *

params_scores, best_params = train_test_tune_umap(features_3d_array, labels, patient_id, strat_kfold)
best_params = load_best_params()

# #run imputate on train_ep, train_no_ep, test_ep, test_no_ep
# data1 = run_imputate(result_4d[0])
# data2 = run_imputate(result_4d[1])
# data3 = run_imputate(result_4d[2])
# data4 = run_imputate(result_4d[3])

# print("imputed data sizes:")
# print(data1.shape)
# print(data2.shape)
# print(data3.shape)
# print(data4.shape)
# imputed_data = [data1, data2, data3, data4]
# non_empty_arrays = [arr for arr in imputed_data if arr.size > 0]

# #concatenate training data for deep learning 
# training_time = np.concatenate((non_empty_arrays[0], non_empty_arrays[1]), axis = 0)
# #concatenate testing data for deep learning
# testing_time = np.concatenate((non_empty_arrays[2], non_empty_arrays[3]), axis =0 )

# # validate the models
# # def validate(train_data, 
# #              train_labels, 

# #              validation_data, 
# #              validation_labels, 

# #              deep_data_train, 
# #              deep_data_test, 

# #              parameters
# # ):

# training_time = np.float32(training_time)
# testing_time = np.float32(testing_time)

# validate(train_data = concat, 
#          train_labels = label_res_concat, 
#          validation_data = concat_test, 
#          validation_labels = concat_test_labels, 
#          deep_data_train = training_time, 
#          deep_data_test = testing_time, 
#          parameters = best_params)
