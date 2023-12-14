
#only import stuff you call in this file 
import os
import numpy as np
import pandas as pd

# Load preprocessed data
from load_data import *

# Order: Train EP, Train No EP, Test EP, Test No EP
# data_list = list of 4 lists that contain 2D numpy arrays
# label_list = list of 4 1D numpy arrays
# patientID_list = list of 4 1D numpy arrays
#data_file_path = '/Users/katarinapejcinovic/Desktop/test_data_pre/'
data_file_path = 'data/'

[data_list, label_list, patientID_list] = load_data(data_file_path)

print(len(data_list))
print(len(label_list))
print(len(patientID_list))
print(len(data_list[0]))
print(label_list[0].shape)
print(patientID_list[0].shape)
print("finished loading data")

from feature_selection.cut_segments import *
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
print(len(result_4d[2]))
print(label_result[2].shape)
print(patientID_result[2].shape)
print("-----------------")
print(len(result_4d[3]))
print(label_result[3].shape)
print(patientID_result[3].shape)

from preprocessing.imputate import * 

data1 = run_imputate(result_4d[0])
data2 = run_imputate(result_4d[1])
data3 = run_imputate(result_4d[2])
data4 = run_imputate(result_4d[3])

print("imputed data sizes")
print(data1.shape)
print(data2.shape)
print(data3.shape)
print(data4.shape)

imputed_data = [data1, data2, data3, data4]
non_empty_arrays = [arr for arr in imputed_data if arr.size > 0]


print("number of non-zero arrays", len(non_empty_arrays))
print("impute runs")

from feature_selection.get_features import *
nSamples = 150000 #number of samples per segment
fs = 250 #sampling rate
wavelet_name = 'db1'
features_list = get_features(non_empty_arrays, wavelet_name)

print(features_list[0].shape)
print(features_list[1].shape)
print(len(features_list))

print("new", patientID_result)
print("old", patientID_list)

# from classical_ML.train_test_tune import * 
# concat = np.concatenate((features_list[0], features_list[1]))
# label_res_concat = np.concatenate((label_result[0], label_result[1]))
# patientID_concat = np.concatenate((patientID_result[0], patientID_result[1]))
# params_table, best_params = train_test_tune(concat, label_res_concat, 
#                                             patientID_concat)

#from deep_learning.cnn import *
#from validation.validate import *

# # tune parameters for the classical ml model
# params_table, best_params = train_test_tune(training, labels_train, patient_ids_train)
# # best_params = [{'svc__C': 1, 'svc__kernel': 'linear', 'umap__n_components': 1, 'umap__n_neighbors': 10}, {'randomforestclassifier__max_features': 25, 'randomforestclassifier__min_samples_leaf': 1, 'randomforestclassifier__n_estimators': 10, 'umap__n_components': 1, 'umap__n_neighbors': 10}, {'umap__n_components': 1, 'umap__n_neighbors': 5, 'xgbclassifier__learning_rate': 0.01, 'xgbclassifier__max_depth': 2, 'xgbclassifier__n_estimators': 100}, {'gaussianmixture__init_params': 'k-means++', 'umap__n_components': 1, 'umap__n_neighbors': 5}]

# # validate the models
# validate(preprocessed_train, labels_train_dp, preprocessed_test, labels_test_dp, training, labels_train, testing, labels_test, best_params, pre_train, pre_test)
