
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
print("finished loading data")

print(len(data_list[0]))
print(label_list[0].shape)
print(patientID_list[0].shape)

from feature_selection.cut_segments import *
result_4d, label_result, patientID_result = cut_segments(data_list, label_list, patientID_list)
print(result_4d)

#from classical_ML.train_test_tune import * 
#from deep_learning.cnn import *
#from validation.validate import *

# # tune parameters for the classical ml model
# params_table, best_params = train_test_tune(training, labels_train, patient_ids_train)
# # best_params = [{'svc__C': 1, 'svc__kernel': 'linear', 'umap__n_components': 1, 'umap__n_neighbors': 10}, {'randomforestclassifier__max_features': 25, 'randomforestclassifier__min_samples_leaf': 1, 'randomforestclassifier__n_estimators': 10, 'umap__n_components': 1, 'umap__n_neighbors': 10}, {'umap__n_components': 1, 'umap__n_neighbors': 5, 'xgbclassifier__learning_rate': 0.01, 'xgbclassifier__max_depth': 2, 'xgbclassifier__n_estimators': 100}, {'gaussianmixture__init_params': 'k-means++', 'umap__n_components': 1, 'umap__n_neighbors': 5}]

# # validate the models
# validate(preprocessed_train, labels_train_dp, preprocessed_test, labels_test_dp, training, labels_train, testing, labels_test, best_params, pre_train, pre_test)
