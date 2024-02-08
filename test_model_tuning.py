# from sklearn.metrics import fbeta_score, make_scorer
# ftwo_scorer = make_scorer(fbeta_score, beta=2, average='micro')
# print(ftwo_scorer)
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# grid = GridSearchCV(SVC(), param_grid={'C': [1, 10]}, scoring=ftwo_scorer, verbose=2)

# from sklearn import datasets
# iris = datasets.load_iris()

# X, y = iris.data, iris.target

# # print(y)

# grid.fit(X, y)


#only import stuff you call in this file 
import os
import numpy as np
import pandas as pd
from load_data import *
from data_organization.new_data_struct import *
from data_organization.patient_id_dict import *
from preprocessing.impute import * 
from classical_ML.train_test_tune_selectkbest import * 
from classical_ML.train_test_tune_umap import * 
from classical_ML.load_best_params import *
from feature_selection.get_features import *
from feature_selection.cut_segments import *
# from deep_learning.cnn import *
# from deep_learning.rnn import *
# from validation.validate import *


# Batch Processing
if __name__ == "__main__":
    print('Running sample file')
    data_file_path = 'data/'
else:
    print('Running batch file(s)')
    from batch_processing import data_file_batch
    data_file_path = data_file_batch

[data_list, label_list, patientID_list] = load_data(data_file_path)


#cut segmets into 5 min slices
result_4d, label_result, patientID_result = cut_segments(data_list, label_list, patientID_list)


with open(data_file_path + 'full_3d_array.pkl', 'rb') as f:
    full_data_array = pickle.load(f)


# Break down data structure
data_full = full_data_array[:, 3:, :]
labels = full_data_array[0, 0, :]
patient_id = full_data_array[0, 1, :]
num_segments = full_data_array.shape[2]
num_channels = full_data_array.shape[0]
num_data = full_data_array.shape[1] - 3

# Create Stratified CV by patient
data_reshape = np.reshape(data_full, (num_segments, num_channels, num_data))

# wavelet_name = 'db1'
# features_list = get_features(data_reshape, wavelet_name)

with open('results/temp_features.pkl', 'rb') as f:
    features_list = pickle.load(f)


features_list = np.nan_to_num(features_list, nan=0.0)

splits = 2
strat_kfold_object = StratifiedKFold(n_splits=splits, shuffle=True, random_state=10)
strat_kfold = strat_kfold_object.split(data_reshape, patient_id)


params_scores, best_params = train_test_tune_selectkbest(features_list, labels, patient_id, strat_kfold)
# params_scores, best_params = train_test_tune_umap(features_list, labels, patient_id, strat_kfold)

print('Done')