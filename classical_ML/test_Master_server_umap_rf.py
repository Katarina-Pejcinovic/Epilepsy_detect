
#only import stuff you call in this file 
import os
import numpy as np
import pandas as pd

from classical_ML.train_test_tune_umap import * 
from classical_ML.train_test_tune_ica import * 
from classical_ML.find_best_feat_select import * 
# from classical_ML.load_best_params import *


# data/
# Batch Processing
if __name__ == "__main__":
    print('Running sample file')
    # data_file_path = 'data/'
    data_file_path = '/radraid/kpejcinovic/data/'
    
else:
    print('Running batch file(s)')
    from batch_processing import data_file_batch
    data_file_path = data_file_batch

save_path = '/raid/smtam/results/'
# save_path = 'results/tuning_results/'



print("Loading in data files")
with open(save_path + 'patient_ID.pkl', 'rb') as f:
    patient_id = pickle.load(f)

with open(save_path + 'data_reshape.pkl', 'rb') as f:
    data_reshape = pickle.load(f)

with open(save_path + 'labels.pkl', 'rb') as f:
    labels = pickle.load(f)


# # Load in features after it has been generated locally
print("Loading in features")
with open(data_file_path + 'features_3d_array.pkl', 'rb') as f:
    features_3d_array = pickle.load(f)

print("Train features array", features_3d_array.shape)

# Create Stratified Cross Validation object
splits = 5
strat_kfold_object = StratifiedKFold(n_splits=splits, shuffle=True, random_state=10)
strat_kfold = strat_kfold_object.split(data_reshape, patient_id)

tuning_path = save_path + 'tuning_results/'
# Run once when training models

# ica_scores, ica_params = train_test_tune_ica(features_3d_array, labels, patient_id, strat_kfold, save_file)
# umap_scores, umap_params = train_test_tune_umap(features_3d_array, labels, patient_id, strat_kfold, save_path)
# svc_umap_params, svc_umap_scores, svc_umap_estimator = train_test_tune_umap_svc(features_3d_array, labels, patient_id, strat_kfold, tuning_path)
rf_umap_params, rf_umap_scores, rf_umap_estimator = train_test_tune_umap_rf(features_3d_array, labels, patient_id, strat_kfold, tuning_path)
# xg_umap_params, xg_umap_scores, xg_umap_estimator = train_test_tune_umap_xg(features_3d_array, labels, patient_id, strat_kfold, tuning_path)
# gmm_umap_params, gmm_umap_scores, gmm_umap_estimator = train_test_tune_umap_gmm(features_3d_array, labels, patient_id, strat_kfold, tuning_path)

#Load in locally generated
# with open('results/best_umap_params_dict.pkl', 'rb') as f:
#     umap_params = pickle.load(f)
# with open('results/best_ica_params_dict.pkl', 'rb') as f:
#     ica_params = pickle.load(f)
# with open('results/best_umap_scores.pkl', 'rb') as f:
#     umap_scores = pickle.load(f)
# with open('results/best_ica_scores.pkl', 'rb') as f:
#     ica_scores = pickle.load(f)

# Find best feature selection method and keep those parameters
# best_model_params, best_model_params_scores = find_best_feat_select(umap_params, umap_scores, ica_params, ica_scores)

#Load in best determined model params
# with open('results/best_params_dict.pkl', 'rb') as f:
#     best_model_params = pickle.load(f)
# with open('results/classical_ml_scores.pkl', 'rb') as f:
#     best_model_params_scores = pickle.load(f)

print('Full pipeline finished')
