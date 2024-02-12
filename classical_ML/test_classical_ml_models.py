from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import umap.umap_ as umap

from classical_ml_models import *
from find_best_feat_select import *

# # Test with data in the correct shape
# patients = 5
# files = 5*patients
# channels = 30
# features = 21
# data = np.random.rand(files, channels, features)
# labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

# # Sample validation data
# patients_val = 2
# files_val = 5*patients_val
# val_data = np.random.rand(files_val, channels, features)
# val_labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# Sample parameters
umap_params = [{'svc__C': 0.1, 'svc__degree': 2, 'svc__kernel': 'linear', 'umap__metric': 'euclidean', 'umap__min_dist': 0.1, 'umap__n_components': 5, 'umap__n_neighbors': 5}, 
               {'randomforestclassifier__max_depth': 2, 'randomforestclassifier__min_samples_leaf': 5, 
                'randomforestclassifier__n_estimators': 100, 'umap__metric': 'euclidean', 'umap__min_dist': 0.1, 'umap__n_components': 5, 'umap__n_neighbors': 5}, 
                {'xgbclassifier__learning_rate': 0.1, 'xgbclassifier__max_depth': 5, 'xgbclassifier__n_estimators': 50, 'umap__metric': 'euclidean', 'umap__min_dist': 0.1, 'umap__n_components': 5, 'umap__n_neighbors': 5},
                {'gaussianmixture__init_params': 'k-means++', 'gaussianmixture__covariance_type': 'full', 'umap__metric': 'euclidean', 'umap__min_dist': 0.1, 'umap__n_components': 5, 'umap__n_neighbors': 5}]


kbest_params = [{'svc__C': 0.1, 'svc__degree': 2, 'svc__kernel': 'linear', 'selectkbest__k':10}, 
               {'randomforestclassifier__max_depth': 2, 'randomforestclassifier__min_samples_leaf': 5, 
                'randomforestclassifier__n_estimators': 100, 'selectkbest__k':10}, 
                {'xgbclassifier__learning_rate': 0.1, 'xgbclassifier__max_depth': 5, 'xgbclassifier__n_estimators': 50, 'selectkbest__k':10},
                {'gaussianmixture__init_params': 'k-means++', 'gaussianmixture__covariance_type': 'full', 'selectkbest__k':10}]

ica_params = [{'svc__C': 0.1, 'svc__degree': 2, 'svc__kernel': 'linear', 'fastica__n_components':10}, 
               {'randomforestclassifier__max_depth': 2, 'randomforestclassifier__min_samples_leaf': 5, 
                'randomforestclassifier__n_estimators': 100, 'fastica__n_components':10},  
                {'xgbclassifier__learning_rate': 0.1, 'xgbclassifier__max_depth': 5, 'xgbclassifier__n_estimators': 50, 'fastica__n_components':10},
                {'gaussianmixture__init_params': 'k-means++', 'gaussianmixture__covariance_type': 'full', 'fastica__n_components':10}]

umap_scores = [0, 0, 0, 1]
kbest_scores = [0, 1, 0, 0]
ica_scores = [1, 0, 1, 0]


new_params, new_scores = find_best_feat_select(umap_params, umap_scores, kbest_params, kbest_scores, ica_params, ica_scores)



# [svm_results, svm_pred] = svm_model(data, labels, val_data, test_params[0])
# [rf_results, rf_pred] = random_forest_model(data, labels, val_data, test_params[1])
# kmeans_results = kmeans_model(data, labels, val_data, test_params[2])
# [xg_results, xg_pred] = xg_boost_model(data, labels, val_data, test_params[3])
# [gmm_results, gmm_pred] = gmm_model(data,labels,val_data, test_params[4])

# svm_score = accuracy_score(svm_results, val_labels)
# rf_score = accuracy_score(rf_results, val_labels)
# kmeans_score = accuracy_score(kmeans_results, val_labels)
# xg_score = accuracy_score(xg_results, val_labels)
# gmm_score = accuracy_score(gmm_results, val_labels)

# print("SVM Score: ", svm_score)
# print("Random Forest Score: ", rf_score)
# print("KMeans Score: ", kmeans_score)
# print("XG Boost Score: ", xg_score)
# print("Gaussian Mixture Score: ", gmm_score)