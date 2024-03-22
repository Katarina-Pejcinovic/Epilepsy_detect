# -*- coding: utf-8 -*-
"""classical_ml_models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f4ZuWpch2j-WfEY6MyBLpcoF_o5AvqSS
"""

from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier
import pandas as pd
import umap as umap
from sklearn.metrics import make_scorer,fbeta_score
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import FastICA

# Use for validation
# Sample parameters for now

def svm_model(data, labels, val_data, svm_param):

  # For validation, train on full 3/4 data and then test on untouched 1/4 validation data?

  # PARAMETERS: kernal, C, gamma, degree
  # {'svc__C': 10, 'svc__kernel': 'linear', 'umap__n_components': 5, 'umap__n_neighbors': 5}

  # Reshape data
  num_files = data.shape[0]
  num_channels = data.shape[1]
  num_features = data.shape[2]
  data_reshape = np.reshape(data, (num_files, num_channels*num_features))

  num_files_val = val_data.shape[0]
  num_channels_val = val_data.shape[1]
  num_features_val = val_data.shape[2]

  val_data_reshape = np.reshape(val_data, (num_files_val, num_channels_val*num_features_val))

  X_train = data_reshape
  y_train = labels
  X_test = val_data_reshape

  keys = list(svm_param.keys())
  
  ######################################### UMAP #########################################

  if any('umap' in item for item in keys):
    svc_C = svm_param["svc__C"]
    svc_kernel = svm_param["svc__kernel"]
    svc_degree = svm_param["svc__degree"]
    umap_components = svm_param["umap__n_components"]
    umap_neighbors = svm_param["umap__n_neighbors"]
    umap_metric = svm_param["umap__metric"]
    umap_dist = svm_param["umap__min_dist"]

    # Pipeline w/ UMAP + SVM

    # Create UMAP object
    reducer = umap.UMAP(n_components=umap_components, n_neighbors=umap_neighbors, min_dist=umap_dist, metric=umap_metric)

    # Turn data into z-scores
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # Feature selection
    X_train = reducer.fit_transform(X_train)
    X_test = reducer.transform(X_test)

    # Train the model
    svm_model = SVC(kernel=svc_kernel, C=svc_C, gamma='auto', degree=svc_degree, random_state=42, probability=True)
    svm_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = svm_model.predict_proba(X_test)
    y_pred = svm_model.predict(X_test)

    return y_pred, y_pred_proba
  
  ######################################### SelectKBest #########################################
  if any('selectkbest' in item for item in keys):

    svc_C = svm_param["svc__C"]
    svc_kernel = svm_param["svc__kernel"]
    svc_degree = svm_param["svc__degree"]
    selectkbest_k = svm_param["selectkbest__k"]

    # Pipeline w/ SelectKBest + SVM

    # Create SelectKBest object
    f2_score = make_scorer(fbeta_score, beta=2, average='micro')
    reducer = SelectKBest(k=selectkbest_k)

    # Turn data into z-scores
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # Feature selection
    X_train = reducer.fit_transform(X_train, y_train)
    X_test = reducer.transform(X_test)

    # Train the model
    svm_model = SVC(kernel=svc_kernel, C=svc_C, gamma='auto', degree=svc_degree, random_state=42, probability=True)
    svm_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = svm_model.predict_proba(X_test)
    y_pred = svm_model.predict(X_test)

    return y_pred, y_pred_proba

  ######################################### ICA #########################################
  if any('fastica' in item for item in keys):

    svc_C = svm_param["svc__C"]
    svc_kernel = svm_param["svc__kernel"]
    svc_degree = svm_param["svc__degree"]
    fastica_n_components = svm_param["fastica__n_components"]

    # Pipeline w/ ICA + SVM

    # Create ICA object
    reducer = FastICA(n_components=fastica_n_components)

    # Turn data into z-scores
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # Feature selection
    X_train = reducer.fit_transform(X_train)
    X_test = reducer.transform(X_test)

    # Train the model
    svm_model = SVC(kernel=svc_kernel, C=svc_C, gamma='auto', degree=svc_degree, random_state=42, probability=True)
    svm_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = svm_model.predict_proba(X_test)
    y_pred = svm_model.predict(X_test)

    return y_pred, y_pred_proba

  


def random_forest_model(data, labels, val_data, rf_param):

  # PARAMETERS: n_estimators, min_samples_leaf, max_features
  # {'randomforestclassifier__min_samples_leaf': 5, 'randomforestclassifier__max_depth': 5, 
  # 'randomforestclassifier__n_estimators': 100, 'umap__n_components': 10, 'umap__n_neighbors': 5}

  # Reshape data
  num_files = data.shape[0]
  num_channels = data.shape[1]
  num_features = data.shape[2]

  data_reshape = np.reshape(data, (num_files, num_channels*num_features))

  num_files_val = val_data.shape[0]
  num_channels_val = val_data.shape[1]
  num_features_val = val_data.shape[2]

  val_data_reshape = np.reshape(val_data, (num_files_val, num_channels_val*num_features_val))

  X_train = data_reshape
  y_train = labels
  X_test = val_data_reshape

  keys = list(rf_param.keys())

  ######################################### UMAP #########################################
  if any('umap' in item for item in keys):
    rf_depth = rf_param["randomforestclassifier__max_depth"]
    rf_samples = rf_param["randomforestclassifier__min_samples_leaf"]
    rf_estimators = rf_param["randomforestclassifier__n_estimators"]
    umap_components = rf_param["umap__n_components"]
    umap_neighbors = rf_param["umap__n_neighbors"]
    umap_metric = rf_param["umap__metric"]
    umap_dist = rf_param["umap__min_dist"]

    # Pipeline w/ UMAP + RF

    # Create UMAP object
    reducer = umap.UMAP(n_components=umap_components, n_neighbors=umap_neighbors, min_dist=umap_dist, metric=umap_metric)

    # Turn data into z-scores
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # Feature selection
    X_train = reducer.fit_transform(X_train)
    X_test = reducer.transform(X_test)

    # Train the model
    rf_model = RandomForestClassifier(max_depth=rf_depth, n_estimators=rf_estimators, min_samples_leaf=rf_samples)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = rf_model.predict_proba(X_test)
    y_pred = rf_model.predict(X_test)

    return y_pred, y_pred_proba
  
  ######################################### SelectKBest #########################################
  if any('selectkbest' in item for item in keys):
    rf_depth = rf_param["randomforestclassifier__max_depth"]
    rf_samples = rf_param["randomforestclassifier__min_samples_leaf"]
    rf_estimators = rf_param["randomforestclassifier__n_estimators"]
    selectkbest_k = rf_param["selectkbest__k"]

    # Pipeline w/ SelectKBest + RF

    # Create SelectKBest object
    reducer = SelectKBest(k=selectkbest_k)

    # Turn data into z-scores
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # Feature selection
    X_train = reducer.fit_transform(X_train, y_train)
    X_test = reducer.transform(X_test)

    # Train the model
    rf_model = RandomForestClassifier(max_depth=rf_depth, n_estimators=rf_estimators, min_samples_leaf=rf_samples)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = rf_model.predict_proba(X_test)
    y_pred = rf_model.predict(X_test)

    return y_pred, y_pred_proba

  ######################################### ICA #########################################
  if any('fastica' in item for item in keys):
    rf_depth = rf_param["randomforestclassifier__max_depth"]
    rf_samples = rf_param["randomforestclassifier__min_samples_leaf"]
    rf_estimators = rf_param["randomforestclassifier__n_estimators"]
    fastica_n_components = rf_param["fastica__n_components"]

    # Pipeline w/ ICA + RF

    # Create ICA object
    reducer = FastICA(n_components=fastica_n_components)

    # Turn data into z-scores
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # Feature selection
    X_train = reducer.fit_transform(X_train)
    X_test = reducer.transform(X_test)

    # Train the model
    rf_model = RandomForestClassifier(max_depth=rf_depth, n_estimators=rf_estimators, min_samples_leaf=rf_samples)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = rf_model.predict_proba(X_test)
    y_pred = rf_model.predict(X_test)

    return y_pred, y_pred_proba


def xg_boost_model(data, labels, val_data, xg_param):

  # PARAMETERS: max_depth, n_estimators, learning_rate
  # {'umap__n_components': 10, 'umap__n_neighbors': 5, 'xgbclassifier__learning_rate': 0.1, 'xgbclassifier__max_depth': 5, 
  # 'xgbclassifier__n_estimators': 50}

  # Reshape data
  num_files = data.shape[0]
  num_channels = data.shape[1]
  num_features = data.shape[2]

  data_reshape = np.reshape(data, (num_files, num_channels*num_features))

  num_files_val = val_data.shape[0]
  num_channels_val = val_data.shape[1]
  num_features_val = val_data.shape[2]

  val_data_reshape = np.reshape(val_data, (num_files_val, num_channels_val*num_features_val))

  X_train = data_reshape
  y_train = labels
  X_test = val_data_reshape

  keys = list(xg_param.keys())

  ######################################### UMAP #########################################
  if any('umap' in item for item in keys):
    xg_lr = xg_param["xgbclassifier__learning_rate"]
    xg_depth = xg_param["xgbclassifier__max_depth"]
    xg_estimators = xg_param["xgbclassifier__n_estimators"]
    umap_components = xg_param["umap__n_components"]
    umap_neighbors = xg_param["umap__n_neighbors"]
    umap_metric = xg_param["umap__metric"]
    umap_dist = xg_param["umap__min_dist"]

    # Pipeline w/ UMAP + XG Boost

    # Create UMAP object
    reducer = umap.UMAP(n_components=umap_components, n_neighbors=umap_neighbors, min_dist=umap_dist, metric=umap_metric)

    # Turn data into z-scores
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # Feature selection
    X_train = reducer.fit_transform(X_train)
    X_test = reducer.transform(X_test)

    # Train the model
    xg_model = XGBClassifier(max_depth=xg_depth, n_estimators=xg_estimators, learning_rate=xg_lr)
    xg_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = xg_model.predict_proba(X_test)
    y_pred = xg_model.predict(X_test)

    return y_pred, y_pred_proba
  
  ######################################### SelectKBest #########################################
  if any('selectkbest' in item for item in keys):
    xg_lr = xg_param["xgbclassifier__learning_rate"]
    xg_depth = xg_param["xgbclassifier__max_depth"]
    xg_estimators = xg_param["xgbclassifier__n_estimators"]
    selectkbest_k = xg_param["selectkbest__k"]

    # Pipeline w/ SelectKBest + XG Boost

    # Create SelectKBest object
    reducer = SelectKBest(k=selectkbest_k)

    # Turn data into z-scores
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # Feature selection
    X_train = reducer.fit_transform(X_train, y_train)
    X_test = reducer.transform(X_test)

    # Train the model
    xg_model = XGBClassifier(max_depth=xg_depth, n_estimators=xg_estimators, learning_rate=xg_lr)
    xg_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = xg_model.predict_proba(X_test)
    y_pred = xg_model.predict(X_test)

    return y_pred, y_pred_proba
  
  ######################################### ICA #########################################
  if any('fastica' in item for item in keys):
    xg_lr = xg_param["xgbclassifier__learning_rate"]
    xg_depth = xg_param["xgbclassifier__max_depth"]
    xg_estimators = xg_param["xgbclassifier__n_estimators"]
    fastica_n_components = xg_param["fastica__n_components"]

    # Pipeline w/ ICA + XG Boost

    # Create ICA object
    reducer = FastICA(n_components=fastica_n_components)

    # Turn data into z-scores
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # Feature selection
    X_train = reducer.fit_transform(X_train)
    X_test = reducer.transform(X_test)

    # Train the model
    xg_model = XGBClassifier(max_depth=xg_depth, n_estimators=xg_estimators, learning_rate=xg_lr)
    xg_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = xg_model.predict_proba(X_test)
    y_pred = xg_model.predict(X_test)

    return y_pred, y_pred_proba


def gmm_model(data, labels, val_data, gmm_param):

  # PARAMETERS: n_clusters, init
  # {gaussianmixture__init_params=k-means++, gaussianmixture__n_components=2, umap__n_components=5, umap__n_neighbors=10},
  
  # Reshape data
  num_files = data.shape[0]
  num_channels = data.shape[1]
  num_features = data.shape[2]

  data_reshape = np.reshape(data, (num_files, num_channels*num_features))

  num_files_val = val_data.shape[0]
  num_channels_val = val_data.shape[1]
  num_features_val = val_data.shape[2]

  val_data_reshape = np.reshape(val_data, (num_files_val, num_channels_val*num_features_val))

  X_train = data_reshape
  y_train = labels
  X_test = val_data_reshape

  keys = list(gmm_param.keys())

  ######################################### UMAP #########################################
  if any('umap' in item for item in keys):

    gmm_init = gmm_param["gaussianmixture__init_params"]
    gmm_cov = gmm_param["gaussianmixture__covariance_type"]
    umap_components = gmm_param["umap__n_components"]
    umap_neighbors = gmm_param["umap__n_neighbors"]
    umap_metric = gmm_param["umap__metric"]
    umap_dist = gmm_param["umap__min_dist"]

    # Pipeline w/ UMAP + Gaussian Mixture

    # Create UMAP object
    reducer = umap.UMAP(n_components=umap_components, n_neighbors=umap_neighbors, min_dist=umap_dist, metric=umap_metric)

    # Turn data into z-scores
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # Feature selection
    X_train = reducer.fit_transform(X_train)
    X_test = reducer.transform(X_test)

    # Train the model
    gmm_model = GaussianMixture(n_components=2, init_params=gmm_init, covariance_type=gmm_cov)
    gmm_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = gmm_model.predict_proba(X_test)
    y_pred = gmm_model.predict(X_test)

    return y_pred, y_pred_proba
  
  ######################################### SelectKBest #########################################
  if any('selectkbest' in item for item in keys):
    gmm_init = gmm_param["gaussianmixture__init_params"]
    gmm_cov = gmm_param["gaussianmixture__covariance_type"]
    selectkbest_k = gmm_param["selectkbest__k"]

    # Pipeline w/ SelectKBest + Gaussian Mixture

    # Create SelectKBest object
    reducer = SelectKBest(k=selectkbest_k)

    # Turn data into z-scores
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # Feature selection
    X_train = reducer.fit_transform(X_train, y_train)
    X_test = reducer.transform(X_test)

    # Train the model
    gmm_model = GaussianMixture(n_components=2, init_params=gmm_init, covariance_type=gmm_cov)
    gmm_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = gmm_model.predict_proba(X_test)
    y_pred = gmm_model.predict(X_test)

    return y_pred, y_pred_proba
  
  ######################################### ICA #########################################
  if any('fastica' in item for item in keys):
    gmm_init = gmm_param["gaussianmixture__init_params"]
    gmm_cov = gmm_param["gaussianmixture__covariance_type"]
    fastica_n_components = gmm_param["fastica__n_components"]

    # Pipeline w/ ICA + Gaussian Mixture

    # Create ICA object
    reducer = FastICA(n_components=fastica_n_components)

    # Turn data into z-scores
    scl = StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # Feature selection
    X_train = reducer.fit_transform(X_train)
    X_test = reducer.transform(X_test)

    # Train the model
    gmm_model = GaussianMixture(n_components=2, init_params=gmm_init, covariance_type=gmm_cov)
    gmm_model.fit(X_train, y_train)

    # Make predictions
    y_pred_proba = gmm_model.predict_proba(X_test)
    y_pred = gmm_model.predict(X_test)

    return y_pred, y_pred_proba