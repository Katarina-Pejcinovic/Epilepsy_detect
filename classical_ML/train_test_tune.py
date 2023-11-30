## Tune model hyperparameters
## From feature extraction -> (np array) -> UMAP Feature Selection -> (np array) -> ML model
## Calculate F2 score of results -> determine best model parameters

# Use F2 score (weigh recall higher) because in epilepsy detection, it is most important to detect ALL true positives

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
import pandas as pd
import umap.umap_ as umap

def create_svc_pipeline(group_kfold):

  pipeline = make_pipeline(StandardScaler(), umap.UMAP(), SVC())

  param_grid = {
      'umap__n_components':[1, 3],
      'umap__n_neighbors':[5, 10],
      'svc__kernel':['linear', 'rbf'],
      'svc__C':[1, 10],
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring="balanced_accuracy",
          cv=group_kfold,
          verbose=2
        )
  
  return param_search


def create_rf_pipeline(group_kfold):
  pipeline = make_pipeline(StandardScaler(), umap.UMAP(), RandomForestClassifier())

  param_grid = {
      'umap__n_components':[1, 3],
      'umap__n_neighbors':[5, 10],
      'randomforestclassifier__n_estimators':[10, 100],
      'randomforestclassifier__min_samples_leaf':[1, 5],
      'randomforestclassifier__max_features':[25, 50],
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring="balanced_accuracy",
          cv=group_kfold,
          verbose=2
        )

  return param_search


def create_kmeans_pipeline(group_kfold):
  pipeline = make_pipeline(StandardScaler(), umap.UMAP(), KMeans(n_clusters=2, n_init='auto'))

  param_grid = {
      'umap__n_components':[1, 3],
      'umap__n_neighbors':[5, 10],
      'kmeans__n_clusters':[2, 3],
      'kmeans__init':['k-means++', 'random'],
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring="balanced_accuracy",
          cv=group_kfold,
          verbose=2
        )

  return param_search

def create_gmm_pipeline(group_kfold):
  pipeline = make_pipeline(StandardScaler(), umap.UMAP(), GaussianMixture(n_components=2))

  param_grid = {
      'umap__n_components':[1, 3],
      'umap__n_neighbors':[5, 10],
      'gaussianmixture__init_params':['k-means++', 'random'],
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring="balanced_accuracy",
          cv=group_kfold,
          verbose=2
        )

  return param_search


def create_xg_pipeline(group_kfold):
  pipeline = make_pipeline(StandardScaler(), umap.UMAP(), XGBClassifier(objective= 'binary:logistic'))

  param_grid = {
      'umap__n_components':[1, 3],
      'umap__n_neighbors':[5, 10],
      'xgbclassifier__max_depth':[2, 5],
      'xgbclassifier__n_estimators': [50, 100],
      'xgbclassifier__learning_rate': [0.01, 0.1],
    }
  
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring="balanced_accuracy",
          cv=group_kfold,
          verbose=2
        )

  return param_search

def train_test_tune(data, labels, groups):
  # Reshape data
  # Cross validate loop
  # Inside loop - UMAP + model
  # Use grid search cv with group kfold
  # return best parameters per model
  # May not work for unsupervised models

  # Inputs: numpy array of data (size: [# files, 32 channels, 177 features])
  #         numpy array of labels (size: [# files, 1]) -- Epilepsy or No Epilepsy
  #         numpy array of patient ID per file (size: [# files, 1])

  # Outputs: best hyperparameters for each classical ml model

  ## Reshape data
  num_files = data.shape[0]
  num_channels = data.shape[1]
  num_features = data.shape[2]
  num_patients = np.size(np.unique(groups))

  # data_reshape = np.reshape(data, (num_files, num_channels*num_features))
  data_reshape = data

  group_kfold = GroupKFold(n_splits=num_patients)

  ## Create pipelines

  ## SVC
  svc_param_search = create_svc_pipeline(group_kfold)
  svc_param_search.fit(data_reshape, labels, groups=groups)

  svc_best_params = svc_param_search.best_params_
  svc_results = pd.DataFrame(svc_param_search.cv_results_)
  svc_params = svc_results[['param_umap__n_components', 'param_umap__n_neighbors',
                            'param_svc__C', 'param_svc__kernel', 'mean_test_score']]

  ## RF
  rf_param_search = create_rf_pipeline(group_kfold)
  rf_param_search.fit(data_reshape, labels, groups=groups)

  rf_best_params = rf_param_search.best_params_
  rf_results = pd.DataFrame(rf_param_search.cv_results_)
  rf_params = rf_results[['param_umap__n_components', 'param_umap__n_neighbors',
                          'param_randomforestclassifier__n_estimators', 'param_randomforestclassifier__min_samples_leaf',
                          'param_randomforestclassifier__max_features', 'mean_test_score']]

  ## K Means
  kmeans_param_search = create_kmeans_pipeline(group_kfold)
  kmeans_param_search.fit(data_reshape, labels, groups=groups)

  kmeans_best_params = kmeans_param_search.best_params_
  kmeans_results = pd.DataFrame(kmeans_param_search.cv_results_)
  kmeans_params = kmeans_results[['param_umap__n_components', 'param_umap__n_neighbors',
                                  'param_kmeans__init', 'param_kmeans__n_clusters','mean_test_score']]

  ## GMM
  gmm_param_search = create_gmm_pipeline(group_kfold)
  gmm_param_search.fit(data_reshape, labels, groups=groups)

  gmm_best_params = gmm_param_search.best_params_
  gmm_results = pd.DataFrame(gmm_param_search.cv_results_)
  gmm_params = gmm_results[['param_umap__n_components', 'param_umap__n_neighbors',
                            'param_gaussianmixture__init_params', 'mean_test_score']]

  ## XG Boost
  xg_param_search = create_xg_pipeline(group_kfold)
  xg_param_search.fit(data_reshape, labels, groups=groups)

  xg_best_params = xg_param_search.best_params_
  xg_results = pd.DataFrame(xg_param_search.cv_results_)
  xg_params = xg_results[['param_umap__n_components', 'param_umap__n_neighbors',
                          'param_xgbclassifier__n_estimators', 'param_xgbclassifier__max_depth', 
                          'param_xgbclassifier__learning_rate', 'mean_test_score']]

  ## Results
  print('Cross validate to determine optimal feature selection and model hyperparameters')

  # Return all of best parameters of each model as a multidimensional list
  params_full = [svc_params, rf_params, kmeans_params, xg_params, gmm_params]
  params_best = [svc_best_params, rf_best_params, kmeans_best_params, xg_best_params, gmm_best_params]

  return params_full, params_best

# # Run with fake test data
# patients = 5
# files = 5*patients
# channels = 2
# features = 10
# data = np.random.rand(files, channels, features)
# labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

# # Return list of pd dataframes that contain every combo of parameters + mean_test_score
# # Return list of dict for each model with the best parameters
# [params, best_params] = train_test_tune(data, labels, groups)

# print('Done')