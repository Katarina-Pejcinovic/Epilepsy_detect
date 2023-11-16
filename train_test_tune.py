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
import pandas as pd
from sklearn.datasets import make_classification
import umap.umap_ as umap

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

  data_reshape = np.reshape(data, (num_files, num_channels*num_features))

  group_kfold = GroupKFold(n_splits=num_patients)

  ## Create pipelines

  ## SVC
  # svc_pipeline = make_pipeline(StandardScaler(), umap.UMAP(), SVC())

  # svc_param_grid = {
  #     'umap__n_components':[5, 10],
  #     'umap__n_neighbors':[5, 10],
  #     'svc__kernel':['linear', 'rbf'],
  #     'svc__C':[1, 10],
  #   }

  # # Parameter search
  # svc_param_search = GridSearchCV(
  #         estimator = svc_pipeline,
  #         param_grid = svc_param_grid,
  #         n_jobs=1,
  #         scoring="accuracy",
  #         cv=group_kfold,
  #         verbose=2
  #       )

  # svc_param_search.fit(data_reshape, labels, groups=groups)

  # svc_best_params = svc_param_search.best_params_
  # svc_results = pd.DataFrame(svc_param_search.cv_results_)
  # svc_params = svc_results[['param_umap__n_components', 'param_svc__C', 'mean_test_score']]

  ## RF
  rf_pipeline = make_pipeline(StandardScaler(), umap.UMAP(), RandomForestClassifier())

  rf_param_grid = {
      'umap__n_components':[5, 10],
      # 'umap__n_neighbors':[5, 10],
      'randomforestclassifier__n_estimators':[10, 100],
      # 'randomforestclassifier__min_samples_leaf':[1, 5],
      # 'randomforestclassifier__max_features':[25, 50],
    }

  # Parameter search
  rf_param_search = GridSearchCV(
          estimator = rf_pipeline,
          param_grid = rf_param_grid,
          n_jobs=1,
          scoring="accuracy",
          cv=group_kfold,
          verbose=2
        )

  rf_param_search.fit(data_reshape, labels, groups=groups)

  rf_best_params = rf_param_search.best_params_
  rf_results = pd.DataFrame(rf_param_search.cv_results_)
  rf_params = rf_results[['param_umap__n_components', 'param_randomforestclassifier__n_estimators', 'mean_test_score']]

  print('Cross validate to determine optimal feature selection and model hyperparameters')

  # return all of best parameters of each model as a multidimensional list
  parameters = [['kernel', 'C', 'gamma', 'degree'], ['n_estimators', 'min_samples_leaf', 'max_features'],
                ['_covariance_type'], ['n_clusters'], ['n_components', 'n_neighbors', 'min_dist', 'metrics']]

  return rf_params, rf_best_params

patients = 5
files = 5*patients
channels = 2
features = 10
data = np.random.rand(files, channels, features)
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

data_reshape = np.reshape(data, (files, channels*features));
print(data_reshape.shape)

[params, best_params] = train_test_tune(data, labels, groups)

print(params)
print(best_params)