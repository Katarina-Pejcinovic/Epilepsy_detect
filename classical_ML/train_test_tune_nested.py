## Tune model hyperparameters
## Limited parameter range due to small initial dataset

from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap
import pickle

def create_svc_pipeline(group_kfold):

  pipeline = make_pipeline(StandardScaler(), umap.UMAP(), SVC())

  param_grid = {
      'umap__n_components':[3, 5],
      'umap__n_neighbors':[2, 3],
      'svc__kernel':['linear', 'rbf'],
      'svc__C':[1, 10],
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring="accuracy",
          cv=group_kfold,
          verbose=2
        )
  
  return param_search


def create_rf_pipeline(group_kfold):
  pipeline = make_pipeline(StandardScaler(), umap.UMAP(), RandomForestClassifier())

  param_grid = {
      'umap__n_components':[3, 5],
      'umap__n_neighbors':[2, 3],
      'randomforestclassifier__n_estimators':[10, 100],
      # 'randomforestclassifier__n_estimators':[10],
      'randomforestclassifier__min_samples_leaf':[1, 5],
      'randomforestclassifier__max_depth':[3, 5],
      # 'randomforestclassifier__max_features':[25]
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring="accuracy",
          cv=group_kfold,
          verbose=2
        )

  return param_search


def create_gmm_pipeline(group_kfold):
  pipeline = make_pipeline(StandardScaler(), umap.UMAP(), GaussianMixture(n_components=2))

  param_grid = {
      'umap__n_components':[3, 5],
      'umap__n_neighbors':[2, 3],
      'gaussianmixture__init_params':['k-means++', 'random'],
      # 'gaussianmixture__init_params':['k-means++'],
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring="accuracy",
          cv=group_kfold,
          verbose=2
        )

  return param_search


def create_xg_pipeline(group_kfold):
  pipeline = make_pipeline(StandardScaler(), umap.UMAP(), XGBClassifier(objective= 'binary:logistic'))

  param_grid = {
      'umap__n_components':[3, 5],
      'umap__n_neighbors':[2, 3],
      'xgbclassifier__max_depth':[3, 5],
      # 'xgbclassifier__max_depth':[2],
      'xgbclassifier__n_estimators': [50, 100],
      'xgbclassifier__learning_rate': [0.01, 0.1],
      # 'xgbclassifier__learning_rate': [0.01],
    }
  
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring="accuracy",
          cv=group_kfold,
          verbose=2
        )

  return param_search

def train_test_tune_nested(data, labels, groups):
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
  # data_reshape = data

  group_kfold = GroupKFold(n_splits=num_patients)

  ## Create pipelines
  
  # SVM
  svc_best_params_list = []
  svc_scores_list = []

  for train_idx, test_idx in group_kfold.split(data_reshape, labels, groups=groups):
    X_train, X_test = data_reshape[train_idx], data_reshape[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    groups_train = groups[train_idx]
    num_patients_train = np.size(np.unique(groups_train))
    group_kfold_inner = GroupKFold(n_splits=num_patients_train)

    svc_param_search = create_svc_pipeline(group_kfold_inner)

    svc_param_search.fit(X_train, y_train, groups=groups_train)
    svc_best_params_list.append(svc_param_search.best_params_)

    svc_scores = svc_param_search.score(X_test, y_test)
    svc_scores_list.append(svc_scores)

  best_svc_model_score = np.argmax(svc_scores_list)
  svc_best_score = np.max(svc_scores_list)
  svc_best_params = svc_best_params_list[best_svc_model_score]
  

  ## RF
  rf_best_params_list = []
  rf_scores_list = []

  for train_idx, test_idx in group_kfold.split(data_reshape, labels, groups=groups):
    X_train, X_test = data_reshape[train_idx], data_reshape[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    groups_train = groups[train_idx]
    num_patients_train = np.size(np.unique(groups_train))
    group_kfold_inner = GroupKFold(n_splits=num_patients_train)

    rf_param_search = create_rf_pipeline(group_kfold_inner)

    rf_param_search.fit(X_train, y_train, groups=groups_train)
    rf_best_params_list.append(rf_param_search.best_params_)

    rf_scores = rf_param_search.score(X_test, y_test)
    rf_scores_list.append(rf_scores)

  best_rf_model_score = np.argmax(rf_scores_list)
  rf_best_score = np.max(rf_scores_list)
  rf_best_params = rf_best_params_list[best_rf_model_score]


  ## XG Boost
  xg_best_params_list = []
  xg_scores_list = []

  for train_idx, test_idx in group_kfold.split(data_reshape, labels, groups=groups):
    X_train, X_test = data_reshape[train_idx], data_reshape[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    groups_train = groups[train_idx]
    num_patients_train = np.size(np.unique(groups_train))
    group_kfold_inner = GroupKFold(n_splits=num_patients_train)

    xg_param_search = create_xg_pipeline(group_kfold_inner)

    xg_param_search.fit(X_train, y_train, groups=groups_train)
    xg_best_params_list.append(xg_param_search.best_params_)

    xg_scores = xg_param_search.score(X_test, y_test)
    xg_scores_list.append(xg_scores)

  best_xg_model_score = np.argmax(xg_scores_list)
  xg_best_score = np.max(xg_scores_list)
  xg_best_params = xg_best_params_list[best_xg_model_score]


  ## GMM
  gmm_best_params_list = []
  gmm_scores_list = []

  for train_idx, test_idx in group_kfold.split(data_reshape, labels, groups=groups):
    X_train, X_test = data_reshape[train_idx], data_reshape[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    groups_train = groups[train_idx]
    num_patients_train = np.size(np.unique(groups_train))
    group_kfold_inner = GroupKFold(n_splits=num_patients_train)

    gmm_param_search = create_gmm_pipeline(group_kfold_inner)

    gmm_param_search.fit(X_train, y_train, groups=groups_train)
    gmm_best_params_list.append(gmm_param_search.best_params_)

    gmm_scores = gmm_param_search.score(X_test, y_test)
    gmm_scores_list.append(gmm_scores)

  best_gmm_model_score = np.argmax(gmm_scores_list)
  gmm_best_score = np.max(gmm_scores_list)
  gmm_best_params = gmm_best_params_list[best_gmm_model_score]


  ## Results
  print('Cross validate to determine optimal feature selection and model hyperparameters')

  # Return all of best parameters of each model as a multidimensional list
  param_scores = [svc_best_score, rf_best_score, xg_best_score, gmm_best_score]
  param_best = [svc_best_params, rf_best_params, xg_best_params, gmm_best_params]

  # Save best params to text file
  file = open('results/best_params.txt','w')
  for item in param_best:
    for key, value in item.items():
      file.write('%s: %s\n' % (key, value))
    file.write('\n')
  file.close()

  # Save best params to load later
  with open('results/best_params_dict.pkl', 'wb') as f:
    pickle.dump(param_best, f)

  # Return
  return param_scores, param_best

# # Run with fake test data
# patients = 5
# files = 5*patients
# channels = 2
# features = 10
# data = np.random.rand(files, channels, features)
# labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

# # # Return list of pd dataframes that contain every combo of parameters + mean_test_score
# # # Return list of dict for each model with the best parameters
# [scores, best_params] = train_test_tune(data, labels, groups)

# print('Done')