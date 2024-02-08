## Tune model hyperparameters
## Limited parameter range due to small initial dataset

from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import make_scorer,fbeta_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import SelectKBest
import pickle


def create_svc_pipeline(stratified_kfold, f2_score):

  pipeline = make_pipeline(StandardScaler(), SelectKBest(score_func=f2_score), SVC())

  param_grid = {
      # 'selectkbest__k':np.linspace(10, 100, 10, endpoint=True),
      # 'svc__kernel':['linear', 'rbf', 'poly', 'sigmoid'],
      # 'svc__C':[0.1, 1, 10, 100],
      # 'svc__degree': [2, 3, 4, 5]
    'selectkbest__k':[10, 20],
    'svc__kernel':['linear'],
    'svc__C':[0.1],
    'svc__degree': [2]
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring=f2_score,
          cv=stratified_kfold,
          verbose=2,
          error_score='raise',
        )
  
  return param_search


def create_rf_pipeline(stratified_kfold, f2_score):
  pipeline = make_pipeline(StandardScaler(), SelectKBest(score_func=f2_score), RandomForestClassifier())

  param_grid = {
      # 'selectkbest__k':np.linspace(10, 100, 10, endpoint=True),
      # 'randomforestclassifier__n_estimators':[1, 2, 4, 8, 16, 32, 64, 100],
      # 'randomforestclassifier__min_samples_leaf':np.linspace(50, 400, 8, endpoint=True),
      # 'randomforestclassifier__max_depth':np.linspace(2, 20, 10, endpoint=True),
      'selectkbest__k':[10, 20],
      'randomforestclassifier__n_estimators':[8],
      'randomforestclassifier__min_samples_leaf':[50],
      'randomforestclassifier__max_depth':[2],
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring=f2_score,
          cv=stratified_kfold,
          verbose=2
        )

  return param_search


def create_gmm_pipeline(stratified_kfold, f2_score):
  pipeline = make_pipeline(StandardScaler(), SelectKBest(score_func=f2_score), GaussianMixture(n_components=2))

  param_grid = {
      # 'selectkbest__k':np.linspace(10, 100, 10, endpoint=True),
      # 'gaussianmixture__init_params':['k-means++', 'random'],
      # 'gaussianmixture__covariance_type': ['full', 'tied', 'diag', 'spherical'],
      'selectkbest__k':[10, 20],
      'gaussianmixture__init_params':['k-means++'],
      'gaussianmixture__covariance_type': ['full'],
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring=f2_score,
          cv=stratified_kfold,
          verbose=2
        )

  return param_search


def create_xg_pipeline(stratified_kfold, f2_score):
  pipeline = make_pipeline(StandardScaler(), SelectKBest(score_func=f2_score), XGBClassifier(objective= 'binary:logistic'))

  param_grid = {
      # 'selectkbest__k':np.linspace(10, 100, 10, endpoint=True),
      # 'xgbclassifier__max_depth':np.linspace(3, 10, 8, endpoint=True),
      # 'xgbclassifier__n_estimators': np.linspace(100, 500, 5, endpoint=True),
      # 'xgbclassifier__learning_rate': [0.01, 0.1],
      'selectkbest__k':[10, 20],
      'xgbclassifier__n_estimators': [50],
      'xgbclassifier__max_depth':[3],
      'xgbclassifier__learning_rate': [0.01],
    }
  
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring=f2_score,
          cv=stratified_kfold,
          verbose=2
        )

  return param_search

def train_test_tune_selectkbest(data, labels, patient_id, stratified_cv):
  # Reshape data
  # Cross validate loop
  # Inside loop - UMAP + model
  # Use grid search cv with group kfold
  # return best parameters per model
  # May not work for unsupervised models

  # Inputs: numpy array of data (size: [# files, 26 channels, # features (225)])
  #         numpy array of labels (size: [# files, 1]) -- Epilepsy or No Epilepsy
  #         numpy array of patient ID per file (size: [# files, 1])
  #         stratified_cv object

  # Outputs: best hyperparameters for each classical ml model

  ## Reshape data
  num_segments = data.shape[0]
  num_channels = data.shape[1]
  num_features = data.shape[2]

  data_reshape = np.reshape(data, (num_segments, num_channels*num_features))

  # num_patients = np.size(np.unique(patient_id))
  splits = 2

  stratified_cv = list(stratified_cv)

  ## Create pipelines
  
  # SVM
  svc_best_params_list = []
  svc_scores_list = []

  for i, (train_idx, test_idx) in enumerate(stratified_cv):
    X_train, X_test = data_reshape[train_idx], data_reshape[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    group_train = patient_id[train_idx]

    strat_kfold_object = StratifiedKFold(n_splits=splits, shuffle=True)
    f2_score = make_scorer(fbeta_score, beta=2, average='micro')
    # strat_kfold_inner = strat_kfold_object.split(X_train, group_train)

    # for j, (train_index, test_index) in enumerate(strat_kfold_inner):
    #     print(f"Fold {j}:")
    #     print(f"  Train: index={train_index}")
    #     print(f"  Test:  index={test_index}")

    svc_param_search = create_svc_pipeline(strat_kfold_object.split(X_train, group_train), f2_score)
    svc_param_search.fit(X_train, y_train)

    svc_best_params_list.append(svc_param_search.best_params_)

    svc_scores = svc_param_search.score(X_test, y_test)
    svc_scores_list.append(svc_scores)

  best_svc_model_score = np.argmax(svc_scores_list)
  svc_best_score = np.max(svc_scores_list)
  svc_best_params = svc_best_params_list[best_svc_model_score]

  # Save best params to text file
  file = open('results/best_svc_params_selectk.txt','w')
  for item, score in zip(svc_best_params_list, svc_scores_list):
    for key, value in item.items():
      file.write('%s: %s\n' % (key, value))
    # file.write('\n')
    file.write('F2 Score: %s\n\n' % (score))
  file.close()
  

  ## RF
  rf_best_params_list = []
  rf_scores_list = []

  for i, (train_idx, test_idx) in enumerate(stratified_cv):
    X_train, X_test = data_reshape[train_idx], data_reshape[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    groups_train = patient_id[train_idx]

    strat_kfold_object = StratifiedKFold(n_splits=splits, shuffle=True)
    f2_score = make_scorer(fbeta_score, beta=2, average='micro')

    rf_param_search = create_rf_pipeline(strat_kfold_object.split(X_train, group_train), f2_score)
    rf_param_search.fit(X_train, y_train)

    rf_best_params_list.append(rf_param_search.best_params_)

    rf_scores = rf_param_search.score(X_test, y_test)
    rf_scores_list.append(rf_scores)

  best_rf_model_score = np.argmax(rf_scores_list)
  rf_best_score = np.max(rf_scores_list)
  rf_best_params = rf_best_params_list[best_rf_model_score]

    # Save best params to text file
  file = open('results/best_rf_params_selectk.txt','w')
  for item, score in zip(rf_best_params_list, rf_scores_list):
    for key, value in item.items():
      file.write('%s: %s\n' % (key, value))
    # file.write('\n')
    file.write('F2 Score: %s\n\n' % (score))
  file.close()


  ## XG Boost
  xg_best_params_list = []
  xg_scores_list = []

  for i, (train_idx, test_idx) in enumerate(stratified_cv):
    X_train, X_test = data_reshape[train_idx], data_reshape[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    groups_train = patient_id[train_idx]

    strat_kfold_object = StratifiedKFold(n_splits=splits, shuffle=True)
    f2_score = make_scorer(fbeta_score, beta=2, average='micro')

    xg_param_search = create_xg_pipeline(strat_kfold_object.split(X_train, group_train), f2_score)
    xg_param_search.fit(X_train, y_train)

    xg_best_params_list.append(xg_param_search.best_params_)

    xg_scores = xg_param_search.score(X_test, y_test)
    xg_scores_list.append(xg_scores)

  best_xg_model_score = np.argmax(xg_scores_list)
  xg_best_score = np.max(xg_scores_list)
  xg_best_params = xg_best_params_list[best_xg_model_score]

    # Save best params to text file
  file = open('results/best_xg_params_selectk.txt','w')
  for item, score in zip(xg_best_params_list, xg_scores_list):
    for key, value in item.items():
      file.write('%s: %s\n' % (key, value))
    # file.write('\n')
    file.write('F2 Score: %s\n\n' % (score))
  file.close()


  ## GMM
  gmm_best_params_list = []
  gmm_scores_list = []

  for i, (train_idx, test_idx) in enumerate(stratified_cv):
    X_train, X_test = data_reshape[train_idx], data_reshape[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    groups_train = patient_id[train_idx]
    
    strat_kfold_object = StratifiedKFold(n_splits=splits, shuffle=True)
    f2_score = make_scorer(fbeta_score, beta=2, average='micro')

    gmm_param_search = create_gmm_pipeline(strat_kfold_object.split(X_train, group_train), f2_score)
    gmm_param_search.fit(X_train, y_train)

    gmm_best_params_list.append(gmm_param_search.best_params_)

    gmm_scores = gmm_param_search.score(X_test, y_test)
    gmm_scores_list.append(gmm_scores)

  best_gmm_model_score = np.argmax(gmm_scores_list)
  gmm_best_score = np.max(gmm_scores_list)
  gmm_best_params = gmm_best_params_list[best_gmm_model_score]

  # Save best params to text file
  file = open('results/best_gmm_params_selectk.txt','w')
  for item, score in zip(gmm_best_params_list, gmm_scores_list):
    for key, value in item.items():
      file.write('%s: %s\n' % (key, value))
    # file.write('\n')
    file.write('F2 Score: %s\n\n' % (score))
  file.close()


  ## Results
  print('Cross validate to determine optimal feature selection and model hyperparameters')

  # Return all of best parameters of each model as a multidimensional list
  param_scores = [svc_best_score, rf_best_score, xg_best_score, gmm_best_score]
  param_best = [svc_best_params, rf_best_params, xg_best_params, gmm_best_params]

  # Save best params to text file
  file = open('results/best_params_selectk.txt','w')
  for item in param_best:
    for key, value in item.items():
      file.write('%s: %s\n' % (key, value))
    file.write('\n')
  file.close()

  # Save best params to load later
  with open('results/best_params_dict_selectk.pkl', 'wb') as f:
    pickle.dump(param_best, f)

  # Return
  return param_scores, param_best









# Run with fake test data
# patients = 5
# files = 5*patients
# channels = 2
# features = 10
# data = np.random.rand(files, channels, features)
# labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
# groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

# # # Return list of pd dataframes that contain every combo of parameters + mean_test_score
# # # Return list of dict for each model with the best parameters
# [scores, best_params] = train_test_tune_nested(data, labels, groups)