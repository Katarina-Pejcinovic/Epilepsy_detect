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
from sklearn.decomposition import FastICA
import pickle
from sklearn.metrics import precision_score, recall_score, accuracy_score


def create_svc_pipeline(stratified_kfold, scoring_methods):

  pipeline = make_pipeline(StandardScaler(), FastICA(), SVC())

  param_grid = {
      'fastica__n_components':np.linspace(10, 100, 10, endpoint=True).astype(int),
      'svc__kernel':['linear', 'rbf', 'poly', 'sigmoid'],
      'svc__C':[0.1, 1, 10, 100],
      'svc__degree': [2, 3, 4, 5],
      # 'fastica__n_components':[10, 20],
      # 'svc__kernel':['linear'],
      # 'svc__C':[0.1, 10],
      # 'svc__degree': [2, 3]
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring=scoring_methods,
          refit='Accuracy',
          cv=stratified_kfold,
          verbose=2
        )
  
  return param_search


def create_rf_pipeline(stratified_kfold, scoring_methods):
  pipeline = make_pipeline(StandardScaler(), FastICA(), RandomForestClassifier())

  param_grid = {
      'fastica__n_components':np.linspace(10, 100, 10, endpoint=True).astype(int),
      'randomforestclassifier__n_estimators':[1, 2, 4, 8, 16, 32, 64, 100],
      'randomforestclassifier__min_samples_leaf':np.linspace(50, 400, 8, endpoint=True),
      'randomforestclassifier__max_depth':np.linspace(2, 20, 10, endpoint=True),
      # 'fastica__n_components':[10, 20],
      # 'randomforestclassifier__n_estimators':[8, 16],
      # 'randomforestclassifier__min_samples_leaf':[50, 200],
      # 'randomforestclassifier__max_depth':[2, 10],
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring=scoring_methods,
          refit='Accuracy',
          cv=stratified_kfold,
          verbose=2
        )

  return param_search

def create_xg_pipeline(stratified_kfold, scoring_methods):
  pipeline = make_pipeline(StandardScaler(), FastICA(), XGBClassifier(objective= 'binary:logistic'))

  param_grid = {
      'fastica__n_components':np.linspace(10, 100, 10, endpoint=True).astype(int),
      'xgbclassifier__max_depth':np.linspace(3, 10, 8, endpoint=True),
      'xgbclassifier__n_estimators': np.linspace(100, 500, 5, endpoint=True),
      'xgbclassifier__learning_rate': [0.01, 0.1],
      # 'fastica__n_components':[10, 20],
      # 'xgbclassifier__n_estimators': [50],
      # 'xgbclassifier__max_depth':[3, 5],
      # 'xgbclassifier__learning_rate': [0.01, 0.1],
    }
  
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring=scoring_methods,
          refit='Accuracy',
          cv=stratified_kfold,
          verbose=2
        )

  return param_search

def create_gmm_pipeline(stratified_kfold, scoring_methods):
  pipeline = make_pipeline(StandardScaler(), FastICA(), GaussianMixture(n_components=2))

  param_grid = {
      'fastica__n_components':np.linspace(10, 100, 10, endpoint=True).astype(int),
      'gaussianmixture__init_params':['k-means++', 'random'],
      'gaussianmixture__covariance_type': ['full', 'tied', 'diag', 'spherical'],
      # 'fastica__n_components':[10, 20],
      # 'gaussianmixture__init_params':['k-means++'],
      # 'gaussianmixture__covariance_type': ['full'],
    }

  # Parameter search
  param_search = GridSearchCV(
          estimator = pipeline,
          param_grid = param_grid,
          n_jobs=1,
          scoring=scoring_methods,
          refit='Accuracy',
          cv=stratified_kfold,
          verbose=2
        )

  return param_search

def calc_f2_score(precision, recall, beta):
  num = (1 + pow(beta, 2)) * (precision) * (recall)
  denom = (pow(beta, 2) * precision) + recall
  f2_score = num/denom
  return f2_score

def train_test_tune_ica(data, labels, patient_id, stratified_cv):
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
  splits = 3

  stratified_cv = list(stratified_cv)

  ## Create pipelines
  
  # SVM
  svc_best_params_list = []
  svc_accuracy_list = []
  svc_precision_list = []
  svc_recall_list = []
  svc_f2_list = []

  rf_best_params_list = []
  rf_accuracy_list = []
  rf_precision_list = []
  rf_recall_list = []
  rf_f2_list = []

  xg_best_params_list = []
  xg_accuracy_list = []
  xg_precision_list = []
  xg_recall_list = []
  xg_f2_list = []

  gmm_best_params_list = []
  gmm_accuracy_list = []
  gmm_precision_list = []
  gmm_recall_list = []
  gmm_f2_list = []

  for i, (train_idx, test_idx) in enumerate(stratified_cv):
    X_train, X_test = data_reshape[train_idx], data_reshape[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    group_train = patient_id[train_idx]

    strat_kfold_object = StratifiedKFold(n_splits=splits, shuffle=True)
    scoring = {"Precision": 'precision', "Recall": 'recall', "Accuracy": 'balanced_accuracy'}
    # strat_kfold_inner = strat_kfold_object.split(X_train, group_train)

    # for j, (train_index, test_index) in enumerate(strat_kfold_inner):
    #     print(f"Fold {j}:")
    #     print(f"  Train: index={train_index}")
    #     print(f"  Test:  index={test_index}")

    ################################################## SVC ##################################################
    svc_param_search = create_svc_pipeline(strat_kfold_object.split(X_train, group_train), scoring)
    svc_param_search.fit(X_train, y_train)

    # Get best set of params based on F2 scoring
    svc_results = svc_param_search.cv_results_
    svc_accuracy = svc_results['mean_test_Accuracy']
    svc_precision = svc_results['mean_test_Precision']
    svc_recall = svc_results['mean_test_Recall']
    svc_f2 = calc_f2_score(svc_precision, svc_recall, 2)
    best_f2_index = np.nanargmax(svc_f2)
    svc_best_params = svc_results['params'][best_f2_index]

    # Save best set of params
    svc_best_params_list.append(svc_best_params)
    svc_param_search.best_params_ = svc_best_params

    # Update estimator with F2 params for validation
    for step_name, step_params in svc_best_params.items():
      step, param_name = step_name.split('__', 1)
      svc_param_search.best_estimator_.named_steps[step].set_params(**{param_name: step_params})

    best_estimator = svc_param_search.best_estimator_

    # Get accuracy, precision, recall, F2 score of best param set
    y_pred_test = best_estimator.predict(X_test)
    svc_accuracy_test = accuracy_score(y_test, y_pred_test)
    svc_precision_test = precision_score(y_test, y_pred_test)
    svc_recall_test = recall_score(y_test, y_pred_test)
    svc_f2_test = calc_f2_score(svc_precision_test, svc_recall_test, 2)

    # Save metrics per fold
    svc_accuracy_list.append(svc_accuracy_test)
    svc_precision_list.append(svc_precision_test)
    svc_recall_list.append(svc_recall_test)
    svc_f2_list.append(svc_f2_test)

    ################################################## RF ##################################################
    rf_param_search = create_rf_pipeline(strat_kfold_object.split(X_train, group_train), scoring)    
    rf_param_search.fit(X_train, y_train)

    # Get best set of params based on F2 scoring
    rf_results = rf_param_search.cv_results_
    rf_accuracy = rf_results['mean_test_Accuracy']
    rf_precision = rf_results['mean_test_Precision']
    rf_recall = rf_results['mean_test_Recall']
    rf_f2 = calc_f2_score(rf_precision, rf_recall, 2)
    rf_best_f2_index = np.nanargmax(rf_f2)
    rf_best_params = rf_results['params'][rf_best_f2_index]

    # Save best set of params
    rf_best_params_list.append(rf_best_params)
    rf_param_search.best_params_ = rf_best_params

    # Update estimator with F2 params for validation
    for step_name, step_params in rf_best_params.items():
      step, param_name = step_name.split('__', 1)
      rf_param_search.best_estimator_.named_steps[step].set_params(**{param_name: step_params})

    best_estimator = rf_param_search.best_estimator_

    # Get accuracy, precision, recall, F2 score of best param set
    y_pred_test = best_estimator.predict(X_test)
    rf_accuracy_test = accuracy_score(y_test, y_pred_test)
    rf_precision_test = precision_score(y_test, y_pred_test)
    rf_recall_test = recall_score(y_test, y_pred_test)
    rf_f2_test = calc_f2_score(rf_precision_test, rf_recall_test, 2)

    # Save metrics per fold
    rf_accuracy_list.append(rf_accuracy_test)
    rf_precision_list.append(rf_precision_test)
    rf_recall_list.append(rf_recall_test)
    rf_f2_list.append(rf_f2_test)

    ################################################## XG Boost ##################################################
    xg_param_search = create_xg_pipeline(strat_kfold_object.split(X_train, group_train), scoring)

    xg_param_search.fit(X_train, y_train)

    # Get best set of params based on F2 scoring
    xg_results = xg_param_search.cv_results_
    xg_accuracy = xg_results['mean_test_Accuracy']
    xg_precision = xg_results['mean_test_Precision']
    xg_recall = xg_results['mean_test_Recall']
    xg_f2 = calc_f2_score(xg_precision, xg_recall, 2)
    xg_best_f2_index = np.nanargmax(xg_f2)
    xg_best_params = xg_results['params'][xg_best_f2_index]

    # Save best set of params
    xg_best_params_list.append(xg_best_params)
    xg_param_search.best_params_ = xg_best_params

    # Update estimator with F2 params for validation
    for step_name, step_params in xg_best_params.items():
      step, param_name = step_name.split('__', 1)
      xg_param_search.best_estimator_.named_steps[step].set_params(**{param_name: step_params})

    best_estimator = xg_param_search.best_estimator_

    # Get accuracy, precision, recall, F2 score of best param set
    y_pred_test = best_estimator.predict(X_test)
    xg_accuracy_test = accuracy_score(y_test, y_pred_test)
    xg_precision_test = precision_score(y_test, y_pred_test)
    xg_recall_test = recall_score(y_test, y_pred_test)
    xg_f2_test = calc_f2_score(xg_precision_test, xg_recall_test, 2)

    # Save metrics per fold
    xg_accuracy_list.append(xg_accuracy_test)
    xg_precision_list.append(xg_precision_test)
    xg_recall_list.append(xg_recall_test)
    xg_f2_list.append(xg_f2_test)

    ################################################## GMM ##################################################
    gmm_param_search = create_gmm_pipeline(strat_kfold_object.split(X_train, group_train), scoring)
    gmm_param_search.fit(X_train, y_train)

    # Get best set of params based on F2 scoring
    gmm_results = gmm_param_search.cv_results_
    gmm_accuracy = gmm_results['mean_test_Accuracy']
    gmm_precision = gmm_results['mean_test_Precision']
    gmm_recall = gmm_results['mean_test_Recall']
    gmm_f2 = calc_f2_score(gmm_precision, gmm_recall, 2)
    gmm_best_f2_index = np.nanargmax(gmm_f2)
    gmm_best_params = gmm_results['params'][gmm_best_f2_index]

    # Save best set of params
    gmm_best_params_list.append(gmm_best_params)
    gmm_param_search.best_params_ = gmm_best_params

    # Update estimator with F2 params for validation
    for step_name, step_params in gmm_best_params.items():
      step, param_name = step_name.split('__', 1)
      gmm_param_search.best_estimator_.named_steps[step].set_params(**{param_name: step_params})

    best_estimator = gmm_param_search.best_estimator_

    # Get accuracy, precision, recall, F2 score of best param set
    y_pred_test = best_estimator.predict(X_test)
    gmm_accuracy_test = accuracy_score(y_test, y_pred_test)
    gmm_precision_test = precision_score(y_test, y_pred_test)
    gmm_recall_test = recall_score(y_test, y_pred_test)
    gmm_f2_test = calc_f2_score(gmm_precision_test, gmm_recall_test, 2)

    # Save metrics per fold
    gmm_accuracy_list.append(gmm_accuracy_test)
    gmm_precision_list.append(gmm_precision_test)
    gmm_recall_list.append(gmm_recall_test)
    gmm_f2_list.append(gmm_f2_test)


  ########################## Find best set of params from all of the outer folds ##########################
  try:
    svc_best_score = np.nanmax(svc_f2_list)
  except:
    svc_best_score = np.nan
  best_svc_model_score = np.where(svc_f2_list == svc_best_score)
  svc_best_params = svc_best_params_list[best_svc_model_score[0][0]]

  # Save svc params to text file
  file = open('results/best_ica_svc_params.txt','w')
  for item, score in zip(svc_best_params_list, svc_f2_list):
    for key, value in item.items():
      file.write('%s: %s\n' % (key, value))
    # file.write('\n')
    file.write('F2 Score: %s\n\n' % (score))
  file.close()
  
  try:
    rf_best_score = np.nanmax(rf_f2_list)
  except:
    rf_best_score = np.nan
  best_rf_model_score = np.where(rf_f2_list == rf_best_score)
  rf_best_params = rf_best_params_list[best_rf_model_score[0][0]]

  # Save rf params to text file
  file = open('results/best_ica_rf_params.txt','w')
  for item, score in zip(rf_best_params_list, rf_f2_list):
    for key, value in item.items():
      file.write('%s: %s\n' % (key, value))
    # file.write('\n')
    file.write('F2 Score: %s\n\n' % (score))
  file.close()

  try:
    xg_best_score = np.nanmax(xg_f2_list)
  except:
    xg_best_score = np.nan
  best_xg_model_score = np.where(xg_f2_list == xg_best_score)
  xg_best_params = xg_best_params_list[best_xg_model_score[0][0]]

  # Save xg params to text file
  file = open('results/best_ica_xg_params.txt','w')
  for item, score in zip(xg_best_params_list, xg_f2_list):
    for key, value in item.items():
      file.write('%s: %s\n' % (key, value))
    # file.write('\n')
    file.write('F2 Score: %s\n\n' % (score))
  file.close()

  try:
    gmm_best_score = np.nanmax(gmm_f2_list)
  except:
    gmm_best_score = np.nan
  best_gmm_model_score = np.where(gmm_f2_list == gmm_best_score)
  gmm_best_params = gmm_best_params_list[best_gmm_model_score[0][0]]

  # Save gmm params to text file
  file = open('results/best_ica_gmm_params.txt','w')
  for item, score in zip(gmm_best_params_list, gmm_f2_list):
    for key, value in item.items():
      file.write('%s: %s\n' % (key, value))
    # file.write('\n')
    file.write('F2 Score: %s\n\n' % (score))
  file.close()

  # Return all of best parameters of each model as a multidimensional list

  svc_scores_list = [svc_accuracy_list, svc_precision_list, svc_recall_list, svc_f2_list]
  rf_scores_list = [rf_accuracy_list, rf_precision_list, rf_recall_list, rf_f2_list]
  xg_scores_list = [xg_accuracy_list, xg_precision_list, xg_recall_list, xg_f2_list]
  gmm_scores_list = [gmm_accuracy_list, gmm_precision_list, gmm_recall_list, gmm_f2_list]
  
  param_scores = [svc_best_score, rf_best_score, xg_best_score, gmm_best_score]
  param_best = [svc_best_params, rf_best_params, xg_best_params, gmm_best_params]
  all_scores = [svc_scores_list, rf_scores_list, xg_scores_list, gmm_scores_list]

  # Save best params to load later
  with open('results/best_ica_params_dict.pkl', 'wb') as f:
    pickle.dump(param_best, f)

  # Save best params scores to load later
  with open('results/best_ica_scores.pkl', 'wb') as f:
    pickle.dump(param_scores, f)

  # Save all scores to load later
  with open('results/ica_scores.pkl', 'wb') as f:
    pickle.dump(all_scores, f)

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