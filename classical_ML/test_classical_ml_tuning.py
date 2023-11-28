import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier
import pandas as pd
import umap.umap_ as umap

from train_test_tune import *

### Classical ML Parameter Tuning

def eval_tuning(data, labels, groups):
  [params, best_params] = train_test_tune(data, labels, groups)

  # Check the evaluation scores of each test parameter combination
  svc_params = params[0].sort_values(by=['mean_test_score'], ascending=False)
  rf_params = params[1].sort_values(by=['mean_test_score'], ascending=False)
  kmeans_params = params[2].sort_values(by=['mean_test_score'], ascending=False)
  xg_params = params[3].sort_values(by=['mean_test_score'], ascending=False)
  gmm_params = params[4].sort_values(by=['mean_test_score'], ascending=False)

  svc_tune_scores = svc_params.mean_test_score
  rf_tune_scores = rf_params.mean_test_score
  kmeans_tune_scores = kmeans_params.mean_test_score
  xg_tune_scores = xg_params.mean_test_score
  gmm_tune_scores = gmm_params.mean_test_score

  best_svc_score = max(svc_tune_scores)
  avg_svc_score = np.mean(svc_tune_scores)

  best_rf_score = max(rf_tune_scores)
  avg_rf_score = np.mean(rf_tune_scores)

  best_kmeans_score = max(kmeans_tune_scores)
  avg_kmeans_score = np.mean(kmeans_tune_scores)

  best_xg_score = max(xg_tune_scores)
  avg_xg_score = np.mean(xg_tune_scores)

  best_gmm_score = max(gmm_tune_scores)
  avg_gmm_score = np.mean(gmm_tune_scores)

  print('SVC Results')
  print('Best score: ', best_svc_score)
  print('Average score: ', avg_svc_score)
  print('Best parameters: ', svc_params.iloc[:3], '\n')

  print('Random Forest Results')
  print('Best score: ', best_rf_score)
  print('Average score: ', avg_rf_score)
  print('Best parameters: ', rf_params.iloc[:3], '\n')
    
  print('K Means Results')
  print('Best score: ', best_kmeans_score)
  print('Average score: ', avg_kmeans_score)
  print('Best parameters: ', kmeans_params.iloc[:3], '\n')
    
  print('XG Boost Results')
  print('Best score: ', best_xg_score)
  print('Average score: ', avg_xg_score)
  print('Best parameters: ', xg_params.iloc[:3], '\n')

  print('Gaussian Mixture Results')
  print('Best score: ', best_gmm_score)
  print('Average score: ', avg_gmm_score)
  print('Best parameters: ', gmm_params.iloc[:3], '\n')
  
  return

# Run with fake test data
patients = 5
files = 5*patients
channels = 2
features = 10
data = np.random.rand(files, channels, features)
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

# Return list of pd dataframes that contain every combo of parameters + mean_test_score
# Return list of dict for each model with the best parameters
eval_tuning(data, labels, groups)