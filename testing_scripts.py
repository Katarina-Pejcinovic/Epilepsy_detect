
### Call individual functions for the component they are testing (currently does not need to do anything else)

# Should read in files that would be used for testing (should represent idealized output that the function would receive)
# Results should be saved in a variable so they can be evaluated

### Testing Functions

# !pip install mne
# !pip install umap-learn
# !pip install pyedflib
# !pip install xgboost
# !pip install PyWavelets
# !pip install neurokit2
# !pip install torch

import mne
import pandas as pd
import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.signal import periodogram
import pywt
from collections import Counter
from tqdm.notebook import tqdm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
import umap.umap_ as umap

from sklearn.metrics import accuracy_score
from glob import glob
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import shutil
from scipy.signal import welch
#from utils import *
import logging
import zipfile
import pyedflib
from pyedflib import highlevel
import io
import tempfile

# Import processing functions

# Now, you can import your module
from preprocessing.dataset import *

from classical_ML.classical_ml_models import *
from classical_ML.get_features import *
from classical_ML.train_test_tune import *
from deep_learning.cnn import *
from deep_learning.rnn import *
from validation.validate import *

# In the future this will be a full folder, but for now it will be one edf file

# [UPDATE PATH]
data_file_path  = '/content/gdrive/Shareddrives/BE_223A_Seizure_Project/Code/aaaaaajy_s001_t000.edf'
labels = [0]

# # Import processing functions


### Pre-Processing

def eval_pre_processing(edf_path):

  eeg_data_pair = EEGDataPair(data_file_path)

  # Run the preprocessing pipeline
  edf_file = eeg_data_pair.processing_pipeline()

  # Store the original raw for visualization
  raw_before = eeg_data_pair.raw.copy()

  # Channels that are present after preprocessing
  common_chs = [ch for ch in raw_before.ch_names if ch in eeg_data_pair.raw.ch_names]

  # Same chanels for both plots
  raw_before.pick_channels(common_chs)
  eeg_data_pair.raw.pick_channels(common_chs)

  # Visualize EEG data BEFORE preprocessing
  raw_before.plot(title="Before Preprocessing", n_channels=20, scalings="auto", show=True)

  # Visualize EEG data AFTER preprocessing
  eeg_data_pair.raw.plot(title="After Preprocessing", n_channels=20, scalings="auto", show=True)

  print(edf_file.shape)

  print(edf_file.size)

  return edf_file

### Feature Extraction

# Test the get_features function by asserting it outputs a matrix with the right dimensions
def eval_get_features(list_signals, wavelet_name):
  features = get_features(list_signals, wavelet_name)
  print('The features matrix has dimensions ' + str(features.shape))
  if features.shape == (32, 177):
    print('Those are the correct dimensions!')
  if features.shape != (32, 177):
    print('Those are the wrong dimensions! The correct dimensions are (32, 177)')
  return features

### Classical ML Parameter Tuning

def eval_tuning(data, labels, groups):
  [params, best_params] = train_test_tune(data, labels, groups)

  # Check the evaluation scores of each test parameter combination
  svc_params = params[0].sort_values(by=['mean_test_score'], ascending=False)
  rf_params = params[1].sort_values(by=['mean_test_score'], ascending=False)
  kmeans_params = params[2].sort_values(by=['mean_test_score'], ascending=False)
  xg_params = params[3].sort_values(by=['mean_test_score'], ascending=False)

  svc_tune_scores = svc_params.mean_test_score
  rf_tune_scores = rf_params.mean_test_score
  kmeans_tune_scores = kmeans_params.mean_test_score
  xg_tune_scores = xg_params.mean_test_score

  best_svc_score = max(svc_tune_scores)
  avg_svc_score = np.mean(svc_tune_scores)

  best_rf_score = max(rf_tune_scores)
  avg_rf_score = np.mean(rf_tune_scores)

  best_kmeans_score = max(kmeans_tune_scores)
  avg_kmeans_score = np.mean(kmeans_tune_scores)

  best_xg_score = max(xg_tune_scores)
  avg_xg_score = np.mean(xg_tune_scores)

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

### Deep Learning

def eval_rnn(data, labels, val_data, test_data, learning_rate, gradient_threshold=1, batch_size=32, epochs=2):
  predictions = rnn_model(data,labels, val_data, test_data, learning_rate, gradient_threshold, batch_size, epochs)
  
  return predictions


def eval_cnn(edf_file, labels):
  weights = run_cnn(edf_file, labels)
  return weights

### Validation

def eval_validation(train_data, train_labels, val_data, val_labels, parameters):

  results = validate(train_data, train_labels, val_data, val_labels, parameters)

  # Check that validation methods are working and returning results in proper format

  return results


