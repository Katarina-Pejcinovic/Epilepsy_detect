
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
from sklearn.mixture import GaussianMixture
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

### Deep Learning

def eval_rnn(data, labels, val_data, test_data, learning_rate, gradient_threshold=1, batch_size=32, epochs=2):
  predictions = rnn_model(data,labels, val_data, test_data, learning_rate, gradient_threshold, batch_size, epochs)
  
  return predictions

### Validation

def eval_validation(train_data, train_labels, val_data, val_labels, parameters):

  results = validate(train_data, train_labels, val_data, val_labels, parameters)

  # Check that validation methods are working and returning results in proper format

  return results


