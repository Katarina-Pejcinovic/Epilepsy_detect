
import mne
import pandas as pd
import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.signal import periodogram
import neurokit2 as nk
import pywt
from collections import Counter
from tqdm.notebook import tqdm
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
import umap.umap_ as umap
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
import sys

nSamples = 8000 # number of samples per segment
fs = 400 # sampling rate in Hz
wavelet_name = 'db1' # name of discrete mother wavelet used in Discrete Wavelet Transform
# In the future this will be a full folder, but for now it will be one edf file

#practive training data 
data_file_path  = 'data/aaaaaebo_s001_t000.edf'
labels = [0]

#import functions from other files 
from preprocessing import *
from classical_ML.classical_ml_models import *
from classical_ML.get_features import *
from classical_ML.train_test_tune import *
from deep_learning.cnn import *
from deep_learning.rnn import *
from validate import *

'''Preprocessing'''
data_file_path = "aaaaaanr_s001_t001.edf"
eeg_data_pair = EEGDataPair(data_file_path)

# Store the original raw for visualization
raw_before = eeg_data_pair.raw.copy()

# Run the preprocessing pipeline
edf_file = eeg_data_pair.processing_pipeline()

# Channels that are present after preprocessing
common_chs = [ch for ch in raw_before.ch_names if ch in eeg_data_pair.raw.ch_names]

# Same chanels for both plots
raw_before.pick_channels(common_chs)
eeg_data_pair.raw.pick_channels(common_chs)

# Visualize EEG data BEFORE preprocessing
raw_before.plot(title="Before Preprocessing", n_channels=20, scalings="auto", show=True)

# Visualize EEG data AFTER preprocessing
eeg_data_pair.raw.plot(title="After Preprocessing", n_channels=20, scalings="auto", show=True)

# print(edf_file.shape)

# print(edf_file.size)

# print(edf_file)

# from get_features import *
features = get_features(edf_file, wavelet_name)

# from train_test_tune import *

# Inputs: numpy array of data (size: [# files, 32 channels, 177 features])
#         numpy array of labels (size: [# files, 1]) -- Epilepsy or No Epilepsy
#         numpy array of patient ID per file (size: [# files, 1])

# Adding extra fake data to extracted features to simulate having more than one file
data_rand = np.random.rand([24, 32, 177])
data = np.concatenate((features, data_rand), axis=0)
labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
groups = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

[params, best_params] = train_test_tune(data, labels, groups)

svc_params = best_params[0]
rf_params = best_params[1]
kmeans_params = best_params[2]
xg_params = best_params[3]

'''CNN'''
#create training datset
edf_data = mne.io.read_raw_edf('aaaaaebo_s001_t000.edf', preload=True)
multichannel_data_train, time = edf_data[:, :]
labels = np.array([1])
model = run_CNN(multichannel_data_train, labels)

'''RNN'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
import pandas as pd
from tensorflow.keras.utils import to_categorical

data = [0,1,2]
labels = [1,0,0]
val_data = [0,1,0]
parameters = [1,2,3]

rnn_pred = rnn_model(data,labels,val_data,parameters)

# from validate import *

# data:(# of samples, 32 channels, 177 features)
# labels: (# of samples)

# 3/4 of total data that has been used for training of the model
train_data = np.zeros([15, 32, 177])
train_labels = np.zeros(15)

# 1/4 of total data that was put aside at the start
validation_data = np.zeros([5, 32, 177])
validation_labels = np.zeros(5)

# model parameters (SVM, RF, XG Boost, Kmeans, UMAP)
parameters = best_params

validate(train_data, train_labels, validation_data, validation_labels, parameters)