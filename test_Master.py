import mne
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.signal import periodogram
import neurokit2 as nk
import pywt
from collections import Counter
import scipy.stats as stats

#run get_features()
from feature_selection.get_features import * 

#get preprocessed data 
preprocessed_train_ep = np.load('data/training/epilepsy/preprocessed_aaaaaanr_epilepsy_edf_aaaaaanr_s007_t000.edf.npy')
preprocessed_train_noep = np.load('data/training/no_epilepsy/preprocessed_aaaaaebo_no_epilepsy_edf_aaaaaebo_s001_t000.edf.npy')
preprocessed_test_ep = np.load('data/testing/epilepsy/preprocessed_aaaaalug_epilepsy_edf_aaaaalug_s001_t000.edf.npy')
preprocessed_test_noep = np.load('data/testing/no_epilepsy/preprocessed_aaaaappo_no_epilepsy_edf_aaaaappo_s001_t001.edf.npy')

features_one = get_features(preprocessed_train_ep)
features_two = get_features(preprocessed_train_noep)
features_three = get_features(preprocessed_test_ep)
features_four = get_features(preprocessed_test_noep)
print(features_one.shape)
print(features_two.shape)
print(features_three.shape)
print(features_four.shape)

#this is what is sent to the cnn, rnn, and classical 
training = np.append(features_one, features_two, axis =0)
testing = np.append(features_three, features_four, axis =0 )
print(training.shape, testing.shape)

labels_train = np.append(np.ones(16), np.zeros(16), axis =0)
labels_test = labels_train

patient_ids_train = np.append(np.ones(16), np.ones(16)*2)
patient_ids_test = np.append(np.ones(16)*3, np.ones(16)*4)
print(patient_ids_train)




