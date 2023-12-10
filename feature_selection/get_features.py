# -*- coding: utf-8 -*-
"""get_features for download.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PThw4tUuW4tYDODZq2bpyh4kEVdluAQ2
"""

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

nSamples = 150000 #number of samples per segment
fs = 250 #sampling rate
wavelet_name = 'db1' # name of discrete mother wavelet used in Discrete Wavelet Transform

# Iterate over a list of 3D arrays, iterate over the 10-min segments in each 3D array
# iterate over the channels in each segments and for each channel, extract 237 features:
# - Extract 16 statistical features from the raw EEG signal (nonlinear energy, line length, entropy, etc.)
# - Extract the power in each of the 5 major frequency bands from the raw EEG signal
# - Extract 216 statistical features from the DWT coefficients of the EEG signal by doing the following:
#     - Apply the DWT which returns 18 lists of coefficients.
#     - For each of these lists extract 12 statistical features.
#     - The features calculated from all of the lists of coefficients belonging to one signal
#     - are concatenated together, since they belong to the same signal.

def get_features(list_signals, waveletname):
    list_features = []
    
    for signals in list_signals:  # Iterate through the list of 3D arrays
        features_per_array = []
        
        for signal in signals:  # Iterate through each segment in the 3D array
            features_per_channel = []
            
            for channel in range(signal.shape[0]):  # Iterate through each channel in the segment
                time_features = get_time_features(signal[channel])
                freq_features = get_freq_features(signal[channel])
                dwt_coeff = get_dwt_coeff(signal[channel], waveletname)
                
                features = []
                for coeff in dwt_coeff:
                    features += get_time_freq_features(coeff)
                    
                for ff in freq_features:
                    features.append(ff)
                    
                for tf in time_features:
                    features.append(tf)
                    
                features_per_channel.append(features)
            
            features_per_array.append(features_per_channel)
        
        list_features.append(features_per_array)
    
    for i in range(len(list_features)):
      list_features[i] = np.array(list_features[i])
    
    return list_features

def get_features_for_test(list_signals, waveletname):
    list_features = []
    # iterate over signals in file
    for signal in list_signals:
        time_features = get_time_features(signal)
        freq_features = get_freq_features(signal)
        dwt_coeff = get_dwt_coeff(signal, waveletname)
        features = []
        for coeff in dwt_coeff:
          features += get_time_freq_features(coeff)
        for ff in freq_features:
          features.append(ff)
        for tf in time_features:
          features.append(tf)
        list_features.append(features)
    all_features = np.array(list_features)
    return all_features

# Return list of statistical features from data: time-domain features
def get_time_features(data,fs=fs,nSamples=nSamples):
    result = get_time_freq_features(data)
    result.append(nonlinear_energy(data))
    result.append(line_length(data))
    f, psd = periodogram(data, fs)
    result.append(IWMF(f, psd))
    result.append(IWBW(f, psd))
    return result

# Return list of power band values from data: frequency-domain features
def get_freq_features(data, fs=fs):
  power_bands = nk.eeg_power(data, frequency_band=['Gamma', 'Beta', 'Alpha', 'Theta', 'Delta'], sampling_rate=fs)
  band_power = power_bands.values.tolist()[0][1:]
  return band_power

# Return list of Discrete Wavelet Transform coefficients from data
def get_dwt_coeff(data, waveletname):
  return pywt.wavedec(data, waveletname)

# Return list of simple statistical features from data
def get_time_freq_features(data):
    entropy = calculate_entropy(data)
    crossings = calculate_crossings(data)
    statistics = calculate_statistics(data)
    return [entropy] + crossings + statistics

# Return the nonlinear energy from data
def nonlinear_energy(X):
    return sum(X[1:-1]**2 - X[2:]*X[:-2])

# Return the line length from data
def line_length(X):
    return sum(np.abs(X[:-1]-X[1:]))

# Return the intensity weighted mean frequency (IWMF) from data
def IWMF(F,PSD):
    nPSD = PSD /sum(PSD)
    iwmf = np.dot(PSD,F)
    return iwmf

# Return the intensity weighted bandwidth (IWBW) from data
def IWBW(F, PSD):
    nPSD = PSD /sum(PSD)
    iwmf = IWMF(F, PSD)
    iwbw = np.sqrt(np.mean(((nPSD*(F-iwmf))**2)))
    return iwbw

# Return the entropy from list of values
def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=sp.stats.entropy(probabilities)
    return entropy

# Return simple statistical features from list of values
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

# Return the number of zero crossings and mean crossings from list of values
def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    number_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    number_mean_crossings = len(mean_crossing_indices)
    return [number_zero_crossings, number_mean_crossings]