#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Need this if not use pip
# This is an example for a single file without formating for the only 26 most common channels

import numpy as np
import pandas as pd
from glob import glob
import os
from scipy.signal import welch
import mne
import matplotlib.pyplot as plt
from preprocessing.dataset import *  
import warnings
#import pandas as pd
import shutil
#import os
import mne
#from scipy.signal import welch
#from utils import *
#import matplotlib.pyplot as plt
import logging
import zipfile
import pyedflib
from pyedflib import highlevel
import io
import tempfile
from collections import Counter
import scipy as sp




#This is an example of a single file

EDF_PATH = '/Users/andresmichel/Documents/EGG_data /v2.0.0/epilepsy_edf/aaaaaawu/s001_2003_11_04/02_tcp_le/aaaaaawu_s001_t001.edf'
EVENT_CSV_PATH = None
TERM_CSV_PATH = None
PATIENT_ID = 'aaaaaawu'
LABEL = 'epilepsy_edf'

#This is how you start the pipeline, you need to provide all arguments. In this case we only have 
#EDF_PATH,  PATIENT_ID, and LABEL. Just put None for EVENT_CSV_PATH, TERM_CSV_PATH

eeg_data_pair = EEGDataPair(EDF_PATH, EVENT_CSV_PATH, TERM_CSV_PATH, PATIENT_ID,LABEL)

#Gets before file as raw data

raw_before = eeg_data_pair.raw.copy()

#Preprocess data. Dcctinary returns dictionary in form 
#{}'eeg_data_bipolar':signals_array_tcp,'eeg_data_unipolar': signals_array, 'patient_id': patient_id, 'label': label}

dic = eeg_data_pair.processing_pipeline()

# Get dictionary objects

#want to compare to bipolar
raw_after = eeg_data_pair.raw_tcp

#here is unipolar nunpy array
eeg_array_unipolar = dic['eeg_data_unipolar']

#heere is the bipolar nunpy array
eeg_array_bipolar =dic['eeg_data_bipolar']

#patient ID
patient_id = dic['patient_id']

#label
label = dic['label']





print(eeg_array_unipolar.shape)
print(eeg_array_unipolar.size)
print(eeg_array_unipolar)

print(eeg_array_bipolar.shape)
print(eeg_array_bipolar.size)
print(eeg_array_bipolar)
print("Label:" ,label)
print("Patient ID:", patient_id)


##### THIS IS EXATRA, FOR VIZUALIZATION AND GRAPH COMPARING POWER SECTRA BEFORE AND AFTER #######


# This is for vizualization (Qualitative) and graphical (Quantitative)

number_channels_before = len(raw_before.ch_names)
number_channels_after_bipolar = len(eeg_data_pair.raw_tcp.ch_names)
number_channels_after_unipolar = len(eeg_data_pair.raw.ch_names)

print(number_channels_before)
#print(raw_before.ch_names)
print(number_channels_after_unipolar) 
print(number_channels_after_bipolar) 
#print(eeg_data_pair.raw.ch_names) 

# Visualize EEG data BEFORE preprocessing

raw_before.plot(title="Before Preprocessing", n_channels=number_channels_before, scalings="auto", show=True)

# Visualize EEG data AFTER preprocessing bipolar
raw_after.plot(title="After Preprocessing", n_channels=number_channels_after_bipolar, scalings="auto", show=True)

# Visualize EEG data AFTER preprocessing unipolar
eeg_data_pair.raw.plot(title="After Preprocessing", n_channels=number_channels_after_unipolar, scalings="auto", show=True)


# Make comaprison 

# def _compute_psd(data, fs, n_per_seg=None):
#     if n_per_seg is None:
#         n_per_seg = 256
#     freqs, psds = welch(data, fs=fs, nperseg=n_per_seg)
#     return psds, freqs



# Data and sampling frequency for before preprocessing
# data_before, times_before = raw_before[:]
# fs_before = raw_before.info['sfreq']

# # Data and sampling frequency for after preprocessing
# data_after, times_after = raw_after[:]
# fs_after = raw_after.info['sfreq']

# # PSDs
# psds_before, freqs_before = _compute_psd(data_before, fs_before)
# psds_after, freqs_after = _compute_psd(data_after, fs_after)

# # Average across channels
# psds_before_mean = psds_before.mean(axis=0)
# psds_after_mean = psds_after.mean(axis=0)

# Plotting
# plt.figure(figsize=(10, 5))
# plt.semilogy(freqs_before, psds_before_mean, label='Before Preprocessing')
# plt.semilogy(freqs_after, psds_after_mean, label='After Preprocessing')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Power Spectral Density (dB/Hz)')
# plt.title('Comparing PSD Before and After Preprocessing')
# plt.legend()
# plt.show()

print(len(raw_before.ch_names))
print("Channel names in raw_before:", raw_before.ch_names)

print("Channel names in raw_tcp after preprocessing:", eeg_data_pair.raw_tcp.ch_names)
eeg_data_pair.raw_tcp.plot(title="After Preprocessing", n_channels=len(eeg_data_pair.raw_tcp.ch_names), scalings="auto", show=True)
print(len(eeg_data_pair.raw_tcp.ch_names))




# In[2]:


import numpy as np
import pandas as pd
import os
from scipy.signal import welch
import mne
import matplotlib.pyplot as plt
import logging
import zipfile
import pyedflib
from pyedflib import highlevel
import io
import tempfile
from scipy import signal
import scipy as sp
from collections import Counter
#This is for a single file to and return 26 most common channels we talked about

standard_bipolar_channels = [
    'C3-P3', 'C3-Cz', 'F3-C3', 'T3-C3', 'T3-T5', 'P3-O1', 'Cz-C4', 'F4-C4', 'F7-T3', 'T6-O2',
    'T5-O1', 'Fz-Cz', 'P4-O2', 'T4-T6', 'C4-P4', 'Fz-F4', 'F3-Fz', 'C4-T4', 'Cz-Pz', 'P3-Pz',
    'F8-T4', 'Pz-P4', 'Fp2-F4', 'Fp1-F7', 'Fp2-F8', 'Fp1-F3'

]

def standardize_bipolar_data(bipolar_data, bipolar_channels):
    standardized_data = np.full((len(standard_bipolar_channels), bipolar_data.shape[1]), np.nan)
    for i, channel in enumerate(standard_bipolar_channels):
        if channel in bipolar_channels:
            channel_index = bipolar_channels.index(channel)
            standardized_data[i, :] = bipolar_data[channel_index, :]
    return standardized_data

EDF_PATH = '/Users/andresmichel/Documents/EGG_data /v2.0.0/epilepsy_edf/aaaaaawu/s001_2003_11_04/02_tcp_le/aaaaaawu_s001_t001.edf'
EVENT_CSV_PATH = None
TERM_CSV_PATH = None
PATIENT_ID = 'aaaaaawu'
LABEL = 'epilepsy_edf'

eeg_data_pair = EEGDataPair(EDF_PATH, None, None, PATIENT_ID, LABEL)

# Preprocess data
dic = eeg_data_pair.processing_pipeline()

# Standardize the bipolar data
eeg_array_bipolar = standardize_bipolar_data(dic['eeg_data_bipolar'], eeg_data_pair.raw_tcp.ch_names)

# Visualization and Analysis
# Plot raw data before preprocessing
raw_before = eeg_data_pair.raw.copy()
raw_before.plot(title="Before Preprocessing", n_channels=len(raw_before.ch_names), scalings="auto", show=True)

# Plot bipolar data after preprocessing
raw_after = eeg_data_pair.raw_tcp
raw_after.plot(title="After Preprocessing (Bipolar)", n_channels=len(raw_after.ch_names), scalings="auto", show=True)

print("Shape of standardized bipolar data:", eeg_array_bipolar.shape)



# In[ ]:





# In[ ]:


# In[22]:


# Some sort of way to get Label, Patient ID, Session date, Type, and the name of the file. 
#This preprocess all of the files in the data set without returning te 26 most common channels

base_dir = '/Users/andresmichel/Documents/EGG_data /v2.0.0/'
output_dir = os.path.join(base_dir, 'preprocessed_data')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(base_dir, 'subject_ids_epilepsy.list'), 'r') as file:
    subject_ids_epilepsy = file.read().splitlines()

with open(os.path.join(base_dir, 'subject_ids_no_epilepsy.list'), 'r') as file:
    subject_ids_no_epilepsy = file.read().splitlines()

def save_preprocessed_data(eeg_array, patient_id, label, base_filename):
    # Create subdirectory for the label (epilepsy or no epilepsy)
    label_dir = os.path.join(output_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Create subdirectory for the patient ID
    patient_dir = os.path.join(label_dir, patient_id)
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir)

    # Save the preprocessed data
    preprocessed_filename = f"preprocessed_{patient_id}_{label}_{base_filename}.npy"
    preprocessed_filepath = os.path.join(patient_dir, preprocessed_filename)
    np.save(preprocessed_filepath, eeg_array)
    print(f"Saved preprocessed data to {preprocessed_filepath}")

# Process each file
def process_file(edf_path, patient_id, label):
    eeg_data_pair = EEGDataPair(edf_path, None, None, patient_id, label)
    dic = eeg_data_pair.processing_pipeline()

    # Get the processed data
    eeg_array = dic['eeg_data']

    base_filename = os.path.basename(edf_path)
    save_preprocessed_data(eeg_array, patient_id, label, base_filename)

for label, subject_ids in [('epilepsy_edf', subject_ids_epilepsy), ('no_epilepsy_edf', subject_ids_no_epilepsy)]:
    for patient_id in subject_ids:
        patient_path = os.path.join(base_dir, label, patient_id)
        if os.path.exists(patient_path):
            for edf_file in glob(os.path.join(patient_path, '**/*.edf'), recursive=True):
                process_file(edf_file, patient_id, label)


# In[ ]:


# This process everthing for all data in dataset and returns numpy aray each with 26 most common channels.

standard_bipolar_channels = [
    'C3-P3', 'C3-Cz', 'F3-C3', 'T3-C3', 'T3-T5', 'P3-O1', 'Cz-C4', 'F4-C4', 'F7-T3', 'T6-O2',
    'T5-O1', 'Fz-Cz', 'P4-O2', 'T4-T6', 'C4-P4', 'Fz-F4', 'F3-Fz', 'C4-T4', 'Cz-Pz', 'P3-Pz',
    'F8-T4', 'Pz-P4', 'Fp2-F4', 'Fp1-F7', 'Fp2-F8', 'Fp1-F3'
]

base_dir = '/Users/andresmichel/Documents/EGG_data /v2.0.0/'
output_dir = os.path.join(base_dir, 'preprocessed_data_bipolar_OG')


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(base_dir, 'subject_ids_epilepsy.list'), 'r') as file:
    subject_ids_epilepsy = file.read().splitlines()

with open(os.path.join(base_dir, 'subject_ids_no_epilepsy.list'), 'r') as file:
    subject_ids_no_epilepsy = file.read().splitlines()

# Counter for channel frequencies
channel_frequencies = Counter()

def save_preprocessed_data(eeg_array, patient_id, label, base_filename):
    label_dir = os.path.join(output_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    patient_dir = os.path.join(label_dir, patient_id)
    if not os.path.exists(patient_dir):
        os.makedirs(patient_dir)

    preprocessed_filename = f"preprocessed_{patient_id}_{label}_{base_filename}.npy"
    preprocessed_filepath = os.path.join(patient_dir, preprocessed_filename)
    np.save(preprocessed_filepath, eeg_array)
    print(f"Saved preprocessed data to {preprocessed_filepath}")

def process_file(edf_path, patient_id, label):
    global channel_frequencies

    eeg_data_pair = EEGDataPair(edf_path, None, None, patient_id, label)
    dic = eeg_data_pair.processing_pipeline()

    eeg_array_bipolar = dic['eeg_data_bipolar']
    bipolar_channel_names = eeg_data_pair.raw_tcp.ch_names
    standardized_bipolar_data = standardize_bipolar_data(eeg_array_bipolar, bipolar_channel_names)

    channel_frequencies.update(bipolar_channel_names)

    base_filename = os.path.basename(edf_path)
    save_preprocessed_data(standardized_bipolar_data, patient_id, label, base_filename)

def standardize_bipolar_data(bipolar_data, bipolar_channels):
    standardized_data = np.full((len(standard_bipolar_channels), bipolar_data.shape[1]), np.nan)
    for i, channel in enumerate(standard_bipolar_channels):
        if channel in bipolar_channels:
            channel_index = bipolar_channels.index(channel)
            standardized_data[i, :] = bipolar_data[channel_index, :]
    return standardized_data

# Process each patient's files
for label, subject_ids in [('epilepsy_edf', subject_ids_epilepsy), ('no_epilepsy_edf', subject_ids_no_epilepsy)]:
    for patient_id in subject_ids:
        patient_path = os.path.join(base_dir, label, patient_id)
        if os.path.exists(patient_path):
            for edf_file in glob(os.path.join(patient_path, '**/*.edf'), recursive=True):
                process_file(edf_file, patient_id, label)

# Plot for after
channels, frequencies = zip(*channel_frequencies.most_common())
plt.figure(figsize=(15, 5))
plt.bar(channels, frequencies)
plt.xticks(rotation=90)
plt.title('Frequency of Channels across EEG Files')
plt.xlabel('Channel')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# In[ ]:




