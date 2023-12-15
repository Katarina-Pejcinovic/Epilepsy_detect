#!/usr/bin/env python
# coding: utf-8

# In[23]:


#Need this if not use pip

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


#This is an example of a single file

EDF_PATH = '/Users/andresmichel/Documents/EGG_data /v2.0.0/epilepsy_edf/aaaaaawu/s001_2003_11_04/02_tcp_le/aaaaaawu_s001_t001.edf'
EVENT_CSV_PATH = None
TERM_CSV_PATH = None
PATIENT_ID = 'aaaaaawu'
LABEL = 'epilepsy_edf'

#This is how you start the pipeline, you need to provide all arguments. In this case we only have 
#EDF_PATH,  PATIENT_ID, and LABEL. Just put None for EVENT_CSV_PATH, TERM_CSV_PATH

eeg_data_pair = EEGDataPair(EDF_PATH, EVENT_CSV_PATH, TERM_CSV_PATH, PATIENT_ID,LABEL)

#Gets before file

raw_before = eeg_data_pair.raw.copy()

#Preprocess data. DIctinary returns dictionary in form {'eeg_data': signals_array, 'patient_id': patient_id, 'label': label}


dic = eeg_data_pair.processing_pipeline()

# Get dictionary objects

raw_after = eeg_data_pair.raw
eeg_array = dic['eeg_data']
patient_id = dic['patient_id']
label = dic['label']

#testing

print(eeg_array.shape)
print(eeg_array.size)
print(eeg_array)
print("Label:" ,label)
print("Patient ID:", patient_id)


##### THIS IS EXATRA, FOR VIZUALIZATION AND GRAPH COMPARING POWER SECTRA BEFORE AND AFTER #######


# This is for vizualization (Qualitative) and graphical (Quantitative)

number_channels_before = len(raw_before.ch_names)
number_channels_after = len(eeg_data_pair.raw.ch_names)

#print(number_channels_before)
#print(raw_before.ch_names)
#print(number_channels_after) 
#print(eeg_data_pair.raw.ch_names) 

# Visualize EEG data BEFORE preprocessing

raw_before.plot(title="Before Preprocessing", n_channels=number_channels_before, scalings="auto", show=True)

# Visualize EEG data AFTER preprocessing
raw_after.plot(title="After Preprocessing", n_channels=number_channels_after, scalings="auto", show=True)

# Make comaprison 

def _compute_psd(data, fs, n_per_seg=None):
    if n_per_seg is None:
        n_per_seg = 256
    freqs, psds = welch(data, fs=fs, nperseg=n_per_seg)
    return psds, freqs



# Data and sampling frequency for before preprocessing
data_before, times_before = raw_before[:]
fs_before = raw_before.info['sfreq']

# Data and sampling frequency for after preprocessing
data_after, times_after = raw_after[:]
fs_after = raw_after.info['sfreq']

# PSDs
psds_before, freqs_before = _compute_psd(data_before, fs_before)
psds_after, freqs_after = _compute_psd(data_after, fs_after)

# Average across channels
psds_before_mean = psds_before.mean(axis=0)
psds_after_mean = psds_after.mean(axis=0)

# Plotting
plt.figure(figsize=(10, 5))
plt.semilogy(freqs_before, psds_before_mean, label='Before Preprocessing')
plt.semilogy(freqs_after, psds_after_mean, label='After Preprocessing')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB/Hz)')
plt.title('Comparing PSD Before and After Preprocessing')
plt.legend()
plt.show()






# In[ ]:


# Loop to run multiple files

EDF_PATH = '/Users/andresmichel/Documents/EGG_data /v2.0.0/epilepsy_edf/aaaaaawu/s001_2003_11_04/02_tcp_le/aaaaaawu_s001_t001.edf'


# In[22]:


# Some sort of way to get Label, Patient ID, Session date, Type, and the name of the file. 

# Base directory and output directory
base_dir = '/Users/andresmichel/Documents/EGG_data /v2.0.0/'
output_dir = os.path.join(base_dir, 'preprocessed_data')

# Create the output directory 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load subject IDs from the lists
with open(os.path.join(base_dir, 'subject_ids_epilepsy.list'), 'r') as file:
    subject_ids_epilepsy = file.read().splitlines()

with open(os.path.join(base_dir, 'subject_ids_no_epilepsy.list'), 'r') as file:
    subject_ids_no_epilepsy = file.read().splitlines()

# Subdirectories and save the preprocessed file
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
    # Replace EEGDataPair with your actual preprocessing class
    eeg_data_pair = EEGDataPair(edf_path, None, None, patient_id, label)
    dic = eeg_data_pair.processing_pipeline()

    # Get the processed data
    eeg_array = dic['eeg_data']

    # Save the preprocessed data
    base_filename = os.path.basename(edf_path)
    save_preprocessed_data(eeg_array, patient_id, label, base_filename)

# Process each patient's files
for label, subject_ids in [('epilepsy_edf', subject_ids_epilepsy), ('no_epilepsy_edf', subject_ids_no_epilepsy)]:
    for patient_id in subject_ids:
        patient_path = os.path.join(base_dir, label, patient_id)
        if os.path.exists(patient_path):
            for edf_file in glob(os.path.join(patient_path, '**/*.edf'), recursive=True):
                process_file(edf_file, patient_id, label)


# In[19]:





# In[ ]:




