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
from dataset import *  
from glob import glob

    # Counter for channel frequencies
channel_frequencies = Counter()


def save_preprocessed_data(eeg_array, patient_id, label, base_filename,output_dir):
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

def process_file(edf_path, patient_id, label,output_dir):
    global channel_frequencies

    eeg_data_pair = EEGDataPair(edf_path, None, None, patient_id, label)
    dic = eeg_data_pair.processing_pipeline()

    eeg_array_bipolar = dic['eeg_data_bipolar']
    bipolar_channel_names = eeg_data_pair.raw_tcp.ch_names
    standardized_bipolar_data = standardize_bipolar_data(eeg_array_bipolar, bipolar_channel_names,output_dir)

    channel_frequencies.update(bipolar_channel_names)

    base_filename = os.path.basename(edf_path)
    save_preprocessed_data(standardized_bipolar_data, patient_id, label, base_filename,output_dir)

def standardize_bipolar_data(bipolar_data, bipolar_channels,output_dir):
    
    standard_bipolar_channels = [
    'C3-P3', 'C3-Cz', 'F3-C3', 'T3-C3', 'T3-T5', 'P3-O1', 'Cz-C4', 'F4-C4', 'F7-T3', 'T6-O2',
    'T5-O1', 'Fz-Cz', 'P4-O2', 'T4-T6', 'C4-P4', 'Fz-F4', 'F3-Fz', 'C4-T4', 'Cz-Pz', 'P3-Pz',
    'F8-T4', 'Pz-P4', 'Fp2-F4', 'Fp1-F7', 'Fp2-F8', 'Fp1-F3'
]
    
    standardized_data = np.full((len(standard_bipolar_channels), bipolar_data.shape[1]), np.nan)
    for i, channel in enumerate(standard_bipolar_channels):
        if channel in bipolar_channels:
            channel_index = bipolar_channels.index(channel)
            standardized_data[i, :] = bipolar_data[channel_index, :]
    return standardized_data

def master_prepro(base_dir):
        # This process everthing for all data in dataset and returns numpy aray each with 26 most common channels.

    standard_bipolar_channels = [
        'C3-P3', 'C3-Cz', 'F3-C3', 'T3-C3', 'T3-T5', 'P3-O1', 'Cz-C4', 'F4-C4', 'F7-T3', 'T6-O2',
        'T5-O1', 'Fz-Cz', 'P4-O2', 'T4-T6', 'C4-P4', 'Fz-F4', 'F3-Fz', 'C4-T4', 'Cz-Pz', 'P3-Pz',
        'F8-T4', 'Pz-P4', 'Fp2-F4', 'Fp1-F7', 'Fp2-F8', 'Fp1-F3'
    ]

    #base_dir = 'data/'   # Change this depending on where data is
    output_dir = os.path.join(base_dir, 'preprocessed_data_bipolar_OG') # Change this depending ojn where you want it

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(base_dir, 'subject_ids_epilepsy.txt'), 'r') as file:
        subject_ids_epilepsy = file.read().splitlines()

    with open(os.path.join(base_dir, 'subject_ids_no_epilepsy.txt'), 'r') as file:
        subject_ids_no_epilepsy = file.read().splitlines()

    #counter

    # Process each patient's files
    for label, subject_ids in [('epilepsy_edf', subject_ids_epilepsy), ('no_epilepsy_edf', subject_ids_no_epilepsy)]:
        for patient_id in subject_ids:
            patient_path = os.path.join(base_dir, label, patient_id)
            if os.path.exists(patient_path):
                for edf_file in glob(os.path.join(patient_path, '**/*.edf'), recursive=True):
                    process_file(edf_file, patient_id, label,output_dir)

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
    plt.close()
