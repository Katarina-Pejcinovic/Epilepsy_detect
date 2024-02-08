import numpy as np
import pandas as pd
import shutil
import os
import mne
from scipy.signal import welch
#from utils import *
import matplotlib.pyplot as plt
import logging
import zipfile
import pyedflib
from pyedflib import highlevel
import io
import tempfile
from scipy import signal
import scipy as sp



class EEGDataPair:
    
    def display_channel_names(self):
        """
        Display the number of channels and their names present in the raw data.
        """
        channel_names = self.raw.ch_names
        num_channels = len(channel_names)
        return num_channels, channel_names

    def _compute_psd(self, data, fs, n_per_seg=None):
        """
        Compute the power spectral density (PSD) using Welch's method.
        """
        if n_per_seg is None:
            n_per_seg = 256

        freqs, psds = welch(data, fs=fs, nperseg=n_per_seg)
        return psds, freqs

    def _define_channels_based_on_psd(self, fmin=0.5, fmax=50, method='zscore', threshold_factor=1.5):
        """
        Compute the power spectral density (PSD) for each channel and classify
        channels as "bad" or "good" based on threshold.
        """
        data, times = self.raw[:]
        fs = self.raw.info['sfreq']
        psds, freqs = self._compute_psd(data, fs)

        idx = np.where((freqs >= fmin) & (freqs <= fmax))
        psds = psds[:, idx].squeeze()

        channel_power = np.sum(psds, axis=1)
        bad_channels = []
        keep_channels = []

        if method == 'zscore':
            z_scores = (channel_power - np.mean(channel_power)) / np.std(channel_power)
            bad_channels = [self.raw.ch_names[i] for i, z in enumerate(z_scores) if np.abs(z) > threshold_factor]
        elif method == 'iqr':
            q75, q25 = np.percentile(channel_power, [75, 25])
            iqr = q75 - q25
            bad_channels = [self.raw.ch_names[i] for i, p in enumerate(channel_power) if p < q25 - threshold_factor * iqr or p > q75 + threshold_factor * iqr]

        keep_channels = [ch for ch in self.raw.ch_names if ch not in bad_channels]

        return bad_channels, keep_channels

    def __init__(self, edf_path, event_csv_path=None, term_csv_path=None, patient_id=None, label=None):
        self.edf_file = edf_path
        self.patient_id = patient_id
        self.label = label

        self.raw = mne.io.read_raw_edf(edf_path, preload=True)
        self.bad_channels, self.keep_channels = self._define_channels_based_on_psd()

        number_channels, channel_names = self.display_channel_names()
        print(f"Total number of channels: {number_channels}")
        print("Channel names:")
        for ch_name in channel_names:
            print(ch_name)

        try:
            self.raw.drop_channels(self.bad_channels)
        except ValueError:
            pass

        self.raw.pick_channels(self.keep_channels)
        self.raw.rename_channels(lambda ch: ch.replace('EEG ', '').replace('-REF', '').replace('-LE', '').capitalize())
        self._set_montage()

    def _set_montage(self):
        missing_channels = [ch for ch in self.raw.ch_names if ch not in mne.channels.make_standard_montage('standard_1020').ch_names]
        if missing_channels:
            print(f"The following channels are not in the standard_1020 montage: {missing_channels}. These channels will be set to 'misc'.")
            self.raw.set_channel_types({ch: 'misc' for ch in missing_channels})
        self.raw.set_montage('standard_1020', on_missing='ignore')  
    
    
    
    
    def make_tcp(self):
        """
        Create a transverse central parietal (TCP) montage.
        This function will create a new MNE Raw object that's suitable for event-based predictions.
        The processed data will be stored in the class instance for further access.
        Returns the Raw object with TCP montage.
        """

        # Define the bipolar pairs
#         bipolar_pairs = [
#             ('Fp1', 'F7'), ('F7', 'T3'), ('Fp2', 'F8'), ('F8', 'T4'),
#             ('T3', 'T5'), ('T5', 'O1'), ('T4', 'T6'), ('T6', 'O2'),
#             ('F3', 'C3'), ('C3', 'P3'), ('F4', 'C4'), ('C4', 'P4'),
#             ('Fz', 'Cz'), ('Cz', 'Pz'), ('Fp1', 'F3'), ('Fp2', 'F4'),
#             ('T3', 'C3'), ('T4', 'C4'), ('P3', 'O1'), ('P4', 'O2'),
#             ('F3', 'Fz'), ('Fz', 'F4'), ('C3', 'Cz'), ('Cz', 'C4'),
#             ('P3', 'Pz'), ('Pz', 'P4')
#         ]
        
        
        
        bipolar_pairs = [
            ('Fp1', 'F7'), ('Fp2', 'F8'), ('F7', 'T3'), ('F8', 'T4'),
            ('T3', 'T5'), ('T4', 'T6'), ('T5', 'O1'), ('T6', 'O2'), 
            ('A1','T3'), ('T4', 'A2'), ('T3', 'C3'), ('C4', 'T4'),
            ('C3', 'Cz'),('Cz', 'C4'), ('Fp1', 'F3'), ('Fp2', 'F4'),
            ('F3', 'C3'), ('F4', 'C4'), ('C3', 'P3'), ('C4', 'P4'),
            ('P3', 'O1'), ('P4', 'O2'), ('Pz', 'P4'), ('Fz', 'Cz'),
            ('Cz', 'Pz'), ('Pz', 'Oz'), ('Fz', 'Fpz'), ('Fpz', 'Fp1'),
            ('Fpz', 'Fp2'), ('O1', 'Oz'), ('O2', 'Oz'), 
            ('P3', 'Pz'), ('Fz', 'F4'), ('F3', 'Fz')
            
        ]

        if not hasattr(self, 'raw'):
            logging.error("Raw data not initialized.")
            return None

        # Bipolar pairs from the raw data
        exist_bppairs = [(a, c) for a, c in bipolar_pairs if a in self.raw.ch_names and c in self.raw.ch_names]
        print("Existing bipolar pairs:", exist_bppairs)

        # Bipolar pairs exist in the raw data
        if not exist_bppairs:
            print("No matching bipolar pairs found in the data. Returning raw data without any changes.")
            return self.raw

        # Set bipolar references and drop the original channels
        self.raw_tcp = mne.set_bipolar_reference(self.raw, anode=[i[0] for i in exist_bppairs], cathode=[i[1] for i in exist_bppairs], copy=True)

        # Remove duplicate channels from the list and drop only existing channels
        channels_to_drop = list(set([chan for pair in exist_bppairs for chan in pair]))
        channels_to_drop = [chan for chan in channels_to_drop if chan in self.raw_tcp.ch_names]
        print("Channels to drop:", channels_to_drop)

        self.raw_tcp.drop_channels(channels_to_drop)
        print("Existing channels after creating TCP montage:", self.raw_tcp.ch_names)
        print(len(self.raw_tcp.ch_names))

        return self.raw_tcp



    
    def plot_raw_data(self, title="Raw EEG Data"):
            """
            Plots a segment of the raw EEG data for visual inspection.
            
            """
            number_channels = len(self.raw.ch_names)
            #print(number_channels)
            #print('arrreeeeeee')
            
            self.raw.plot(n_channels=number_channels, title=title, scalings='auto', show=True, block=True)
            plt.show()

    def preprocess_raw(self):
        """
        Function to preprocess and clean EEG data.
        Includes notch filtering, standard filtering, and ICA-based artifact removal.
        """
        standard_channels = [
            'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'A1', 'A2', 'FZ', 'CZ', 'PZ'
        ]
        potential_suffixes = ['-REF', '-LE', '']
        potential_channel_names = [ch + suffix for ch in standard_channels for suffix in potential_suffixes]

        # Process unipolar data (self.raw)
        self._preprocess_individual_data(self.raw, potential_channel_names, apply_ica=True)

        # Check if bipolar data (self.raw_tcp) exists and process it
        if hasattr(self, 'raw_tcp'):
            self._preprocess_individual_data(self.raw_tcp, potential_channel_names, apply_ica=False)

        print('Processed successfully.')

    def _preprocess_individual_data(self, eeg_data, potential_channel_names, apply_ica):
        """
        Helper function to preprocess individual EEG data (unipolar or bipolar).
        """
        if apply_ica:
            eeg_channels_in_data = [ch for ch in eeg_data.ch_names if ch.split()[0] in potential_channel_names]
            channels_to_remove = [ch for ch in eeg_data.ch_names if ch.split()[0] not in potential_channel_names]
            valid_channels_to_remove = [ch for ch in channels_to_remove if ch in eeg_data.ch_names]

            if valid_channels_to_remove:
                eeg_data.drop_channels(valid_channels_to_remove)
                print(f"Removed {', '.join(valid_channels_to_remove)} from the data.")

        # Apply filters
        l_freq, h_freq = 0.5, 40
        notch_freq = 60.0  
        eeg_data.filter(l_freq, h_freq, fir_design='firwin')
        eeg_data.notch_filter(np.arange(notch_freq, eeg_data.info['sfreq']//2, notch_freq), fir_design='firwin')

        # Apply ICA for artifact removal only to unipolar data (self.raw)
        if apply_ica:
            self._apply_ica(eeg_data)

    def _apply_ica(self, eeg_data):
        """
        Apply ICA for artifact removal on unipolar data.
        """
        picks = mne.pick_types(eeg_data.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
        n_components_for_ica = min(10, len(picks))
        print(f"Number of ICA components: {n_components_for_ica}")

        ica = mne.preprocessing.ICA(n_components_for_ica, random_state=97, max_iter=800)
        ica.fit(eeg_data, picks=picks)

        # Find and exclude bad components (eye and heart artifacts)
        eye_channels = [ch for ch in eeg_data.ch_names if 'Fp' in ch or 'F7' in ch or 'F8' in ch or 'T3' in ch or 'T4' in ch or 'T5' in ch or 'T6' in ch or 'A1' in ch or 'A2' in ch]
        eog_inds, _ = ica.find_bads_eog(eeg_data, ch_name=eye_channels)
        ekg_channels = [ch for ch in eeg_data.ch_names if 'EKG' in ch or 'Ekg' in ch]
        ecg_inds = ica.find_bads_ecg(eeg_data, ch_name=ekg_channels[0])[0] if ekg_channels else []

        ica.exclude = eog_inds + ecg_inds
        eeg_data = ica.apply(eeg_data)

            

            
    def processing_pipeline(self):
            """
            Process an EDF file and return a dictionary containing the processed data as a numpy array and the patient ID.
            """
            signals, signal_headers, header = highlevel.read_edf(self.edf_file)
            patient_id = self.patient_id  # Patient ID
            label = self.label

            # Preprocessing
            self.make_tcp()
            #self.apply_filter()
            self.preprocess_raw()

            if label == "epilepsy_edf":
                label = "Epilepsy"
            else:
                label = "No Epilepsy"

            # Convert the processed data to Numpy array
            data, times = self.raw[:]
            print(data.shape[1] / round(times[-1]))
            print(data.shape)
            downsampled_list_signals = sp.signal.resample(data, round(times[-1])*250, axis=1)
            signals_array = np.array(downsampled_list_signals)
            
            #bipolar one
            data_tcp, times_tcp = self.raw_tcp[:]
            downsampled_list_signals_tcp = sp.signal.resample(data_tcp, round(times_tcp[-1])*250, axis=1)
            signals_array_tcp = np.array(downsampled_list_signals_tcp)
            print(downsampled_list_signals_tcp.shape[1] / round(times[-1]))
            print(downsampled_list_signals_tcp.shape)
            
            print("Hereeeeeeee")
            

            return {'eeg_data_bipolar':signals_array_tcp,'eeg_data_unipolar': signals_array, 'patient_id': patient_id, 'label': label}



    
class EEGPipeline:
    def __init__(self, root_directory):
        self.root_directory = root_directory
        self.data_pairs = []
        
    def load_edf_csv_pairs(self, partition='train', n_files=None):
        partition_path = os.path.join(self.root_directory, 'edf', partition)
        print(partition_path)
        for root, dirs, files in os.walk(partition_path):
            edf_files = [f for f in files if f.endswith('.edf')]
            event_csv_files = [f for f in files if f.endswith('.csv')]
            term_csv_files = [f for f in files if f.endswith('.csv_bi')]
            
            for edf_file in edf_files:
                print(root, dirs, edf_file)
                base_name = os.path.splitext(edf_file)[0]
                corresponding_event_csv = f"{base_name}.csv"
                corresponding_term_csv = f"{base_name}.csv_bi"
                
                if corresponding_event_csv in event_csv_files and corresponding_term_csv in term_csv_files:
                    edf_path = os.path.join(root, edf_file)
                    event_csv_path = os.path.join(root, corresponding_event_csv)
                    term_csv_path = os.path.join(root, corresponding_term_csv)
                    
                    patient_id = os.path.basename(os.path.dirname(root))
                    
                    try:
                        data_pair = EEGDataPair(edf_path, event_csv_path, term_csv_path, patient_id)
                        self.data_pairs.append(data_pair)
                    except:
                        continue
                    
                    if n_files is not None and len(self.data_pairs) >= n_files:
                        return

root_directory = 'data/'
pipeline = EEGPipeline(root_directory)
pipeline.load_edf_csv_pairs(partition='dev', n_files=10)