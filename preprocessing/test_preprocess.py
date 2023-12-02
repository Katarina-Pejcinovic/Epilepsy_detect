### Pre-Processing
from preprocessing.dataset import *
def eval_pre_processing(edf_path):

  eeg_data_pair = EEGDataPair(edf_path)

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
