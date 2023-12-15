#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from glob import glob
import os
import mne
import matplotlib.pyplot as plt
from preprocessing.dataset import *  
import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

PATIENT_ID = "aaaaaanr"  # Example patient ID
SESSION = "s007_2013_02_05"  # Example session
TYPE = "01_tcp_ar"  # Example type
LABEL = "epilepsy_edf"


#This depends on the file you are using 
EDF_PATH = '/Users/andresmichel/Documents/EGG_data /v2.0.0/epilepsy_edf/aaaaaanr/s007_2013_02_05/01_tcp_ar/aaaaaanr_s007_t000.edf'
EVENT_CSV_PATH = None
TERM_CSV_PATH = None


eeg_data_pair = EEGDataPair(EDF_PATH, EVENT_CSV_PATH, TERM_CSV_PATH, PATIENT_ID,LABEL)

dictionary = eeg_data_pair.processing_pipeline()

array = dictionary['eeg_data']
patient_id = dictionary['patient_id']
label = dictionary['label']




