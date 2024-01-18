# Create dict for patient IDs
# 100 ep train (100-199), 100 noep (200-299)
# Assign numerical value to each patient ID for 3D numpy array
# Save to file

import pandas as pd
import numpy as np
import os
import pickle
import csv

def patient_id_dict(patient_list_folder):

    ep_path = patient_list_folder + 'subject_ids_epilepsy.txt'
    noep_path = patient_list_folder + 'subject_ids_no_epilepsy.txt'

    # Load in the txt files instead as a list of ID names
    ep_df = pd.read_table(ep_path, delimiter="\t")
    ep_ID = ep_df["IDs"].values.tolist()
    noep_df = pd.read_table(noep_path, delimiter="\t")
    noep_ID = noep_df["IDs"].values.tolist()

    # Create dictionary for patient IDs -> int
    patient_dict = {}

    patient_num = 100

    for patient in ep_ID:
        patient_dict[patient] = patient_num
        patient_num = patient_num + 1
    
    for patient in noep_ID:
        patient_dict[patient] = patient_num
        patient_num = patient_num + 1

    # Save dictionary as to load later
    with open(patient_list_folder + 'patient_ID_dictionary.pkl', 'wb') as f:
        pickle.dump(patient_dict, f)
    
    # Save dictionary as csv 
    with open(patient_list_folder + 'patient_ID_dictionary.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for row in patient_dict.items():
            writer.writerow(row)



def load_patient_dict(patient_list_folder):

    with open(patient_list_folder + 'patient_ID_dictionary.pkl', 'rb') as f:
        dict = pickle.load(f)

    return dict


# patient_list_folder = 'data/'
# patient_id_dict(patient_list_folder)
# dict = load_patient_dict(patient_list_folder)
# print(dict)