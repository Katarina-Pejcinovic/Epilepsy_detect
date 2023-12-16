import numpy as np
import pandas as pd
import os

# Get preprocessed data 
def load_data(data_file_path):
 
    # Beta subset: ep_train = [aaaaaanr], ep_test = [aaaaalug], noep_train = [aaaaaebo], noep_test = [aaaaappo]
    # ep_train_ID = ['aaaaaanr']
    # ep_test_ID = ['aaaaalug']
    # noep_train_ID = ['aaaaaebo']
    # noep_test_ID = ['aaaaappo']

    # Load in the txt files instead as a list of ID names
    ep_train_df = pd.read_table('data/subject_ids_epilepsy_training.txt', delimiter="\t")
    ep_train_ID = ep_train_df["IDs"].values.tolist()
    noep_train_df = pd.read_table('data/subject_ids_no_epilepsy_training.txt', delimiter="\t")
    noep_train_ID = noep_train_df["IDs"].values.tolist()

    ep_test_df = pd.read_table('data/subject_ids_epilepsy_testing.txt', delimiter="\t")
    ep_test_ID = ep_test_df["IDs"].values.tolist()
    noep_test_df = pd.read_table('data/subject_ids_no_epilepsy_testing.txt', delimiter="\t")
    noep_test_ID = noep_test_df["IDs"].values.tolist()

    # Get all patient names
    preprocessed_path = data_file_path + 'preprocessed_data/'
    state = ['epilepsy_edf/', 'no_epilepsy_edf/']

    ep_patients_train = [filename for filename in os.listdir(preprocessed_path + state[0]) 
                        if filename in ep_train_ID]

    noep_patients_train = [filename for filename in os.listdir(preprocessed_path + state[1])
                            if filename in noep_train_ID]
    ep_patients_test = [filename for filename in os.listdir(preprocessed_path + state[0])
                        if filename in ep_test_ID]
    noep_patients_test = [filename for filename in os.listdir(preprocessed_path + state[1])
                        if filename in noep_test_ID]
    

    preprocessed_train_ep = []
    preprocessed_train_noep = []
    preprocessed_test_ep = []
    preprocessed_test_noep = []

    ep_patients_train_list = []
    noep_patients_train_list = []
    ep_patients_test_list = []
    noep_patients_test_list = []

    # Load in all the pre-processed files
    for patient in ep_patients_train:
        patient_folder_path = preprocessed_path + state[0] + "/" + patient
        processed_files = [filename for filename in os.listdir(patient_folder_path) 
                        if filename.startswith("p")]
        for file in processed_files:
            file = np.load(patient_folder_path + "/" + file)
            preprocessed_train_ep.append(file)
            ep_patients_train_list.append(patient)
        
    for patient in noep_patients_train:
        patient_folder_path = preprocessed_path + state[1] + "/" + patient
        processed_files = [filename for filename in os.listdir(patient_folder_path) 
                        if filename.startswith("p")]
        for file in processed_files:
            file = np.load(patient_folder_path + "/" + file)
            preprocessed_train_noep.append(file)
            noep_patients_train_list.append(patient)

    for patient in ep_patients_test:
        patient_folder_path = preprocessed_path + state[0] + "/" + patient
        processed_files = [filename for filename in os.listdir(patient_folder_path) 
                        if filename.startswith("p")]
        for file in processed_files:
            file = np.load(patient_folder_path + "/" + file)
            preprocessed_test_ep.append(file)
            ep_patients_test_list.append(patient)

    for patient in noep_patients_test:
        patient_folder_path = preprocessed_path + state[1] + "/" + patient
        processed_files = [filename for filename in os.listdir(patient_folder_path) 
                        if filename.startswith("p")]
        for file in processed_files:
            file = np.load(patient_folder_path + "/" + file)
            preprocessed_test_noep.append(file)
            noep_patients_test_list.append(patient)

    data_list = [preprocessed_train_ep, preprocessed_train_noep, preprocessed_test_ep, preprocessed_test_noep]
    label_list = [np.ones(len(preprocessed_train_ep)), np.zeros(len(preprocessed_train_noep)), 
                  np.ones(len(preprocessed_test_ep)), np.zeros(len(preprocessed_test_noep))]
    patientID_list = [np.array(ep_patients_train_list), np.array(noep_patients_train_list), 
                      np.array(ep_patients_test_list), np.array(noep_patients_test_list)]

    return data_list, label_list, patientID_list


# # Test function
# data_path = "data/"
# [data_list, label_list, patientID_list] = load_data(data_path)

# print("dont")
