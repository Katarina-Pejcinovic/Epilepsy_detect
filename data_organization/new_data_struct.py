# cut_segments outputs this:
# List of 4 3d numpy arrays --- [ # segments, # channels, # points]
# List of 4 1d arrays of labels -- [ # segments ]
# List of 4 1d arrays of patient IDs -- [ # segments ]

# this function should output:
# 1 3d numpy array --- [ # channels, 3 + # points, # segments ]

# 3 extra columns = [ labels, patient ID, recording # ]
# 1st row is the data, every other row is NaN

import numpy as np
from patient_id_dict import *

# WILL USE PATIENT DICT IN THE FUTURE - NOT PATIENT ID ARRAY FROM CUT SEGMENTS

def create_3d_numpy(results_list, labels_list, patients_list, patient_list_folder):

    array_count = 1

    for results, labels, patients in zip(results_list, labels_list, patients_list):
        num_segments = np.size(results, 0)
        num_channels = np.size(results, 1)
        num_points = np.size(results, 2)
        # print(np.shape(results))

        # Reshape results array
        results = np.reshape(results, [num_channels, num_points, num_seg])

        # print(np.shape(results))
        # print(num_segments)
        # print(num_channels)
        # print(num_timepoints)

        # Convert patient IDs to int using patient dictionary
        patient_dict = load_patient_dict(patient_list_folder)
        new_patients = np.empty([num_segments])

        for count, p in enumerate(patients):
            p_num = patient_dict[p]
            new_patients[count] = p_num

        temp_array = np.empty([num_channels, 3, num_segments])
        temp_array[:] = np.nan;
        temp_array[0, 0, :] = labels
        temp_array[0, 1, :] = new_patients

        unique_patients, counts = np.unique(new_patients, return_counts=True)
        patient_counts = np.column_stack((unique_patients, counts))
        patient_counts = patient_counts.astype(int)
        # print(patient_counts)

        seg_index = 0;
        for count, patient in enumerate(patient_counts[:, 0]):
            # print(patient)
            recording_count = 0
            for recording in range(patient_counts[count, 1]):
                # print(recording_count)
                temp_array[0, 2, seg_index] = recording_count
                recording_count = recording_count + 1
                seg_index = seg_index + 1

        # print("Old shape: ", np.shape(temp_array))
        # print(temp_array[:, :, 1])

        new_data_array = np.concatenate([temp_array, results], axis=1)

        # print("New shape: ", np.shape(new_data_array))
        # print(new_data_array[:, :, 1])

        if array_count == 1:
            full_data_array = new_data_array
        else:
            full_data_array = np.concatenate([full_data_array, new_data_array], axis = 2)
            # print("Full shape: ", np.shape(full_data_array))

        array_count = array_count + 1
        # print("count: ", array_count)

    return full_data_array


# TEST

num_seg = 5;
num_channels = 4
num_points = 5

test_array = np.ones([num_seg, num_channels, num_points])
array_list = [test_array, test_array, test_array, test_array]

test_labels = np.ones([num_seg])*2
labels_list = [test_labels, test_labels, test_labels, test_labels]

test_patients= np.array(['aaaaaawu', 'aaaaaawu', 'aaaaamor', 'aaaaanla', 'aaaaanla'])
# 101 101 161 253 253
patients_list = [test_patients, test_patients, test_patients, test_patients]

patient_list_folder = 'data/'

full_data_array = create_3d_numpy(array_list, labels_list, patients_list, patient_list_folder)
print("Full Data Shape: ", np.shape(full_data_array))



