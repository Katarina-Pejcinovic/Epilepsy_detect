# cut_segments outputs this:
# 4 3d numpy arrays --- [ # segments, # channels, # points]
# 4 1d arrays of labels -- [ # segments ]
# 4 1d arrays of patient IDs -- [ # segments ]

# this function should output:
# 1 3d numpy array --- [ # channels, 3 + # points, # segments ]

# 3 extra columns = [ labels, patient ID, recording # ]
# 1st row is the data, every other row is NaN

import numpy as np

# WILL USE PATIENT DICT IN THE FUTURE - NOT PATIENT ID ARRAY FROM CUT SEGMENTS
def create_3d_numpy(results, labels, patients):
    num_segments = np.size(results, 2)
    num_channels = np.size(results, 0)
    num_timepoints = np.size(results, 1)
    # print(num_segments)
    # print(num_channels)
    # print(num_timepoints)

    temp_array = np.empty([num_channels, 3, num_segments])
    temp_array[:] = np.nan;
    temp_array[0, 0, :] = labels
    temp_array[0, 1, :] = patients

    unique_patients, counts = np.unique(patients, return_counts=True)
    patient_counts = np.column_stack((unique_patients, counts))
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

    print("Old shape: ", np.shape(temp_array))
    # print(temp_array[:, :, 1])

    new_data_array = np.concatenate([temp_array, results], axis=1)
    print("New shape: ", np.shape(new_data_array))
    # print(new_data_array[:, :, 1])

    return new_data_array


# TEST

num_seg = 3;
num_channels = 2
num_points = 10
test_array = np.ones([num_seg, num_channels, num_points])
test_array = np.reshape(test_array, [num_channels, num_points, num_seg])
test_labels = np.ones([num_seg])*2
test_patients= np.array([1, 1, 2])

new_data_array = create_3d_numpy(test_array, test_labels, test_patients) 




