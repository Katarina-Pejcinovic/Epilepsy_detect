import numpy as np

def cut_segments(data_list, label_list, patientID_list):
# Inputs: 
# - a list 'data_list' of 4 lists of n 2D numpy arrays where each 2D numpy array of size (x, y)
# contains x channels and y data samples per channel, with sampling rate 250 Hz
# - a list 'label_list' of 4 1D numpy arrays where each 1D numpy array of size (n,) has a label for each 2D numpy array in data_list
# - a list 'patientID_list' of 4 1D numpy arrays where each 1D numpy array of size (n,) has the patient ID for each 2D numpy array in data_list
# The function cuts segments of 5 minutes (75,000 samples) for all channels from each 2D np array
# Outputs:
# - a list 'called result_4d' of 4 3D np arrays, where each 3D np array is of size (s, x, 75000) where s is the number of 5-minute segments
# - a list 'label_result' of 4 1D numpy arrays where each 1D numpy array of size (s,) has the label for each 5-minute segment in the 3D np arrays in result_4d
# - a list 'patientID_result' of 4 1D numpy arrays where each 1D numpy array of size (s,) has the patientID for each 5-minute segment in the 3D np arrays in result_4d

    result_4d = []
    label_result = []
    patientID_result = []

    for idx, input_arrays in enumerate(data_list):
        segments = []
        labels = []
        patientIDs = []

        for i, array in enumerate(input_arrays):
            num_channels, signal_length = array.shape
            num_segments_array = signal_length // 75000

            if num_segments_array >= 1:
                segments_array = np.array_split(array[:, :num_segments_array * 75000], num_segments_array, axis=1)
                segments.extend(segments_array)
                labels.extend(np.full(num_segments_array, label_list[idx][i]))
                patientIDs.extend(np.full(num_segments_array, patientID_list[idx][i]))

        #result_4d.append(segments)
        if segments:
            result = np.stack(segments, axis=0)
        else:
            result = segments
        result_4d.append(result)
        label_result.append(np.array(labels))
        patientID_result.append(np.array(patientIDs))

    return result_4d, label_result, patientID_result