import numpy as np
import os
import pickle

def split(data, train_file_path, test_file_path):
    # Calculate the split index for 3/4 training, 1/4 testing
    num_recordings = data.shape[2]
    split_index = int(num_recordings * 0.75)

    # Shuffle the recording indices
    indices = np.arange(num_recordings)
    np.random.seed(7)
    np.random.shuffle(indices)

    # Split the data by recordings
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]

    train_data = data[:, :, train_indices]
    test_data = data[:, :, test_indices]

    # Save the data
    os.makedirs(train_file_path, exist_ok=True)
    os.makedirs(test_file_path, exist_ok=True)

    train_file = os.path.join(train_file_path, 'train_data.pkl')
    test_file = os.path.join(test_file_path, 'test_data.pkl')

    with open(train_file, 'wb') as f:
        pickle.dump(train_data, f)

    with open(test_file, 'wb') as f:
        pickle.dump(test_data, f)

    return train_data, test_data


# Call split function
#train_data, test_data = split(full_data_array, train_file_path, test_file_path)

#print("Shape of train Data: ", train_data.shape)
#print("Shape of test Data: ", test_data.shape)
