import numpy as np
import os
import pickle

def split(data, train_file_path, test_file_path):
    # Calculate the split index for 3/4 training, 1/4 testing
    patient_id = train_data[0, 1, :]
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



data_file_path = 'data/'

with open(data_file_path + 'full_3d_array.pkl', 'rb') as f:
     full_data_array = pickle.load(f)
print("Full data array shape:", full_data_array.shape)

patient_id = full_data_array[0, 1, :]

with open(data_file_path + 'patient_ID_dictionary.pkl', 'rb') as f:
     patient_ID_dict = pickle.load(f)

# print(patient_ID_dict)
num_patients = 100

# ep_patients = np.array(patient_ID_dict.values())[0:100]
ep_patients = np.array(list(patient_ID_dict.values()))[0:100]
noep_patients = np.array(list(patient_ID_dict.values()))[100:]

split_index = int(num_patients*0.75)

# Shuffle the patient indices
indices = np.arange(100)
np.random.seed(7)
np.random.shuffle(indices)
# print(indices)

noep_indices = np.arange(100)
np.random.seed(10)
np.random.shuffle(noep_indices)
# print(noep_indices)

ep_train_indices = indices[:split_index].tolist()
ep_test_indices = indices[split_index:]
noep_train_indices = noep_indices[:split_index]
noep_test_indices = noep_indices[split_index:]

ep_train_patients = ep_patients[ep_train_indices]
ep_test_patients = ep_patients[ep_test_indices]
noep_train_patients = noep_patients[noep_train_indices]
noep_test_patients = noep_patients[noep_test_indices]

train_patients = np.concatenate((ep_train_patients, noep_train_patients))
test_patients = np.concatenate((ep_test_patients, noep_test_patients))

# Split the full data array into train and test
find_train_indices = np.where(np.isin(patient_id, train_patients))[0]
find_test_indices = np.where(np.isin(patient_id, test_patients))[0]

train_data = full_data_array[:, :, find_train_indices]
test_data = full_data_array[:, :, find_test_indices]
print(train_data.shape)
print(test_data.shape)


# Call split function
#train_data, test_data = split(full_data_array, train_file_path, test_file_path)

#print("Shape of train Data: ", train_data.shape)
#print("Shape of test Data: ", test_data.shape)
