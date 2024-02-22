
from sklearn.pipeline import make_pipeline
import numpy as np
from sklearn.metrics import make_scorer,fbeta_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest
import pickle
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Stratify patients based on age --> perform feature selection and see how the top features change

data_file_path = 'data/'

with open(data_file_path + 'train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
with open(data_file_path + 'test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

with open('data/features_3d_array.pkl', 'rb') as f:
    features_3d_array = pickle.load(f)
with open('data/features_3d_array_test.pkl', 'rb') as f:
    features_3d_array_test = pickle.load(f)

# Break down train data structure
data_full = train_data[:, 3:, :]
labels = train_data[0, 0, :]
patient_id = train_data[0, 1, :]
num_segments = train_data.shape[2]
num_channels = train_data.shape[0]
num_data = train_data.shape[1] - 3
data_reshape = np.reshape(data_full, (num_segments, num_channels, num_data))
print("Train data reshape ran")
print(data_reshape.shape)

# Break down test data structure
data_full_test = test_data[:, 3:, :]
labels_test = test_data[0, 0, :]
patient_id_test = test_data[0, 1, :]
num_segments_test = test_data.shape[2]
num_channels_test = test_data.shape[0]
num_data_test = test_data.shape[1] - 3
data_reshape_test = np.reshape(data_full_test, (num_segments_test, num_channels_test, num_data_test))
print("Train data reshape ran")
print(data_reshape_test.shape)


channel_1 = features_3d_array[:, 0, :]
print(channel_1.shape)
## Run selectkbest by channel

selector = SelectKBest(k=20)

# Turn data into z-scores
scl = StandardScaler()
X_train = scl.fit_transform(channel_1)

# Feature selection
X_train = selector.fit_transform(channel_1, labels)

selected_feature_indices = selector.get_support(indices=True)

print(selected_feature_indices)