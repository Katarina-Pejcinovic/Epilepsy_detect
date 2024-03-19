import numpy as np 

def downsample(data):

    # Extract positive samples
    positive_indices = np.where(data[:, 0, 0] == 1)[0]

    # Count the number of positive samples
    num_positive_samples = len(positive_indices)
    print("NUM POS SAMPLE", num_positive_samples)

    # Extract negative samples
    negative_indices = np.where(data[:, 0, 0] == 0)[0]

    # Count the number of negative samples
    num_negative_samples = len(negative_indices)
    print("NUM NEG SAMPLE", num_negative_samples)

    # Determine the desired number of positive samples to keep to balance the ratio
    desired_num_positive_samples = num_negative_samples
    print("desired", desired_num_positive_samples)

    # Randomly select positive samples to keep
    selected_positive_indices = np.random.choice(positive_indices, size=desired_num_positive_samples, replace=False)

    # Combine selected positive indices with all negative indices
    selected_indices = np.concatenate((selected_positive_indices, negative_indices))

    # Shuffle the indices
    np.random.shuffle(selected_indices)

    # Create the balanced dataset
    balanced_data = data[selected_indices]

    return balanced_data
