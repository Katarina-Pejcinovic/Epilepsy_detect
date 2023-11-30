
# Loop to run multiple files

def process_multiple_files(base_dir):

    # Some sort of way to get Label, Patient ID, Session date, Type, and the name of the file. 

    # Base directory and output directory
    # base_dir = '/Users/andresmichel/Documents/EGG_data /v2.0.0/'
    output_dir = os.path.join(base_dir, 'preprocessed_data')

    # Create the output directory 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load subject IDs from the lists
    with open(os.path.join(base_dir, 'subject_ids_epilepsy.list'), 'r') as file:
        subject_ids_epilepsy = file.read().splitlines()

    with open(os.path.join(base_dir, 'subject_ids_no_epilepsy.list'), 'r') as file:
        subject_ids_no_epilepsy = file.read().splitlines()

    # Subdirectories and save the preprocessed file
    def save_preprocessed_data(eeg_array, patient_id, label, base_filename):
        # Create subdirectory for the label (epilepsy or no epilepsy)
        label_dir = os.path.join(output_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Create subdirectory for the patient ID
        patient_dir = os.path.join(label_dir, patient_id)
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)

        # Save the preprocessed data
        preprocessed_filename = f"preprocessed_{patient_id}_{label}_{base_filename}.npy"
        preprocessed_filepath = os.path.join(patient_dir, preprocessed_filename)
        np.save(preprocessed_filepath, eeg_array)
        print(f"Saved preprocessed data to {preprocessed_filepath}")

    # Process each file
    def process_file(edf_path, patient_id, label):
        # Replace EEGDataPair with your actual preprocessing class
        eeg_data_pair = EEGDataPair(edf_path, None, None, patient_id, label)
        dic = eeg_data_pair.processing_pipeline()

        # Get the processed data
        eeg_array = dic['eeg_data']

        # Save the preprocessed data
        base_filename = os.path.basename(edf_path)
        save_preprocessed_data(eeg_array, patient_id, label, base_filename)

    # Process each patient's files
    for label, subject_ids in [('epilepsy_edf', subject_ids_epilepsy), ('no_epilepsy_edf', subject_ids_no_epilepsy)]:
        for patient_id in subject_ids:
            patient_path = os.path.join(base_dir, label, patient_id)
            if os.path.exists(patient_path):
                for edf_file in glob(os.path.join(patient_path, '**/*.edf'), recursive=True):
                    process_file(edf_file, patient_id, label)