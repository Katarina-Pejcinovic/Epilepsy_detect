
import numpy as np
from sklearn.impute import SimpleImputer

#run_imputate takes in the first index in the list list[0], which is a list of 
# 4 3D numpy arrays. run_imputate takes in a 3D numpy array 
def run_impute(preprocessed_og):

    #cut off metadata 
    y_length = preprocessed_og.shape[1]

    preprocessed = preprocessed_og[:, 3:y_length, :]
 
    #print("preprocessed_shape", '\n', preprocessed.shape)
    
    #Make a copy of the 3D numpy array 
    copy_preprocessed_3D = np.copy(preprocessed)

    #loop through slices of 3D array representing patients
    for i in range(copy_preprocessed_3D.shape[2]):

        #Take a slice of the data 
        patient_slice = copy_preprocessed_3D[:, :, i]
        #print("patient slice ",'\n',  patient_slice)

        #instantiate the imputer class
        imputer = SimpleImputer(missing_values = np.nan, 
                        strategy ='mean')

        #calculate the values 
        # Fitting the data to the imputer object
        imputer = imputer.fit(patient_slice)
        
        # Imputing the data     
        imputed_values = imputer.transform(patient_slice)

        #print("imputed values", '\n', imputed_values)

        copy_preprocessed_3D[:, :, i] = imputed_values

    #add back on the metadata
    metadata = np.concatenate((preprocessed_og[:, 0:3, :],copy_preprocessed_3D), axis=1)
    #print("metadata", '\n', metadata.shape)
    #print("metadata patient 1",'\n', metadata[:, :, 0])
    return metadata

        
