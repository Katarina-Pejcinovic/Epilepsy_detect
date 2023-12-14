
import numpy as np
from sklearn.impute import SimpleImputer

#run_imputate takes in the first index in the list list[0], which is a list of 
# 4 3D numpy arrays. run_imputate takes in a 3D numpy array 
def run_imputate(preprocessed):

    #Make a copy of the 3D numpy array 
    copy_preprocessed_3D = np.copy(preprocessed)

    #loop through slices of 3D array representing patients
    for i in range(copy_preprocessed_3D.shape[0]):

        #Take a slice of the data 
        patient_slice = copy_preprocessed_3D[i, :, :]
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

        copy_preprocessed_3D[i, :, :] = imputed_values


    return copy_preprocessed_3D

        
