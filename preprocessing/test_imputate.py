import numpy as np
from imputate import *


#fake preprocessed data 
preprocessed = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]])

#preprocessed = np.load('data/training/epilepsy/preprocessed_aaaaaanr_epilepsy_edf_aaaaaanr_s007_t000.edf.npy')

# Create a row of NaN values
row_of_nans = np.full_like(preprocessed[0, :], np.nan)

# Append the row of NaN values to the 2D array
preprocessed = np.append(preprocessed, [row_of_nans], axis=0)
print(preprocessed)

#call run_imputate()
imputated = run_imputate(preprocessed)
print(imputated )