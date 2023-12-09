import numpy as np
from imputate import *

# Create two random 2D arrays with shape (3, 4) filled with random floats
array1 = np.random.rand(3, 4)
array2 = np.random.rand(3, 4)

# Replace one row in array1 with NaN
row_to_replace = 1  # You can change this value to choose a different row
array1[row_to_replace, :] = np.nan

# Stack the arrays along a new axis to create a 3D array
stacked_array = np.stack((array1, array2), axis=0)

# print("Array1:")
# print(array1)
# print("\nArray2:")
# print(array2)
# print("\nStacked Array:")
# print(stacked_array)



#call run_imputate()
imputated = run_imputate(stacked_array)
print("imputated", '\n', imputated)
#print(imputated )