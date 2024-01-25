import numpy as np
from impute import *

# Create two random 2D arrays with shape (3, 1) filled with random floats
array1 = np.random.rand(3, 1)
array2 = np.random.rand(3, 1)

# print("data_array", array1)

# Replace one row in array1 with NaN
row_to_replace = 1  # You can change this value to choose a different row
array1[row_to_replace, :] = np.nan
array2[row_to_replace, :] = np.nan


metadata_np =np.random.rand(3,3)

array1 = np.concatenate((metadata_np,array1 ), axis = 1)
array2 = np.concatenate((metadata_np,array2 ), axis = 1)

# print(array1.shape)
# print(array1)
# print(array2.shape)
# print(array2)
# Stack the arrays along a new axis to create a 3D array
stacked_array = np.stack((array1, array2), axis=0)
print("original", '\n', stacked_array)
print("original size", '\n', stacked_array.shape)

# Transpose the array to shape (3, 4, 2)
transposed_array = np.transpose(stacked_array, (1, 2, 0))
print("final", '\n', transposed_array.shape)
print("final", '\n', transposed_array)

print("one slice", '\n', transposed_array[:, :, 0])
print("two slice", '\n', transposed_array[:, :, 1])
# print("Array1:")
# print(array1)
# print("\nArray2:")
# print(array2)
# print("\nStacked Array:")
# print(stacked_array)


# print(stacked_array.shape)
# print(stacked_array)
# #call run_imputate()
imputated = run_impute(transposed_array)
##print("imputated", '\n', imputated)
#print(imputated.shape)