import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def run_imputate(preprocessed):

    #make a copy and make a mask of the missing data 
    preprocessed_copy = np.copy(preprocessed)
    missing_mask = np.isnan(preprocessed_copy)

    #instantiate the imputer class
    imputer = IterativeImputer(max_iter=10, random_state=0)

    #calculate the values 
    imputed_values = imputer.fit_transform(preprocessed_copy)

    preprocessed_copy[missing_mask] = imputed_values[missing_mask]

    return preprocessed_copy

        
