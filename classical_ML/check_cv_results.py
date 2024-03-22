
import numpy as np
import pickle
import pandas as pd

path = '/raid/smtam/results/tuning_results/'

with open(path + 'svc_cv_results_umap.pkl', 'rb') as f:
    cv_results = pickle.load(f)

cv_results_df = pd.DataFrame(cv_results)

cv_results_df.to_csv(path + 'svc_cv_results_umap.csv', index=False)



