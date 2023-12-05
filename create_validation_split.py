
import numpy as np
import pandas as pd

# ids = pd.read_table('data/subject_ids_epilepsy.txt', delimiter="\t")
ids = pd.read_table('data/subject_ids_no_epilepsy.txt', delimiter="\t")
print(ids.head())

num_patients = ids.shape[0]
print(num_patients)

num_training = round(num_patients*0.75)

training = ids.sample(num_training)
print(training.head())
print(training.shape[0])

testing = ids[~ids['IDs'].isin(training['IDs'])]
print(testing.head())
print(testing.shape[0])

# training.to_csv('data/subject_ids_epilepsy_training.txt', sep="\t", index=False)
# testing.to_csv('data/subject_ids_epilepsy_testing.txt', sep="\t", index=False)
training.to_csv('data/subject_ids_no_epilepsy_training.txt', sep="\t", index=False)
testing.to_csv('data/subject_ids_no_epilepsy_testing.txt', sep="\t", index=False)