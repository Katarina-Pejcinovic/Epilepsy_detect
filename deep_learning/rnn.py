# -*- coding: utf-8 -*-
"""rnn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vED7-RLAMIeY71HroSjMSsyv-hk3NUf7
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd  /content/gdrive/Shareddrives/BE_223A_Seizure_Project/Code/

import mne
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.signal import periodogram
import neurokit2 as nk
import pywt
from collections import Counter
import scipy.stats as stats

import tensorflow as tf
from keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from keras.layers import LSTM
from tensorflow.python.keras.layers import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.optimizers.legacy import Adam
from tensorflow.python.keras.layers import Input
from keras.layers import Bidirectional
from tensorflow.python.keras.layers import Dropout
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
from keras.preprocessing.sequence import pad_sequences


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def rnn_model(train_df, learning_rate=0.001, gradient_threshold=1, batch_size=32, epochs=2, n_splits=5):
    model_save_path = 'deep_learning/rnn_saved_model'
    train_data = train_df[:,:,3:]
    n_channels = train_data.shape[1]
    train_label = train_df[:,0,0]
    groups = train_df[:,0,1]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    val_predictions_list = {}
    val_predictions_binary_list = {}
    model = Sequential()
    model.add(Bidirectional(LSTM(200, return_sequences=False), input_shape=(n_channels, train_data.shape[2])))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    opt = Adam(learning_rate=learning_rate, clipnorm=gradient_threshold)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    counter = 0
    for train_index, val_index in skf.split(train_data, train_label, groups=groups):
        counter+=1
        X_train, X_val = train_data[train_index], train_data[val_index]
        y_train, y_val = train_label[train_index], train_label[val_index]

        X_train_reshaped = X_train.reshape(X_train.shape[0], n_channels, X_train.shape[2])

        history = model.fit(
            X_train_reshaped,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1  # Set verbose to 0 to suppress output during training
        )

        X_val_reshaped = X_val.reshape(X_val.shape[0], n_channels, X_val.shape[2])

        val_predictions = model.predict(X_val_reshaped)
        val_predictions_list[f'fold{counter}'] = val_predictions

        # Convert predictions to binary (0 or 1)
        val_predictions_binary = [1 if pred >= 0.50 else 0 for pred in val_predictions]
        val_predictions_binary_list[f'fold{counter}'] = [val_predictions_binary, y_val]
    
    model.save(model_save_path)
    return val_predictions_binary_list, val_predictions_list
        
def rnn_model_test(test_df):
    predictions = []
    preds_proba = []
    model = load_model('deep_learning/rnn_saved_model')
    test_data = test_df[:,:,3:]
    n_channels = test_data.shape[1]
    # Evaluate the model on the test data
    X_test_reshaped = test_data.reshape(test_data.shape[0], n_channels, test_data.shape[2])
    test_predictions = model.predict(X_test_reshaped)
    preds_proba.append(test_predictions)

    # Convert test predictions to binary (0 or 1)
    test_predictions_binary = [1 if pred >= 0.50 else 0 for pred in test_predictions]
    predictions.extend(test_predictions_binary)

    return predictions, preds_proba