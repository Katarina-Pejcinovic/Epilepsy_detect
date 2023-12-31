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
from keras.models import Sequential
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


def rnn_model(eeg_array, label, test_data, learning_rate=0.001, gradient_threshold=1, batch_size=32, epochs=2):
    train_array = eeg_array
    train_label = label

    n_channels = eeg_array[0].shape[0]
    
    # Find the maximum sequence length in the entire dataset
    max_sequence_length = max(len(seq.T) for seq in train_array)

    model = Sequential()
    model.add(Bidirectional(LSTM(200, return_sequences=False), input_shape=(n_channels, max_sequence_length)))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    learning_rate = learning_rate
    gradient_threshold = 1
    opt = Adam(learning_rate=learning_rate, clipnorm=gradient_threshold)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    for id, (X, y) in enumerate(zip(train_array, train_label)):
        X = X.T
        y = np.array([y])

        X_padded = pad_sequences([X], maxlen=max_sequence_length, padding='post', truncating='post')[0]
        X_reshaped = X_padded.reshape((1, n_channels, max_sequence_length))

        history = model.fit(
            X_reshaped,
            y,
            batch_size=batch_size,
            epochs=epochs
        )

    predictions = []
    preds_proba = []
    for X_test in test_data:
        X_test = X_test.T
        
        # Pad/truncate each inner array to the maximum length
        X_test_padded = pad_sequences([X_test], maxlen=max_sequence_length, padding='post', truncating='post')[0]
        X_test_reshaped = X_test_padded.reshape((1, n_channels, max_sequence_length))
        
        prediction = model.predict(X_test_reshaped)
        preds_proba.append(prediction[0][0])
        if prediction < 0.50:
            predictions.append(0)
        if prediction > 0.50:
            predictions.append(1)

    return predictions, preds_proba