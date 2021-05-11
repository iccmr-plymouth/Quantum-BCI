import numpy as np
from scipy import signal

from biosppy import storage
from biosppy.signals.eeg import eeg
from biosppy.signals.eeg import get_power_features

import threading
import tkinter as tk
import time

import os
import math
import sys

from numpy import genfromtxt

import pickle

expt_path = ""

data_file = input("Enter path of data file (Example: data.csv): ")

my_data = genfromtxt(data_file, delimiter=',')

# Declare headset-related variables


# Declare variables for alpha BCI. `data_block` stores the most recent data and uses it for analysis.
sampling_rate = 250
analysis_block_window = 1.0
data_block = np.zeros((int(sampling_rate * analysis_block_window), 8))
alpha_threshold = 0.02

data_2 = np.zeros((1, 9))
b_notch, a_notch = signal.iirnotch(50.0, 30.0, 250.0)

data_file_alpha = "processed_EEG.csv"
file_alpha = open(data_file_alpha, "w")

# Create a new variable called 'data_trials' that stores the features of 
# each trial and corresponding labels.

data_trials = []

for i in range(0, my_data.shape[0]):
    data_2[0, 0:8] = my_data[i, 0:8]
    mind_status_label = my_data[i, 8]
    data_2[0, 8] = mind_status_label

    # This deletes the oldest EEG sample and inserts the newest sample. In other words, it acts a FIFO queue.
    data_block = np.roll(data_block, -1, axis=0)
    data_block[-1, :] = my_data[i, 0:8]


    # Code for alpha BCI and data preprocessing.

    # The below if statement discards the first analysis_block_window and checks if data_block is ready for analysis.
    if i > int(sampling_rate * analysis_block_window) and i % int(sampling_rate * analysis_block_window) == 0:
        
        # Apply a notch filter to the signal
        y_notched = signal.filtfilt(b_notch, a_notch, data_block, axis = 0)

        # Extract the powers of EEG bands using the 'biosppy.signals.eeg' package
        ts, filtered, features_ts, theta, alpha_low, alpha_high, beta, gamma, plf_pairs, plf = eeg(signal=y_notched,
            sampling_rate=sampling_rate, show=False)
        
        
        features = np.stack((alpha_low, alpha_high, beta, theta), axis=0)

        
with open("processed_EEG.pickle", 'wb') as f:
  pickle.dump(data_trials, f, pickle.HIGHEST_PROTOCOL)