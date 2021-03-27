# Quantum-BCI
Controlling a quantum computer with a BCI

# Offline Experimental set-up
To start the experiment, I wear the EEG headset and run the 'alpha_offline_expt.py' file. The ground truth is initially set to -1 as I am not ready yet. Once I am ready, I press a key on my keyboard that sets the ground truth label. When I go to a relaxed state (or close my eyes), the ground truth is set to 1. When I go to an aroused state (or open my eyes), the ground truth is set to 0. I press a different key on the keyboard to choose between different mental states. Just before I end the experiment, I set the ground truth back to -1.


# Format of csv files
## data.csv
There are 9 columns: 8 EEG channels + 1 label. The label is the ground truth that specifies the mental state. '1' is for relaxed state (or eyes closed) and '0' is for aroused state (or eyes open). '-1' is for unknown state. The unknown state useful is useful for the beginning and end of each trial because I am either getting ready or just about to end the experiment. You can discard these values when training the machine learning model.

## data_alpha.csv
This file might not be of relevance to train your machine learning model. In this file, I am processing the EEG data to calculate an output based on the power of alpha waves (every 1 second). I have only calculated a threshold-based output, considering the first EEG channel.
