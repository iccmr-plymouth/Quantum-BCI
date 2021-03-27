# Quantum-BCI
Controlling a quantum computer with a BCI

# Offline Experimental set-up
Initially, when the trial is started, the ground 

# Format of csv files
## data.csv
There are 9 columns. 8 EEG channels + 1 label. The label is the ground truth that specifies the mental state. '1' is for relaxed state (or eyes closed) and '0' is for aroused state (or eyes open). '-1' is for unknown state. The unknown state useful is useful for the beginning and end of each trial because I am either not ready or just about to end the experiment. You can discard these values when training the machine learning model.
