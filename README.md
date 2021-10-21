# Quantum-BCI
Controlling a quantum computer with a BCI

# Offline Experimental set-up
To start the experiment, I wear the EEG headset and run the [alpha_offline_expt.py](https://github.com/satvik-venkatesh/Quantum-BCI/blob/main/alpha_offline_expt.py) file. The ground truth is initially set to -1 as I am not ready yet. Once I am ready, I press a key on my keyboard that sets the ground truth label. When I go to a relaxed state, the ground truth is set to 1. When I go to an aroused state, the ground truth is set to 0. I press a different key on the keyboard to choose between different mental states. Just before I end the experiment, I set the ground truth back to -1.


# Format of csv files
## data.csv
There are 9 columns: 8 EEG channels + 1 label. The label is the ground truth that specifies the mental state. '1' is for relaxed state (or eyes closed) and '0' is for aroused state (or eyes open). '-1' is for unknown state. The unknown state useful is useful for the beginning and end of each trial because I am either getting ready or just about to end the experiment. You can discard these values when training the machine learning model.

## data_alpha.csv
The first column contains the alpha power, the second column contains the threshold-based output, the third column contains the ground truth. This file might not be of relevance to train your machine learning model. In this file, I am processing the EEG data to calculate an output based on the power of alpha waves (every 1 second). I have only calculated a threshold-based output, considering the first EEG channel.
 
 
 
# Types of experiments
Please check the [experiments](https://github.com/satvik-venkatesh/Quantum-BCI/tree/main/experiments) folder.

As mentioned above, data.csv contains the raw EEG + ground truth labels. Here, I am altering my mental state. In data_alpha.csv, you will see a basic calculation using threshold-dependent values for alpha power without the machine learning.


# Data preprocessing
Before performing analysis on EEG, please apply a notch filter of 50 Hz and a bandpass filter (perhaps, from 2 to 30Hz). The filtering has been done in the [alpha_offline_expt.py](https://github.com/satvik-venkatesh/Quantum-BCI/blob/main/alpha_offline_expt.py) file. Please read through line 156 to 183 that contain the code for filtering. I have used a Python package called [biosppy.signals.eeg](https://github.com/PIA-Group/BioSPPy/blob/212c3dcbdb1ec43b70ba7199deb5eb22bcb78fd0/biosppy/signals/eeg.py). [Here](https://biosppy.readthedocs.io/en/stable/biosppy.signals.html#biosppy-signals-eeg) is the documentation for the Python package. It was useful to extract alpha and beta frequency bands, which can be used as feastures for your machine learning algorithm.

# Online System
Primarily, there are two files for the online experiment, which actually controls the qubit using the BCI. The program adopts a client-server architecture. 'quantum_server.py' controls qubit and 'morse_BCI.py' encondes the brain waves into brain codes. First run the 'quantum_server.py' through the commandline and 'morse_BCI.py' through another command line, as shown in the video below.

# Demo
Turn on sound in the video to listen to the click.

https://user-images.githubusercontent.com/52415916/137694698-4b6ea069-4d9f-4538-b3f7-81ab47d8188e.mp4
