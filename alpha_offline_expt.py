import UnicornPy
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

def main():
    # Specifications for the data acquisition.
    #-------------------------------------------------------------------------------------
    global bci_exited

    TestsignaleEnabled = False;
    FrameLength = 1;
    AcquisitionDurationInSeconds = 300;
    DataFile = "data.csv";
    data_file_alpha = "data_alpha.csv"
    
    print("Unicorn Acquisition Example")
    print("---------------------------")
    print()

    b, a = signal.butter(8, [1/125.0, 30/125.0], 'band')
    b_notch, a_notch = signal.iirnotch(50.0, 30.0, 250.0)


    try:
        # Get available devices.
        #-------------------------------------------------------------------------------------

        # Get available device serials.
        deviceList = UnicornPy.GetAvailableDevices(True)

        if len(deviceList) <= 0 or deviceList is None:
            raise Exception("No device available.Please pair with a Unicorn first.")

        # Print available device serials.
        print("Available devices:")
        i = 0
        for device in deviceList:
            print("#%i %s" % (i,device))
            i+=1

        # Request device selection.
        print()
        deviceID = int(input("Select device by ID #"))
        if deviceID < 0 or deviceID > len(deviceList):
            raise IndexError('The selected device ID is not valid.')

        # Open selected device.
        #-------------------------------------------------------------------------------------
        print()
        print("Trying to connect to '%s'." %deviceList[deviceID])
        device = UnicornPy.Unicorn(deviceList[deviceID])
        print("Connected to '%s'." %deviceList[deviceID])
        print()

        # Create a file to store data.
        file = open(DataFile, "wb")
        file_alpha = open(data_file_alpha, "w")


        # Initialize acquisition members.
        #-------------------------------------------------------------------------------------
        numberOfAcquiredChannels= device.GetNumberOfAcquiredChannels()
        configuration = device.GetConfiguration()

        # Print acquisition configuration
        print("Acquisition Configuration:");
        print("Sampling Rate: %i Hz" %UnicornPy.SamplingRate);
        print("Frame Length: %i" %FrameLength);
        print("Number Of Acquired Channels: %i" %numberOfAcquiredChannels);
        print("Data Acquisition Length: %i s" %AcquisitionDurationInSeconds);
        print();

        # Allocate memory for the acquisition buffer.
        receiveBufferBufferLength = FrameLength * numberOfAcquiredChannels * 4
        receiveBuffer = bytearray(receiveBufferBufferLength)

        # `data_2` is an EEG sample that would contain only the EEG data and the associated labels (aroused/relaxed)
        # That is, 8 EEG channels + 1 label
        data_2 = np.zeros((1, 9))

        # Declare variables for alpha BCI. `data_block` stores the most recent data and uses it for analysis.
        analysis_block_window = 1.0
        data_block = np.zeros((int(UnicornPy.SamplingRate * analysis_block_window), 8))
        alpha_threshold = 0.02

        try:
            # Start data acquisition.
            #-------------------------------------------------------------------------------------
            device.StartAcquisition(TestsignaleEnabled)
            os.system("cls")

            print("Data acquisition started.\n")

            # Calculate number of get data calls.
            numberOfGetDataCalls = int(AcquisitionDurationInSeconds * UnicornPy.SamplingRate / FrameLength);
        
            # Limit console update rate to max. 25Hz or slower to prevent acquisition timing issues.                   
            consoleUpdateRate = int((UnicornPy.SamplingRate / FrameLength) / 25.0);
            if consoleUpdateRate == 0:
                consoleUpdateRate = 1

            # Acquisition loop.
            #-------------------------------------------------------------------------------------
            for i in range (0,numberOfGetDataCalls):
                # Receives the configured number of samples from the Unicorn device and writes it to the acquisition buffer.

                if should_exit:
                    break

                device.GetData(FrameLength,receiveBuffer,receiveBufferBufferLength)

                # Convert receive buffer to numpy float array 
                data = np.frombuffer(receiveBuffer, dtype=np.float32, count=numberOfAcquiredChannels * FrameLength)
                data = np.reshape(data, (FrameLength, numberOfAcquiredChannels))

                # Copy only the EEG data and remove unnecessary columns
                # `mind_status_label` is the ground truth for the mind_status (aroused/relaxed)
                data_2[0, 0:8] = data[:, 0:8]
                data_2[0, 8] = mind_status_label

                # This deletes the oldest EEG sample and inserts the newest sample. In other words, it acts a FIFO queue.
                data_block = np.roll(data_block, -1, axis=0)
                data_block[-1, :] = data[:, 0:8]

                np.savetxt(file,data_2,delimiter=',',fmt='%.3f',newline='\n')
                

                # Code for alpha BCI

                # The below if statement discards the first analysis_block_window and checks if data_block is ready for analysis.
                if i > int(UnicornPy.SamplingRate * analysis_block_window) and i % int(UnicornPy.SamplingRate * analysis_block_window) == 0:
                    
                    # Apply a notch filter to the signal
                    y_notched = signal.filtfilt(b_notch, a_notch, data_block, axis = 0)

                    # Extract the powers of EEG bands using the 'biosppy.signals.eeg' package
                    ts, filtered, features_ts, theta, alpha_low, alpha_high, beta, gamma, plf_pairs, plf = eeg(signal=y_notched,
                        sampling_rate=UnicornPy.SamplingRate, show=False)

                    a_low_pow = np.mean(alpha_low[:, :1])
                    a_high_pow = np.mean(alpha_high[:, :1])
                    avg_alpha_pow = (a_low_pow + a_high_pow) / 2.0


                    mind_status = "unknown"

                    if avg_alpha_pow > alpha_threshold:
                        mind_status = "relaxed"
                    else:
                        mind_status = "aroused"

                    data_line = "{:.5f}, {}\n".format(avg_alpha_pow, mind_status)
                    print(data_line)

                    file_alpha.write(data_line)



            # Stop data acquisition.
            #-------------------------------------------------------------------------------------
            device.StopAcquisition();
            print()
            print("Data acquisition stopped.");

        except UnicornPy.DeviceException as e:
            print(e)
        except Exception as e:
            print("An unknown error occured. %s" %e)
        finally:
            # release receive allocated memory of receive buffer
            del receiveBuffer

            #close file
            file.close()
            file_alpha.close()

            # Close device.
            #-------------------------------------------------------------------------------------
            del device
            print("Disconnected from Unicorn")

    except Unicorn.DeviceException as e:
        print(e)
    except Exception as e:
        print("An unknown error occured. %s" %e)

    # input("\n\nPress ENTER key to exit")
    bci_exited = True

def on_close():
    global should_exit

    if not should_exit:
        should_exit = True

    while not bci_exited:
        time.sleep(1.0)
        continue
    top.destroy()   

def select_angle():
    pass

def key_press(event):
    global mind_status_label
    key = event.char
    if key == '1':
        mind_status_label = 1
        print("Relaxed")
    elif key == '2':
        mind_status_label = 0
        print("aroused")
    elif key == '3':
        mind_status_label = 0
        print("unknown")

#execute main
should_exit = False
bci_exited = False
mind_status_label = -1
bci_thread = threading.Thread(target = main)
bci_thread.start()


top = tk.Tk()
top.title("Quantum BCI")
top.protocol("WM_DELETE_WINDOW", on_close)
top.bind('<Key>', key_press)


top.mainloop()