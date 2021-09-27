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

import soundfile as sf
import pyaudio

import sys

import zmq

def calculate_mental_state(features):
    """
    This function should return 'relaxed' or 'aroused'.
    `features` is a 4x7x8 dimensional vector which can be fed into the machine learning algorithm in real-time.
    """

def main():
    # Specifications for the data acquisition.
    #-------------------------------------------------------------------------------------
    global bci_exited
    global wf_seek

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
    
    morse_code = np.zeros((2,))
    morse_index = 8


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
        # file = open(DataFile, "wb")
        # file_alpha = open(data_file_alpha, "w")


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

        # Declare variables for alpha BCI. `data_block` stores the most recent data and uses it for analysis.
        analysis_block_window = 1.0
        data_block = np.zeros((int(UnicornPy.SamplingRate * analysis_block_window), 8))
        alpha_threshold = 0.015
        click_time = 1.0

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

                # This deletes the oldest EEG sample and inserts the newest sample. In other words, it acts a FIFO queue.
                data_block = np.roll(data_block, -1, axis=0)
                data_block[-1, :] = data[:, 0:8]

                # np.savetxt(file,data,delimiter=',',fmt='%.3f',newline='\n')
                

                # Code for alpha BCI
                
                if i % int(UnicornPy.SamplingRate * click_time) == 0:
                    wf_seek = -1                    

                # The below if statement discards the first analysis_block_window and checks if data_block is ready for analysis.
                if i % int(UnicornPy.SamplingRate * analysis_block_window) == 0:
                    
                    # Apply a notch filter to the signal
                    y_notched = signal.filtfilt(b_notch, a_notch, data_block, axis = 0)

                    # Extract the powers of EEG bands using the 'biosppy.signals.eeg' package
                    ts, filtered, features_ts, theta, alpha_low, alpha_high, beta, gamma, plf_pairs, plf = eeg(signal=y_notched,
                        sampling_rate=UnicornPy.SamplingRate, show=False)
                    
                    features = np.stack((alpha_low, alpha_high, beta, theta), axis=0)

                    mind_status = "unknown"
                    
                    #calculate_mental_state can return two possible strings --- "relaxed" or "aroused"

                    mind_status = calculate_mental_state(features)                    
                    
                    if mind_status == "relaxed":
                        if morse_index % 12 == 3:
                            morse_code[0] = 0
                        elif morse_index % 12 == 7:
                            morse_code[1] = 0

                    elif mind_status == "aroused":
                        if morse_index % 12 == 4:
                            morse_code[0] = 1
                        elif morse_index % 12 == 8:
                            morse_code[1] = 1

                    # data_line = "{:.5f}, {}\n".format(avg_alpha_pow, mind_status)
                    # print(data_line)

                    if morse_index >= 12 and morse_index % 12 == 8:
                        # pass
                        # print("morse code: {}".format(morse_code))
                        socket.send_pyobj(morse_code)
                        # sys.stdout.write("morse code:")
                    
                    morse_index += 1

                    # file_alpha.write(data_line)



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
            # file.close()
            # file_alpha.close()

            # Close device.
            #-------------------------------------------------------------------------------------
            del device
            print("Disconnected from Unicorn")

    except UnicornPy.DeviceException as e:
        print(e)
    except Exception as e:
        print("An unknown error occured. %s" %e)

    # input("\n\nPress ENTER key to exit")
    bci_exited = True

def on_close():
    global should_exit
    global is_playing

    
    if is_playing:
        is_playing = False

    if not should_exit:
        should_exit = True
        
        

    while not bci_exited:
        time.sleep(1.0)
        continue
    
    socket.send_pyobj(np.array([1, 0]))
    
    top.destroy()   

def select_angle():
    pass

def play_audio():
    """This function is a thread. It starts when the play button is clicked."""
    global is_playing
    global wf_seek
        
    wf1Blocks = [block for block in
           sf.blocks('click-16k.wav', blocksize = chunk, overlap = 0, fill_value=0, dtype = 'float32')]

    # while len(wf1Blocks) < chunks_per_file:
    #     wf1Blocks.append(np.zeros((chunk)))

    # wf1Blocks = wf1Blocks[0:chunks_per_file]

    p = pyaudio.PyAudio()

    stream = p.open(format = pyaudio.paFloat32,
                    channels = 1,
                    rate = 16000,
                    output = True)

    while is_playing: # is_playing to stop playing. I have changed '' to b'', which is the syntax for new python.

        if wf_seek < len(wf1Blocks) - 1:           
            wf_seek += 1
            data = wf1Blocks[wf_seek].astype(np.float32).tobytes()
                    
            stream.write(data)
            
        else:
            time.sleep(0.0001)
    
    is_playing = False;
    stream.stop_stream()
    stream.close()
    p.terminate()


socket = zmq.Context(zmq.REP).socket(zmq.PUB)
socket.bind("tcp://*:1234")

# Audio thread
audio_thread = threading.Thread(target = play_audio)
chunk = 2000
wf_seek = 500000
is_playing = True
audio_thread.start()

#execute main
should_exit = False
bci_exited = False
bci_thread = threading.Thread(target = main)
bci_thread.start()


top = tk.Tk()
top.title("Quantum BCI")
top.protocol("WM_DELETE_WINDOW", on_close)

# angle_var = tk.DoubleVar()
# angle_Value = 0.0
# angle_scale = tk.Scale(top, variable = angle_var, orient = tk.HORIZONTAL, length = 450, from_ = 0.0, to = 180.0, command = select_angle)
# angle_scale.pack()

top.mainloop()
