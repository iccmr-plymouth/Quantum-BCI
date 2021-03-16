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
    AcquisitionDurationInSeconds = 60;
    DataFile = "data.csv";
    data_file_alpha = "data_alpha.csv"
    
    print("Unicorn Acquisition Example")
    print("---------------------------")
    print()

    b, a = signal.butter(8, [1/125.0, 30/125.0], 'band')
    # b2, a2 = signal.butter(8, [1/125.0, 30/125.0], 'band')
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
        file_alpha = open(data_file_alpha, "a")


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

        # Declare new variables for alpha BCI
        analysis_block_window = 1.0
        data_block = np.zeros((int(UnicornPy.SamplingRate * analysis_block_window), 8))
        alpha_threshold = 0.02

        quantum_angle = 0.0

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

                data_block = np.roll(data_block, -1, axis=0)
                data_block[-1, :] = data[:, 0:8]

                np.savetxt(file,data,delimiter=',',fmt='%.3f',newline='\n')
                
                # Update console to indicate that the data acquisition is running.
                if i % consoleUpdateRate == 0:
                    #print('.',end='',flush=True)
                    pass


                # Code alpha BCI


                if i > int(UnicornPy.SamplingRate * analysis_block_window) and i % int(UnicornPy.SamplingRate * 1.0) == 0:
                    # y = signal.filtfilt(b, a, data_block, axis = 0)
                    y_notched = signal.filtfilt(b_notch, a_notch, data_block, axis = 0)
                    # y_notched = data_block
                    # print("Reached here!!!")

                    # print("data_block: {}".format(data_block))
                    # ts, theta, alpha_low, alpha_high, beta, gamma = get_power_features(signal=y_notched,
                    #     sampling_rate=250, size=3.0, overlap=0)

                    ts, filtered, features_ts, theta, alpha_low, alpha_high, beta, gamma, plf_pairs, plf = eeg(signal=y_notched,
                        sampling_rate=UnicornPy.SamplingRate, show=False)

                    # print("alpha_low.shape: {} and alpha_high.shape: {}".format(alpha_low.shape, alpha_high.shape))
                    # print("alpha_low: {} \t alpha_high: {}".format(alpha_low[-1], alpha_high[-1]))
                    a_low_pow = np.mean(alpha_low[:, :1])
                    a_high_pow = np.mean(alpha_high[:, :1])
                    m = (a_low_pow + a_high_pow) / 2.0
                    # print("alpha_low.shape: {}".format(alpha_low[:, -1].shape))
                    # print("alpha_low: {} and alpha_high: {}".format(np.mean(alpha_low[:, -1]), np.mean(alpha_high[:, -1])))
                    # print("alpha power: {:.2E}".format(m))

                    # print("{:.2E}".format(m))


                    # print("{:.2f}".format(m*100))

                    eye_status = "open"
                    if m >= alpha_threshold:
                        quantum_angle = min(quantum_angle + 30.0, 180.0)
                        eye_status = "closed"
                    else:
                        quantum_angle = max(quantum_angle - 30.0, 0.0)
                        eye_status = "closed"

                    # os.system("cls")
                    # print("Angle: {}{}".format(quantum_angle, u"\u00b0"))

                    data_line = "{:.5f},{}\n".format(m, eye_status)
                    print(data_line)

                    file_alpha.write(data_line)


                    # if m >= 1.5e-2:
                    #     print("Closed")
                    # else:
                    #     print("Open")


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