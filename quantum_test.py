import warnings
warnings.filterwarnings("ignore")

import time
import pygame
import numpy as np
import pandas as pd
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.bloch import Bloch
from qiskit import QuantumCircuit
from matplotlib import get_backend
import matplotlib.pyplot as plt
from dataclasses import dataclass
from joblib import load

import UnicornPy
from scipy import signal
import os
import threading
from biosppy.signals.eeg import eeg
from biosppy.signals.eeg import get_power_features

import tkinter

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


# Define some globals
should_exit = False
pending_bci_update = False
mind_status = "unknown"
bci_exited = False
q_exited = False


def bci_thread():
    global pending_bci_update
    global mind_status
    global qubit

    """
    This function is a thread
    """
    # Specifications for the data acquisition.
    #-------------------------------------------------------------------------------------
    global bci_exited

    TestsignaleEnabled = False;
    FrameLength = 1;
    AcquisitionDurationInSeconds = 20;
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

                # This deletes the oldest EEG sample and inserts the newest sample. In other words, it acts a FIFO queue.
                data_block = np.roll(data_block, -1, axis=0)
                data_block[-1, :] = data[:, 0:8]

                np.savetxt(file,data,delimiter=',',fmt='%.3f',newline='\n')
                

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

                    
                    # while pending_bci_update:
                    #     time.sleep(0.01)

                    if avg_alpha_pow > alpha_threshold:
                        mind_status = "relaxed"
                        qubit.control(dphi=+1e-2*np.pi, dtheta=0)
                    else:
                        mind_status = "aroused"
                        qubit.control(dphi=-1e-2*np.pi, dtheta=0)

                    pending_bci_update = True

                    data_line = "{:.5f}, {}\n".format(avg_alpha_pow, mind_status)
                    print(data_line)

                    file_alpha.write(data_line)



            # Stop data acquisition.
            #-------------------------------------------------------------------------------------
            device.StopAcquisition();
            print()
            print("Data acquisition stopped.");

        except UnicornPy.DeviceException as e:
            print("It is a device exception!")
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

    except UnicornPy.DeviceException as e:
        print(e)
    except Exception as e:
        print("An unknown error occured. %s" %e)

    # input("\n\nPress ENTER key to exit")
    bci_exited = True


def get_df(input_file):
    #input_file = '../' + input_file
    df = pd.read_csv(input_file, header=None, names= ['alpha', 'beta', 'gamma', 'theta', 'mind_status', 'ms_label'])
    df = df[df.ms_label != -1]
    X = df.loc[:, ['alpha', 'beta', 'gamma', 'theta']]
    y = df.loc[:, 'ms_label']
    return X, y, df


@dataclass
class Qubit:
    """Our qubit object, 
    """    
    phi: float = 0
    theta: float = 0

    def __post_init__(self):
        assert (self.phi >= 0) and (self.phi <= 2*np.pi), 'phi must be in range [0, 2pi]'
        assert (self.theta >= 0) and (self.theta <= np.pi), 'theta must be in range [0, pi]'
    
    def to_qiskit(self):
        qc = QuantumCircuit(1)  # Create a quantum circuit with one qubit
        initial_state = [np.cos(0.5*self.theta),np.exp(1j*self.phi)*np.sin(0.5*self.theta)]   # Define initial_state as |1>
        return qc.initialize(initial_state, 0)
    
    def control(self, dphi=1e-4, dtheta=1e-4):
        self.phi += dphi*np.pi
        self.theta += dtheta*np.pi
        if self.phi > 2*np.pi:
            self.phi = 2*np.pi
        if self.phi < 0:
            self.phi = 0
        if self.theta > np.pi:
            self.theta = np.pi
        if self.theta < 0:
            self.theta = 0


class TextPrint(object):
    """
    This is a simple class that will help us print to the screen
    It has nothing to do with the joysticks, just outputting the
    information.

    Sample Python/Pygame Programs
    Simpson College Computer Science
    http://programarcadegames.com/
    http://simpson.edu/computer-science/
    """
    def __init__(self):
        """ Constructor """
        self.reset()
        self.x_pos = 10
        self.y_pos = 10
        self.font = pygame.font.Font(None, 20)
 
    def print(self, my_screen, text_string):
        """ Draw text onto the screen. """
        text_bitmap = self.font.render(text_string, True, BLACK)
        my_screen.blit(text_bitmap, [self.x_pos, self.y_pos])
        self.y_pos += self.line_height
 
    def reset(self):
        """ Reset text to the top of the screen. """
        self.x_pos = 10
        self.y_pos = 10
        self.line_height = 15
 
    def indent(self):
        """ Indent the next line of text """
        self.x_pos += 10
 
    def unindent(self):
        """ Unindent the next line of text """
        self.x_pos -= 10




def run_qubit(qubit, realtime=True, model=None, data=None):
    global should_exit
    global pending_bci_update
    global mind_status
    global q_exited
    
    if model is not None:
        mod = load(model)
        print(mod)
    if data is not None:
        X, y, df = get_df(data) #'oc2.csv'
        preds = mod.predict(X)
    
    # Loop until the user clicks the close button.
    done = False
    
    # Used to manage how fast the screen updates    
    fig = plt.figure()
    B = Bloch(fig)
    bloch = [0,0,0]


    # -------- Main Program Loop -----------
    while not should_exit: # real time
        # EVENT PROCESSING STEP

        r, theta, phi = 1, qubit.theta, qubit.phi
        bloch[0] = r*np.sin(theta)*np.cos(phi)
        bloch[1] = r*np.sin(theta)*np.sin(phi)
        bloch[2] = r*np.cos(theta)
        B.add_vectors(bloch)
        B.render(title='1-qubit Bloch Sphere')
        plt.pause(0.1)
        plt.draw()
        B.clear()
        fig.clear()
        
    q_exited = True

def qubit_thread():
    """
    This function is a thread
    """
    run_qubit(qubit, realtime=True)

def on_close():
    global should_exit

    if not should_exit:
        should_exit = True

    while not bci_exited or not q_exited:
        time.sleep(1.0)
        continue
    
    print("Going to destroy!!")
    # top.quit()
    # top.destroy()   

qubit = Qubit(phi=0, theta=0.5*np.pi)# default values 0, 0

def on_key_press(event):
    print("you pressed {}".format(event.key))
    key_press_handler(event, canvas, toolbar)




def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


root = tkinter.Tk()
root.wm_title("Embedding in Tk")

fig = Figure(figsize=(5, 4), dpi=100)
# t = np.arange(0, 3, .01)
# fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

B = Bloch(fig)
bloch = [0,0,0]
B.add_vectors(bloch)
B.render(title='1-qubit Bloch Sphere')



canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


canvas.mpl_connect("key_press_event", on_key_press)


button = tkinter.Button(master=root, text="Quit", command=_quit)
button.pack(side=tkinter.BOTTOM)

tkinter.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.

