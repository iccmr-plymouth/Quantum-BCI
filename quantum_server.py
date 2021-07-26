# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:02:31 2021

@author: satvi
"""

import zmq

import numpy as np
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.bloch import Bloch
from qiskit import QuantumCircuit
from dataclasses import dataclass
import matplotlib.pyplot as plt
import os


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


def main(qubit, realtime=True, model=None, data=None):
    global should_exit
    global pending_bci_update
    global mind_status
    
    fig = plt.figure()
    B = Bloch(fig)
    bloch = [0,0,0]

    r, theta, phi = 1, qubit.theta, qubit.phi
    bloch[0] = r*np.sin(theta)*np.cos(phi)
    bloch[1] = r*np.sin(theta)*np.sin(phi)
    bloch[2] = r*np.cos(theta)
    B.add_vectors(bloch)
    B.render(title='1-qubit Bloch Sphere')
    plt.pause(1.0)
    plt.draw()

def parse_morse_code(n, current_angle):
    if (n == np.array([0, 1])).all():
        command = "start-programme"
    
    elif (n == np.array([1, 0])).all():
        if current_angle == 0:
            current_angle += 1
            command = "change-angle"

        elif current_angle == 1:
            command = "end-programme"

    elif (n == np.array([1, 1])).all():
        command = "increase-angle"

    elif (n == np.array([0, 0])).all():
        command = "decrease-angle"
    
    else:
        command = "unknown"
    
    return command, current_angle


if __name__ == '__main__':
    socket = zmq.Context(zmq.REP).socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    socket.connect('tcp://127.0.0.1:1234')

    qubit = Qubit(phi=0, theta=0.5*np.pi)# default values 0, 0
    # main(qubit, realtime=True)
    

    # plt.draw()
    # B.clear()
    # fig.clear()
    
    morse_code = np.array([-1, -1])
    command = "unknown"
    
    has_started = False
    current_angle = 0
    os.system("cls")

    while command != "end-programme":
        morse_code = socket.recv_pyobj()
        command, current_angle = parse_morse_code(morse_code, current_angle)
        
        if command == "start-programme" and not has_started:
            fig = plt.figure()
            B = Bloch(fig)
            bloch = [0,0,0]
        
            B.clear()
            r, theta, phi = 1, qubit.theta, qubit.phi
            bloch[0] = r*np.sin(theta)*np.cos(phi)
            bloch[1] = r*np.sin(theta)*np.sin(phi)
            bloch[2] = r*np.cos(theta)
            B.add_vectors(bloch)
            plt.pause(0.01)
            B.render(title='1-qubit Bloch Sphere')

            print("Morse code: {}, Command: {}\n".format(morse_code, command))            
            plt.pause(2.0)
            has_started = True
            
        elif has_started:
            
            if command == "increase-angle" and current_angle == 0:                
                qubit.control(dphi=+4e-2*np.pi, dtheta=0)
            elif command == "decrease-angle" and current_angle == 0:                
                qubit.control(dphi=-4e-2*np.pi, dtheta=0)
    
            elif command == "increase-angle" and current_angle == 1:                
                qubit.control(dphi=0, dtheta=+4e-2*np.pi)
            elif command == "decrease-angle" and current_angle == 1:                
                qubit.control(dphi=0, dtheta=-4e-2*np.pi)
                
            
            B.clear()
            r, theta, phi = 1, qubit.theta, qubit.phi
            bloch[0] = r*np.sin(theta)*np.cos(phi)
            bloch[1] = r*np.sin(theta)*np.sin(phi)
            bloch[2] = r*np.cos(theta)
            B.add_vectors(bloch)
            # B.plot_vectors()
            # B.arr.
            # plt.pause(0.01)
            B.render()        
            
            print("Morse code: {}, Command: {}\n".format(morse_code, command))
            plt.pause(2.0)
            
        else:
            print("Morse code: {}, Invalid-command\n".format(morse_code))
            
