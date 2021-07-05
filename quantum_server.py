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

if __name__ == '__main__':
    socket = zmq.Context(zmq.REP).socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    socket.connect('tcp://127.0.0.1:1234')

    qubit = Qubit(phi=0, theta=0.5*np.pi)# default values 0, 0
    # main(qubit, realtime=True)
    
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

    # plt.draw()
    # B.clear()
    # fig.clear()
    
    mind_status = ""
    plt.pause(2.0)
    
    while mind_status != "end":
        mind_status = socket.recv_pyobj()

        if mind_status == "relaxed":                
            qubit.control(dphi=+1e-2*np.pi, dtheta=0)
        elif mind_status == "aroused":                
            qubit.control(dphi=-1e-2*np.pi, dtheta=0)
            
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
        
        print(mind_status)
        plt.pause(2.0)
