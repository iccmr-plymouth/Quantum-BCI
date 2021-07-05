# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 09:02:31 2021

@author: satvi
"""

import zmq

socket = zmq.Context(zmq.REP).socket(zmq.SUB)
socket.setsockopt_string(zmq.SUBSCRIBE, '')
socket.connect('tcp://127.0.0.1:1234')

mind_status = ""
while mind_status != "end":
    mind_status = socket.recv_pyobj()
    print(mind_status)
