
import sys
import time
import qiskit
import collections
import numpy as np
import pyqtgraph as pg
import pyqtgraph.parametertree as ptree

from dataclasses import dataclass
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
from pyqtgraph.ptime import time
from pyqtgraph.graphicsItems.ScatterPlotItem import _USE_QRECT








@dataclass
class Qubit:
    """Our qubit object, 
    """    
    phi: float = 0
    theta: float = 0

    def __post_init__(self):
        assert (self.phi >= 0) and (self.phi <= 2*np.pi), 'phi must be in range [0, 2pi]'
        assert (self.theta >= 0) and (self.theta <= np.pi), 'theta must be in range [0, pi]'
    
    def __eq__(self, target_state):
        # checks if a qubit is equal or close enough to another
        qubit_original = self.to_qiskit()
        qubit_target = target_state.to_qiskit()
        fidelity = qiskit.quantum_info.state_fidelity(qubit_original, qubit_target)
        return fidelity > 0.9

    def to_qiskit(self):
        #qc = QuantumCircuit(1)  # Create a quantum circuit with one qubit
        initial_state = [np.cos(0.5*self.theta),np.exp(1j*self.phi)*np.sin(0.5*self.theta)]   # Define initial_state as |1>
        return qiskit.quantum_info.Statevector(initial_state)#qc.initialize(initial_state, 0)
    
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





@dataclass
class bloch_visualization:

    qubit: Qubit = None
    debug: bool = True

    def __post_init__(self):
        # Must construct a QApplication before a QWidget
        self.app = pg.mkQApp()
        # Block I: Defining parameters for the sliders
        translate = QtCore.QCoreApplication.translate
        
        self.param = ptree.Parameter.create(
            name=translate('ScatterPlot', 'Parameters'),
            type='group',
            children=[
                dict(name='paused', title=translate('ScatterPlot', 'Paused:    '), type='bool', value=False),
                dict(name='_USE_QRECT', title='_USE_QRECT:    ', type='bool', value=_USE_QRECT)
            ]
        )
        (c.setDefault(c.value()) for c in self.param.children())

        
        # Block II: Setting up the zones/widgets
        splitter = QtWidgets.QSplitter()
        # II.1: Left side of the screen, parameter's sliders
        pt = ptree.ParameterTree(showHeader=False)
        pt.setParameters(self.param) # adds the default parameters to the sliders
        # II.2: Right side of the screen, plotting zone
        self.plt = pg.PlotWidget()
        self.plt.setRange(xRange=[0, 2*np.pi], yRange=[0, np.pi])
        splitter.addWidget(pt)
        splitter.addWidget(self.plt)
        splitter.show()

        # Block III: Setting up data-related parameters
        self.item = pg.ScatterPlotItem()
        self.ptr = 0 # index of the data used for the plotting
        self.lastTime = time()
        self.fps = None
        self.timer = QtCore.QTimer()

        self._testData()

        self.param.child('paused').sigValueChanged.connect(lambda _, v: self.timer.stop() if v else self.timer.start())
        self.timer.timeout.connect(self._update)
        self.timer.start(0)

        try:
            self.app.exec_()
        except IndexError:
            sys.exit(1)
        print("done")

    def _mkItem(self):
        _USE_QRECT = self.param['_USE_QRECT']
        self.item = pg.ScatterPlotItem(pxMode=True, **self._getData())
        self.item.opts['useCache'] = True#param['useCache']
        self.plt.clear()
        self.plt.addItem(self.item)

    def _testData(self):
        scale = 0.01*np.pi
        loc = np.pi/2
        if self.debug:
            self.data = np.random.normal(size=(50, 2), scale=scale, loc=loc)
        else:
            # To Satvik: Here you should either use pass or input the data
            # self.data = np.array([[self.qubit.phi,self.qubit.theta]]) # np.array with shape (1,2)
            pass
        self._mkItem()



    def _getData(self):
        if self.debug:
            # This loops through self.data, offline
            dataReturn = dict(x=self.data[self.ptr % 50], y=self.data[(self.ptr + 1) % 50])
        else:
            # This takes only the last instance, online
            dataReturn = dict(x=[self.data[-1,0]], y=[self.data[-1,1]])
        return dataReturn
    
    def _getfps(self):
        now = time()
        dt = now - self.lastTime
        self.lastTime = now
        if self.fps is None:
            self.fps = 1.0/dt
        else:
            s = np.clip(dt*3., 0, 1)
            self.fps = self.fps*(1-s)+(1.0/dt)*s
    


    
    def _update(self):
        self.item.setData(**self._getData())
        self.ptr += 1
        self._getfps()
        self.plt.setTitle('%0.2f fps' % self.fps)
        self.plt.repaint()

bloch_visualization(Qubit(1,1), debug=True)