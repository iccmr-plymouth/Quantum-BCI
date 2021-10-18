import time
import pygame
import numpy as np
import pandas as pd
import qiskit
from qiskit.visualization import plot_bloch_vector
from qiskit.visualization.bloch import Bloch
from qiskit import QuantumCircuit
from matplotlib import get_backend
import matplotlib.pyplot as plt
from dataclasses import dataclass
from joblib import load


# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def get_df(input_file):
    #input_file = '../' + input_file
    #df = pd.read_csv(input_file, header=None, names= ['alpha', 'beta', 'gamma', 'theta', 'mind_status', 'ms_label'])
    #df = df[df.ms_label != -1]
    #X = df.loc[:, ['alpha', 'beta', 'gamma', 'theta']]
    #y = df.loc[:, 'ms_label']
    df = pd.read_csv(input_file)
    X = df.iloc[:, 0:32]
    y = df.label
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


def main(qubit, realtime=True, model=None, data=None):
    if model is not None:
        mod = load(model)
        print(mod)
    if data is not None:
        X, y, df = get_df(data) #'oc2.csv'
        preds = mod.predict(X)
    pygame.init()
    # Set the width and height of the screen [width,height]
    size = [700, 300]
    screen = pygame.display.set_mode(size)
    
    pygame.display.set_caption("CLICK IN THE WHITE AREA to control 1-qubit Bloch Sphere")
    
    # Loop until the user clicks the close button.
    done = False
    
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()
    
    fig = plt.figure()
    B = Bloch(fig)
    bloch = [0,0,0]

    # Get ready to print
    textPrint = TextPrint()

    # -------- Main Program Loop -----------
    if realtime:
        # times for the small time_window to average results so that we don't
        # introduce fast variations when moving the angle in the qubit
        t = 0 # from 0 to end of predictions, preds
        t_max = 10 # time-step
        time_window = []
        # times for the 3-step process of moving phi, theta and measuring.
        # this requires the time_windows to be much larger so a human is able to
        # react in time.
        t3 = 0
        t3_max = 1*250 # 3 seconds for each angle and 250 Hz sampling frequency gives us 3*250 time-steps
        t3_step = 0 # initial step, 0 for phi, 1 for theta and 2 for measuring
        measure_voting = [] # for the third step we need to store values to decide wether to measure or not

        while not done: # real time
            # EVENT PROCESSING STEP
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            screen.fill(WHITE)
            textPrint.reset()

            r, theta, phi = 1, qubit.theta, qubit.phi
            bloch[0] = r*np.sin(theta)*np.cos(phi)
            bloch[1] = r*np.sin(theta)*np.sin(phi)
            bloch[2] = r*np.cos(theta)
            B.add_vectors(bloch)
            B.render(title='1-qubit Bloch Sphere')
            plt.pause(0.01)
            plt.draw()
            B.clear()
            fig.clear()

            # To Satvik: There must be a way to feed the data from the headset
            # to the model. I will leave that part to you as I dont have one.
            # Meanwhile I will feed already collected data.
            # Don't hesitate to ask me if you need help with my code.
            
            if len(time_window) < t_max: # list will continously grow until it reaches t_max
                time_window.append(preds[t])
            else: # one goes in one goes out so that len(time_window) keeps being t_max
                time_window.pop(0)
                time_window.append(preds[t])
                # predictions are 0 or 1, so 1 means angle increases and 0 decreases
                variation = sum(map(lambda x: (-1)**(x-1), time_window))
                # 0 maps to -1 and and 1 to 1, the sum is the total variation
                if t3_step == 0:
                    # phi moving
                    # 1e-2 so variation is not huge, np.pi for units and variation for user input
                    qubit.control(dphi=1e-2*np.pi*variation, dtheta=0)
                elif t3_step == 1:
                    # theta moving
                    qubit.control(dphi=0, dtheta=1e-2*np.pi*variation)
                else:
                    # measuring
                    measure_voting.append(preds[t])
                    if len(measure_voting) == t3_max: # when it reaches its full length, we vote:
                        voting = sum(measure_voting)
                        if voting > t3_max/2:
                            measure = True
                            textPrint.print(screen, "MEASURE")
                        else:
                            measure = False
                            textPrint.print(screen, "NOT MEASURE")
                        measure_voting = [] # resets the voting list for the next iteration

                if t3 == t3_max - 1:
                    print(t3)
                    t3_step = (t3_step + 1) % 3 # t3_step = 0,1,2,0,1,2,...
            t += 1 # position of the prediction
            textPrint.print(screen, str(t))
            textPrint.print(screen, str(t3))
            t3 = (t3 + 1) % t3_max # once t3 reaches t3_max-1, it goes back to zero
            


            #if pygame.key.get_pressed()[pygame.K_LEFT] == True:
            #    qubit.control(dphi=-1e-3*np.pi, dtheta=0)
            #    print(f'key left pressed, phi: {qubit.phi:.3f}')
            #if pygame.key.get_pressed()[pygame.K_RIGHT] == True:
            #    qubit.control(dphi=+1e-3*np.pi, dtheta=0)
            #    print(f'key right pressed, phi: {qubit.phi:.3f}')  
            #if pygame.key.get_pressed()[pygame.K_UP] == True:
            #    qubit.control(dphi=0, dtheta=+1e-3*np.pi)
            #    print(f'key up pressed, theta: {qubit.theta:.3f}')
            #if pygame.key.get_pressed()[pygame.K_DOWN] == True:
            #    qubit.control(dphi=0, dtheta=-1e-3*np.pi)
            #    print(f'key down pressed, theta: {qubit.theta:.3f}')
            
            steps = ['PHI_MOVING', 'THETA_MOVING', 'MEASURING']
            textPrint.print(screen, "STEP: {}".format(steps[t3_step]))
            textPrint.print(screen, "phi: {:.3f}".format(qubit.phi))
            textPrint.print(screen, "theta: {:.3f}".format(qubit.theta))
            textPrint.print(screen, "r: {:.3f}".format(r))
            textPrint.print(screen, "x: {:.3f}".format(bloch[0]))
            textPrint.print(screen, "y: {:.3f}".format(bloch[1]))
            textPrint.print(screen, "z: {:.3f}".format(bloch[2]))
            
            #xvar, yvar = joystick.get_axis(3), joystick.get_axis(1)
            #pygame.draw.circle(screen, color=BLACK, center=(round(250+xvar*200), round(350+yvar*200)), radius=5)

            # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT
        
            # Go ahead and update the screen with what we've drawn.
            pygame.display.flip()
        
            # Limit to 60 frames per second, should change later with sample rate
            clock.tick(250)
    else:
        i=0
        imax = preds.shape[0]
        while i < imax and not done:
            # EVENT PROCESSING STEP
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

            screen.fill(WHITE)
            textPrint.reset()

            r, theta, phi = 1, qubit.theta, qubit.phi
            bloch[0] = r*np.sin(theta)*np.cos(phi)
            bloch[1] = r*np.sin(theta)*np.sin(phi)
            bloch[2] = r*np.cos(theta)
            B.add_vectors(bloch)
            B.render(title='1-qubit Bloch Sphere')
            plt.pause(0.01)
            plt.draw()
            B.clear()
            fig.clear()

            # if pygame.key.get_pressed()[pygame.K_LEFT] == True:
            #     qubit.control(dphi=-1e-3*np.pi, dtheta=0)
            #     print(f'key left pressed, phi: {qubit.phi:.3f}')
            # if pygame.key.get_pressed()[pygame.K_RIGHT] == True:
            #     qubit.control(dphi=+1e-3*np.pi, dtheta=0)
            #     print(f'key right pressed, phi: {qubit.phi:.3f}')
            if preds[i] == 0:
                qubit.control(dphi=-1e-3*np.pi, dtheta=0)
            if preds[i] == 1:
                qubit.control(dphi=+1e-3*np.pi, dtheta=0)
            if pygame.key.get_pressed()[pygame.K_UP] == True:
                qubit.control(dphi=0, dtheta=+1e-3*np.pi)
                print(f'key up pressed, theta: {qubit.theta:.3f}')
            if pygame.key.get_pressed()[pygame.K_DOWN] == True:
                qubit.control(dphi=0, dtheta=-1e-3*np.pi)
                print(f'key down pressed, theta: {qubit.theta:.3f}')
                
            textPrint.print(screen, "phi: {:.3f}".format(qubit.phi))
            textPrint.print(screen, "theta: {:.3f}".format(qubit.theta))
            textPrint.print(screen, "r: {:.3f}".format(r))
            textPrint.print(screen, "x: {:.3f}".format(bloch[0]))
            textPrint.print(screen, "y: {:.3f}".format(bloch[1]))
            textPrint.print(screen, "z: {:.3f}".format(bloch[2]))
                
                #xvar, yvar = joystick.get_axis(3), joystick.get_axis(1)
                #pygame.draw.circle(screen, color=BLACK, center=(round(250+xvar*200), round(350+yvar*200)), radius=5)

                # ALL CODE TO DRAW SHOULD GO ABOVE THIS COMMENT
            
                # Go ahead and update the screen with what we've drawn.
            pygame.display.flip()
            
                # Limit to 60 frames per second
            clock.tick(20)
            i = i+1
            # Close the window and quit.
            # If you forget this line, the program will 'hang'
            # on exit if running from IDLE.
    pygame.quit()


if __name__ == '__main__':
    qubit = Qubit(phi=0.5*np.pi, theta=0.5*np.pi)# default values 0, 0
    qubit2 = Qubit(phi=0.499*np.pi, theta=0.495*np.pi)
    print(qubit==qubit2)
    #main(qubit, realtime=True, model="mod_knn_oc.joblib",data='./reports/df_oc.csv')

    