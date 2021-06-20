# -*- coding: utf-8 -*-
'''
This module demonstrates how non-linear control algorithm works for two-arm manipulator

Example
-------

    $ python main.py -a -c="passivity adaptive" -m="tracking sine function"

Notes
-------
    there are 3 command line option
    -a, --animation: default False, if True, enable task space animation
    -c: default "inverse dynamics", can choose which algorithm to control
    -m: default "regulation", can choose which problem model to solve

Notation
-------
    two arm manipulator has two controllable variable, q1 and q2
    q1 is angle between x-axis and first arm(fixed link to non-fixed second link), in radian
    q2 is angle between first arm and second arm(non-fixed second link to end effector), in radian
    suffix _1 means something about first arm
    suffix _2 means something about second arm
    suffix _dot means first derivative in time of that vector 
    suffix _ddot means second derivative in time of that vector 
    suffix _0 means initial state of that vector
    suffix _list,_set means memory of that vectors
    suffix _desired means desired state of that vector
    suffix _e,_tilda means error between desired state and current state of that vector, (·)_tilda = (·) − (·)_desired

'''

import numpy as np
import argparse

from control import Control
from dynamics import Kinematics,Dynamics
from utility import angle_normalize, desired_set_generator, show_result

parser = argparse.ArgumentParser()
parser.add_argument("-a","--animation",action="store_true", help="enable animation")
parser.add_argument("-c", action="store", \
    choices=["inverse dynamics","robust","passivity robust","passivity adaptive","sliding mode"], \
    dest="control_algorithm", type=str, help="choose control algorithm")
parser.add_argument("-m", action="store", \
    choices=["regulation","tracking parabolic function","tracking sine function"], \
    dest="problem_model", type=str, help="choose control problem model")
args = parser.parse_args()

if __name__ == '__main__':
    '''
    Notation
    -------
    sampling frequency: f = 1kHz
    simulation time: t_max = 20sec
    time list: t_list, 1 x int(t_max*f) list
    sampling time: dt = 1/f = 1ms

    state: X = [q1 q2 q1_dot q2_dot], 4 x 1 vector
    initial state: [pi/3 pi/6 0 0], 4 x 1 vector
    state memory(to plot, only for angle): q1_list, q2_list, 1 x len(t_list) list
    desired state memory(to control and plot): q_desired_set, q1_desired_set, q2_desired_set, 1 x len(t_list) list
    error memory(to plot): q1_e_list, q2_e_list, 1 x len(t_list) list

    Pesudo code
    -------
    initialization for state X
    for t=1:t_max
        dynamics update with current state X
        control input calculation for desired state
        simulate next state X with current state X and control input tau
        save simulate result
    plot and animate result

    '''
    f = 1e3
    t_max = 20
    t_list = np.linspace(0,t_max,int(t_max*f))
    dt = 1/f

    X = np.mat([[np.pi/3],[np.pi/6],[0],[0]])

    q1_list = [X[0,0]]
    q2_list = [X[1,0]]

    q_desired_set = desired_set_generator(t_list,args.problem_model)
    q1_desired_set = [angle_normalize(q1_d) for q1_d in q_desired_set[:,0]]
    q2_desired_set = [angle_normalize(q2_d) for q2_d in q_desired_set[:,1]]

    q1_e_list = [q_desired_set[:,0][0]-X[0,0]]
    q2_e_list = [q_desired_set[:,1][0]-X[1,0]]

    control = Control(dt,option=args.control_algorithm)
    kinematics = Kinematics()
    theta = kinematics.theta_nominal()

    for i in range(len(t_list)-1):
        dyn = Dynamics(X,dt)
        
        tau,theta = control.control(dyn,X,q_desired_set[i+1],theta)

        X = dyn.dyn_update(tau)
        
        q1_list.append(angle_normalize(X[0,0]))
        q2_list.append(angle_normalize(X[1,0]))

        q1_e_list.append(angle_normalize(q_desired_set[:,0][i+1]-X[0,0]))
        q2_e_list.append(angle_normalize(q_desired_set[:,1][i+1]-X[1,0]))

    show_result(q1_list,q2_list,q1_desired_set,q2_desired_set,q1_e_list,q2_e_list,t_list,dt,args.animation,args.control_algorithm,args.problem_model)