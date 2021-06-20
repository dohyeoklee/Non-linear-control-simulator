# -*- coding: utf-8 -*-
'''
This module is a set of utility function

Notation
-------
    mainly refer to main.py's docstring

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dynamics import Kinematics

def angle_normalize(theta):
    '''
    Returns normailzed angle, -pi to pi
    
    Parameters
    ----------
    theta : float
        angle in radian
    
    Returns
    -------
    normaized theta : float
        angle restricted -pi to pi, in radian
    
    Example
    -------
    >>> theta = 2*np.pi + 0.01
    >>> angle_normalize(theta)
    0.009999999999999787 (which equals 0.01)

    '''
    return (((theta+np.pi) % (2*np.pi)) - np.pi)

def desired_set_generator(t,option="regulation"):
    '''
    Returns set of desired state, chose by command line option
    There are three control problem model,
    -regulation
    want to converge q1 to 0.5[rad], q2 to 0.6[rad]

    -tracking parabolic function
    want to follow q1 to 1/2*0.1*t^2, q2 to 1/2*0.3*t^2

    -tracking sine function
    want to follow q1 to 0.5*sin(t), converge q2 to np.pi/3
    
    Parameters
    ----------
    t : 1 x int(t) list
        time linspace list, 0 to t_max, each component difers dt(=1/f) with adjacent component

    option : string
        choose which problem model to solve, determine which set of desired state to return
    
    Returns
    -------
    q_desired_set : int(t) x 6 np.array
        array of list (q1_desired,q2_desired,q1_desired_dot,q2_desired_dot,q1_desired_ddot,q2_desired_ddot) along series of time t
    
    Example
    -------
    >>> t = [0,0.1,0.2,0.3]
    >>> desired_set_generator(t,option)
    array([[0.5, 0.6, 0. , 0. , 0. , 0. ],
       [0.5, 0.6, 0. , 0. , 0. , 0. ],
       [0.5, 0.6, 0. , 0. , 0. , 0. ],
       [0.5, 0.6, 0. , 0. , 0. , 0. ]])

    '''
    if option == "regulation":
        q1_desired = [0.5 for _ in t]
        q2_desired = [0.6 for _ in t]
        q1_desired_dot = [0 for _ in t]
        q2_desired_dot = [0 for _ in t]
        q1_desired_ddot = [0 for _ in t]
        q2_desired_ddot = [0 for _ in t]
        q_desired_set = np.array([d_set for d_set in zip(q1_desired,q2_desired,q1_desired_dot,q2_desired_dot,q1_desired_ddot,q2_desired_ddot)])

    elif option == "tracking parabolic function":
        quad_form_a = np.array([[0.1],[0.3]])
        q1_desired = [1/2*quad_form_a[0,0]*t_i**2 for t_i in t]
        q2_desired = [1/2*quad_form_a[1,0]*t_i**2 for t_i in t]
        q1_desired_dot = [quad_form_a[0,0]*t_i for t_i in t]
        q2_desired_dot = [quad_form_a[1,0]*t_i for t_i in t]
        q1_desired_ddot = [quad_form_a[0,0] for _ in t]
        q2_desired_ddot = [quad_form_a[1,0] for _ in t]
        q_desired_set = np.array([d_set for d_set in zip(q1_desired,q2_desired,q1_desired_dot,q2_desired_dot,q1_desired_ddot,q2_desired_ddot)])

    elif option == "tracking sine function":
        q1_desired = [0.5*np.sin(t_i) for t_i in t]
        q2_desired = [np.pi/3 for t_i in t]
        q1_desired_dot = [0.5*np.cos(t_i) for t_i in t]
        q2_desired_dot = [0 for _ in t]
        q1_desired_ddot = [-0.5*np.sin(t_i) for t_i in t]
        q2_desired_ddot = [0 for _ in t]
        q_desired_set = np.array([d_set for d_set in zip(q1_desired,q2_desired,q1_desired_dot,q2_desired_dot,q1_desired_ddot,q2_desired_ddot)])
        
    return q_desired_set

def show_result(q1_list,q2_list,q1_desired,q2_desired,q1_e,q2_e,t,dt,enable_animation,algorithm_option,problem_option):
    '''
    plot and animate simulation result

    Parameters
    ----------
    q1_list : 1 x int(t) list
        list of controlled and simulated q1

    q2_list : 1 x int(t) list
        list of controlled and simulated q2

    q1_desired : 1 x int(t) list
        list of desired q1, which generated by function desired_set_generator(t,option)

    q2_desired : 1 x int(t) list
        list of desired q2, which generated by function desired_set_generator(t,option)

    q1_e : 1 x int(t) list
        list of error(=q1_simulated - q1_desired) of q1

    q2_e : 1 x int(t) list
        list of error(=q2_simulated - q2_desired) of q2

    t : 1 x int(t) list
        time linspace list, 0 to t_max, each component difers dt(=1/f) with adjacent component

    dt : float
        difference between each sampled time

    enable_animation : string
        option to enable task space animation

    algorithm_option : string
        option for title plots about control algorithm

    problem_option : string
        option for title plots about control problem model

    Returns
    -------
    None
    
    '''
    plt.figure(1)
    line1,=plt.plot(t,q1_list,'r-')
    line2,=plt.plot(t,q2_list,'b-')
    line1_d,=plt.plot(t,q1_desired,'g--')
    line2_d,=plt.plot(t,q2_desired,'c--')
    plt.legend(handles=(line1,line2,line1_d,line2_d),labels=("q1","q2","q1 desired","q2 desired"))
    plt.xlabel("time[sec]")
    plt.ylabel("theta[rad]")
    plt.title(problem_option + " with " + algorithm_option + " control")
    
    plt.figure(2)
    line3,=plt.plot(t,q1_e,'r-')
    line4,=plt.plot(t,q2_e,'b-')
    line0,=plt.plot(t,[0 for _ in range(len(q1_e))],'g--')
    plt.legend(handles=(line3,line4,line0),labels=("q1_error","q2_error"))
    plt.xlabel("time[sec]")
    plt.ylabel("theta[rad]")
    plt.title("regulation error" + " with " + algorithm_option + " control")

    if problem_option == "regulation":
        plt.title("regulation error" + " with " + algorithm_option + " control")
    else:
        plt.title("tracking error" + " with " + algorithm_option + " control")

    if enable_animation:
        x1,y1,x2,y2 = Kinematics().jacobian(q1_list,q2_list)

        fig = plt.figure(3)
        x_lim = Kinematics.L_1 + Kinematics.L_2
        y_lim = Kinematics.L_1 + Kinematics.L_2
        ax = fig.add_subplot(111,xlim=[-x_lim,x_lim], ylim=[-y_lim,y_lim])
        plt.title("task space simulation" + " with " + algorithm_option + " control")
        ax.grid()

        line5, = ax.plot([], [], 'o-', lw=2)
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

        def init():
            line5.set_data([], [])
            time_text.set_text('')
            return line5, time_text

        def animate(i):
            thisx = [0, x1[i], x2[i]]
            thisy = [0, y1[i], y2[i]]

            line5.set_data(thisx, thisy)
            time_text.set_text(time_template % (i*dt))
            return line5, time_text

        ani = animation.FuncAnimation(fig, animate, np.arange(1, len(t)),\
            interval=1, blit=True, init_func=init)

    plt.show()