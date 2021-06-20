# -*- coding: utf-8 -*-
'''
This module describes the application of each non-linear control algorithm for two-arm manipulator

Notation
-------
    mainly refer to main.py's docstring and,
    notation of dynamics and control is mainly refer to:
    "Robot Dynamics and Control, 2nd edition, Mark W.Spong, Seth Hutchinson, and M. Vidyasagar"

'''

import numpy as np
from scipy import linalg

class Control(object):
    '''
    Describe about non-linear control algorithms for two-arm manipulator

    Parameters
    ----------
    dt : float
        difference between each sampled time

    option : string
        name of control algorithm to use by command line option

    Attributes
    ----------
    option_selector : function pointer
        reference of function for selected control algorithm

    dt : float
        difference between each sampled time

    '''
    def __init__(self,dt,option="inverse dynamics"):
        if option == "inverse dynamics":
            self.option_selector = self.inverse_dynamics_control
        elif option == "robust":
            self.option_selector = self.robust_control
        elif option == "passivity robust":
            self.option_selector = self.passivity_based_robust_control
        elif option == "passivity adaptive":
            self.option_selector = self.passivity_based_adaptive_control
        elif option == "sliding mode":
            self.option_selector = self.sliding_mode_control

        self.dt = dt

    def control(self,dyn,X,d_set,theta):
        '''
        Returns return of selected control algorithm function
    
        Parameters
        ----------
        dyn : class instance
            instance of class Dynamics() in dynamics.py

        X : 4 x 1 vector
            state vector at certain time t

        d_set : 1 x 6 list
            desired state at certain time t

        theta : 5 x 1 vector
            regression parameter vector
        
        Returns
        -------
        tau : 2 x 1 vector
            control input
            
        Example
        -------
        >>> X = np.mat([[0],[0],[0],[0]])
        >>> dt = 0.001
        >>> q_desired_set = np.array([1,2,3,4,5,6])
        >>> option = "sliding mode"
        >>> kinematics = Kinematics()
        >>> theta = kinematics.theta_nominal()
        >>> dyn = Dynamics(X,dt)
        >>> control = Control(dt,option)
        >>> control.control(dyn,X,q_desired_set,theta)
        (array([[160.42468867, 160.42468867],
        [ 58.1632766 ,  58.1632766 ]]), None)

        '''
        return self.option_selector(dyn,X,d_set,theta)

    @staticmethod
    def inverse_dynamics_control(dyn,X,d_set,_):
        '''
        Returns control input calculated by structural inverse dynamics control algorithm

        Hyper parameters
        ----------------
        w_1 : float
            w_1 > 0, natural frequency which determines the speed of response of the first joint

        w_2 : float
            w_2 > 0, natural frequency which determines the speed of response of the second joint

        Parameters
        ----------
        dyn : class instance
            instance of class Dynamics() in dynamics.py

        X : 4 x 1 vector
            state vector at certain time t

        d_set : 1 x 6 list
            desired state at certain time t
        
        Returns
        -------
        tau : 2 x 1 vector
            control input

        '''
        w_1 = 1
        w_2 = 2

        q_desired = np.array([[d_set[0]],[d_set[1]]])
        q_desired_dot = np.array([[d_set[2]],[d_set[3]]])
        q_desired_ddot = np.array([[d_set[4]],[d_set[5]]])

        q = X[0:2,0]
        q_dot = X[2:4,0]
        
        q_tilda = q-q_desired
        q_tilda_dot = q_dot-q_desired_dot

        k_0 = np.array([[w_1**2,0],[0,w_2**2]])
        k_1 = np.array([[2*w_1,0],[0,2*w_2]])

        a_q = q_desired_ddot - k_0@q_tilda - k_1@q_tilda_dot
        u = dyn.D@a_q + dyn.C@q_dot + dyn.g
        
        return u,None

    @staticmethod
    def robust_control(dyn,X,d_set,_):
        '''
        Returns control input calculated by robust control algorithm

        Hyper parameters
        ----------------
        w_1 : float
            w_1 > 0, natural frequency which determines the speed of response of the first joint

        w_2 : float
            w_2 > 0, natural frequency which determines the speed of response of the second joint

        alpha,r_1,r_2,r_3 : float
            alpha < 1, r_I >= 0, r_1,r_2: const, r_3: time-varying, hyper parameters for bounding uncertainty

        scale_factor : float
            scale_factor < 1, scale factor that mapping epsilon_max to epsilon

        Q : 4 x 4 matrix
            Q > 0(positive definite), hyper parameter to solve lyapunov equation
    
        Parameters
        ----------
        dyn : class instance
            instance of class Dynamics() in dynamics.py

        X : 4 x 1 vector
            state vector at certain time t

        d_set : 1 x 6 list
            desired state at certain time t
        
        Returns
        -------
        tau : 2 x 1 vector
            control input

        '''
        w_1 = 5
        w_2 = 5
        alpha = 0.5
        r_1 = 0.1
        r_2 = 0.15
        r_3 = 0.2
        scale_factor = 0.7
        Q = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        q_desired = np.array([[d_set[0]],[d_set[1]]])
        q_desired_dot = np.array([[d_set[2]],[d_set[3]]])
        q_desired_ddot = np.array([[d_set[4]],[d_set[5]]])

        q = X[0:2,0]
        q_dot = X[2:4,0]
        
        q_tilda = q-q_desired
        q_tilda_dot = q_dot-q_desired_dot

        k_1 = np.array([[2*w_1,0],[0,2*w_2]])
        k_0 = np.array([[w_1**2,0],[0,w_2**2]])

        e = np.array([[q_tilda[0,0]],[q_tilda[1,0]],[q_tilda_dot[0,0]],[q_tilda_dot[1,0]]])
        norm_e = np.linalg.norm(e)

        rho = (r_1*norm_e + r_2*norm_e**2 + r_3)/(1-alpha)
        A = np.array([[0,0,1,0],[0,0,0,1],[-w_1**2,0,-2*w_1,0],[0,-w_2**2,0,-2*w_2]])
        B = np.array([[0,0],[0,0],[1,0],[0,1]])
        P = linalg.solve_continuous_lyapunov(A,Q)
        ev_Q,_ = np.linalg.eig(Q)
        ev_Q_min = min(ev_Q)

        E = B.T @ P @ e
        norm_E = np.linalg.norm(E)

        epsilon_max = 2*ev_Q_min*norm_e**2/rho
        
        epsilon = epsilon_max*scale_factor
        
        if norm_E > epsilon:
            del_a = -rho*E/norm_E
        else:
            del_a = -rho*E/epsilon
        
        a_q = q_desired_ddot - k_0@q_tilda - k_1@q_tilda_dot + del_a
        u = dyn.D@a_q + dyn.C@q_dot + dyn.g
        
        return u,None

    @staticmethod
    def passivity_based_robust_control(dyn,X,d_set,theta_0):
        '''
        Returns control input calculated by passivity based robust control algorithm

        Hyper parameters
        ----------------
        scale_factor : float
            scale_factor < 1, scale factor that mapping epsilon_max to epsilon

        Lambda : 2 x 2 matrix
            Lambda > 0, diagonal, hyper parameter for intermidiate variables. kind of cut-off frequency

        k : 2 x 2 matrix
            k > 0, diagonal, hyper parameter for intermidiate variables.

        rho : float
            rho > 0, bound for uncertainty of parameter vector
    
        Parameters
        ----------
        dyn : class instance
            instance of class Dynamics() in dynamics.py

        X : 4 x 1 vector
            state vector at certain time t

        d_set : 1 x 6 list
            desired state at certain time t

        theta_0 : 5 x 1 vector
            nominal regression parameter vector
        
        Returns
        -------
        tau : 2 x 1 vector
            control input

        '''
        scale_factor = 0.7
        Lambda = np.array([[10,0],[0,10]])
        k = np.array([[25,0],[0,10]])
        rho = 0.5
        
        q_desired = np.array([[d_set[0]],[d_set[1]]])
        q_desired_dot = np.array([[d_set[2]],[d_set[3]]])
        q_desired_ddot = np.array([[d_set[4]],[d_set[5]]])

        q = X[0:2,0]
        q_dot = X[2:4,0]
        
        q_tilda = q-q_desired
        q_tilda_dot = q_dot-q_desired_dot

        e = np.array([[q_tilda[0,0]],[q_tilda[1,0]],[q_tilda_dot[0,0]],[q_tilda_dot[1,0]]])
        norm_e = np.linalg.norm(e)

        v = q_desired_dot - Lambda@q_tilda
        a = q_desired_ddot - Lambda@q_tilda_dot
        r = q_tilda_dot + Lambda@q_tilda

        Y_mat = dyn.Y_passivity(a,v)

        E = Y_mat.T@r
        norm_E = np.linalg.norm(E)

        ev_Q_1,_ = np.linalg.eig(Lambda.T*k*Lambda)
        ev_Q_2,_ = np.linalg.eig(Lambda*k)
        ev_Q_min = min(min(ev_Q_1),min(ev_Q_2))

        epsilon_max = 4*ev_Q_min*norm_e**2/rho
        
        epsilon = epsilon_max*scale_factor

        if norm_E > epsilon:
            del_theta = -rho*E/norm_E
        else:
            del_theta = -rho*E/epsilon
        
        theta_hat = theta_0 + del_theta
        u = Y_mat@theta_hat - k@r
        
        return u,theta_0

    def passivity_based_adaptive_control(self,dyn,X,d_set,theta_0):
        '''
        Returns control input calculated by passivity based adaptive control algorithm

        Hyper parameters
        ----------------
        gain : 5 x 5 matrix
            gain > 0, constant. determines the speed of convergence and stability

        Lambda : 2 x 2 matrix
            Lambda > 0, diagonal, hyper parameter for intermidiate variables. kind of cut-off frequency

        k : 2 x 2 matrix
            k > 0, diagonal, hyper parameter for intermidiate variables.
    
        Parameters
        ----------
        dyn : class instance
            instance of class Dynamics() in dynamics.py

        X : 4 x 1 vector
            state vector at certain time t

        d_set : 1 x 6 list
            desired state at certain time t

        theta_0: 5 x 1 vector
            regression parameter vector
        
        Returns
        -------
        tau : 2 x 1 vector
            control input

        '''
        gain = 2*np.identity(5)
        Lambda = np.array([[2,0],[0,2]])
        k = np.array([[100,0],[0,100]])
        
        q_desired = np.array([[d_set[0]],[d_set[1]]])
        q_desired_dot = np.array([[d_set[2]],[d_set[3]]])
        q_desired_ddot = np.array([[d_set[4]],[d_set[5]]])

        q = X[0:2,0]
        q_dot = X[2:4,0]
        
        q_tilda = q-q_desired
        q_tilda_dot = q_dot-q_desired_dot

        v = q_desired_dot - Lambda*q_tilda
        a = q_desired_ddot - Lambda*q_tilda_dot
        r = q_tilda_dot + Lambda*q_tilda

        Y_mat = dyn.Y_passivity(a,v)

        theta_hat_dot = -np.linalg.inv(gain)@Y_mat.T@r
        theta_hat = theta_0 + theta_hat_dot*self.dt
        u = Y_mat@theta_hat - k*r
        
        return u,theta_hat

    @staticmethod
    def sliding_mode_control(dyn,X,d_set,_):
        '''
        Returns control input calculated by sliding mode control algorithm

        Hyper parameters
        ----------------
        Lambda : float
            Lambda > 0, slope of sliding surface

        k : float
            eta > 0, bounding parameter of lyapunov function

        Parameters
        ----------
        dyn : class instance
            instance of class Dynamics() in dynamics.py

        X : 4 x 1 vector
            state vector at certain time t

        d_set : 1 x 6 list
            desired state at certain time t
        
        Returns
        -------
        tau : 2 x 1 vector
            control input

        '''
        Lambda = 0.3
        k = 2
        
        q_desired = np.array([[d_set[0]],[d_set[1]]])
        q_desired_dot = np.array([[d_set[2]],[d_set[3]]])
        q_desired_ddot = np.array([[d_set[4]],[d_set[5]]])

        q = X[0:2,0]
        q_dot = X[2:4,0]
        
        q_tilda = q-q_desired
        q_tilda_dot = q_dot-q_desired_dot

        S = q_tilda_dot + Lambda*q_tilda
        f = -dyn.D_inv @ (dyn.C@q_dot + dyn.g)
        u_hat = q_desired_ddot - Lambda*q_tilda_dot - f
 
        u = dyn.D @ (u_hat - k*np.sign(S))
        
        return u,None