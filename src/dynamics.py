# -*- coding: utf-8 -*-
'''
This module describes the kinematics and dynamics of two-arm manipulator

Notation
-------
    mainly refer to main.py's docstring and,
    notation of dynamics and control is mainly refer to:
    "Robot Dynamics and Control, 2nd edition, Mark W.Spong, Seth Hutchinson, and M. Vidyasagar"

    dynamic equation of two-arm manipulator: D*q_ddot + C*q_dot + g = Y*theta = tau
    mathmatical aspect of this equation is written at Readme.md 

    D(q): inertia matrix, symmetric and positive definite for each vector q in R^n
        in this case, 2 x 2 matrix
    C(q,q_dot): christoffel symbol matrix, include centrifugal term and coriolis term
        in this case, 2 x 2 matrix
    g(q): arise from differentiating the potential energy
        in this case, 2 x 1 vector
    Y(q,q_dot,q_ddot): linear regressor
        in this case, 2 x 5 matrix
    theta: parameter vector, only contains about robot's physical parameter(mass,length, etc..)
        in this case, 5 x 1 vector
    tau: control input calculated by certain control algorithm
        in this case, 2 x 1 vector

'''

import numpy as np

class Kinematics(object):
    '''
    Describe about kinematics and physical parameters of two-arm manipulator

    Attributes
    ----------
    m_1 : float
        mass of first arm

    m_2 : float
        mass of second arm

    L_c1 : float
        length of first link to first arm's center of mass

    L_c2 : float
        length of second link to second arm's center of mass

    L_1 : float
        length of first arm

    L_2 : float
        length of second arm

    I_1 : float
        inertia moment of first arm with respect to first link

    I_1 : float
        inertia moment of second arm with respect to second link

    G : float
        gravitational constant at earth

    '''
    m_1 = 1.0
    m_2 = 1.0
    L_c1 = 1.0
    L_c2 = 1.0
    L_1 = 2*L_c1
    L_2 = 2*L_c2
    I_1 = (m_1*L_1**2)/3
    I_2 = (m_2*L_2**2)/3
    G = 9.8

    def theta_nominal(self):
        '''
        Returns nominal regression parameter, theta, 
        nomial means that modeling of dynamics equation(D,C,g) is ideal
        theta only can be changed by changing physical parameter of robot(mass,length, etc..)
    
        Parameters
        ----------
        None
        
        Returns
        -------
        theta : 5 x 1 vector
            regression parameter vector
            
        Example
        -------
        >>> kinematics = Kinematics()
        >>> kinematics.theta_nominal()
        array([[8.66666667],
        [2.        ],
        [2.33333333],
        [3.        ],
        [1.        ]])

        '''
        theta_1 = self.m_1*self.L_c1**2 + self.m_2*(self.L_1**2+self.L_c2**2) + self.I_1 + self.I_2
        theta_2 = self.L_1*self.L_c2
        theta_3 = self.m_2*self.L_c2**2 + self.I_2
        theta_4 = self.m_1*self.L_c1 + self.m_2*self.L_1
        theta_5 = self.m_2*self.L_c2
        return np.array([[theta_1],[theta_2],[theta_3],[theta_4],[theta_5]])

    def jacobian(self,q1_list,q2_list):
        '''
        Returns (x,y) coordinate of second link and end effector
    
        Parameters
        ----------
        q1_list : 1 x int(t) list
            list of controlled and simulated q1

        q2_list : 1 x int(t) list
            list of controlled and simulated q2
        
        Returns
        -------
        x1 : 1 x int(t) list
            list of x coordinate of the second link

        y1 : 1 x int(t) list
            list of y coordinate of the second link

        x2 : 1 x int(t) list
            list of x coordinate of the end effector

        y2 : 1 x int(t) list
            list of y coordinate of the end effector

        Example
        -------
        >>> q1_list = [0,0]
        >>> q2_list = [0,0]
        >>> kinematics = Kinematics()
        >>> kinematics.jacobian(q1_list,q2_list)
        (array([2., 2.]), array([0., 0.]), array([4., 4.]), array([0., 0.]))

        '''

        x1 = self.L_1*np.cos(np.array(q1_list))
        y1 = self.L_1*np.sin(np.array(q1_list))
        x2 = self.L_2*np.cos(np.array(q1_list)+np.array(q2_list)) + x1
        y2 = self.L_2*np.sin(np.array(q1_list)+np.array(q2_list)) + y1

        return x1,y1,x2,y2

class Dynamics(Kinematics):
    '''
    Describe about dynamics of two-arm manipulator

    Parameters
    ----------
    X : 4 x 1 vector
        state vector

    dt : float
        difference between each sampled time
    
    Attributes
    ----------
    X : 4 x 1 vector
        state vector

    dt : float
        difference between each sampled time

    D : 2 x 2 matrix
        inertia matrix

    D_inv : 2 x 2 matrix
        inverse of inertia matrix

    C : 2 x 2 matrix
        christoffel symbol matrix

    g : 2 x 1 vector
        potential energy term

    '''
    def __init__(self,X,dt):
        self.X = X
        self.dt =dt
        self.D = self.get_D()
        self.D_inv = self.get_D_inv()
        self.C = self.get_C()
        self.g = self.get_g()

    def get_D(self):
        '''
        Returns inertia matrix D for certain state X(initialized by member variable)
    
        Parameters
        ----------
        None
        
        Returns
        -------
        D : 2 x 2 matrix
            inertia matrix of two-arm manipulator

        Example
        -------
        >>> X = np.array([[0],[0],[0],[0]])
        >>> dt = 0.001
        >>> dyn = Dynamics(X,dt)
        >>> dyn.get_D()
        array([[12.66666667,  4.33333333],
        [ 4.33333333,  2.33333333]])

        '''
        q_2 = self.X[1,0]

        d_11 = self.m_1*self.L_c1**2 + self.m_2*(self.L_1**2+self.L_c2**2+2*self.L_1*self.L_c2*np.cos(q_2))+self.I_1+self.I_2
        d_12 = self.m_2*(self.L_c2**2+self.L_1*self.L_c2*np.cos(q_2))+self.I_2
        d_22 = self.m_2*self.L_c2**2+self.I_2
        return np.array([[d_11,d_12],[d_12,d_22]])

    def get_D_inv(self):
        '''
        Returns invers of inertia matrix, D_inv for certain state X(initialized by member variable)
    
        Parameters
        ----------
        None
        
        Returns
        -------
        D_inv : 2 x 2 matrix
            inverse of inertia matrix of two-arm manipulator

        Example
        -------
        >>> X = np.array([[0],[0],[0],[0]])
        >>> dt = 0.001
        >>> dyn = Dynamics(X,dt)
        >>> dyn.get_D_inv()
        array([[ 0.21649485, -0.40206186],
        [-0.40206186,  1.17525773]])

        '''
        q_2 = self.X[1,0]

        d_11 = self.m_1*self.L_c1**2 + self.m_2*(self.L_1**2+self.L_c2**2+2*self.L_1*self.L_c2*np.cos(q_2))+self.I_1+self.I_2
        d_12 = self.m_2*(self.L_c2**2+self.L_1*self.L_c2*np.cos(q_2))+self.I_2
        d_22 = self.m_2*self.L_c2**2+self.I_2
        det = d_11*d_22-d_12**2
        inv_11 = d_22/det
        inv_12 = -d_12/det
        inv_22 = d_11/det
        return np.array([[inv_11,inv_12],[inv_12,inv_22]])

    def get_C(self):
        '''
        Returns christoffel symbol matrix C for certain state X(initialized by member variable)
    
        Parameters
        ----------
        None
        
        Returns
        -------
        C : 2 x 2 matrix
            christoffel symbol matrix of two-arm manipulator

        Example
        -------
        >>> X = np.array([[1],[1],[1],[1]])
        >>> dt = 0.001
        >>> dyn = Dynamics(X,dt)
        >>> dyn.get_C()
        array([[-1.68294197, -3.36588394],
        [ 1.68294197,  0.        ]])

        '''
        q_2 = self.X[1,0]
        q_dot_1 = self.X[2,0]
        q_dot_2 = self.X[3,0]

        h = -self.m_2*self.L_1*self.L_c2*np.sin(q_2)
        c_11 = h*q_dot_2
        c_12 = h*q_dot_2 + h*q_dot_1
        c_21 = -h*q_dot_1
        return np.array([[c_11,c_12],[c_21,0]])

    def get_g(self):
        '''
        Returns potential energy term g for certain state X(initialized by member variable)
    
        Parameters
        ----------
        None
        
        Returns
        -------
        g : 2 x 1 vector
            potential energy term of two-arm manipulator

        Example
        -------
        >>> X = np.array([[0],[0],[0],[0]])
        >>> dt = 0.001
        >>> dyn = Dynamics(X,dt)
        >>> dyn.get_g()
        array([[39.2],
        [ 9.8]])

        '''
        q_1 = self.X[0,0]
        q_2 = self.X[1,0]

        g_1 = (self.m_1*self.L_c1+self.m_2*self.L_1)*self.G*np.cos(q_1) + self.m_2*self.L_c2*self.G
        g_2 = self.m_2*self.L_c2*self.G*np.cos(q_1+q_2)
        return np.array([[g_1],[g_2]])

    def Y_passivity(self,a,v):
        '''
        Returns linear regressor Y, calculated by intermidiate variable a,v 
        and certain state X(initialized by member variable) for passivity-based control algorithms
    
        Parameters
        ----------
        v : 2 x 1 vector
            intermidiate variable for passivity-based control algorithms, which equals q_desired_dot - Lambda*q_tilda

        a : 2 x 1 vector
            intermidiate variable for passivity-based control algorithms, which equals to v_dot
        
        Returns
        -------
        Y : 2 x 5 matrix
            linear regressor of two-arm manipulator dynamics

        Example
        -------
        >>> X = np.array([[0],[0],[0],[0]])
        >>> v = np.array([[0],[0]])
        >>> a = np.array([[0],[0]])
        >>> dt = 0.001
        >>> dyn = Dynamics(X,dt)
        >>> dyn.Y_passivity(a,v)
        array([[0. , 0. , 0. , 9.8, 9.8],
        [0. , 0. , 0. , 0. , 9.8]])

        '''
        q1 = self.X[0,0]
        q2 = self.X[1,0]
        
        y_11 = a[0,0]
        y_12 = (2*a[0,0]+a[1,0])*np.cos(q2) - (v[1,0]**2+2*v[0,0]*v[1,0])*np.sin(q2)
        y_13 = v[1,0]
        y_14 = self.G*np.cos(q1)
        y_15 = self.G*np.cos(q1+q2)

        y_21 = 0
        y_22 = a[0,0]*np.cos(q2) + v[0,0]**2*np.sin(q2)
        y_23 = a[0,0]+a[1,0]
        y_24 = 0
        y_25 = self.G*np.cos(q1+q2)

        Y = np.array([[y_11,y_12,y_13,y_14,y_15],[y_21,y_22,y_23,y_24,y_25]])
        
        return Y

    def dyn_update(self,tau):
        '''
        Returns updated state X at t+1, calculated by
        certain state X(initialized by member variable) at t and control input tau,
        using RK1 for updating state
    
        Parameters
        ----------
        tau : 2 x 1 vector
            control input
        
        Returns
        -------
        X : 4 x 1 vector
            updated state

        Example
        -------
        >>> X = np.array([[0],[0],[0],[0]])
        >>> dt = 0.001
        >>> tau = np.array([[1],[0]])
        >>> dyn.dyn_update(tau)
        matrix([[-2.16494845e-06],
        [ 1.92061856e-06],
        [-4.32989691e-03],
        [ 3.84123711e-03]])

        '''
        dt = self.dt

        q_ddot = self.D_inv @ (tau - self.C@np.array([[self.X[2,0]],[self.X[3,0]]]) - self.g)
        return self.X + np.mat([[self.X[2,0]*dt + (q_ddot[0,0]*dt**2)/2],\
            [self.X[3,0]*dt + (q_ddot[1,0]*dt**2)/2],[q_ddot[0,0]*dt],[q_ddot[1,0]*dt]])