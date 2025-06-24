import casadi as ca
import numpy as np
from control.matlab import lqr, ctrb, obsv, ss, c2d, dlqr
from abc import ABC, abstractmethod

# ____________________________________________________________________
# 
#  This is an interface for all our control models. It requires all models
#  to implement these functions with these specific parameters. It allows
#  us to create a simulator class that will produce simulations for any 
#  model that implements this interface. The simulator doesn't need to know
#  details about what kind of model it is to simulate it.
#
#
class ControlModel(ABC):
    @abstractmethod
    def __init__(self, 
                 state, 
                 right_hand_side, 
                 u, 
                 constants, 
                 constant_values, 
                 dt,
                 state_names = None,
                 name='unnamed model'
    ):
        pass

    @abstractmethod
    def next_state(self, x, u ,y):
        pass

    @abstractmethod
    def control_input(self, x):
        pass

    @abstractmethod
    def state_size(self):
        pass

    @abstractmethod
    def input_size(self):
        pass

    @abstractmethod
    def state_names(self):
        pass

    @abstractmethod
    def model_name(self):
        pass

    @abstractmethod
    def time_step(self):
        pass

    @abstractmethod
    def has_kalman_filter(self):
        pass


class LQRModel(ControlModel):
    def __init__(self, 
                 state, 
                 right_hand_side, 
                 u, 
                 constants, 
                 constant_values, 
                 dt,
                 state_names = None,
                 name='unnamed model'
    ):

        self.dt = dt
        self.name = name
        self.x_size = state.size1()
        self.u_size = u.size1()
        self.constant_values = constant_values
        if state_names == None:
            self.x_names = [s.name() for s in ca.vertsplit(state)]
        else:
            self.x_names = state_names      

        # make our casadi equation 
        self.f = ca.Function('f', [state, u, constants], [right_hand_side], ['state', 'u', 'constants'], ['dx'])
    
        # now set up empty values for our lqr data structures
        self.Q = np.array([])
        self.R = np.array([])
        self.A = np.array([])
        self.B = np.array([])
        self.K = np.array([])
        self.x_ref = np.array([])
        self.u_ref = np.array([])


    def set_up_K(self, Q, R, 
                 x_reference = None, 
                 u_reference = None):
        self.Q = Q
        self.R = R
        self.x_ref = x_reference
        if x_reference == None:
            self.x_ref = np.zeros(self.x_size)
        self.u_ref = u_reference
        if u_reference == None:
            self.u_ref = np.zeros(self.u_size)
            
        # compute the jacobian of the system
        f_jacobian = self.f.jacobian()

        # linearize in the pendulum up position
        linearized = f_jacobian(state = self.x_ref, u=self.u_ref, constants = self.constant_values)

        self.A = linearized['jac_dx_state'].full()
        self.B = linearized['jac_dx_u'].full()

        # next we check if system is controlable
        # we check to see that the controllability matrix
        # has full rank
        if np.linalg.matrix_rank(ctrb(self.A, self.B)) == self.x_size:
            print('\nSystem is controllable! \n')
        else:
            print('\nSystem is not controllable')

        # compute the input rule
        self.K, S, E = lqr(self.A,self.B,self.Q,self.R) 
    
        # print system eigenvalues
        sys_matrix = self.A - self.B @ self.K
        eigenvalues, eigenvectors = np.linalg.eig(sys_matrix)
        print('Eigenvalues for A-BK matrix of', self.name, 'system')
        print(eigenvalues)
        recommended_dt = abs(0.5 / min(eigenvalues))
        print(recommended_dt, 'ms is the maximum recommended dt for a Forward Euler integrator')


    def __repr__(self):
        name = self.name
        states = '\tstates= [' +  ', '.join(self.x_names) + ']'
        dt = '\tdt= ' + str(self.dt)
        A = 'A= \n'
        if self.A.any():
            A += str(self.A)
        B = 'B= \n'
        if self.B.any():
            B += str(self.B)
        K = 'K= \n'
        if self.K.any():
            K += str(self.K)
        return f"LQRModel: {name},\n{states},\n{dt},\n{A},\n{B},\n{K}"

    # ____________________________________________________________________
    #
    # these are all the functions required by the ControlModel interface
    # ____________________________________________________________________

    def next_state(self, x, u ,y):
        dx = self.A@(x - self.x_ref) + self.B@(u - self.u_ref)
        return x + dx*self.dt

    def control_input(self, x):
        return -self.K@(x - self.x_ref)

    def state_size(self):
        return self.x_size

    def input_size(self):
        return self.u_size

    def state_names(self):
        return self.x_names

    def model_name(self):
        return self.name

    def time_step(self):
        return self.dt

    def has_kalman_filter(self):
        return False


class LQGModel(LQRModel):
    def __init__(self, 
                 state, 
                 right_hand_side, 
                 u, 
                 constants, 
                 constant_values, 
                 dt,
                 state_names = None,
                 name='unnamed model'
    ):
        super().__init__(state, 
                         right_hand_side, 
                         u, 
                         constants, 
                         constant_values, 
                         dt,
                         state_names,
                         name)
        
        # set up data structures for Kalman Filter
        self.C = np.array([])
        self.V_d = np.array([])
        self.V_n = np.array([])
        self.Kf = np.array([])
        
    def set_up_K(self, Q, R, 
                 x_reference = None, 
                 u_reference = None):
        super().set_up_K(Q, R, x_reference, u_reference)


    def set_up_kalman_filter(self, C, V_d, V_n):

        # C is our measurement model
        self.C = C

        # our state disturbance matrix
        assert self.A.shape == V_d.shape
        self.V_d = V_d

        # our sensor noise matrix
        assert V_n.shape == (C.shape[0], C.shape[0])
        self.V_n = V_n

        # next we check if system is observable
        # we check to see that the controllability matrix
        # has full rank
        if np.linalg.matrix_rank(obsv(self.A, self.C).transpose()) == self.x_size:
            print('\nSystem is observable! \n')
        else:
            print('\nSystem is not observable')

        # Generate our Kalman Filter
        self.Kf = lqr(self.A.transpose(), self.C.transpose(), self.V_d, self.V_n)[0].transpose()


    def __repr__(self):
        Kf = '\nKf= \n'
        if self.Kf.any():
            Kf += str(self.Kf)
        return 'LQGModel with Kalman filter \n' + super().__repr__() + Kf


    def next_state(self, x, u ,y):
        dx = self.A@(x - self.x_ref) + self.B@(u - self.u_ref) + self.Kf@(y - self.C@x)
        return x + dx*self.dt

    def has_kalman_filter(self):
        return True
    

class LQRDModel(LQRModel):
    def __init__(self, 
                 state, 
                 right_hand_side, 
                 u, 
                 constants, 
                 constant_values, 
                 dt,
                 state_names = None,
                 name='unnamed model'
    ):
        super().__init__(state, 
                         right_hand_side, 
                         u, 
                         constants, 
                         constant_values, 
                         dt,
                         state_names,
                         name)

 

    def __repr__(self):
        return 'LQRDModel \n' 



    def set_up_K(self, Q, R, 
                 x_reference = None, 
                 u_reference = None):
        
        super().set_up_K(Q, R, x_reference, u_reference)

        sys_c = ss(self.A, self.B, np.eye(self.state_size()), np.zeros_like(self.B))
        sys_d = c2d(sys_c, self.dt, 'zoh')

        # replace continuous matrices with new discrete ones
        self.A = sys_d.A
        self.B = sys_d.B
        self.K, S, E = dlqr(sys_d, self.Q, self.R)

    # def state_size(self):
    #     return self.lqm.state_size()
    
    # def input_size(self):
    #     return self.lqm.input_size()
    
    # def get_next_state(self, x0, u0):
    #     xr = self.lqm.goal_state
    #     ur = self.lqm.goal_u
    #     return self.sys_d.A@(x0 - xr) + self.sys_d.B@(u0 - ur) + xr
    
    # def get_next_state_simulator(self, x0, u0, dt):
    #     return self.get_next_state(x0,u0)

    # def get_control_input(self, x0):
    #     return -self.K_d@(x0 - self.lqm.goal_state)

    # def get_name(self):
    #     return self.name
    
    # def get_state_names(self):
    #     return self.lqm.state_names
    
    # def get_goal_state(self):
    #     return self.lqm.goal_state
    
# class LQGDiscreteModel():
#     def __init__(self, lqm, name=''):
#         self.lqm = lqm
#         self.dt = lqm.dt
#         self.name = name
#         if name == '':
#             self.name = lqm.name

#         self.C = lqm.C
#         sys_c = ss(lqm.A_kf, lqm.B_kf, lqm.C_kf, lqm.D_kf)
#         self.sys_d = c2d(sys_c, lqm.dt, 'zoh')

#         R_diag = np.concatenate((lqm.R.diagonal(), lqm.R_kf.diagonal()), axis=0)
#         self.K_d, S, E = dlqr(self.sys_d, lqm.Q, np.diag(R_diag))

#         self.adj_input_size = self.lqm.input_size() + self.lqm.C.shape[0]
    
#     def state_size(self):
#         return self.lqm.state_size()
    
#     def input_size(self):
#         return self.adj_input_size
    
#     def get_next_state(self, x0, u0):
#         xr = self.lqm.goal_state
#         ur = self.lqm.goal_u_kf
#         return self.sys_d.A@(x0 - xr) + self.sys_d.B@(u0 - ur) + xr
    
#     def get_next_state_simulator(self, x0, u0, dt):
#         return self.get_next_state(x0,u0)

#     def get_control_input(self, x0):
#         return -self.K_d@(x0 - self.lqm.goal_state)
    
#     def get_name(self):
#         return self.name
    
#     def get_state_names(self):
#         return self.lqm.state_names
    
#     def get_goal_state(self):
#         return self.lqm.goal_state
    

# class LQGDiscreteModel2:
#     def __init__(self, lqrm, name=''):
        
#         self.state = lqrm.state
#         self.u = lqrm.u
#         self.dt = lqrm.dt
#         self.name = name
#         self.state_names = lqrm.state_names
#         self.goal_state = lqrm.goal_state
#         self.goal_u = lqrm.goal_u

#         # do I need to keep this?
#         self.model_c = lqrm

#         sys_c = ss(lqrm.A, lqrm.B, np.eye(lqrm.state_size()), np.zeros_like(lqrm.B))
#         sys_d = c2d(sys_c, self.dt, 'zoh')
#         self.A = sys_d.A
#         self.B = sys_d.B
#         self.C = lqrm.C

#         Qd = lqrm.Q * self.dt         
#         Rd = lqrm.R * self.dt
#         self.K, _, _ = dlqr(self.A, self.B, Qd, Rd)

    
#         Vd = lqrm.Q_kf * self.dt         # process-noise cov.
#         Wd = lqrm.R_kf / self.dt         # measurement-noise cov. (typical scaling)
#         self.Kf = dlqr(self.A.transpose(), self.C.transpose(), Vd, Wd)[0].transpose()

#     def state_size(self):
#         return self.state.size1()
    
#     def input_size(self):
#         return self.u.size1()
    
#     def get_name(self):
#         return self.name
    
#     def get_state_names(self):
#         return self.state_names
    
#     def get_next_state(self, x, u, y):
#         xr = self.goal_state
#         ur = self.goal_u
#         print('Ax + Bu', self.A@(x - xr) + self.B@(u - ur))
#         print('Kfy', self.Kf@y)
#         print('KfCx', -self.Kf@self.C@x)
#         print('\n')
#         return self.A@(x - xr) + self.B@(u - ur) + self.Kf@(y - self.C@x) + xr
    
    
#     def get_control_input(self, x):
#         return -self.K@(x - self.goal_state)