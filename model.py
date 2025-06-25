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
        self.eigenvalues = np.array([])
        self.recommended_dt = 0
        self.controllable = False
        


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
        self.controllable = np.linalg.matrix_rank(ctrb(self.A, self.B)) == self.x_size

        # compute the input rule
        self.K, S, E = lqr(self.A,self.B,self.Q,self.R) 
    
        # print system eigenvalues
        sys_matrix = self.A - self.B @ self.K
        self.eigenvalues, eigenvectors = np.linalg.eig(sys_matrix)
        self.recommended_dt = abs(0.5 / min(self.eigenvalues))


    def __repr__(self):
        name = self.name
        states = '\tstates= [' +  ', '.join(self.x_names) + ']'
        dt = '\tdt= ' + str(self.dt)
        setup = ''
        if self.A.any():
            A = 'A: \n' + str(self.A)
            B = 'B: \n' + str(self.B)
            K = 'K: \n' + str(self.K)
            E = 'Eigenvalues: \n' + str(self.eigenvalues)
            C = 'System is not controllable'
            if self.controllable:
                C = 'System is controllable'
            r_dt = 'Euler forward recommended dt: ' + str(self.recommended_dt)
            setup = f"{A},\n{B},\n{K},\n{E},\n{C},\n{r_dt}"
        return f"LQRModel: {name},\n{states},\n{dt},\n{setup}"

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
        self.observable = False


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
        self.observable = np.linalg.matrix_rank(obsv(self.A, self.C).transpose()) == self.x_size

        # Generate our Kalman Filter
        self.Kf = lqr(self.A.transpose(), self.C.transpose(), self.V_d, self.V_n)[0].transpose()


    def __repr__(self):
        setup = ''
        if self.Kf.any():
            O = 'System is not observable'
            if self.observable:
                O = 'System is observable'
            Kf = 'Kalman Filter: \n' + str(self.Kf)
            setup = f"\n{O},\n{Kf}"

        return 'LQGModel with Kalman filter \n' + super().__repr__() + setup

    # ____________________________________________________________________
    #
    #  these are all the functions required by the ControlModel interface
    #  those not listed here are inherited fro LQRModel
    # ____________________________________________________________________

    def next_state(self, x, u , y):
        dx = self.A@(x - self.x_ref) + self.B@(u - self.u_ref) + self.Kf@(y - self.C@x)
        return x + dx*self.dt

    def has_kalman_filter(self):
        return True
    

class LQRDModel(LQRModel):

    # __init_ function is inherited from LQRModel

    def __repr__(self):
        return 'LQRDModel \n' + super().__repr__()

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

    # ____________________________________________________________________
    #
    #  these are all the functions required by the ControlModel interface
    #  those not listed here are inherited fro LQRModel
    # ____________________________________________________________________


    def next_state(self, x, u ,y):
        return self.A@(x - self.x_ref) + self.B@(u - self.u_ref) + self.x_ref


    def control_input(self, x):
        return -self.K@(x - self.x_ref)


class LQGDModel(LQRDModel):
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
        self.observable = False

    def set_up_kalman_filter(self, C, V_d, V_n):
        
        self.C = C         # C is our measurement model
        self.V_d = V_d     # our state disturbance matrix
        self.V_n = V_n     # our sensor noise matrix
    
        self.V_d = self.V_d * self.dt         # process-noise cov.
        self.V_n = self.V_n / self.dt         # measurement-noise cov. (typical scaling)
        self.Kf = dlqr(self.A.transpose(), self.C.transpose(), self.V_d, self.V_n)[0].transpose()

    def __repr__(self):
        setup = ''
        if self.Kf.any():
            Kf = 'Kalman Filter: \n' + str(self.Kf)
            setup = f"\n{Kf}"

        return 'LQGModel with Kalman filter \n' + super().__repr__() + setup    

    def next_state(self, x, u , y):
        return self.A@(x - self.x_ref) + self.B@(u - self.u_ref) + self.Kf@(y - self.C@x) + self.x_ref

    def has_kalman_filter(self):
        return True   


    


    