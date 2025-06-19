import casadi as ca
import numpy as np
from control.matlab import lqr, ctrb, obsv



class LinearModel:
    def __init__(self, 
                 state, 
                 right_hand_side, 
                 u, 
                 constants, 
                 constant_values, 
                 name='unamed model',
                 state_names=''):

        # these are all our casadi variables and symbols
        self.state = state
        self.u = u
        self.constants = constants
        self.name = name
        if state_names == '':
            self.state_names = [s.name() for s in ca.vertsplit(state)]
        else:
            self.state_names = state_names

        # these are all default values
        # there are functions to set these to specific values
        self.goal_state = [0.0] * state.size1()
        self.Q = np.eye(state.size1())        
        self.R = np.eye(u.size1())       

        # data structures for a Kalman Filter are
        # set to None.  They can be set with the 
        # setupKalmanFilter function if a filter is needed
        self.C = None
        self.D = None
        self.Q_kf = None
        self.R_kf = None
        self.A_kf = None
        self.B_kf = None
        self.C_kf = None
        self.D_kf = None

        assert constants.size1() == len(constant_values)
        self.constant_values = constant_values

        # make our casadi equation 
        self.f = ca.Function('f', [state, u, constants], [right_hand_side], ['state', 'u', 'constants'], ['dx'])

    def set_goal_state(self, goal_state):
        assert len(goal_state) == len(self.goal_state)
        self.goal_state = goal_state  

    # this should come in as a numpy array
    def set_Q(self, Q):
        assert self.Q.shape == Q.shape
        self.Q = Q  

    def set_R(self, R):
        assert self.R.shape == R.shape
        self.R = R

    def state_size(self):
        return self.state.size1()
    
    def input_size(self):
        return self.u.size1()
    
    
    def set_up(self):
        # compute the jacobian of the system
        f_jacobian = self.f.jacobian()

        # linearize in the pendulum up position
        linearized = f_jacobian(state = self.goal_state, u=0.0, constants = self.constant_values)

        self.A = linearized['jac_dx_state'].full()
        self.B = linearized['jac_dx_u'].full()

        # next we check if system is controlable
        # we check to see that the controllability matrix
        # has full rank
        if np.linalg.matrix_rank(ctrb(self.A, self.B)) == self.state.size1():
            print('\nSystem is controllable! \n')
        else:
            print('\nSystem is not controllable')

        # compute the input rule
        self.K = lqr(self.A,self.B,self.Q,self.R)[0] 
    
        # print system eigenvalues
        sys_matrix = self.A - self.B @ self.K
        eigenvalues, eigenvectors = np.linalg.eig(sys_matrix)
        print('eigenvalues of A-BK: \n', eigenvalues)


    def set_up_Kalman_Filter(self, C, Q_kf, R_kf):

        # C is our measurement model
        self.C = C
        self.D = np.zeros_like(self.B)

        # This is our state disturbance matrix
        assert self.A.shape == Q_kf.shape
        self.Q_kf = Q_kf 

        assert R_kf.shape == (C.shape[0], C.shape[0])
        self.R_kf = R_kf

        # next we check if system is observable
        # we check to see that the controllability matrix
        # has full rank
        if np.linalg.matrix_rank(obsv(self.A, self.C).transpose()) == self.state.size1():
            print('\nSystem is observable! \n')
        else:
            print('\nSystem is not observable')

        # Generate our Kalman Filter
        self.Kf = lqr(self.A.transpose(), self.C.transpose(), self.Q_kf, self.R_kf)[0].transpose()

        # These are our A,B,C,D matrices for the Kalman Filter system
        self.A_kf = self.A - (self.Kf @ self.C)    
        self.B_kf = np.concatenate((self.B, self.Kf), axis=1)
        self.C_kf = np.eye(self.state.size1())
        self.D_kf = np.zeros_like(self.B_kf)

    def get_next_state(self, x0, u0, dt):
        dx = self.f(state=x0, u=u0, constants=self.constant_values)['dx']
        return x0 + dx*dt

    def get_control_input(self, x0):
        return -self.K@(x0 - self.goal_state)




