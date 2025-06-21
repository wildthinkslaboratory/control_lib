import casadi as ca
import numpy as np
from control.matlab import lqr, ctrb, obsv, ss, c2d, dlqr



class LQRModel:
    def __init__(self, 
                 state, 
                 right_hand_side, 
                 u, 
                 constants, 
                 constant_values, 
                 dt,
                 name,
                 state_names):

        # these are all our casadi variables and symbols
        self.state = state
        self.rhs = right_hand_side
        self.u = u
        self.constants = constants

        assert constants.size1() == len(constant_values)
        self.constant_values = constant_values
        self.dt = dt
        self.name = name

        if state_names == '':
            self.state_names = [s.name() for s in ca.vertsplit(state)]
        else:
            self.state_names = state_names

        # these are all default values
        # there are functions to set these to specific values
        self.goal_state = [0.0] * state.size1()
        self.goal_u = [0.0] * u.size1()
        self.Q = np.eye(state.size1())        
        self.R = np.eye(u.size1())       

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
    
    def get_name(self):
        return self.name
    
    def get_state_names(self):
        return self.state_names
    
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
        self.K, S, E = lqr(self.A,self.B,self.Q,self.R) 
    
        # print system eigenvalues
        sys_matrix = self.A - self.B @ self.K
        eigenvalues, eigenvectors = np.linalg.eig(sys_matrix)
        print('eigenvalues of A-BK: \n', eigenvalues)
        recommended_dt = abs(0.5 / min(eigenvalues))
        print(recommended_dt, 'ms is the maximum recommended dt for a Forward Euler integrator')


    # The next functions for getting the state are for actual
    # working implementations. They hopefully are fast
    def get_next_state(self, x0, u0):
        dx = self.A@(x0 - self.goal_state) + self.B@(u0 - self.goal_u)
        return x0 + dx*self.dt
    
    
    def get_next_state_nonlinear(self, x0, u0):
        dx = self.f(state=x0, u=u0, constants=self.constant_values)['dx']
        return x0 + dx*self.dt

    def get_next_state_simulator(self, x0, u0, dt):
        return self.get_next_state(x0, u0)

    def get_control_input(self, x0):
        return -self.K@(x0 - self.goal_state)


class LQGModel(LQRModel):
    def __init__(self, lqrm, C, Q_kf, R_kf, name=''):
        
        model_name = name
        if model_name == '':
            model_name = lqrm.name
    
        super().__init__(lqrm.state, 
                         lqrm.rhs, 
                         lqrm.u, 
                         lqrm.constants, 
                         lqrm.constant_values, 
                         lqrm.dt,
                         model_name,
                         lqrm.state_names)


        self.goal_state = lqrm.goal_state
        self.goal_u = lqrm.goal_u
        self.Q = lqrm.Q
        self.R = lqrm.R
        self.f = lqrm.f
        self.set_up()
        
        # C is our measurement model
        self.C = C
        self.D = np.zeros_like(self.B)
        self.goal_u_kf = np.concatenate((self.goal_u, self.C@self.goal_state), axis=0)

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

        self.goal_u_kf = np.concatenate((self.goal_u, self.C@self.goal_state), axis=0)

        
    def get_next_state(self, x0, u0):
        dx = self.A_kf@(x0 - self.goal_state) + self.B_kf@(u0 - self.goal_u_kf)
        return x0 + dx*self.dt
    
    def get_next_state_simulator(self, x0, u0, dt):
        u = np.concatenate((u0, self.C@x0), axis=0)
        dx = self.A_kf@(x0 - self.goal_state) + self.B_kf@(u - self.goal_u_kf)
        return x0 + dx*self.dt
    

class LQRDiscreteModel():
    def __init__(self, lqm, name=''):
        self.lqm = lqm
        self.dt = lqm.dt
        self.name = name
        if name == '':
            self.name = lqm.name

        sys_c = ss(lqm.A, lqm.B, np.eye(4), np.zeros_like(lqm.B))
        self.sys_d = c2d(sys_c, lqm.dt, 'zoh')

        self.K_d, S, E = dlqr(self.sys_d, lqm.Q, lqm.R)

    def state_size(self):
        return self.lqm.state_size()
    
    def input_size(self):
        return self.lqm.input_size()
    
    def get_next_state(self, x0, u0):
        xr = self.lqm.goal_state
        ur = self.lqm.goal_u
        return self.sys_d.A@(x0 - xr) + self.sys_d.B@(u0 - ur) + xr
    
    def get_next_state_simulator(self, x0, u0, dt):
        return self.get_next_state(x0,u0)

    def get_control_input(self, x0):
        return -self.K_d@(x0 - self.lqm.goal_state)

    def get_name(self):
        return self.name
    
    def get_state_names(self):
        return self.lqm.state_names
    
class LQGDiscreteModel():
    def __init__(self, lqm, name=''):
        self.lqm = lqm
        self.dt = lqm.dt
        self.name = name
        if name == '':
            self.name = lqm.name
        sys_c = ss(lqm.A_kf, lqm.B_kf, lqm.C_kf, lqm.D_kf)
        self.sys_d = c2d(sys_c, lqm.dt, 'zoh')

        R_diag = np.concatenate((lqm.R.diagonal(), lqm.R_kf.diagonal()), axis=0)
        self.K_d, S, E = dlqr(self.sys_d, lqm.Q, np.diag(R_diag))

    def state_size(self):
        return self.lqm.state_size()
    
    def input_size(self):
        return self.lqm.input_size()
    
    def get_next_state(self, x0, u0):
        xr = self.lqm.goal_state
        ur = self.lqm.goal_u_kf
        return self.sys_d.A@(x0 - xr) + self.sys_d.B@(u0 - ur) + xr
    
    def get_next_state_simulator(self, x0, u0, dt):
        u = np.concatenate((u0, self.lqm.C@x0), axis=0)
        return self.get_next_state(x0,u)

    # def get_control_input(self, x0):
    #     return -self.K_d@(x0 - self.lqm.goal_state)
    def get_control_input(self, x0):
        return -self.lqm.K@(x0 - self.lqm.goal_state)
    
    def get_name(self):
        return self.name
    
    def get_state_names(self):
        return self.lqm.state_names
    
