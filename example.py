import casadi as ca
from casadi import sin, cos
from model import LQRModel, LQGModel, LQRDModel, LQGDModel
from simulator import Simulator, NoisySimulator
import numpy as np


# We create Casadi symbolic variables for the state
x = ca.MX.sym('x')  
xdot = ca.MX.sym('xdot')
theta = ca.MX.sym('theta')
thetadot = ca.MX.sym('thetadot')
u = ca.MX.sym('u')             # control input
state = ca.vertcat(
    x,
    xdot,
    theta,
    thetadot,
)

# create casadi constants
M = ca.MX.sym('M') 
m = ca.MX.sym('m') 
L = ca.MX.sym('L') 
g = ca.MX.sym('g') 
d = ca.MX.sym('d')


constants = ca.vertcat( M, m, L, g, d )

# we build the nonlinear function f for the equations of motion
# we begin with some partial expressions to make the formulas easier to build
denominator = M + m*(sin(theta)**2)
n0 = -m*g*sin(theta)*cos(theta)
n1 = m*L*(sin(theta))*(thetadot)**2 - d*xdot + u
n2 = (m + M)*g*sin(theta)
RHS = ca.vertcat( 
    xdot, 
    (n0 + n1) / denominator, 
    thetadot, 
    (n2+(-cos(theta))*(n1)) / (L*denominator)
    )

# the constant values are imported from the build file
M_ = 0.05       # wheels plus motors (kilograms) 
m_ = 1          # rest of the robot (kilograms)
L_ = 1.22        # length of pendulum (meters)
g_ = -9.81       # gravity, (meters / sec^2)
d_ = 0.01       # d is a damping factor

constant_values = [M_, m_, L_, g_, d_]

# I made latex names for my states. They look nice in the simulation plots
my_state_names = ['$x$ ','$\\dot{x}$ ','$\\theta$ ','$\\dot{\\theta}$ ']

dt = 0.01
# Now we make our model.
lqrBot = LQRModel(state, 
                RHS, 
                u, 
                constants, 
                constant_values, 
                dt,
                state_names=my_state_names,
                name='balancing robot LQR')

# set up the K matrix
Q = np.eye(4)
R = np.eye(1)
goal_state = [0.0, 0.0, np.pi, 0.0]
goal_u = [0.0]

lqrBot.set_up_K(Q, R, goal_state, goal_u)

lqgBot = LQGModel(state, 
                RHS, 
                u, 
                constants, 
                constant_values, 
                dt,
                state_names=my_state_names,
                name='balancing robot LQG')

lqgBot.set_up_K(Q, R, goal_state, goal_u)

# If we want a Kalman Filter we need to pass in a measurement model
C = np.array([[1, 0, 0, 0], \
            [0, 0, 1, 0], \
            [0, 0, 0, 1]]) 

V_d = np.eye(4)
V_n = np.eye(3) * 0.001

lqgBot.set_up_kalman_filter(C, V_d, V_n)

lqrdBot = LQRDModel(state, 
                RHS, 
                u, 
                constants, 
                constant_values, 
                dt,
                state_names=my_state_names,
                name='balancing robot LQRD')

lqrdBot.set_up_K(Q, R, goal_state, goal_u)


lqgdBot = LQGDModel(state, 
                RHS, 
                u, 
                constants, 
                constant_values, 
                dt,
                state_names=my_state_names,
                name='balancing robot LQGD')


lqgdBot.set_up_K(Q, R, goal_state, goal_u)
lqgdBot.set_up_kalman_filter(C, V_d, V_n)


if __name__ == "__main__":
    # now we can rum a simulation
    u0 = np.array([0.0])
    x0 = np.array([1.0,0,np.pi + 0.3, 0.0]) # Initial condition
    sim_length = 4 # in seconds

    # simulator = Simulator(lqgBot, x0, u0, sim_length)
    # simulator.run()
    
    simulator = NoisySimulator(lqgBot, x0, u0, sim_length)
    simulator.run()