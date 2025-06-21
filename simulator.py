import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

class Simulator:
    def __init__(self, model, x0, u0, timespan):
        self.model = model
        self.x0 = x0
        self.u0 = u0
        self.dt = self.model.dt
        self.tspan = np.arange(0,timespan, self.dt)
        self.data = np.empty([len(self.tspan),self.model.state_size() + self.model.input_size()])
        self.input_bound = np.array([])

    def add_intput_bound(self, bound):
        assert len(bound) == len(self.u0)
        self.input_bound = bound

    def run(self):
        x = self.x0
        u = self.u0
        start_time = perf_counter()
        for i in range(len(self.tspan)):
            x = self.model.get_next_state_simulator(x,u,self.dt)
            u = self.model.get_control_input(x)
            self.data[i] =  np.append(x, u)
        print('Time for {} iterations of {} is {}'.format(len(self.tspan), self.model.get_name(), perf_counter() - start_time))


        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({
        "text.usetex": True,
        })

        ns = self.model.state_size()

        for i in range(ns):
            plt.plot(self.tspan,self.data[:,i],linewidth=2,label=self.model.get_state_names()[i])
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title(self.model.get_name())
        plt.legend(loc='lower right')
        plt.show()


        nu = self.model.input_size()
        fig, axs = plt.subplots(ns + nu)
        fig.set_figheight(8)
        fig.suptitle(self.model.get_name())
        for i in range(ns):
            axs[i].plot(self.tspan,self.data[:,i],linewidth=2)
            axs[i].set_ylabel(self.model.get_state_names()[i])
        for i in range(nu):
            axs[i+ns].plot(self.tspan,self.data[:,i+ns],linewidth=2)
            #axs[i+ns].set_ylabel(self.model.u[i].name())

            if self.input_bound.any():
                upper = np.full((len(self.tspan),), self.input_bound[i][0])
                lower = np.full((len(self.tspan),), self.input_bound[i][1])
                axs[i+ns].plot(self.tspan,upper,linewidth=1)
                axs[i+ns].plot(self.tspan,lower,linewidth=1)
                
        plt.xlabel('Time')
        plt.show()



class NoisySimulator:
    def __init__(self, model, x0, u0, timespan, dt):
        self.model = model
        self.x0 = x0
        self.u0 = u0
        self.dt = dt
        self.tspan = np.arange(0,timespan,dt)
        self.true_data = np.empty([len(self.tspan),self.model.state_size() + self.model.input_size()])
        self.noisy_data = np.empty([len(self.tspan),self.model.state_size() + self.model.input_size()])
        self.kf_data = np.empty([len(self.tspan),self.model.state_size() + self.model.input_size()])
    
        # set the default noise
        self.noise = np.full((self.model.C.shape[0], ), 0.0000026)

    def run(self):
        num_measurements = self.model.C.shape[0]
        x_noise = self.x0
        x_kf = self.x0
        x_true = self.x0

        u_kf = np.concatenate((self.u0, self.model.C@self.x0), axis=0)
        u_noise = self.u0
        u_true = self.u0

        noise = np.zeros(num_measurements)
        sensors = np.zeros(num_measurements)

        for i in range(len(self.tspan)):
            # generate some noise
            for j in range(num_measurements):
                noise[j] = np.random.normal(0.0,np.sqrt(self.noise[j]))

            x_true = self.model.get_next_state_nonlinear(x_true,u_true,self.dt)
            u_true = self.model.get_control_input(x_true)
            self.true_data[i] = np.append(x_true, u_true)

            # perturb our state with some noise
            x_noise = x_noise + noise@self.model.C
            x_noise = self.model.get_next_state_linear(x_noise,u_noise,self.dt)
            u_noise = self.model.get_control_input(x_noise)
            self.noisy_data[i] = np.append(x_noise, u_noise)

            x_kf = self.model.get_next_state_kf(x_kf, u_kf, self.dt)
            u_kf[0] = self.model.get_control_input(x_kf)
            u_kf[1:4] = noise + self.model.C@x_kf
            self.kf_data[i] = np.append(x_kf, u_kf[0])



        # plt.rcParams['figure.figsize'] = [8, 8]
        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({
        "text.usetex": True,
        })
        
        m = np.ones((num_measurements,))
        measurements = m@self.model.C

        for i in range(self.model.state_size()):
            if measurements[i] != 0.0:
                plt.plot(self.tspan, self.noisy_data[:,i],linewidth=1,label=('true + noise'))
                plt.plot(self.tspan,self.kf_data[:,i],linewidth=2,label='Kalman filter')
                plt.plot(self.tspan,self.true_data[:,i],linewidth=1,label='true')
                plt.xlabel('time')
                plt.ylabel(self.model.state_names[i])
                plt.legend()
                plt.title(self.model.name)
                # plt.savefig("documents/KFangular_velocity.pdf", format="pdf", bbox_inches="tight")
                plt.show()


