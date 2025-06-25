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

        # if our model doesn't have a Kalman Filter then y
        # will just be a dummy variable
        y = self.x0
        if self.model.has_kalman_filter():
            y = self.model.C@self.x0 

        start_time = perf_counter()
        for i in range(len(self.tspan)):
            x = self.model.next_state(x,u,y)
            u = self.model.control_input(x)

            # if we have a Kalman Filter then just set
            # the sensor values to the state
            if self.model.has_kalman_filter():
                y = self.model.C@x

            self.data[i] =  np.append(x, u)
        print('Time for {} iterations of {} is {}'.format(len(self.tspan), self.model.model_name(), perf_counter() - start_time))


        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({
        "text.usetex": True,
        })

        ns = self.model.state_size()

        for i in range(ns):
            plt.plot(self.tspan,self.data[:,i],linewidth=2,label=self.model.state_names()[i])
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title(self.model.model_name())
        plt.legend(loc='lower right')
        plt.show()


        nu = self.model.input_size()
        fig, axs = plt.subplots(ns + nu)
        fig.set_figheight(8)
        fig.suptitle(self.model.model_name())
        for i in range(ns):
            axs[i].plot(self.tspan,self.data[:,i],linewidth=2)
            axs[i].set_ylabel(self.model.state_names()[i])
        for i in range(nu):
            axs[i+ns].plot(self.tspan,self.data[:,i+ns],linewidth=2)
            #axs[i+ns].set_ylabel(self.model.u[i].name())

            if self.input_bound.any():
                if self.input_bound[i].any():
                    upper = np.full((len(self.tspan),), self.input_bound[i][0])
                    lower = np.full((len(self.tspan),), self.input_bound[i][1])
                    axs[i+ns].plot(self.tspan,upper,linewidth=1)
                    axs[i+ns].plot(self.tspan,lower,linewidth=1)
                
        plt.xlabel('Time')
        plt.show()


class NoisySimulator:
    def __init__(self, model, x0, u0, timespan, noise = []):
        self.model = model
        self.x0 = x0
        self.u0 = u0
        self.dt = model.dt
        self.tspan = np.arange(0,timespan,self.dt)
        self.true_data = np.empty([len(self.tspan),self.model.state_size()])
        self.sensor_data = np.empty([len(self.tspan),self.model.state_size()])
        self.md_data = np.empty([len(self.tspan),self.model.state_size()])

        self.C = np.eye(self.model.state_size())
        if self.model.has_kalman_filter():
            self.C = self.model.C    

        self.num_measurements = self.C.shape[0]
        self.noise_var = noise
        if noise == []:
            self.noise_var = np.ones(self.num_measurements) * 0.001
        


    def run(self):
        x_true = self.x0
        u = self.u0
        x_md = x_true
        noise = np.zeros(self.num_measurements)
        y = self.C@self.x0 

        for i in range(len(self.tspan)):

            # generate some noise
            for j, var in enumerate(self.noise_var):
                noise[j] = np.random.normal(0.0,np.sqrt(var))

            # get the Kalman filter estimate 
            # sensors are reading the true state plus some noise
            y = self.C @ x_true + noise
            x_md = self.model.next_state(x_md,u,y)
           
            # calculate where we really are if the sensors
            # were perfect
            dx = self.model.f(state=x_true, u=u, constants=self.model.constant_values)['dx']
            x_true = x_true + dx*self.dt

            # sensor readings are the true state perturbed by some noise
            x_sensors = x_true + self.C.transpose() @ noise

            # get the control input based on our estimated state
            u = self.model.control_input(x_md)

            self.true_data[i] =  np.reshape(x_true, (4,))
            self.md_data[i] = np.reshape(x_md, (4,))
            self.sensor_data[i] = np.reshape(x_sensors, (4,))
    


        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({
        "text.usetex": True,
        })

        ns = self.model.state_size()
        s_i = self.C @ [i for i in range(ns)]
        s_i = [int(i) for i in s_i]
        for i in s_i:
            plt.plot(self.tspan,self.true_data[:,i],linewidth=2,label=self.model.state_names()[i] + ' true')
            plt.plot(self.tspan,self.sensor_data[:,i],linewidth=2,label=self.model.state_names()[i] + ' sensors')
            plt.plot(self.tspan,self.md_data[:,i],linewidth=2,label=self.model.state_names()[i] + ' kf')
            plt.xlabel('Time')
            plt.ylabel('State')
            plt.title(self.model.model_name())
            plt.legend(loc='lower right')
            plt.show()





# class NoisySimulator:
#     def __init__(self, model, x0, u0, timespan):
#         self.model = model
#         self.x0 = x0
#         self.u0 = u0
#         self.dt = model.dt
#         self.tspan = np.arange(0,timespan,self.dt)
#         self.true_data = np.empty([len(self.tspan),self.model.state_size() + self.model.input_size()])
#         self.noisy_data = np.empty([len(self.tspan),self.model.state_size() + self.model.input_size()])
#         self.kf_data = np.empty([len(self.tspan),self.model.state_size() + self.model.input_size()])
    
#         # set the default noise
#         self.noise = np.array([0.07, 0.0225])

#     def run(self):
#         num_measurements = self.model.C.shape[0]
#         x_noise = self.x0
#         x_kf = self.x0
#         x_true = self.x0

#         u_kf = np.concatenate((self.u0, self.model.C@self.x0), axis=0)
#         u_noise = self.u0
#         u_true = self.u0

#         noise = np.zeros(num_measurements)
#         sensors = np.zeros(num_measurements)

#         for i in range(len(self.tspan)):
#             # generate some noise
#             for j in range(num_measurements):
#                 noise[j] = np.random.normal(0.0,np.sqrt(self.noise[j]))

#             x_true = self.model.get_next_state_nonlinear(x_true,u_true,self.dt)
#             u_true = self.model.get_control_input(x_true)
#             self.true_data[i] = np.append(x_true, u_true)

#             # perturb our state with some noise
#             x_noise = x_noise + noise@self.model.C
#             x_noise = self.model.get_next_state_linear(x_noise,u_noise,self.dt)
#             u_noise = self.model.get_control_input(x_noise)
#             self.noisy_data[i] = np.append(x_noise, u_noise)

#             x_kf = self.model.get_next_state_kf(x_kf, u_kf, self.dt)
#             u_kf[0] = self.model.get_control_input(x_kf)
#             u_kf[1:4] = noise + self.model.C@x_kf
#             self.kf_data[i] = np.append(x_kf, u_kf[0])



#         # plt.rcParams['figure.figsize'] = [8, 8]
#         plt.rcParams.update({'font.size': 12})
#         plt.rcParams.update({
#         "text.usetex": True,
#         })
        
#         m = np.ones((num_measurements,))
#         measurements = m@self.model.C

#         for i in range(self.model.state_size()):
#             if measurements[i] != 0.0:
#                 plt.plot(self.tspan, self.noisy_data[:,i],linewidth=1,label=('true + noise'))
#                 plt.plot(self.tspan,self.kf_data[:,i],linewidth=2,label='Kalman filter')
#                 plt.plot(self.tspan,self.true_data[:,i],linewidth=1,label='true')
#                 plt.xlabel('time')
#                 plt.ylabel(self.model.state_names[i])
#                 plt.legend()
#                 plt.title(self.model.name)
#                 # plt.savefig("documents/KFangular_velocity.pdf", format="pdf", bbox_inches="tight")
#                 plt.show()





