import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

class Simulator:
    def __init__(self, model, x0, u0, timespan, dt):
        self.model = model
        self.x0 = x0
        self.u0 = u0
        self.dt = dt
        self.tspan = np.arange(0,timespan,dt)
        self.data = np.empty([len(self.tspan),self.model.state_size() + self.model.input_size()])
    
    def run(self):
        x = self.x0
        u = self.u0
        start_time = perf_counter()
        for i in range(len(self.tspan)):
            x = self.model.get_next_state(x,u,self.dt)
            u = self.model.get_control_input(x)
            self.data[i] =  np.append(x, u)
        print('Time for {} iterations of {} is {}'.format(len(self.tspan), self.model.name, perf_counter() - start_time))


        plt.rcParams.update({'font.size': 12})
        plt.rcParams.update({
        "text.usetex": True,
        })

        ns = self.model.state.size1()

        for i in range(ns):
            plt.plot(self.tspan,self.data[:,i],linewidth=2,label=self.model.state_names[i])
        plt.xlabel('Time')
        plt.ylabel('State')
        plt.title(self.model.name)
        plt.legend(loc='lower right')
        plt.show()


        nu = self.model.u.size1()
        fig, axs = plt.subplots(ns + nu)
        fig.set_figheight(8)
        fig.suptitle(self.model.name)
        for i in range(ns):
            axs[i].plot(self.tspan,self.data[:,i],linewidth=2)
            axs[i].set_ylabel(self.model.state_names[i])
        for i in range(nu):
            axs[i+ns].plot(self.tspan,self.data[:,i+ns],linewidth=2)
            axs[i+ns].set_ylabel(self.model.u[i].name())
        plt.xlabel('Time')
        plt.show()




