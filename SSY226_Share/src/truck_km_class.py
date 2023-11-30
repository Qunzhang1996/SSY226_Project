import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import casadi as cs
from enum import IntEnum
from vehicle_class import C, ST,C_k,Vehicle
from car_km_class import Car_km

import warnings
warnings.simplefilter("error")
nt=4
L=8.47
class Truck_CC(Car_km):
    def __init__(self, state, dt):
        super().__init__(state, dt)
        self.steps_between_cc_replanning = 100
        self.v_ref = state[C.V_S]
        self.a = 1
        self.b = 1.5
        self.delta = 4
        self.T = 3
        self.s0 = 20
        self.counter_cc_replanning = 0
        self.history_v_ref = [state[C.V_S]]

    def truck_f(self, x, u):
        """Continues truck model"""
        # x: v, s, n, t
        return np.array((u, x[0], 0, 1))

    def truck_F(self, x, u, dt):
        """Discrete truck model"""
        k1 = self.truck_f(x, u)
        k2 = self.truck_f(x+dt/2*k1, u)
        k3 = self.truck_f(x+dt/2*k2, u)
        k4 = self.truck_f(x+dt*k3, u)
        return x+dt/6*(k1+2*k2+2*k3+k4)

    def CC(self,v_ref, state):
        """Compute new state using CC after steps_between_cc_replanning*dt seconds"""
        history_velocity = []
        for _ in range(self.steps_between_cc_replanning):
            a_free = self.a*(1-np.power(state[ST.V]/v_ref, self.delta)) \
                if state[ST.V] <= v_ref \
                else -self.b*(1-np.power(v_ref/state[ST.V], self.a*self.delta/self.b))
            state = self.truck_F(state, a_free, self.dt)
            history_velocity.append(state[ST.V])
        return state, history_velocity   

    def func_v_ref(self, v_ref, s_final, state, velocity_intermediate_points):
        """Compute the difference between the desired final distance s_final and output of CC for a give v_ref"""
        cc_state, history_velocity = self.CC(v_ref, state)
        J = np.power(s_final - cc_state[ST.S],2) # cost to deviate from the given s_final
        J += 10*np.sum(np.multiply(np.exp(np.arange(-self.steps_between_cc_replanning+1,1)),np.power(history_velocity-velocity_intermediate_points,2))) # cost to deviated from the given velocity profile
        return J 
    
    def compute_v_ref(self):
        # Compute v_ref using the CC model for a given trajectory
        state = self.get_state()
        tt = self.planned_trajectory[C.T, :]
        S_state_guess = scipy.interpolate.interp1d(tt,self.planned_XU0[C.S::(self.nx+self.nu)],kind='cubic')(state[ST.T] + self.dt*self.steps_between_cc_replanning)
        t_intermediate_points = self.dt*np.arange(self.steps_between_cc_replanning) + self.dt + state[ST.T]
        velocity_intermediate_points = scipy.interpolate.interp1d(tt, self.planned_XU0[C.V_S::(self.nx+self.nu)], fill_value='extrapolate', kind='cubic')(t_intermediate_points)
        # root_v_ref, infodict, ier, mesg = scipy.optimize.fsolve(func_v_ref,state_guess[ST.V],args=(state_guess[ST.S],state, velocity_intermediate_points),full_output=True,xtol=0.1)
        res = scipy.optimize.minimize_scalar(self.func_v_ref,bounds=(16,26),args=(S_state_guess,state, velocity_intermediate_points),method='bounded')
        print(state[ST.T], res.x, res.nfev, res.success, res.message)
        self.v_ref = res.x 
    
    
    def simulate(self, dt,outside_carla_state=np.zeros([5,1])):

        #To test, try to transform the pm state to km state
        # print('this is the state', self.state)
        outside_carla_state=self.transformation_mp2km(self.state).reshape(-1,1)
        # print('this is the state in km', outside_carla_state)

        #transfer the state from Carla to pm
        self.state=self.transformation_km2mp(outside_carla_state).ravel()
        # print('this is the state in pm', self.state)
        # exit()


        if self.counter_cc_replanning == 0:
            self.compute_v_ref()
            self.counter_cc_replanning = self.steps_between_cc_replanning
        self.counter_cc_replanning -= 1
        
        # Trajectory following using LQR only for a_n
        e = np.zeros([self.nx-1,1])
        tt = self.planned_trajectory[C.T, :]
        t = self.state[C.T]
        e[0] = self.state[C.V_S] - scipy.interpolate.interp1d(tt,self.planned_XU0[C.V_S::(self.nx+self.nu)],kind='cubic')(t)
        e[1] = self.state[C.S] - scipy.interpolate.interp1d(tt,self.planned_XU0[C.S::(self.nx+self.nu)],kind='cubic')(t)
        e[2] = self.state[C.N] - scipy.interpolate.interp1d(tt,self.planned_XU0[C.N::(self.nx+self.nu)],kind='cubic')(t)
        e[3] = self.state[C.V_N] - scipy.interpolate.interp1d(tt,self.planned_XU0[C.V_N::(self.nx+self.nu)],kind='cubic')(t)
        u = -np.matmul(self.R,e)
        a_n = u[1]
        
        # Trajectory following using CC
        s_leading = math.inf
        v_leading = 0
        for name_id in self.vehicles:
            if self.vehicles[name_id].state[ST.S] > self.state[ST.S] and \
                    self.vehicles[name_id].state[ST.N] > -self.lane_width/2 and \
                    self.state[ST.S] < s_leading:
                s_leading = self.vehicles[name_id].state[ST.S]
                v_leading = self.vehicles[name_id].state[ST.V]
        dv = self.state[ST.V] - v_leading
        s = s_leading - self.state[ST.S]
        v0_local = self.v_ref
        s_star = self.s0 + max(0, v0_local*self.T +
                               v0_local*dv/(2*np.sqrt(self.a*self.b)))
        z = s_star/s
        a_free = self.a*(1-np.power(self.state[ST.V]/v0_local, self.delta)) \
            if self.state[ST.V] <= v0_local \
            else -self.b*(1-np.power(v0_local/self.state[ST.V], self.a*self.delta/self.b))
        if self.state[ST.V] < v0_local:  # CHANGE FROM <=
            dvdt = self.a*(1-z**2) if z >= 1 else a_free * \
                (1-np.power(z, min(100, 2*self.a/a_free)))  # MAX IN POWER
        else:
            dvdt = a_free+self.a*(1-z**2) if z >= 1 else a_free
        a_s = dvdt
        u = [a_s, a_n]


        #to get the optimal control input of the truck, using transformation_inv
        u_optimal_km=self.input_transform_inv(u).reshape(-1,1)
        #set the limitation for the control input(accleration between -1 and 1)
        u_optimal_km[0]=min(1,max(-1,u_optimal_km[0]))
        #set the limitation for the control input(steering angle between -0.5 and 0.5)
        u_optimal_km[1]=min(0.5,max(-0.5,u_optimal_km[1]))
        # print('this is the optimal control input of the truck', u_optimal_km)

        self.state = self.car_F(self.state, u, dt).full().ravel()

        self.history_state.append(self.get_state())
        self.history_control.append(u)
        self.history_v_ref.append(self.v_ref)
