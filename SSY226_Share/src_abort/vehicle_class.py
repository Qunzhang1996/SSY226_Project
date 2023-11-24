import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import casadi as cs
from enum import IntEnum

import warnings
warnings.simplefilter("error")

# Structure for name convenience
nt = 4  # number of variables in trajectories
class ST(IntEnum):
    V, S, N, T = range(nt)

class C(IntEnum):
    V_S, S, N, T, V_N = range(5)
# states: v_s longitudinal speed, v_n lateral speed, s longitudinal position, time, n lateral position

class C_k(IntEnum):
    X_km, Y_km, Psi, T, V_km =range(5)

#follow vehicle kinematic, define new C_K
#states: X longitudinal, Y lateral, Psi heading, T time, V velocity

class VehicleTwin:
    """Useful class to represent other vehicles, digital twin"""

    def __init__(self, state):
        self.state = state
        self.desired_trajectory = None
        self.planned_trajectory = None
        self.interpolated_desired_trajectory = None
        self.interpolated_planned_trajectory = None

    def update_state(self, state):
        """Take current state of another vehicle"""
        self.state = state

    def update_planned_desired_trajectories(self, planned_trajectory, desired_trajectory):
        """Take planned and desired trajectories of another vehicle"""
        self.desired_trajectory = desired_trajectory
        self.planned_trajectory = planned_trajectory

    def interpolate_trajectories(self, current_time, planning_points, dt):
        """Compute trajectory points of another vehicles at convenient time steps"""

        if self.planned_trajectory is not None:
            # Interpolate/extrapolate previously received
            self.interpolated_planned_trajectory = scipy.interpolate.interp1d(
                self.planned_trajectory[C_k.T, :], self.planned_trajectory, fill_value='extrapolate', kind='cubic')(current_time+np.arange(planning_points)*dt)
        else:
            # Planned trajectory is either not yet received or not computed by that vehicle
            # Use a generic model to predict future points
            self.interpolated_planned_trajectory = np.zeros(
                (nt+1, planning_points))
            self.interpolated_planned_trajectory[C_k.V_km] = np.ones(planning_points) * self.state[C_k.V_km]
            self.interpolated_planned_trajectory[C_k.X_km] = self.state[C_k.X_km] + (
                current_time - self.state[C_k.T]) * self.state[C_k.V_km] + np.arange(planning_points) * self.state[C_k.V_km] * dt
            self.interpolated_planned_trajectory[C_k.Y_km] = np.ones(planning_points) * self.state[C_k.Y_km]
            self.interpolated_planned_trajectory[C_k.T] = current_time + np.arange(planning_points) * dt
            self.interpolated_planned_trajectory[C_k.Psi, :] = self.calculate_heading_array(current_time, planning_points, dt)

        if self.desired_trajectory is not None:
            self.interpolated_desired_trajectory = scipy.interpolate.interp1d(
                self.desired_trajectory[C_k.T, :], self.desired_trajectory, fill_value='extrapolate', kind='cubic')(current_time+np.arange(planning_points)*dt)
        else:
            # Desired trajectory is not computed by all vehicles at all moments so do nothing
            self.interpolated_desired_trajectory == None

    ####calculate Psi array
    def calculate_heading_array(self, current_time, planning_points, dt):
        psi_array = np.zeros(planning_points)
        for i in range(planning_points - 1):
            current_time_point = current_time + i * dt
            psi_array[i] = self.calculate_heading(current_time_point, dt)
        return psi_array
    
    ###method used to calculate the Psi
    def calculate_heading(self, current_time, dt):
        """Calculate the heading angle (Psi) at the current time using the interpolated trajectory."""
        if self.interpolated_planned_trajectory is not None:
            # Find the index of the current time in the interpolated trajectory
            time_index = np.searchsorted(self.interpolated_planned_trajectory[C_k.T], current_time)
    
            # Ensure we have at least one future point to calculate heading
            if time_index < len(self.interpolated_planned_trajectory[C_k.T]) - 1:
                # Get the current and next trajectory points
                current_point = self.interpolated_planned_trajectory[:, time_index]
                next_point = self.interpolated_planned_trajectory[:, time_index + 1]
                
                # Calculate the difference in position
                delta_s = next_point[C_k.X_km] - current_point[C_k.X_km]
                delta_n = next_point[C_k.Y_km] - current_point[C_k.Y_km]
                
                # Calculate the heading angle (Psi)
                psi = np.arctan2(delta_n, delta_s)         
                return psi
            else:
                # If we are at the end of the trajectory, we can't calculate a heading
                return None
        else:
            # If there is no planned trajectory, we can't calculate a heading
            return None


class Vehicle:
    """Common class for all vehicles"""

    def __init__(self, state):
        self.vehicles = {}
        self.state = state
        self.planning_dt = 0.8*(51/31)*0.7
        self.planning_points = 31
        self.planned_trajectory = None
        self.desired_trajectory = None
        self.control_trajectory = None
        self.lane_width = 3.5 # m

        self.history_state = [np.array(self.get_state())]
        self.history_planned_trajectory = []
        self.history_desired_trajectory = []

        self.name = 'v'

    def add_vehicle_twin(self, name, state):
        self.vehicles[name] = VehicleTwin(state)

    def get_state(self):
        return self.state

    def update_state(self, state):
        """Update the vehicle state with information from CAN"""
        self.state = state

    def update_twin_state(self, name_id, state):
        self.vehicles[name_id].update_state(state)

    def receive_planned_desired_trajectories(self, name_id, planned_trajectory, desired_trajectory):
        self.vehicles[name_id].update_planned_desired_trajectories(
            planned_trajectory, desired_trajectory)
        
    def save_history(self,name):
        import pickle
        with open(name+'.pickle', 'wb') as handle:
            data = {}
            data['history_state'] = np.array(self.history_state).T
            data['history_planned_trajectory'] = self.history_planned_trajectory
            pickle.dump(data, handle)



class Car(Vehicle):
    def __init__(self, state, dt):
        super().__init__(state)
        self.nx = 5
        self.nu = 2
        self.state = np.zeros(self.nx)
        self.state[:nt] = state
        self.state[C.V_N] = 0
        self.desired_XU0 = None
        self.v0 = 25
        self.planned_XU0 = [0] * (self.nx*(self.planning_points)+self.nu*(self.planning_points - 1))
        self.planned_XU0[C.V_S::(self.nx+self.nu)] = np.linspace(self.state[C.V_S], self.v0,self.planning_points)
        self.planned_XU0[C.T::(self.nx+self.nu)] = self.state[C.T] + self.planning_dt*np.arange(self.planning_points)
        local_S = np.zeros(self.planning_points)
        local_S[0] = self.state[C.S]
        for i in range(self.planning_points-1):
            local_S[i+1] = local_S[i] + self.planned_XU0[C.V_S + i*(self.nx+self.nu)]*self.planning_dt
        self.planned_XU0[C.S::(self.nx+self.nu)] = local_S
        self.control_XU0 = self.planned_XU0
        self.desired_sol = None
        self.planned_sol = None
        self.planned_control = None
        # self.desired_control = None
        self.iter_desired = 0
        self.max_iter_desired = 5
        self.n_margin_bottom = 0.3
        self.n_margin_top = 0.3
        # self.n_points_match = 5
        self.n_points_match = 1
        self.P_road_v = [0] * 8
        self.q = np.diag([1,100,100,1])
        self.r = np.diag([1,1])*0.1
        self.create_car_F()
        self.dt_factor = 30
        self.compute_K(dt)
        self.create_opti()
        self.v_p_weight_P_paths = [0, 0]
        self.history_planned_control = []
        self.history_control = []
        self.dt = dt
        
        
    def get_state(self):
        return self.state[:nt]

    def create_car_F(self):  # create a discrete model of the vehicle
        nx = self.nx
        nu = self.nu
        x = cs.SX.sym('x', nx)
        u = cs.SX.sym('u', nu)
        # states: v_s longitudinal speed, v_n lateral speed, s longitudinal position, time, n lateral position
        v_s, s, n, t, v_n = x[0], x[1], x[2], x[3], x[4]
        # controls: a_s longitudinal acceleration, a_n lateral acceleration
        a_s, a_n = u[0], u[1]
        dot_v_s = a_s
        dot_s = v_s
        dot_n = v_n
        dot_t = 1
        dot_v_n = a_n
        dot_x = cs.vertcat(dot_v_s, dot_s, dot_n, dot_t, dot_v_n)
        f = cs.Function('f', [x, u], [dot_x])
        dt = cs.SX.sym('dt', 1)  # time step in optimization
        k1 = f(x, u)
        k2 = f(x+dt/2*k1, u)
        k3 = f(x+dt/2*k2, u)
        k4 = f(x+dt*k3, u)
        x_kp1 = x+dt/6*(k1+2*k2+2*k3+k4)
        F = cs.Function('F', [x, u, dt], [x_kp1])  # x_k+1 = F(x_k, u_k, dt)
        self.car_F = F  # save the vehicle model in the object variable
        self.K_dot_x = cs.vertcat(dot_v_s, dot_s, dot_n, dot_v_n)
        self.K_x = cs.vertcat(v_s, s, n, v_n)
        self.K_u = u

    def compute_K(self, dt_sim):
        
        nx = self.nx
        nu = self.nu
        nx = nx - 1
        x0 = cs.SX.sym('x', nx)
        u0 = cs.SX.sym('u', nu)
        # dot_x = cs.vertcat(dot_v_s, dot_s, dot_n, dot_v_n)
        # x = cs.vertcat(v_s, s, n, v_n)
        xAxb = cs.linearize(self.K_dot_x,self.K_x,x0)
        xAb = cs.linear_coeff(xAxb,self.K_x)
        xA = xAb[0]
        uAxb = cs.linearize(self.K_dot_x,self.K_u,u0)
        uAb = cs.linear_coeff(uAxb,self.K_u)
        uA = uAb[0]
        vx0 = [0] * nx
        vu0 = [0] * nu
        fxA = cs.Function('fxA',[x0,self.K_u],[xA])
        vxA = fxA(vx0,vu0)
        fuA = cs.Function('fuA',[self.K_dot_x,u0],[uA])
        vuA = fuA(vx0,vu0)
        # dt = self.ref_sw.dt
        dt = dt_sim
        newA = vxA*dt + np.eye(nx)
        a = newA.full()
        newB = vuA*dt
        b = newB.full()
        from scipy import linalg as la
        P = la.solve_discrete_are(a, b, self.q, self.r)
        R = la.solve(self.r + b.T.dot(P).dot(b), b.T.dot(P).dot(a))
        self.R = R

        # R gain for predicting simulation
        dt = self.planning_dt/self.dt_factor
        newA = vxA*dt + np.eye(nx)
        a = newA.full()
        newB = vuA*dt
        b = newB.full()
        P = la.solve_discrete_are(a, b, self.q, self.r)
        R = la.solve(self.r + b.T.dot(P).dot(b), b.T.dot(P).dot(a))
        self.R_planning = R


    def create_opti(self):  # create an optimization problem
        N = self.planning_points - 1
        opti = cs.Opti()
        nx = self.nx
        nu = self.nu
        XU = opti.variable(1, nx*(N+1)+nu*N)
        v_s = XU[0::(nx+nu)]
        s = XU[1::(nx+nu)]
        n = XU[2::(nx+nu)]
        t = XU[3::(nx+nu)]
        v_n = XU[4::(nx+nu)]
        a_s = XU[5::(nx+nu)]
        a_n = XU[6::(nx+nu)]
        X = cs.vertcat(v_s, s, n, t, v_n)
        U = cs.vertcat(a_s, a_n)
        P_road = opti.parameter(1, 4*2)
        P_paths = opti.parameter(1, (N+1)*2*2)
        p_weight_P_paths = opti.parameter(1, 2)
        p_x0 = opti.parameter(nx, 1)
        p_nf = opti.parameter(1)
        p_weight_a_s = opti.parameter(1)
        p_weight_a_n = opti.parameter(1)
        p_v_s_nominal = opti.parameter(1)
        p_min_s = opti.parameter(1)
        p_min_n = opti.parameter(1)
        p_slope_s = opti.parameter(1)
        p_slope_n = opti.parameter(1)
        p_weight_match = opti.parameter(1, N)
        P_match = opti.parameter(nx, N)
        p_s_P_switch = opti.parameter(1, 2)

        # parameters to define minimum distance between two vehicles
        opti.set_value(p_min_s, 61+10)
        opti.set_value(p_min_n, 0.25*2)
        opti.set_value(p_slope_s, 0.05*10)
        opti.set_value(p_slope_n, 2)
        p_dt = opti.parameter(1)

        # Penalty on longitudinal acceleration, lateral acceleration, velocity deviation from the nominal velocity, final lateral velocity, deviation from the center point on the main lane
        n_at_end = (P_road[0]/2 * (cs.tanh(P_road[1]*(s[-1]-P_road[2]))+1)+P_road[3]
                  + P_road[4]/2 * (cs.tanh(P_road[5]*(s[-1]-P_road[6]))+1)+P_road[7])/2
        # n_at_end = 1
        J = p_weight_a_s*cs.sumsqr(a_s) + p_weight_a_n * cs.sumsqr(a_n) + 0.1*1e-1*cs.sumsqr(
            v_s - p_v_s_nominal) + 1e3*cs.sumsqr(v_n[-1]) + 1e2*cs.sumsqr(n[-1] - n_at_end) # - p_nf)  # 1e3*(n[-1])**2 #+
        for i in range(N+1):
            # penalty on the distance to first of the surrounding vehicles
            J += p_weight_P_paths[0]*1/2*(cs.tanh(s[i] - p_s_P_switch[0]) + 1)*1/2*(
                cs.tanh(p_slope_s/2*(p_min_s**2 - (s[i]-P_paths[i+(N+1)*0])**2)) + 1)
            # penalty on the distance to second of the surrounding vehicles
            J += p_weight_P_paths[1]*1/2*(cs.tanh(s[i] - p_s_P_switch[1]) + 1)*1/2*(
                cs.tanh(p_slope_s/2*(p_min_s**2 - (s[i]-P_paths[i+(N+1)*2])**2)) + 1)
            # penalty on the vehicle to be too close to the road boundaries
            J += 1e1*cs.sumsqr(cs.fmax((P_road[0]/2*(cs.tanh(P_road[1]*(
                s[i]-P_road[2]))+1)+P_road[3]) - n[i] + self.n_margin_bottom, 0))
            J += 1e1*cs.sumsqr(cs.fmax(n[i] - (P_road[4]/2*(
                cs.tanh(P_road[5]*(s[i]-P_road[6]))+1)+P_road[7]) + self.n_margin_top, 0))

        # Penalty to match first points
        for i in range(N):
            J += p_weight_match[i]*cs.sumsqr((P_match[C.V_S, i]-X[C.V_S, i+1]))
            J += p_weight_match[i]*cs.sumsqr((P_match[C.V_N, i]-X[C.V_N, i+1]))
            J += p_weight_match[i]*cs.sumsqr((P_match[C.S, i]-X[C.S, i+1]))
            J += p_weight_match[i]*cs.sumsqr((P_match[C.N, i]-X[C.N, i+1]))

        opti.minimize(J)
        opti.subject_to(X[:, 0] == p_x0)
        for k in range(N+1):
            if k < N:
                # model constraints
                opti.subject_to(
                    X[:, k+1] - self.car_F(X[:, k], U[:, k], p_dt) == 0)
                # input constraints
                opti.subject_to(opti.bounded(-5, a_s[k], 5))
                opti.subject_to(opti.bounded(-1, a_n[k], 1))
            # velocity constraints
            opti.subject_to(opti.bounded(1, v_s[k], 35))
            # road constraints
            opti.subject_to(n[k] >= P_road[0]/2 *
                            (cs.tanh(P_road[1]*(s[k]-P_road[2]))+1)+P_road[3])
            opti.subject_to(n[k] <= P_road[4]/2 *
                            (cs.tanh(P_road[5]*(s[k]-P_road[6]))+1)+P_road[7])

        # Parameters to define optimization solver
        opti_options = {}
        opti_options['expand'] = True
        # opti_options['ipopt.print_level'] = 5 # print logs of the solver
        opti_options['ipopt.print_level'] = 0  # do not print
        opti_options['print_time'] = False
        # opti_options['ipopt.linear_solver'] = "ma57" # needs to be requested and installed separately
        opti_options['ipopt.sb'] = 'yes'
        opti.solver("ipopt", opti_options)

        # Create a parameterized function to solve the optimization problem
        self.solver = opti.to_function('solver', [p_x0, p_nf, p_weight_a_s, p_weight_a_n, p_v_s_nominal, p_dt, P_road, P_paths, p_weight_P_paths, p_weight_match, P_match, p_s_P_switch, XU],
                                       [X, U, XU, J])

    def get_stats(self):
        # get stats
        # https://gist.github.com/jgillis/9d12df1994b6fea08eddd0a3f0b0737f
        # "Note: this is very hacky."
        dep = None
        f = self.solver
        # Loop over the algorithm
        for k in range(f.n_instructions()):
            if f.instruction_id(k) == cs.OP_CALL:
                d = f.instruction_MX(k).which_function()
                if d.name() == "solver":
                    dep = d
                    break
        if dep is None:
            return {}
        else:
            # print("dep result is",dep.stats(1),"end print")

            return dep.stats(1)

    def compute_planned_desired_trajectory(self):
        """Compute planned and desired trajectory for CAV"""
        current_time = self.state[C.T]  # HACK, take current time from the current vehicle state
        for v in self.vehicles:
            self.vehicles[v].interpolate_trajectories(
                current_time, self.planning_points, self.planning_dt)

        flagDesired = False
        self.P_path_v = np.zeros((4, self.planning_points))
        if len(self.vehicles) > 0: # if we know about at least one vehicle
            vA = list(self.vehicles.keys())[0]
            if self.vehicles[vA].desired_trajectory is not None:
                self.P_path_v[0:2, :] = self.vehicles[vA].interpolated_desired_trajectory[[
            C.S, C.N], :]  # s and n of v1
                flagDesired = True
            else:
                self.P_path_v[0:2, :] = self.vehicles[vA].interpolated_planned_trajectory[[
            C.S, C.N], :]  # s and n of v1
        if len(self.vehicles) > 1: # if we know about two vehicles
            vB = list(self.vehicles.keys())[1]
            if self.vehicles[vA].desired_trajectory is not None:
                self.P_path_v[2:4, :] = self.vehicles[vB].interpolated_desired_trajectory[[
            C.S, C.N], :]  # s and n of v1
                flagDesired = True
            else:
                self.P_path_v[2:4, :] = self.vehicles[vB].interpolated_planned_trajectory[[
            C.S, C.N], :]  # s and n of v1
        else:
            self.P_path_v[2:4, :] = self.P_path_v[0:2, :] # if we know about only one vehicles, we make the second one to be the same as the first

        steer_weight = 1e3
        # x0 = self.state
        # "project" state to the plan;
        # a new plan is not computed from the current state but from the projected state on the previous plan.
        # This is to prevent a sudden change of the error in the track following algorithms, we keep the "old" tracking error.
        if self.planned_sol is not None:
            tt = self.planned_trajectory[C.T, :]
            t = current_time
            x0 = [0] * self.nx
            for i in range(self.nx):
                x0[i] = scipy.interpolate.interp1d(tt,self.planned_XU0[i::(self.nx+self.nu)],kind='cubic')(t)
                

            x0[C.T] = self.state[C.T]
            
        else:
            x0 = self.state
        # print('this is x0')
        # print(x0)
        print(self.name, 'current/projected state for s', self.state[C.S], x0[C.S])
        print(self.name, 'current/projected state for v_s', self.state[C.V_S], x0[C.V_S])

        # At start only planned
        # Check if there is a conflict
        # rename planned to desired
        # plan new planned
        # on next iteration plan both, if desired not None?
        # set limit on number of desired iterations
        # keep following planned

        # We want the first points of a new plan to be similar to the previous plan
        if self.planned_sol is not None:
            local_XU_0 = np.zeros_like(self.planned_XU0)
            old_tt = self.planned_trajectory[C.T, :]
            new_tt = [self.state[C.T] + self.planning_dt *
                      i for i in range(self.planning_points)]
            for i in range(self.nx):
                local_XU_0[i::(self.nx+self.nu)] = scipy.interpolate.interp1d(
                    old_tt, self.planned_XU0[i::(self.nx+self.nu)], fill_value='extrapolate', kind='cubic')(new_tt)
            # print('this is first interp')
            # print(local_XU_0)
            for i in range(self.nx, self.nx+self.nu):
                local_XU_0[i::(self.nx+self.nu)] = scipy.interpolate.interp1d(old_tt[:-1],
                                                                              self.planned_XU0[i::(self.nx+self.nu)], fill_value='extrapolate', kind='cubic')(new_tt[:-1])
            # print('this is second interp')
            # print(local_XU_0)
            v_p_weight_match = 1e2*(1 - 0.5*(np.tanh(10/self.n_points_match*(
                np.arange(self.planning_points-1) - self.n_points_match))+1))
            # v_p_weight_match = np.zeros(self.planning_points-1)
            v_P_match = np.zeros([self.nx, self.planning_points-1])
            v_P_match[C.V_S, :] = local_XU_0[C.V_S::(self.nx+self.nu)][1:]
            v_P_match[C.S, :] = local_XU_0[C.S::(self.nx+self.nu)][1:]
            v_P_match[C.N, :] = local_XU_0[C.N::(self.nx+self.nu)][1:]
            v_P_match[C.V_N, :] = local_XU_0[C.V_N::(self.nx+self.nu)][1:]
        else:
            local_XU_0 = self.planned_XU0
            v_P_match = np.zeros([self.nx, self.planning_points-1])
            v_p_weight_match = np.zeros(self.planning_points-1)

        v_p_s_P_switch = [self.P_road_v[6], self.P_road_v[6]]

        # Check and plan desired trajectory if needed
        if self.desired_sol is not None:
            for i in range(self.nx):
                local_XU_0[i::(self.nx+self.nu)] = scipy.interpolate.interp1d(
                    old_tt, self.desired_XU0[i::(self.nx+self.nu)], fill_value='extrapolate', kind='cubic')(new_tt)
            for i in range(self.nx, self.nx+self.nu):
                local_XU_0[i::(self.nx+self.nu)] = scipy.interpolate.interp1d(old_tt[:-1],
                                                                              self.desired_XU0[i::(self.nx+self.nu)], fill_value='extrapolate', kind='cubic')(new_tt[:-1])
            sol = self.solver(x0,  # optimize a trajectory, initial point is the current state [longitudinal velocity, lateral velocity, longitudinal position, lateral position]
                              self.lane_width/2,  # set the desired final lateral position
                              1e0,  # weight on the longitudinal acceleration
                              steer_weight,
                              self.v0,  # reduce the competition between reaching the max speed and keeping the safe distance by updating the reference speed
                              self.planning_dt, self.P_road_v, self.P_path_v.ravel(), [
                                  0, 0],
                              v_p_weight_match, v_P_match,
                              v_p_s_P_switch,
                              local_XU_0)  # provide previous solution as an initial guess
            status = self.get_stats()
            print(self.name, self.state[C.S], 'planning desired',
                  'Status', status['return_status'], 'J', sol[-1])
            self.desired_XU0 = sol[2].full()[0]
            self.desired_trajectory = sol[0].full()[:4, :]  # eliminate V_N
            self.desired_sol = sol

        # Plan planned trajectory
        # if requested by other vehicles
        if flagDesired:
            v_p_weight_P_paths = [100, 100]
        else:
            v_p_weight_P_paths = self.v_p_weight_P_paths
        sol = self.solver(x0,  # optimize a trajectory, initial point is the current state [longitudinal velocity, lateral velocity, longitudinal position, lateral position]
                          self.lane_width/2,  # set the desired final lateral position
                          1e0,  # weight on the longitudinal acceleration
                          steer_weight,
                          self.v0,  # reduce the competition between reaching the max speed and keeping the safe distance by updating the reference speed
                                self.planning_dt, self.P_road_v, self.P_path_v.ravel(), v_p_weight_P_paths,
                          v_p_weight_match, v_P_match,
                          v_p_s_P_switch,
                          local_XU_0)  # provide previous solution as an initial guess
        status = self.get_stats()
        print(self.name, self.state[C.S], 'planning planned', 'Status',
              status['return_status'], 'J', sol[-1])

        # If conflict detected rename planned as desired, plan planned again
        v_trajectory = sol[0].full()
        # print('this is v_trajectory', v_trajectory,  "this")
        if v_trajectory[C.N, 0] < -self.lane_width/2 and v_trajectory[C.N, -1] > -self.lane_width/2 and self.desired_sol is None:
            # This is only a indication of a potential conflict
            # TODO Check for a conflict of trajectories as well
            print('Conflict detected')
            self.desired_XU0 = sol[2].full()[0]
            self.desired_trajectory = v_trajectory[:4, :]  # eliminate V_N
            self.desired_sol = sol
            self.v_p_weight_P_paths = [1, 1]
            sol = self.solver(x0,  # optimize a trajectory, initial point is the current state [longitudinal velocity, lateral velocity, longitudinal position, lateral position]
                              self.lane_width/2,  # set the desired final lateral position
                              1e0,  # weight on the longitudinal acceleration
                              steer_weight,
                              self.v0,  # reduce the competition between reaching the max speed and keeping the safe distance by updating the reference speed
                              self.planning_dt, self.P_road_v, self.P_path_v.ravel(), self.v_p_weight_P_paths,
                              v_p_weight_match, v_P_match,
                              v_p_s_P_switch,
                              local_XU_0)  # provide previous solution as an initial guess
            status = self.get_stats()
            print(self.name, self.state[C.S], 'planning planned after conflict detection',
                  'Status', status['return_status'], 'J', sol[-1])

        self.planned_XU0 = sol[2].full()[0]
        self.planned_trajectory = sol[0].full()[:4, :]  # eliminate V_N
        self.planned_sol = sol
        self.history_planned_trajectory.append(self.planned_trajectory)
        self.history_desired_trajectory.append(self.desired_trajectory)

        return self.planned_trajectory, self.desired_trajectory
    

    def simulate(self, dt):
        """Compute new state in simulation"""
        # Trajectory following using LQR
        e = np.zeros([self.nx-1,1])
        tt = self.planned_trajectory[C.T, :]
        t = self.state[C.T]
        e[0] = self.state[C.V_S] - scipy.interpolate.interp1d(tt,self.planned_XU0[C.V_S::(self.nx+self.nu)],kind='cubic')(t)
        e[1] = self.state[C.S] - scipy.interpolate.interp1d(tt,self.planned_XU0[C.S::(self.nx+self.nu)],kind='cubic')(t)
        e[2] = self.state[C.N] - scipy.interpolate.interp1d(tt,self.planned_XU0[C.N::(self.nx+self.nu)],kind='cubic')(t)
        e[3] = self.state[C.V_N] - scipy.interpolate.interp1d(tt,self.planned_XU0[C.V_N::(self.nx+self.nu)],kind='cubic')(t)
        u = -np.matmul(self.R,e)
        self.state = self.car_F(self.state, u, dt).full().ravel()
        self.history_state.append(self.get_state())
        self.history_control.append(u)
        

class Truck_CC(Car):
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
    
    
    def simulate(self, dt):
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
        self.state = self.car_F(self.state, u, dt).full().ravel()
        self.history_state.append(self.get_state())
        self.history_control.append(u)
        self.history_v_ref.append(self.v_ref)