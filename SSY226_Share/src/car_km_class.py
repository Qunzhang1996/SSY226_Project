import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import casadi as cs
from enum import IntEnum
from vehicle_class import C, ST,C_k,Vehicle
# from kinematic_test import VehicleKinematic
import warnings
warnings.simplefilter("error")
nt=4
L=4
class Car_km(Vehicle):
    def __init__(self, state,dt, state_km=np.zeros(5)):
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
        self.q = np.diag([10,10,0.1,0.01])
        self.r = np.diag([1,1])
        self.create_car_F()
        self.dt_factor = 30
        self.compute_K(dt)
        self.create_opti()
        self.v_p_weight_P_paths = [0, 0]
        self.history_planned_control = []
        self.history_control = []
        self.dt = dt
        #TODO: add the following two lines with the update of the km of state and input
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.state_km = np.zeros(self.nx)
        self.u_km = np.zeros(self.nu)
        
        
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
        #no need to change, here, we get the s and n, or we can view it as X_km and Y_km
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
            # print("this is status")
            # print(status)
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
    


    def create_car_F_km(self):  # create a discrete model of the vehicle
        nx = self.nx
        nu = self.nu
        x = cs.SX.sym('x', nx)
        u = cs.SX.sym('u', nu)
        # X_km, Y_km, Psi, T, V_km 
        # states: x_km longitudinal speed, y_km lateral speed, psi heading, time, v_km velocity
        x_km, y_km, psi, t, v_km  = x[0], x[1], x[2], x[3], x[4]
        # controls: a_km acceleration, delta steering angle
        a_km, delta = u[0], u[1]
        dot_x_km = v_km*np.cos(psi)
        dot_y_km = v_km*np.sin(psi)
        dot_psi = v_km/L*np.tan(delta)
        dot_t = 1
        dot_v_km = a_km
        dot_x = cs.vertcat(dot_x_km, dot_y_km, dot_psi, dot_t, dot_v_km)
        f = cs.Function('f', [x, u], [dot_x])
        dt = cs.SX.sym('dt', 1)  # time step in optimization
        k1 = f(x, u)
        k2 = f(x+dt/2*k1, u)
        k3 = f(x+dt/2*k2, u)
        k4 = f(x+dt*k3, u)
        x_kp1 = x+dt/6*(k1+2*k2+2*k3+k4)
        F = cs.Function('F', [x, u, dt], [x_kp1])  # x_k+1 = F(x_k, u_k, dt)
        self.car_F_km = F  # save the vehicle model in the object variable
        self.K_dot_x = cs.vertcat(dot_x_km, dot_y_km, dot_psi, dot_v_km)
        self.K_x = cs.vertcat(x_km, y_km, psi, v_km)
        self.K_u = cs.vertcat(a_km, delta)
        self.kinematic_car_model = cs.Function('kinematic_car_model', [self.K_x, self.K_u], [self.K_dot_x])


    def calculate_AB(self,dt_sim):
        nx = self.nx
        nu = self.nu
        nx = nx - 1
        x_op = self.state_km[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km ]]
        u_op = self.u_km
        self.create_car_F_km()
        # self.create_car_F_km()
        # Define state and control symbolic variables
        x = cs.SX.sym('x', nx)
        u = cs.SX.sym('u', nu)

        # Get the state dynamics from the kinematic car model
        state_dynamics = self.kinematic_car_model(x, u)

        # Calculate the Jacobians for linearization
        A = cs.jacobian(state_dynamics, x)
        B = cs.jacobian(state_dynamics, u)
        # Create CasADi functions for the linearized matrices
        f_A = cs.Function('A', [x, u], [A])
        f_B = cs.Function('B', [x, u], [B])

        # Evaluate the Jacobians at the operating point
        A_op = f_A(x_op, u_op)
        B_op = f_B(x_op, u_op)

        # Discretize the linearized matrices
        newA = A_op * dt_sim + np.eye(self.nx-1)
        # print(newA)
        newB = B_op * dt_sim
        # print(newB)

        return newA.full(), newB.full()
    
    def compute_km_K(self):
        a, b = self.calculate_AB(self.dt)
        from scipy import linalg as la
        P = la.solve_discrete_are(a, b, self.q, self.r)
        R = la.solve(self.r + b.T.dot(P).dot(b), b.T.dot(P).dot(a))
        return R


    def compute_km_mpc(self,error0, N=60):
        nx = self.nx
        nu = self.nu
        nx = nx - 1
        a,b=self.calculate_AB(self.dt)
        from scipy import linalg as la
        # Setup the optimization problem
        opti = cs.Opti()  # create QP problem
        # Decision variables for state and inpu
        X = opti.variable(nx, N+1)  
        U = opti.variable(nu, N)  
         # Objective function
        obj = 0  # initate obj func
        for i in range(N):
            obj += cs.mtimes([X[:, i].T, self.q, X[:, i]])  # 状态代价
            obj += cs.mtimes([U[:, i].T, self.r, U[:, i]])  # 控制代价
        # Add the objective function to the optimization problem
        opti.minimize(obj)
        # constraints
        for i in range(N):
            opti.subject_to(X[:, i+1] == cs.mtimes(a, X[:, i]) + cs.mtimes(b, U[:, i]))
        # initial constraints
        opti.subject_to(X[:, 0] == error0)
        # Control input constraints
        # Control input constraints
        u_min = [-5, -np.pi / 6]
        u_max = [5, np.pi / 6]

        for j in range(nu):
            opti.subject_to(opti.bounded(u_min[j], U[j, :], u_max[j]))
        # Constraint to iput lead to problem!!!!!!!!!!!!!!
        # Configure the solver
        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver('ipopt')

        # Solve the optimization problem
        sol = opti.solve()

        # Get the optimal control input sequence
        u_optimal = sol.value(U[:, 0])
        return u_optimal
    

    def find_target_point(trajectory, point, shift_points, last_index=0):
        # Calculate the squared Euclidean distance to each point in the trajectory
        distances = np.sum((trajectory - point) ** 2, axis=1)
        # Find the index of the closest point
        closest_idx = np.argmin(distances)
        target_idx = closest_idx + shift_points
        if target_idx > len(trajectory) - 1:
            target_idx = len(trajectory) - 1
        if target_idx <= last_index:
            target_idx = last_index
        # Return the closest point and its index
        target_point = trajectory[target_idx]
        return target_point, target_idx


    # calculate the direction of the trajectory
    def calculate_direction(x, y):
        dy = np.diff(y, prepend=y[0])
        dx = np.diff(x, prepend=x[0])
        psi = np.arctan2(dy, dx)
        return psi
    


    # here is the transformation between mass_point and km state
    def transformation_km2mp(self, state):
        """Transfer from  km state to mass_point state"""
        x_km, y_km, psi, t, v_km = state[0], state[1], state[2], state[3], state[4]
        x = x_km
        y = y_km
        psi = np.arctan2(np.sin(psi), np.cos(psi))
        v = v_km
        v_s=v*np.cos(psi)
        v_n=v*np.sin(psi)
        s=x
        n=y
        return np.array([v_s, s, n, t, v_n])
    
    def transformation_mp2km(self, state):
        """Transfer from mass_point to km state"""
        v_s, s, n, t, v_n = state[0], state[1], state[2], state[3], state[4]
        x_km = s
        y_km = n
        psi = np.arctan2(np.sin(np.arctan2(v_n, v_s)), np.cos(np.arctan2(v_n, v_s)))
        v_km = np.sqrt(v_s**2+v_n**2)
        return np.array([x_km, y_km, psi, t, v_km])

    def input_transform(self, u):
        """Transfer from km input to mass_point input, a, delta to ax,ay"""
        a, delta = u[0], u[1]
        ax = a*np.cos(delta)
        ay = a*np.sin(delta)
        return np.array([ax, ay])
    
    def input_transform_inv(self, u):
        """Transfer from mass_point input to km input, ax,ay to a, delta"""
        ax, ay = u[0], u[1]
        a = np.sqrt(ax**2+ay**2)
        delta = np.arctan2(ay, ax)
        if delta > np.pi/2:
            delta = delta - np.pi
        elif delta < -np.pi/2:
            delta = delta + np.pi
        return np.array([a, delta])




    def simulate(self, dt, outside_carla_state=np.zeros([5,1])):
        """compute new state in simulation, transfer from mass_point to km state
        and transfer from km state to mass_point state"""

        #To test, try to transform the pm state to km state
        outside_carla_state=self.transformation_mp2km(self.state).reshape(-1,1)


        #TODO: add the carla state
        state_carla=np.zeros([5,1])#"represent this with the state from Carla"   X, Y, PSI, T, V
        state_carla[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.T, C_k.V_km]]=outside_carla_state
        state_mp_planned=np.zeros([self.nx,1])
        tt = self.planned_trajectory[C.T, :]
        t = self.state[C.T]
        state_carla[C_k.T]=t
        
        #get target point of mass_point
        state_mp_planned[C.V_S] = scipy.interpolate.interp1d(tt,self.planned_XU0[C.V_S::(self.nx+self.nu)],kind='cubic')(t)
        state_mp_planned[C.S]= scipy.interpolate.interp1d(tt,self.planned_XU0[C.S::(self.nx+self.nu)],kind='cubic')(t)
        state_mp_planned[C.N]= scipy.interpolate.interp1d(tt,self.planned_XU0[C.N::(self.nx+self.nu)],kind='cubic')(t)
        state_mp_planned[C.V_N]= scipy.interpolate.interp1d(tt,self.planned_XU0[C.V_N::(self.nx+self.nu)],kind='cubic')(t)
        state_mp_planned[C.T]=t
        #transfer from mass_point to km state
        state_km_planned=self.transformation_mp2km(state_mp_planned)

        #make sure the velocity is not zero
        if state_carla[C_k.V_km]==0:
            state_carla[C_k.V_km]=0.001

        #input the state of carla vehicle to state_km to calculate a,b and mpc
        self.state_km=state_carla

        #calculate the error
        e=np.zeros([self.nx-1,1])
        e[0]=state_carla[C_k.X_km]-state_km_planned[C_k.X_km]
        e[1]=state_carla[C_k.Y_km]-state_km_planned[C_k.Y_km]
        e[2]=state_carla[C_k.Psi]-state_km_planned[C_k.Psi]
        e[3]=state_carla[C_k.V_km]-state_km_planned[C_k.V_km]

        #make sure the heading error is in the range of [-pi,pi]
        e[2]=np.arctan2(np.sin(e[2]),np.cos(e[2]))
        # print('this is error', e)
        #calculate the input through MPC
        #TODO:have to check the update in the code above AB Calculation
        u_optimal=self.compute_km_mpc(e)
        # R=self.compute_km_K()
        # u_optimal=-np.matmul(R,e)

        self.u_km=u_optimal
        #TODO:output the input of the km vehicle to the carla vehicle
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        #transfer from km input to mass_point input
        u_optimal_mp=self.input_transform(u_optimal)
        print('this is u_optimal_mp', u_optimal_mp)
        #transfer carla state to mp state
        state_mp_carla=self.transformation_km2mp(state_carla)

        #update the state of the pm vehicle
        self.state = self.car_F(state_mp_carla, u_optimal_mp, dt).full().ravel()
        self.history_state.append(self.get_state())
        self.history_control.append(u_optimal_mp)



    # def simulate(self, dt):
    #     """Compute new state in simulation"""
    #     # Trajectory following using LQR
    #     e = np.zeros([self.nx-1,1])
    #     tt = self.planned_trajectory[C.T, :]
    #     t = self.state[C.T]
    #     e[0] = self.state[C.V_S] - scipy.interpolate.interp1d(tt,self.planned_XU0[C.V_S::(self.nx+self.nu)],kind='cubic')(t+0.5)
    #     e[1] = self.state[C.S] - scipy.interpolate.interp1d(tt,self.planned_XU0[C.S::(self.nx+self.nu)],kind='cubic')(t+0.5)
    #     e[2] = self.state[C.N] - scipy.interpolate.interp1d(tt,self.planned_XU0[C.N::(self.nx+self.nu)],kind='cubic')(t+0.5)
    #     e[3] = self.state[C.V_N] - scipy.interpolate.interp1d(tt,self.planned_XU0[C.V_N::(self.nx+self.nu)],kind='cubic')(t+0.5)
    #     u = -np.matmul(self.R,e)
    #     self.state = self.car_F(self.state, u, dt).full().ravel()
    #     self.history_state.append(self.get_state())
    #     self.history_control.append(u)
