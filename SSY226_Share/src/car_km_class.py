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


#TODO:addparser!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#parser!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class Car_km(Vehicle):
    def __init__(self, state, dt,nt=4,L=4):
        self.nt=nt
        super().__init__(state)
        self.L=L
        self.nx = 5
        self.nu = 2
        self.state = np.zeros(self.nx)
        self.state[:self.nt] = state
        self.state[C_k.V_km] = 0
        self.u=np.zeros(self.nu)
        self.desired_XU0 = None
        self.v0 = 25
        # X_km, Y_km, Psi, T, V_km 
        self.planned_XU0 = [0] * (self.nx*(self.planning_points)+self.nu*(self.planning_points - 1))
        self.planned_XU0[C_k.V_km::(self.nx+self.nu)] = np.linspace(self.state[C_k.V_km], self.v0,self.planning_points)
        self.planned_XU0[C_k.T::(self.nx+self.nu)] = self.state[C_k.T] + self.planning_dt*np.arange(self.planning_points)
        local_S = np.zeros(self.planning_points)#calculate s longitudinal position
        local_S[0] = self.state[C_k.Y_km]
        for i in range(self.planning_points-1):
            # local_S[i+1] = local_S[i] + self.planned_XU0[C.V_S + i*(self.nx+self.nu)]*self.planning_dt
            delta_psi = self.planned_XU0[C_k.Psi + (i+1)*(self.nx+self.nu)] - self.planned_XU0[C_k.Psi + i*(self.nx+self.nu)]
            total_speed = self.planned_XU0[C_k.V_km + i*(self.nx+self.nu)]
            lateral_displacement = total_speed * self.planning_dt * np.sin(delta_psi)
            local_S[i+1] = local_S[i] + lateral_displacement
        self.planned_XU0[C_k.Y_km::(self.nx+self.nu)] = local_S
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
        self.q = np.diag([1,1,100,100])
        self.r = np.diag([1,1])*0.1
        # self.create_car_F()
        
        self.dt_factor = 30
        self.calculate_AB()
        self.compute_km_K()
        self.create_opti()
        self.v_p_weight_P_paths = [0, 0]
        self.history_planned_control = []
        self.history_control = []
        self.dt = dt

    def get_state(self):
        return self.state[:self.nt]

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
        dot_psi = v_km/self.L*np.tan(delta)
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
        self.car_F = F  # save the vehicle model in the object variable
        self.K_dot_x = cs.vertcat(dot_x_km, dot_y_km, dot_psi, dot_v_km)
        self.K_x = cs.vertcat(x_km, y_km, psi, v_km)
        self.K_u = cs.vertcat(a_km, delta)
        self.kinematic_car_model = cs.Function('kinematic_car_model', [self.K_x, self.K_u], [self.K_dot_x])
        
        
    def calculate_AB(self,dt_sim=1):
        nx = self.nx
        nu = self.nu
        nx = nx - 1
        x_op = self.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km ]]
        u_op = self.u
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
        print(newA)
        newB = B_op * dt_sim

        return newA.full(), newB.full()
    
    def compute_km_K(self):
        a,b=self.calculate_AB()
        from scipy import linalg as la
        P = la.solve_discrete_are(a, b, self.q, self.r)
        R = la.solve(self.r + b.T.dot(P).dot(b), b.T.dot(P).dot(a))
        self.R = R

    def compute_K(self, dt_sim):
        a,b=self.calculate_AB()
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
    
    def compute_MPC(self, dt_sim, error0, N=30):
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
        # Setup the optimization problem
        opti = cs.Opti()  # create QP problem
        # Decision variables for state and inpu
        X = opti.variable(nx, N+1)  # 状态变量，每一列对应一个时间步
        U = opti.variable(nu, N)    # 控制变量，每一列对应一个时间步
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
        # u_min = 0    # 
        # u_max = 50   # 
        # opti.subject_to(opti.bounded(u_min, U, u_max)) 
        # Constraint to iput lead to problem!!!!!!!!!!!!!!
        # Configure the solver
        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver('ipopt')

        # Solve the optimization problem
        sol = opti.solve()

        # Get the optimal control input sequence
        u_optimal = sol.value(U[:, 0])

        return u_optimal
    
    def compute_km_mpc(self, dt_sim, error0, N=30):
        nx = self.nx
        nu = self.nu
        nx = nx - 1
        a,b=self.calculate_AB()
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
        # u_min = 0    # 
        # u_max = 50   # 
        # opti.subject_to(opti.bounded(u_min, U, u_max)) 
        # Constraint to iput lead to problem!!!!!!!!!!!!!!
        # Configure the solver
        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver('ipopt')

        # Solve the optimization problem
        sol = opti.solve()

        # Get the optimal control input sequence
        u_optimal = sol.value(U[:, 0])
        return u_optimal


    def create_opti(self):  # create an optimization problem
        N = self.planning_points - 1
        opti = cs.Opti()
        nx = self.nx
        nu = self.nu
        XU = opti.variable(1, nx*(N+1)+nu*N)
        #X_km, Y_km, Psi, T, V_km 
        x_km = XU[0::(nx+nu)]
        y_km = XU[1::(nx+nu)]
        psi = XU[2::(nx+nu)]
        t = XU[3::(nx+nu)]
        v_km = XU[4::(nx+nu)]   
        a_km = XU[5::(nx+nu)]
        delta = XU[6::(nx+nu)]

        X = cs.vertcat(x_km, y_km, psi, v_km)
        U = cs.vertcat(a_km, delta)
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


        #TODO:change the cost func
        # Penalty on longitudinal acceleration, lateral acceleration, velocity deviation from the nominal velocity, final lateral velocity, deviation from the center point on the main lane
        # n_at_end = (P_road[0]/2 * (cs.tanh(P_road[1]*(s[-1]-P_road[2]))+1)+P_road[3]
        #           + P_road[4]/2 * (cs.tanh(P_road[5]*(s[-1]-P_road[6]))+1)+P_road[7])/2
        n_at_end = (P_road[0]/2 * (cs.tanh(P_road[1]*(y_km[-1]-P_road[2]))+1)+P_road[3]
                  + P_road[4]/2 * (cs.tanh(P_road[5]*(y_km[-1]-P_road[6]))+1)+P_road[7])/2
        # n_at_end = 1
        # J = p_weight_a_s*cs.sumsqr(a_s) + p_weight_a_n * cs.sumsqr(a_n) + 0.1*1e-1*cs.sumsqr(
        #     v_s - p_v_s_nominal) + 1e3*cs.sumsqr(v_n[-1]) + 1e2*cs.sumsqr(x_km[-1] - n_at_end) # - p_nf)  # 1e3*(n[-1])**2 #+
        J = p_weight_a_s*cs.sumsqr(a_km) + p_weight_a_n * cs.sumsqr(delta) + 0.1*1e-1*cs.sumsqr(
            v_km - p_v_s_nominal) + 1e3*cs.sumsqr(v_km[-1]) + 1e2*cs.sumsqr(x_km[-1] - n_at_end) # - p_nf)  # 1e3*(n[-1])**2 #+
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
        current_time = self.state[C_k.T]  # HACK, take current time from the current vehicle state
        for v in self.vehicles:
            self.vehicles[v].interpolate_trajectories(
                current_time, self.planning_points, self.planning_dt)

        flagDesired = False
        self.P_path_v = np.zeros((4, self.planning_points))#no need to change here, for the trajectory  contain same info. view s = X_km and n = Y_km, but the num should be changed
        if len(self.vehicles) > 0: # if we know about at least one vehicle
            vA = list(self.vehicles.keys())[0]
            if self.vehicles[vA].desired_trajectory is not None:
                self.P_path_v[0:2, :] = self.vehicles[vA].interpolated_desired_trajectory[[
            C_k.X_km, C_k.Y_km], :]  # s and n of v1
                flagDesired = True
            else:
                self.P_path_v[0:2, :] = self.vehicles[vA].interpolated_planned_trajectory[[
            C_k.X_km, C_k.Y_km], :]  # s and n of v1
        if len(self.vehicles) > 1: # if we know about two vehicles
            vB = list(self.vehicles.keys())[1]
            if self.vehicles[vA].desired_trajectory is not None:
                self.P_path_v[2:4, :] = self.vehicles[vB].interpolated_desired_trajectory[[
            C_k.X_km, C_k.Y_km], :]  # s and n of v1
                flagDesired = True
            else:
                self.P_path_v[2:4, :] = self.vehicles[vB].interpolated_planned_trajectory[[
            C_k.X_km, C_k.Y_km], :]  # s and n of v1
        else:
            self.P_path_v[2:4, :] = self.P_path_v[0:2, :] # if we know about only one vehicles, we make the second one to be the same as the first

        steer_weight = 1e3
        # x0 = self.state
        # "project" state to the plan;
        # a new plan is not computed from the current state but from the projected state on the previous plan.
        # This is to prevent a sudden change of the error in the track following algorithms, we keep the "old" tracking error.
        if self.planned_sol is not None:
            tt = self.planned_trajectory[C_k.T, :]
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
        self.u=u
        #TODO:add u for the motion equation
        # u=self.compute_MPC(dt, e, 30)
        self.state = self.car_F(self.state, u, dt).full().ravel()
        self.history_state.append(self.get_state())
        self.history_control.append(u)



import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import casadi as cs
from enum import IntEnum
import warnings
from vehicle_class import ST,C
from car_km_class import Car_km
from truck_km_class import Truck_CC
def main():
    #################
    # In simulation
    nt=4
    dt = 0.02  # simulations step in seconds
    lane_width = 3.5 # m
    steps_between_replanning = 25
    # steps_between_replanning = 100
    replanning_iterations = 100
    # replanning_iterations = 10
    P_road_v1 = [0, 0.1, 0, -0.5*lane_width,
                0, 0.1, 0, 0.5*lane_width]


    # Create two independent objects to represent two vehicles

    # CC Truck
    v1 = Truck_CC([25, -416.25-70, 0, 0],dt=dt)
    v1.P_road_v = P_road_v1
    v1.name = 'v1'

    # Only used for Car (CAV)
    P_road_v = [lane_width, 0.1, 80, -1.5*lane_width,
                lane_width, 0.1, 0, -0.5*lane_width]
    v2 = Car_km([19, -397.53, -lane_width, 0],dt=dt)
    v2.P_road_v = P_road_v
    v2.lane_width = lane_width
    v2.name = 'v2'

    # Compute planned and desired trajectories of CAV without sending them to others, needed for simulation
    v1.compute_planned_desired_trajectory()
    v2.compute_planned_desired_trajectory()

    # Simulate the system for some time without updates from other vehicles
    for _ in range(steps_between_replanning):
            for v in [v1, v2]:
                v.simulate(dt)


    # Each vehicle outputs own state
    v1_state = v1.get_state()
    v2_state = v2.get_state()

    # We send and receive it over network

    # Create a twin vehicle inside v1 with the current state, CV
    v1.add_vehicle_twin('v2', v2_state)
    # Create a twin vehicle inside v2 with the current state, CAV
    v2.add_vehicle_twin('v1', v1_state)

    for _ in range(replanning_iterations):

        # Compute planned and desired trajectories
        v1_planned_trajectory, v1_desired_trajectory = v1.compute_planned_desired_trajectory()
        v2_planned_trajectory, v2_desired_trajectory = v2.compute_planned_desired_trajectory()

        # Receive planned and desired trajectories of other vehicles inside twin vehicles
        v2.receive_planned_desired_trajectories(
            'v1', v1_planned_trajectory, v1_desired_trajectory)
        v1.receive_planned_desired_trajectories(
            'v2', v2_planned_trajectory, v2_desired_trajectory)

        # Simulate all vehicles for steps_between_replanning steps
        for _ in range(steps_between_replanning):
            for v in [v1, v2]:
                v.simulate(dt)

                # Read states of all vehicles
                v1_state = v1.get_state()
                v2_state = v2.get_state()

                # Update internal twin vehicles
                v1.update_twin_state('v2', v2_state)
                v2.update_twin_state('v1', v1_state)


    # plot road
    plt.subplot(2, 1, 1)
    ss = np.linspace(-900, 2000, 1000)
    road_right = P_road_v[0]/2 * \
        (np.tanh(P_road_v[1]*(ss-P_road_v[2]))+1)+P_road_v[3]
    road_left = P_road_v[4]/2*(np.tanh(P_road_v[5]*(ss-P_road_v[6]))+1)+P_road_v[7]
    plt.plot(ss, road_right, 'k')
    plt.plot(ss, road_left, 'k')

    # plot state and history
    # v0_history_state = np.array(v0.history_state).T
    v1_history_state = np.array(v1.history_state).T
    v2_history_state = np.array(v2.history_state).T

    # v0_history_planned = v0.history_planned_trajectory
    v1_history_planned = v1.history_planned_trajectory
    v2_history_planned = v2.history_planned_trajectory

    # line_v0, = plt.plot(v0_history_state[ST.S], v0_history_state[ST.N])
    line_v1, = plt.plot(v1_history_state[ST.S], v1_history_state[ST.N])
    plt.plot(v1_history_state[ST.S][-1], v1_history_state[ST.N][-1], 'o', color=line_v1.get_color())
    line_v2, = plt.plot(v2_history_state[ST.S], v2_history_state[ST.N])
    plt.plot(v2_history_state[ST.S][-1], v2_history_state[ST.N][-1], 'o', color=line_v2.get_color())

    for vhp in v1_history_planned:
        plt.plot(vhp[ST.S, :], vhp[ST.N, :], ':',
                color=line_v1.get_color(), linewidth=0.7)
    for vhp in v2_history_planned:
        plt.plot(vhp[ST.S, :], vhp[ST.N, :], ':',
                color=line_v2.get_color(), linewidth=0.7)

    plt.subplot(2, 1, 2)
    line_v1, = plt.plot(v1_history_state[ST.S], v1_history_state[ST.V])
    line_v2, = plt.plot(v2_history_state[ST.S], v2_history_state[ST.V])
    for vhp in v1_history_planned:
        plt.plot(vhp[ST.S, :], vhp[ST.V, :], ':',
                color=line_v1.get_color(), linewidth=0.7)
    for vhp in v2_history_planned:
        plt.plot(vhp[ST.S, :], vhp[ST.V, :], ':',
                color=line_v2.get_color(), linewidth=0.7)
    if hasattr(v1, "history_v_ref"):
        plt.plot(v1_history_state[ST.S], v1.history_v_ref, '--')
    if hasattr(v2, "history_v_ref"):
        plt.plot(v2_history_state[ST.S], v2.history_v_ref, '--')
        

    # v1.save_history('v1')
    # v2.save_history('v2')

    plt.show()


if __name__ == "__main__":
    main()