import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import casadi as cs
from enum import IntEnum
import warnings
import sys
sys.path.append('/home/zq/Desktop/SSY226_Project')
from SSY226_Share.src.vehicle_class import C, ST,C_k,Vehicle



class Car_km( ):
    def __init__(self, state, dt=0.2,nt=4,L=4):
        self.L = L
        self.nx = 5
        self.nu = 2
        self.state = np.zeros(self.nx)
        self.state[:nt] = state
        self.state[C_k.V_km] = 0
        self.u = np.zeros(self.nu)
        self.v0 = 10
        self.planning_points = 31
        self.q = np.diag([1.0, 1.0, 1, 0.01])
        self.P=self.q
        self.r = np.diag([0.1, 0.1]) 
        self.dt = dt
        self.dt_factor = 30
        # self.setup_Op_MPC()

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


    def calculate_AB(self,dt_sim):
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
        # print(newA)
        newB = B_op * dt_sim
        # print(newB)

        return newA.full(), newB.full()
    
    def compute_km_K(self):
        a, b = self.calculate_AB(self.dt)
        from scipy import linalg as la
        P = la.solve_discrete_are(a, b, self.q, self.r)
        R = la.solve(self.r + b.T.dot(P).dot(b), b.T.dot(P).dot(a))
        self.R = R
        self.P=P
        return R


    def compute_km_mpc(self,error0, N=30):
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
        u_min = [-5, -np.pi / 4]
        u_max = [5, np.pi / 4]

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

# 定义生成曲线的函数
def generate_curve(A=10, B=0.1, x_max=100):
    x = np.linspace(0, x_max, 50)
    y = A * np.sin(B * x)
    # y=np.linspace(0, 0, 50)
    return x, y

# 计算每个点的方向
def calculate_direction(x, y):
    dy = np.diff(y, prepend=y[0])
    dx = np.diff(x, prepend=x[0])
    psi = np.arctan2(dy, dx)
    return psi

# 生成参考曲线
x_ref, y_ref = generate_curve(A=10, B=0.1, x_max=100)
# concate x_ref and y_ref as 2d array, shape of (N,2)
ref_points = np.vstack([x_ref, y_ref]).T
psi_ref = calculate_direction(x_ref, y_ref)
target_point, target_idx = find_target_point(ref_points, np.array(x_ref[0], y_ref[0]), 10)
# plt.scatter(target_point[0], target_point[1], color='red')
# plt.plot(x_ref, y_ref)
# plt.show()


# 初始化汽车模型
car = Car_km(state=np.array([0, 0, 0, 0]))

# 存储汽车的轨迹
trajectory = []
last_index = 0
# 对每个参考点运行MPC
for i in range(len(x_ref)):
    plt.cla()
    # 设置参考状态（x, y, psi, v）
    # current_position = np.array([x_ref[i], y_ref[i]])  # 假设参考速度为 10 m/s
    current_position = car.state[[C_k.X_km, C_k.Y_km]]

    target_point, target_idx = find_target_point(ref_points, current_position, 1, last_index)
    last_index = target_idx
    ref_state = np.array([target_point[0], target_point[1], psi_ref[target_idx], 10])
    # 计算误差
    car.state[C_k.Psi]=np.arctan2(np.sin(car.state[C_k.Psi]),np.cos(car.state[C_k.Psi]))
    error = car.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]-ref_state
    heading_eerror=np.arctan2(np.sin(error[2]),np.cos(error[2]))
    error[2]=heading_eerror
    print('error: ', error)
    if car.state[C_k.V_km]==0:
        car.state[C_k.V_km]=0.01
    # 计算最优控制输入
    u_optimal = car.compute_km_mpc(error)
    # R=car.compute_km_K()
    # u_optimal = -np.matmul(R,error)
    # if u_optimal[1]>np.pi/4:
    #     u_optimal[1]=np.pi/4
    # elif u_optimal[1]<-np.pi/4:
    #     u_optimal[1]=-np.pi/4
    print('u_optimal',u_optimal)

    # 更新汽车状态
    print(car.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]])
    car.state = car.car_F(car.state, u_optimal, car.dt).full().flatten()

    # # 增加时间状态
    # car.state[C_k.T] += car.dt
    plt.plot(current_position[0], current_position[1], 'o', color='blue')
    plt.plot(x_ref, y_ref)
    plt.scatter(target_point[0], target_point[1], color='red')
    plt.axis("equal")
    plt.pause(2)
    # 记录当前状态
    trajectory.append(car.state.copy())

# 将轨迹转换为NumPy数组
trajectory = np.array(trajectory)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(x_ref, y_ref, label="Reference Path")
plt.plot(trajectory[:, 0], trajectory[:, 1], label="Car Trajectory", linestyle='--', color='red')
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("MPC Tracking Performance")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()