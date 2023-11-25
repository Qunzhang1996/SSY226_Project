import matplotlib.pyplot as plt
import numpy as np
import casadi as cs
from enum import IntEnum
from scipy.interpolate import CubicSpline



class C_k(IntEnum):
    X_km, Y_km, Psi, T, V_km =range(5)


class Car_km():
    def __init__(self, state, dt=0.1, nt=4, L=4):
        self.L = L
        self.nx = 5
        self.nu = 2
        self.state = np.zeros(self.nx)
        self.state[:nt] = state
        self.state[C_k.V_km] = 0
        self.u = np.zeros(self.nu)
        self.q = np.diag([10.0, 10.0, 0.1, 1])
        self.r = np.diag([1, 1]) 
        self.dt = dt
        self.lasy_index = 0

    def create_car_F_km(self):  # Create a discrete model of the vehicle
        nx = self.nx
        nu = self.nu
        x = cs.SX.sym('x', nx)
        u = cs.SX.sym('u', nu)
        # X_km, Y_km, Psi, T, V_km 
        # States: x_km longitudinal speed, y_km lateral speed, psi heading, time, v_km velocity
        x_km, y_km, psi, t, v_km  = x[0], x[1], x[2], x[3], x[4]
        # Controls: a_km acceleration, delta steering angle
        a_km, delta = u[0], u[1]
        dot_x_km = v_km * np.cos(psi)
        dot_y_km = v_km * np.sin(psi)
        dot_psi = v_km / self.L * np.tan(delta)
        dot_t = 1
        dot_v_km = a_km
        dot_x = cs.vertcat(dot_x_km, dot_y_km, dot_psi, dot_t, dot_v_km)
        f = cs.Function('f', [x, u], [dot_x])
        dt = cs.SX.sym('dt', 1)  # Time step in optimization
        k1 = f(x, u)
        k2 = f(x + dt / 2 * k1, u)
        k3 = f(x + dt / 2 * k2, u)
        k4 = f(x + dt * k3, u)
        x_kp1 = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        F = cs.Function('F', [x, u, dt], [x_kp1])  # x_k+1 = F(x_k, u_k, dt)
        self.car_F = F  # Save the vehicle model in the object variable
        self.K_dot_x = cs.vertcat(dot_x_km, dot_y_km, dot_psi, dot_v_km)
        self.K_x = cs.vertcat(x_km, y_km, psi, v_km)
        self.K_u = cs.vertcat(a_km, delta)
        self.kinematic_car_model = cs.Function('kinematic_car_model', [self.K_x, self.K_u], [self.K_dot_x])

    def calculate_AB(self, dt_sim):
        nx = self.nx
        nu = self.nu
        nx = nx - 1
        x_op = self.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]
        if x_op[C_k.V_km-1] == 0:
            x_op[C_k.V_km-1] = 0.01
        x_op[C_k.Psi] = np.arctan2(np.sin(x_op[C_k.Psi]), np.cos(x_op[C_k.Psi]))
        u_op = self.u
        self.create_car_F_km()
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
        newB = B_op * dt_sim

        return newA, newB
    
    def compute_km_K(self):
        a, b = self.calculate_AB(self.dt)
        from scipy import linalg as la
        P = la.solve_discrete_are(a, b, self.q, self.r)
        R = la.solve(self.r + b.T.dot(P).dot(b), b.T.dot(P).dot(a))
        self.R = R
        self.P = P
        return R

    def compute_km_mpc(self, state, error0, ref_points, left_edge_y, right_edge_y, edge_indices, N=10):
        nx = self.nx - 1
        nu = self.nu
        a, b = self.calculate_AB(self.dt)

        opti = cs.Opti()
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)

        penalty_weight = 1  # 惩罚权重
        obj = 0  # 初始化目标函数

        # 时间向量用于插值
        time_vector = np.linspace(0, 1, len(ref_points))
        time_vector_pred = np.linspace(0, 1, N)
        # 创建三次样条插值函数
        spline_x = CubicSpline(time_vector, ref_points[:, 0])

        for i in range(N):
            # 计算动态参考点
            dynamic_ref_x = spline_x(time_vector_pred[i])

            # 将偏差转换为实际状态值
            actual_state_y = X[C_k.Y_km, i] + dynamic_ref_x

            # 状态成本和控制成本
            obj += cs.mtimes([X[:, i].T, self.q, X[:, i]])
            obj += cs.mtimes([U[:, i].T, self.r, U[:, i]])

            # 道路边缘惩罚项
            idx = edge_indices[i] if i < len(edge_indices) else -1
            edge_left = left_edge_y[idx] if idx >= 0 else left_edge_y[-1]
            edge_right = right_edge_y[idx] if idx >= 0 else right_edge_y[-1]

            obj += penalty_weight * cs.fmax(0, edge_left - actual_state_y)**2
            obj += penalty_weight * cs.fmax(0, actual_state_y - edge_right)**2

        opti.minimize(obj)

        # 初始误差约束
        opti.subject_to(X[:, 0] == error0)

        # 动力学约束
        for i in range(N):
            opti.subject_to(X[:, i + 1] == cs.mtimes(a, X[:, i]) + cs.mtimes(b, U[:, i]))

        # 控制输入约束
        u_min = [-5, -np.pi / 3]
        u_max = [5, np.pi / 3]
        for j in range(nu):
            opti.subject_to(opti.bounded(u_min[j], U[j, :], u_max[j]))

        # 配置求解器并解决优化问题
        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver('ipopt', opts)

        try:
            sol = opti.solve()
        except RuntimeError as e:
            print("Solver failed:", e)
            for var in [X, U]:
                print("Value of", var, ":", opti.debug.value(var))
            raise

        u_optimal_list = [sol.value(U[:, i]) for i in range(N)]
        predicted_trajectories = [sol.value(X[:, i]) for i in range(N + 1)]

        return u_optimal_list[0], predicted_trajectories
    
    # Calculate the direction at each point
    def calculate_direction(self,x, y):
        dy = np.diff(y, prepend=y[0])
        dx = np.diff(x, prepend=x[0])
        psi = np.arctan2(dy, dx)
        return psi
    
    def find_target_point(self,trajectory, point, shift_points, last_index):
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
    
    def rad2deg(self,angle):
        return angle * 180 / np.pi
    
    def deg2rad(self,angle):
        return angle * np.pi / 180
    
    def simulate(self, outside_carla_state=np.zeros(4), ref_points=None, psi_ref=None, left_edge_y=None, right_edge_y=None, last_index=None, N=10):
        if last_index is None:
            last_index = self.last_index
        
        state_carla = np.zeros(5)
        state_carla[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]] = outside_carla_state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]
        state_carla[C_k.T] = self.state[C_k.T]

        current_position = state_carla[[C_k.X_km, C_k.Y_km]].T
        target_point, target_idx = self.find_target_point(ref_points, current_position, 1, last_index)
        last_index = target_idx

        ref_state = np.array([target_point[0], target_point[1], psi_ref[target_idx], 10])
        error = state_carla[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]] - ref_state
        heading_error = np.arctan2(np.sin(error[2]), np.cos(error[2]))
        error[2] = heading_error

        edge_indices = np.arange(target_idx, target_idx + N)

        u_optimal, predicted_trajectories = self.compute_km_mpc(state_carla, error, ref_points, left_edge_y, right_edge_y, edge_indices)
        # Visualize the predicted trajectories
        predicted_trajectories = np.array(predicted_trajectories)
        self.state = self.car_F(state_carla, u_optimal, self.dt).full().flatten()
        return u_optimal, predicted_trajectories, last_index



# Define a function to generate a curve
def generate_curve(A=5, B=0.1, x_max=50, road_width=7):
    x = np.linspace(0, x_max, 100)
    y = A * np.sin(B * x)
    psi = np.arctan2(np.diff(y, prepend=y[0]), np.diff(x, prepend=x[0]))

    # 计算道路边界
    left_x = x - road_width / 2 * np.sin(psi)
    left_y = y + road_width / 2 * np.cos(psi)
    right_x = x + road_width / 2 * np.sin(psi)
    right_y = y - road_width / 2 * np.cos(psi)

    return x, y, left_x, left_y, right_x, right_y

def main():
    # 生成参考曲线及道路边界
    x_ref, y_ref, left_x, left_y, right_x, right_y = generate_curve(A=20, B=0.05, x_max=100, road_width=7)
    ref_points = np.vstack([x_ref, y_ref]).T
    psi_ref = np.arctan2(np.diff(y_ref, prepend=y_ref[0]), np.diff(x_ref, prepend=x_ref[0]))

    # 初始化车辆模型
    car = Car_km(state=np.array([0, 0, np.pi/4, 0]))

    # 定义障碍物
    obstacles = [
        {'position': (25, 20 * np.sin(0.05 * 25)), 'radius': 1},  # On the trajectory
        {'position': (40, 20 * np.sin(0.05 * 40) + 2), 'radius': 1.5},  # Near the trajectory
        {'position': (70, 20 * np.sin(0.05 * 70) - 2), 'radius': 1}   # Near the trajectory
    ]

    plt.ion()
    last_index = 0
    for i in range(130):
        plt.cla()

        # 模拟车辆运动
        carla_state = np.zeros(5)
        carla_state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]] = car.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]
        u_optimal, predicted_trajectories, last_index = car.simulate(carla_state, ref_points, psi_ref, left_y, right_y, last_index)

        # 可视化预测轨迹和参考路径
        predicted_trajectories = np.array(predicted_trajectories)
        plt.plot(x_ref, y_ref, 'r')  # 参考路径
        plt.plot(left_x, left_y, 'k--')  # 左侧道路边缘
        plt.plot(right_x, right_y, 'k--')  # 右侧道路边缘
        # plt.plot(*predicted_trajectories[:, [C_k.X_km, C_k.Y_km]].T, '-', color='blue')  # 预测轨迹

        # 可视化障碍物
        for obstacle in obstacles:
            circle = plt.Circle(obstacle['position'], obstacle['radius'], color='blue', fill=False)
            plt.gca().add_patch(circle)

        # 绘制车辆
        car_x = car.state[C_k.X_km]
        car_y = car.state[C_k.Y_km]
        car_angle = np.degrees(car.state[C_k.Psi])
        car_rect = plt.Rectangle((car_x, car_y), 4, 1, angle=car_angle, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(car_rect)

        plt.axis("equal")
        plt.pause(0.1)

    plt.ioff()

    # 可视化最终轨迹
    plt.figure(figsize=(12, 6))
    plt.plot(x_ref, y_ref, label="Reference Path")
    plt.plot(*np.array(predicted_trajectories)[:, [C_k.X_km, C_k.Y_km]].T, label="Car Trajectory", linestyle='--', color='red')
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("MPC Tracking Performance")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()