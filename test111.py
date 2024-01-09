import casadi as cs
import numpy as np
from enum import IntEnum
from scipy import linalg as la
from matplotlib import pyplot as plt
class C_k(IntEnum):
    X_km, Y_km, Psi, V_km = range(4)

class Car_km():
    def __init__(self, state, dt=0.02, nt=4, L=4):
        self.L = L
        self.nx = 5  # 状态维度（包括时间）
        self.nu = 2  # 控制维度
        self.state = np.zeros(self.nx)
        self.state[:nt] = state
        self.state[C_k.V_km] = 20
        self.u = np.zeros(self.nu)
        self.q = np.diag([10, 10.0, 1.0, 10.0])  # 状态代价权重
        self.r = np.diag([0.01, 0.1])  # 控制代价权重
        self.dt = dt
        self.create_car_F_km()  # 创建车辆模型

    def create_car_F_km(self):
        nx = self.nx
        nu = self.nu
        x = cs.SX.sym('x', nx)
        u = cs.SX.sym('u', nu)
        x_km, y_km, psi, t, v_km = x[0], x[1], x[2], x[3], x[4]
        a_km, delta = u[0], u[1]
        dot_x_km = v_km * np.cos(psi)
        dot_y_km = v_km * np.sin(psi)
        dot_psi = v_km / self.L * np.tan(delta)
        dot_t = 1
        dot_v_km = a_km
        dot_x = cs.vertcat(dot_x_km, dot_y_km, dot_psi, dot_t, dot_v_km)
        f = cs.Function('f', [x, u], [dot_x])
        dt = cs.SX.sym('dt', 1)
        k1 = f(x, u)
        k2 = f(x + dt / 2 * k1, u)
        k3 = f(x + dt / 2 * k2, u)
        k4 = f(x + dt * k3, u)
        x_kp1 = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        F = cs.Function('F', [x, u, dt], [x_kp1])
        self.car_F = F

    def compute_km_mpc(self, state, ref_states, N=6):
        nx = self.nx - 1  # 排除时间维度
        nu = self.nu

        opti = cs.Opti()
        X = opti.variable(nx, N+1)
        U = opti.variable(nu, N)
        Ref = opti.parameter(nx, N)

        obj = 0
        for i in range(N):
            state_error = X[:, i] - Ref[:, i]
            obj += cs.mtimes([state_error.T, self.q, state_error])
            obj += cs.mtimes([U[:, i].T, self.r, U[:, i]])
        opti.minimize(obj)

        for i in range(N):
            extended_state = cs.vertcat(X[:, i], 0)  # 添加时间维度
            next_state_pred = self.car_F(extended_state, U[:, i], self.dt)
            opti.subject_to(X[:, i+1] == next_state_pred[:4])

        opti.subject_to(X[:, 0] == state[:4])

        u_min = [-5, -np.pi / 6]
        u_max = [5, np.pi / 6]
        for j in range(nu):
            opti.subject_to(opti.bounded(u_min[j], U[j, :], u_max[j]))

        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver('ipopt', opts)
        opti.set_value(Ref, ref_states)

        sol = opti.solve()
        X_opt = sol.value(X)
        U_opt = sol.value(U)

        return U_opt[:, 0], X_opt[:, :]
def generate_circle_trajectory(center, radius, num_points):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.vstack((x, y)).T

# 示例用法
# 示例用法
def calculate_directions(trajectory):
    directions = np.arctan2(np.diff(trajectory[:, 1], prepend=trajectory[0, 1]),
                            np.diff(trajectory[:, 0], prepend=trajectory[0, 0]))
    return directions

# 示例用法
initial_state = [25, 5, 0, 0]  # 初始状态
car = Car_km(initial_state)
circle_center = [5, 5]  # 圆心坐标
circle_radius = 20      # 圆半径
num_points = 1000      # 轨迹点数量
num_steps = 60000         # MPC 迭代次数

# 生成圆形轨迹
circle_trajectory = generate_circle_trajectory(circle_center, circle_radius, num_points)

# 计算方向角
psi_ref = calculate_directions(circle_trajectory)

# 假设速度为常量
V_ref = 20

# 将圆形轨迹转换为参考状态
ref_states = np.zeros((4, num_points))
ref_states[0, :] = circle_trajectory[:, 0]  # X
ref_states[1, :] = circle_trajectory[:, 1]  # Y
ref_states[2, :] = psi_ref                  # Psi
ref_states[3, :] = V_ref                    # V

# 初始化用于存储预测轨迹的数组
predicted_trajectory = np.zeros((num_steps, 2))

# 迭代地应用 MPC 控制器
for i in range(num_steps):
    # 获取当前参考状态
    current_ref_index = min(i + 6, num_points)
    current_ref = ref_states[:, i:current_ref_index]

    # 检查 current_ref 是否为空
    if current_ref.shape[1] == 0:
        break  # 如果没有更多的参考数据，终止循环

    # 如果参考状态的列数不足 6 列，复制最后一列以填充
    if current_ref.shape[1] < 6:
        num_missing_columns = 6 - current_ref.shape[1]
        last_column = current_ref[:, -1].reshape(4, 1)
        current_ref = np.hstack([current_ref, np.tile(last_column, (1, num_missing_columns))])

    # 应用 MPC 控制器计算下一步控制输入和预测状态
    u_optimal, X_opt = car.compute_km_mpc(car.state, current_ref)

    # 更新车辆状态
    car.state = car.car_F(car.state, u_optimal, car.dt).full().flatten()

    # 记录预测轨迹
    predicted_trajectory[i, :] = car.state[:2]

# 绘制参考轨迹和预测轨迹
plt.figure(figsize=(10, 8))
plt.plot(circle_trajectory[:, 0], circle_trajectory[:, 1], label="Reference Circle Trajectory")
plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], label="Predicted Trajectory", linestyle='--')
plt.scatter(circle_center[0], circle_center[1], color='red', label="Circle Center")
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Circular Trajectory Tracking with MPC')
plt.legend()
plt.axis('equal')
plt.show()
