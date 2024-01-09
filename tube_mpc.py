import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_continuous_are

def lqr_gain(A, B, Q, R):
    """
    Calculate the LQR gain.

    Args:
    - A (numpy.ndarray): The system dynamics matrix.
    - B (numpy.ndarray): The control input matrix.
    - Q (numpy.ndarray): The state cost matrix.
    - R (numpy.ndarray): The control cost matrix.

    Returns:
    - K (numpy.ndarray): The LQR gain matrix.
    """
    # 解Riccati方程
    P = solve_continuous_are(A, B, Q, R)

    # 计算LQR增益
    K = np.dot(np.linalg.inv(R), np.dot(B.T, P))
    return K





# Define system parameters
a = 1  # System dynamics
b = 1.0  # Control dynamics
delta = 0.1  # Tube radius for Tube MPC

# Define time parameters
N = 20  # Prediction horizon
T = 100  # Total time steps

# Initial conditions
x0 = np.array([0])  # Initial position
# Desired position is sin curve # Desired position for one dimensional system
xd = np.sin(np.linspace(0, 2*np.pi, T+1))  # Desired position for one dimensional system

# MPC parameters
Q = np.array([[1]])  # State cost
R = 0.01  # Control cost
# Generate random disturbances
np.random.seed(0)
disturbances = np.random.normal(0, 0.2, T)  # Normal distribution with mean 0 and standard deviation 0.2

# Standard MPC implementation
def standard_mpc(x0, disturbances):
    x_trajectory1 = [x0[0]]
    for t in range(T):
        x = cp.Variable((N+1, 1))
        u = cp.Variable((N, 1))

        constraints = [x[0] == x_trajectory1[-1]]
        cost = 0
        for k in range(N):
            xd_index = min(t + k, len(xd) - 1)
            cost += cp.quad_form(x[k] - xd[xd_index], Q) + R*cp.square(u[k])
            constraints += [x[k+1] == a*x[k] + b*u[k]]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        u0 = u.value[0, 0]
        xt = a * x_trajectory1[-1] + b * u0 + disturbances[t]
        x_trajectory1.append(xt)

    return x_trajectory1

# Tube MPC implementation
def tube_mpc(x0, disturbances):
    x_trajectory = [x0[0]]  # 实际状态轨迹
    u_prev = 0  # 上一个时间步的控制输入，初始为0

    for t in range(T):
        x = cp.Variable((N+1, 1))
        u = cp.Variable((N, 1))

        constraints = [x[0] == x_trajectory[-1]]  # 使用实际状态作为初始条件
        cost = 0
        for k in range(N):
            xd_index = min(t + k, len(xd) - 1)
            cost += cp.quad_form(x[k] - xd[xd_index], Q) + R*cp.square(u[k])
            constraints += [x[k+1] == a*x[k] + b*u[k]]
            constraints += [cp.abs(x[k+1] - xd[xd_index]) <= delta]

        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()

        if problem.status not in ["infeasible", "unbounded"]:
            # 如果优化问题有解，则计算校正后的控制输入
            u0 = u.value[0, 0] + error_correction(x_trajectory[-1], x.value[1, 0], u_prev)
        else:
            # 如果问题无解，使用上一个控制输入
            u0 = u_prev

        # 更新实际状态
        xt = a * x_trajectory[-1] + b * u0 + disturbances[t]
        x_trajectory.append(xt)
        u_prev = u0  # 更新控制输入以用于下一轮迭代

    return x_trajectory

def error_correction(actual_state, predicted_state, prev_control):
    # 根据实际状态与预测状态之间的偏差来计算校正值
    error = actual_state - predicted_state
    # 这里可以定义一个简单的校正逻辑，例如基于误差的比例控制
    correction = -0.5 * error  # Kp 是比例增益，需要根据系统特性来调整
    return correction


# Run simulations
std_mpc_trajectory = standard_mpc(x0, disturbances)
tube_mpc_trajectory = tube_mpc(x0, disturbances)

# Plotting results
# Plotting results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(std_mpc_trajectory, label='Standard MPC Position')
plt.plot(xd, color='g', linestyle='--', label='Desired Position')
plt.title('Standard MPC with Disturbances')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(tube_mpc_trajectory, label='Tube MPC Position')
plt.plot(xd, color='g', linestyle='--', label='Desired Position')
plt.title('Tube MPC with Disturbances')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()

plt.tight_layout()
plt.show()

