import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

# 定义车辆的运动学模型
def create_car_F_km(L):  # create a discrete model of the vehicle
        nx = 5
        nu = 2
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
        car_F_km = F  # save the vehicle model in the object variable
        return car_F_km

# 创建车辆模型
L = 2.8  # 假设车辆轴距为2.5米
car_F_km = create_car_F_km(L)

# 读取控制数据
control_data_path = "/home/zq/Desktop/SSY226_Project/save_car_control.txt"
control_data = np.loadtxt(control_data_path)  # 从文件读取控制数据

# 初始状态
x0 = np.array([-66.72, 37.35755-3.5, 0, 0, 10])  # x_km, y_km, psi, t, v_km
dt = 0.02  # 时间步长
x_list = [x0]

# 模拟
for u in control_data[25:]:
    x = np.array(car_F_km(x0, u, dt)).flatten()
    x_list.append(x)
    x0 = x

# 提取轨迹
x_traj = [state[0] for state in x_list]
y_traj = [state[1] for state in x_list]



plt.ion()  # 开启交互模式

for i in range(len(x_traj)):
    plt.cla()  # 清除当前轴
    plt.plot(x_traj[:i+1], y_traj[:i+1], label='Vehicle Trajectory')  # 绘制到当前点的轨迹
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Vehicle Trajectory Simulation')
    plt.legend()
    plt.grid(True)
    plt.axis("equal")  # 设置轴比例一致
    plt.ylim(33, 40)
    plt.pause(0.01)  # 短暂暂停以更新图形

plt.ioff()  # 关闭交互模式
plt.show()  
