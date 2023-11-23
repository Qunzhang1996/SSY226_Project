import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import casadi as cs
from enum import IntEnum
import warnings
import sys
sys.path.append('/home/zq/Desktop/SSY226_Project')
from SSY226_Share.src.vehicle_class import C, ST, C_k, Vehicle
#define kalman filter class
class ExtendedKalmanFilter:
    def __init__(self, H, Q, R, x0, P0):
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = x0  # Initial state estimate
        self.P = P0  # Initial covariance estimate

    def predict(self, A, B, u):
        print('this is A dot', np.dot(A, self.x))
        print('this is B dot', np.dot(B, u))
        self.x = np.dot(A, self.x) + np.dot(B, u)  # State prediction using AX + BU
        self.P = np.dot(A, np.dot(self.P, A.T)) + self.Q  # Covariance prediction


    def update(self, z):
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)))
        self.x = self.x + np.dot(K, (z - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))



class Car_km():
    def __init__(self, state, dt=0.1, nt=4, L=4):
        self.L = L
        self.nx = 5
        self.nu = 2
        self.state = np.zeros(self.nx)
        self.state[:nt] = state
        self.state[C_k.V_km] = 0
        self.u = np.zeros(self.nu)
        self.q = np.diag([10.0, 10.0, 0.1, 0.01])
        self.r = np.diag([1, 1]) 
        self.dt = dt

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

    def compute_km_mpc(self, state, error0, N=10):
        nx = self.nx
        nu = self.nu
        nx = nx - 1
        a, b = self.calculate_AB(self.dt)
        from scipy import linalg as la
        # Create a QP problem
        opti = cs.Opti()  
        state_prej = state
        # Decision variables for state and input
        X = opti.variable(nx, N+1)
        U = opti.variable(nu, N)
        
        # Objective function
        obj = 0  # Initiate objective function
        for i in range(N):
            obj += cs.mtimes([X[:, i].T, self.q, X[:, i]])  # State cost
            obj += cs.mtimes([U[:, i].T, self.r, U[:, i]])  # Control cost
        
        # Add the objective function to the optimization problem
        opti.minimize(obj)
        
        # Constraints
        for i in range(N):
            opti.subject_to(X[:, i+1] == cs.mtimes(a, X[:, i]) + cs.mtimes(b, U[:, i]))
        
        # Initial constraints
        opti.subject_to(X[:, 0] == error0)
        
        # Control input constraints
        u_min = [-5, -np.pi / 6]
        u_max = [5, np.pi / 6]
        for j in range(nu):
            opti.subject_to(opti.bounded(u_min[j], U[j, :], u_max[j]))
        
        # Configure the solver
        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver('ipopt', opts)

        # Initialize lists to store predicted trajectories and optimal control inputs
        predicted_trajectories = []
        u_optimal_list = []

        # Solve the optimization problem for each time step
        for i in range(N):
            # Solve the optimization problem
            sol = opti.solve()
            
            # Get the optimal control input for this time step
            u_optimal = sol.value(U[:, i])
            u_optimal_list.append(u_optimal)
            
            # Calculate and store the predicted state trajectory for this time step
            predicted_state = self.car_F(state_prej, u_optimal, self.dt).full().flatten()
            predicted_trajectories.append(predicted_state.copy())
            
            # Update the initial error for the next time step
            state_prej = predicted_state

        return u_optimal_list, predicted_trajectories


    
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

# Define a function to generate a curve
def generate_curve(A=5, B=0.1, x_max=50):
    x = np.linspace(0, x_max, 100)
    y = A * np.sin(B * x)
    return x, y

# Calculate the direction at each point
def calculate_direction(x, y):
    dy = np.diff(y, prepend=y[0])
    dx = np.diff(x, prepend=x[0])
    psi = np.arctan2(dy, dx)
    return psi

# Define arrow plot function
def plot_car_direction(ax, car_state, arrow_length=2.0):
    """
    Draw an arrow on the given axis to represent the direction of the car.
    :param ax: Matplotlib axes object for drawing the arrow.
    :param car_state: Array containing the car's state.
    :param arrow_length: Length of the arrow.
    """
    x_arrow = car_state[C_k.X_km]
    y_arrow = car_state[C_k.Y_km]
    dx_arrow = arrow_length * np.cos(car_state[C_k.Psi])
    dy_arrow = arrow_length * np.sin(car_state[C_k.Psi])
    ax.arrow(x_arrow, y_arrow, dx_arrow, dy_arrow, head_width=0.5, head_length=0.5, fc='green', ec='green')

# Generate the reference curve
x_ref, y_ref = generate_curve(A=20, B=0.05, x_max=100)

# concate x_ref and y_ref as 2d array, shape of (N,2)
ref_points = np.vstack([x_ref, y_ref]).T
psi_ref = calculate_direction(x_ref, y_ref)
target_point, target_idx = find_target_point(ref_points, np.array(x_ref[0], y_ref[0]), 10)



# Initialize car model
car = Car_km(state=np.array([0, 0,np.pi/4, 5]))

# Store the car's trajectory
trajectory = []
last_index = 0

u_optimal=np.zeros(2)

# define the car as a rectangle
# Plot the car as a rectangle
car_length = 4.0  # Define the car's length
car_width = 1.0   # Define the car's width


noise_level=0.1
#initialize the kalman filter
H = np.eye(car.nx-1)  # Observation matrix
Q= noise_level**2 * np.eye(car.nx-1)
R = 0.01*noise_level**2 * np.eye(car.nx-1)  # Measurement noise covariance
x0 = np.array([0, 0, np.pi/4, 5])  # Initial state estimate
P0 = np.eye(car.nx-1)  # Initial covariance estimate
ekf = ExtendedKalmanFilter(H, Q, R, x0, P0)
u_optimal=np.zeros(2)





plt.ion()

for i in range(len(x_ref) + 10):
    plt.cla()
    current_position = car.state[[C_k.X_km, C_k.Y_km]]

    target_point, target_idx = find_target_point(ref_points, current_position, 1, last_index)
    last_index = target_idx
    ref_state = np.array([target_point[0], target_point[1], psi_ref[target_idx], 10])
    #start to get measurement
    # calculate the A B matrix of current time step
    A, B = car.calculate_AB(car.dt)

    print('this is A', A)
    print('this is B', B)


    #here, inputting white noise into the car state
    car.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]] += np.random.normal(0, noise_level, 4)
    # add the measurement noise
    measurement_noise = np.random.normal(0, 0.1*noise_level, 4)
    noisy_measurement = car.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]] + measurement_noise
    print('this is u_optimal', u_optimal)
    



    #do kalman filter
    ekf.predict(A, B, u_optimal)
    
    ekf.update(noisy_measurement)
    

    car.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]] = ekf.x


    # keep the heading angle between -pi and pi
    car.state[C_k.Psi] = np.arctan2(np.sin(car.state[C_k.Psi]), np.cos(car.state[C_k.Psi]))
    #calculate the error
    error = car.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]] - ref_state
    #keep the heading angle between -pi and pi
    heading_error = np.arctan2(np.sin(error[2]), np.cos(error[2]))
    error[2] = heading_error
    
    
    #make sure the velocity is not zero
    if car.state[C_k.V_km] == 0:
        car.state[C_k.V_km] = 0.01

    # Call compute_km_mpc to get both u_optimal and predicted_trajectories
    u_optimal_list, predicted_trajectories = car.compute_km_mpc(car.state, error)
    u_optimal = u_optimal_list[0]

    # Visualize the predicted trajectories
    predicted_trajectories = np.array(predicted_trajectories)

    # Plot the reference trajectory
    plt.plot(x_ref, y_ref)

    # Plot the predicted trajectories using vectorized plot
    plt.plot(*predicted_trajectories[:, [C_k.X_km, C_k.Y_km]].T, '-', color='red')

    # Plot the car
    car_x = car.state[C_k.X_km] - 0.5 * car_length * np.cos(car.state[C_k.Psi])
    car_y = car.state[C_k.Y_km] - 0.5 * car_length * np.sin(car.state[C_k.Psi])
    car_angle = np.degrees(car.state[C_k.Psi])
    car_rect = plt.Rectangle((car_x, car_y), car_length, car_width, angle=car_angle, edgecolor='blue', facecolor='none')
    plt.gca().add_patch(car_rect)
    
    # Update car state
    car.state = car.car_F(car.state, u_optimal, car.dt).full().flatten()

    plt.axis("equal")
    plt.pause(0.1)
    trajectory.append(car.state.copy())

plt.ioff()

# Convert trajectory to NumPy array
trajectory = np.array(trajectory)


# Visualize results
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




