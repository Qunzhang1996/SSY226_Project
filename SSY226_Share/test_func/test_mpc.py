import matplotlib.pyplot as plt
import numpy as np
import casadi as cs
from enum import IntEnum



class C_k(IntEnum):
    X_km, Y_km, Psi, T, V_km =range(5)


class Car_km():
    def __init__(self, state, dt=0.02, nt=4, L=4):
        self.L = L
        self.nx = 5
        self.nu = 2
        self.state = np.zeros(self.nx)
        self.state[:nt] = state
        self.state[C_k.V_km] = 10
        self.u = np.zeros(self.nu)
        self.q = np.diag([10.0, 10.0, 1.0, 1.0])
        self.r = np.diag([0.01, 0.1]) 
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

    def calculate_AB(self, dt_sim, state_input=None):
        nx = self.nx
        nu = self.nu
        nx = nx - 1

        # Use the provided state_input if it is given, otherwise use the system's current state
        if state_input is not None:
            x_op = state_input
        else:
            x_op = self.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]

        # Ensure the velocity is not zero by setting a small positive value
        small_velocity = 0.01
        x_op = cs.if_else(x_op[C_k.V_km-1] == 0, cs.vertcat(x_op[:C_k.V_km-1], small_velocity, x_op[C_k.V_km:]), x_op)

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
        newA = A_op * dt_sim + np.eye(nx)
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

    # Calculate the direction at each point
    def compute_km_mpc(self, state, ref_states, N=10):
        nx = self.nx
        nu = self.nu
        nx = nx - 1

        # Create a QP problem
        opti = cs.Opti()  
        # Decision variables for state and input
        X = opti.variable(nx, N+1)
        U = opti.variable(nu, N)
        # Create a parameter for the reference state
        Ref = opti.parameter(nx, N)

        # Objective function
        obj = 0  # Initiate objective function
        for i in range(N):
            state_error = X[:, i] - Ref[:, i]  # State deviation from reference
            obj += cs.mtimes([state_error.T, self.q, state_error])  # State cost
            obj += cs.mtimes([U[:, i].T, self.r, U[:, i]])  # Control cost
            #extra cost for the initial state x,y at the first step
            # obj += 1000*cs.mtimes([state_error[0:2].T, self.q[0:2,0:2], state_error[0:2]])  # State cost
        
        # Add the objective function to the optimization problem
        opti.minimize(obj)        
        # Constraints
        for i in range(N):
            # Recalculate the linearized matrices A and B based on the current predicted state
            C = cs.vertcat(
                self.dt * X[3, i] * cs.sin(X[2, i]) * X[2, i],
                -self.dt * X[3, i] * cs.cos(X[2, i]) * X[2, i],
                0.0,
                -self.dt * X[3, i] * U[1, i] / (self.L * cs.cos(U[1, i]) ** 2))
            

            a, b = self.calculate_AB(self.dt, X[:, i])
            next_state_pred = cs.mtimes(a, X[:, i]) + cs.mtimes(b, U[:, i]) + C
            opti.subject_to(X[:, i+1] == next_state_pred)
        
        # Initial constraints - setting the first state in the prediction to the current state
        opti.subject_to(X[:, 0] == state)
        
        # Control input constraints
        u_min = [-5, -np.pi / 6]
        u_max = [5, np.pi / 6]
        for j in range(nu):
            opti.subject_to(opti.bounded(u_min[j], U[j, :], u_max[j]))
        
        # Configure the solver
        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver('ipopt', opts)

        # Set the value of the reference state parameter
        opti.set_value(Ref, ref_states)

        # Solve the optimization problem
        sol = opti.solve()
        
        # Process the solution
        X_opt = sol.value(X)
        U_opt = sol.value(U)

        return U_opt[:, 0], X_opt[0:2,:]


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
    
    def find_target_trajectory(self, trajectory, point, psi_ref, V_ref=25, N=10):
        # Calculate the squared Euclidean distance to each point in the trajectory
        distances = np.sum((trajectory[:,:2] - point) ** 2, axis=1)
        # Find the index of the closest point
        closest_idx = np.argmin(distances)
        # target_idx = max(closest_idx + shift_points, last_index + 1)
        target_idx = closest_idx 
        # Ensure the index does not exceed the length of the trajectory minus N
        target_idx = min(target_idx, len(trajectory) - N)

        # Initialize an empty array for the N-step reference trajectory
        ref_trajectory = np.zeros((4, N))  # Each column: [X, Y, Psi_ref, V_ref]

        # Fill the array with the N-step trajectory points, Psi_ref, and constant V_ref
        for i in range(N):
            ref_trajectory[0, i] = trajectory[target_idx + i, 0]  # X coordinate
            ref_trajectory[1, i] = trajectory[target_idx + i, 1]  # Y coordinate
            ref_trajectory[2, i] = psi_ref[target_idx + i]        # Psi_ref
            ref_trajectory[3, i] = V_ref                           # Reference velocity

        return ref_trajectory,target_idx

    
    def rad2deg(self,angle):
        return angle * 180 / np.pi
    def deg2rad(self,angle):
        return angle * np.pi / 180
    
    def simulate(self,outside_carla_state=np.zeros(4),ref_points=None, psi_ref=None,last_index=None):
        if last_index is None:
            last_index = self.last_index
        
        #To do the simluation, use the state shown below, otherwise, use the state from carla and comment the following line
        state_carla=np.zeros(5)
        state_carla[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]\
            =outside_carla_state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]
        state_carla[C_k.T]=self.state[C_k.T]
        #here, try to make some difference between the carla and self.state
        # state_carla[C_k.Psi]=state_carla[C_k.Psi]+0.01*np.pi
        # state_carla[C_k.V_km]=state_carla[C_k.V_km]+0.1
        
        
        current_position = state_carla[[C_k.X_km, C_k.Y_km]].T
        # target_point, target_idx = self.find_target_point(ref_points, current_position, \
        #                                                   1,last_index)
        target_trajectory, target_idx = self.find_target_trajectory(ref_points, current_position, \
                                                                    psi_ref)
        last_index = target_idx

        u_optimal, predicted_trajectories = self.compute_km_mpc(state_carla[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]], target_trajectory)
        print('this is the u_optimal',u_optimal)
        # Visualize the predicted trajectories
        self.state = self.car_F(state_carla, u_optimal, self.dt).full().flatten()
        print('this is the state in km', self.state)
        #here, return the u_optimal for carla to use,with degree
        u_optimal[1]=self.rad2deg(u_optimal[1])

        #since the max of carla is form -75 to 75, so we need to limit the steering angle and scale it
        if u_optimal[1]>75:
            u_optimal[1]=75 
        elif u_optimal[1]<-75:
            u_optimal[1]=-75
        u_optimal[1]=u_optimal[1]/75

        return u_optimal, predicted_trajectories,last_index



# Define a function to generate a curve
def generate_curve(A=5, B=0.1, x_max=100):
    x = np.linspace(0, x_max, 200)
    y = A * np.sin(B * x)
    return x, y

def main():
    # Generate the reference curve
    x_ref, y_ref = generate_curve(A=30, B=0.04, x_max=200)
    last_index = 0
    # concate x_ref and y_ref as 2d array, shape of (N,2)
    ref_points = np.vstack([x_ref, y_ref]).T
    # Initialize car model
    car = Car_km(state=np.array([0, 0, np.pi/4, 0]))#notice the forth element is time 
    #To change the initial velocity, change the in the class Car_km
    #TODO:  self.state[C_k.V_km] = 10, CHANGE HERE!!!!!!!!

    psi_ref = car.calculate_direction(x_ref, y_ref)
    # Store the car's trajectory
    trajectory = []
    last_index = 0
    # define the car as a rectangle
    # Plot the car as a rectangle
    car_length = 4.0  # Define the car's length
    car_width = 1.0   # Define the car's width
    plt.ion()
    for i in range(400):
        plt.cla()

        #here!    Simulate the car for one time step
        ##############################################################
        #TODO:here, u can modify it as carla_state
        carla_state=np.zeros(5)

        #TODO:Change Here!  the first elements are x,y,psi,v!!!!!!!!!!!!!!!!
        carla_state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]=car.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]

    

        u_optimal, predicted_trajectories,car.last_index = \
            car.simulate(carla_state, ref_points, psi_ref, last_index)
        ##############################################################

        # Visualize the predicted trajectories
        # Plot the reference trajectory
        plt.plot(x_ref, y_ref)
        # Plot the predicted trajectories using vectorized plot
        plt.plot(predicted_trajectories[C_k.X_km, :], predicted_trajectories[C_k.Y_km, :], '-', color='red', label='Predicted Trajectory')
        # Plot the car
        car_x = car.state[C_k.X_km] - 0.5 * car_length * np.cos(car.state[C_k.Psi])
        car_y = car.state[C_k.Y_km] - 0.5 * car_length * np.sin(car.state[C_k.Psi])
        car_angle = np.degrees(car.state[C_k.Psi])
        car_rect = plt.Rectangle((car_x, car_y), car_length, car_width, angle=car_angle, edgecolor='blue', facecolor='none')
        plt.gca().add_patch(car_rect)
        # Update car state
        plt.axis("equal")
        plt.pause(0.1)
        trajectory.append(car.state.copy().flatten())
    plt.ioff()
        # Visualize results

    # Convert trajectory to NumPy array
    trajectory = np.array(trajectory)
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



if __name__ == '__main__':
    main()