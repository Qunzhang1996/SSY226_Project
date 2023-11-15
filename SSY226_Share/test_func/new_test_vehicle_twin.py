import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import casadi as cs
from enum import IntEnum
import warnings

import sys
sys.path.append('/home/zq/Desktop/SSY226_Project')
from SSY226_Share.src.vehicle_class import *
import matplotlib.pyplot as plt
import numpy as np
import math

class C_k(IntEnum):
    X_km, Y_km, Psi, T, V_km = range(5)

class ST(IntEnum):
    V, S, N, T = range(nt)

def complex_test_interpolate_trajectories_visualization():
    # Example state values: X_km, Y_km, Psi, Time, V_km
    state = np.array([10, 5, 0, 0, 20])  

    # Create a VehicleTwin instance
    vehicle_twin = VehicleTwin(state)

    # Generate test data, considering the initial state
    t = np.linspace(0, 8, 10) + state[C_k.T]  # Time points
    print('here is t', t)
    x_km = 0.5 * (t - state[C_k.T])**2 + state[C_k.X_km]  # Non-linear X_km based on initial X_km
    y_km = 0.5 * np.sin(t - state[C_k.T]) + state[C_k.Y_km]  # Non-linear Y_km based on initial Y_km
    v_km = 2 * (t - state[C_k.T]) + state[C_k.V_km]  # Linear velocity change based on initial V_km

    # Calculate Psi based on X_km and Y_km
    psi = calculate_psi(x_km, y_km)
    print('here is psi', psi)
    # Combine test data into a trajectory
    planned_trajectory = np.array([x_km, y_km, psi, t, v_km])
    print('here is pl_trajectory', planned_trajectory)

    # Update the trajectory
    vehicle_twin.update_planned_desired_trajectories(planned_trajectory, planned_trajectory)

    # Perform interpolation
    current_time = state[C_k.T]  # Start time
    planning_points = 20  # Number of test points
    dt = 0.4  # Time interval
    vehicle_twin.interpolate_trajectories(current_time, planning_points, dt)

    # Visualize the original data and interpolated results
    interpolated_trajectory = vehicle_twin.interpolated_planned_trajectory
    print('here is t', t)
    plot_trajectory_data(t, planned_trajectory, interpolated_trajectory)

def calculate_psi(x_km, y_km):
    # Calculate Psi
    psi = np.zeros_like(x_km)
    for i in range(1, len(x_km)):
        delta_y = y_km[i] - y_km[i - 1]
        delta_x = x_km[i] - x_km[i - 1]
        psi[i] = math.atan2(delta_y, delta_x)
    return psi

def plot_trajectory_data(t, original_data, interpolated_data):
    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Plot original data and interpolated results
    labels = ['X_km', 'Y_km', 'Psi', 'Time', 'V_km']
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.plot(t, original_data[i], 'o-', label='Original')
        plt.plot(np.linspace(t[0], t[-1], len(interpolated_data[i])), interpolated_data[i], 'x-', label='Interpolated')
        plt.title(labels[i])
        plt.legend()

    plt.tight_layout()
    plt.show()

# Call the test function
complex_test_interpolate_trajectories_visualization()
