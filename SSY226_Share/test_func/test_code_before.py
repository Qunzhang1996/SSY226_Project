import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import casadi as cs
from enum import IntEnum
import warnings

import sys
sys.path.append('/home/zq/Desktop/SSY226_Project')
from SSY226_Share.raw_code.simulation_environment_share import VehicleTwin
nt = 4

class ST(IntEnum):
    V, S, N, T = range(nt)

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

# Using your provided class definitions...

def test_particle_model_interpolation():
    # Create an instance of VehicleTwin
    state = np.array([20, 10, 5, 0])  # Example state: V, S, N, T
    vehicle_twin = VehicleTwin(state)

    # Generate test data, considering the initial state
    t = np.linspace(0, 4, 5) + state[ST.T]  # Time points
    s = 0.5 * (t - state[ST.T])**2 + state[ST.S]  # Longitudinal displacement
    n = 0.5 * np.sin(t - state[ST.T]) + state[ST.N]  # Lateral displacement
    v = 2 * (t - state[ST.T]) + state[ST.V]  # Longitudinal velocity

    # Combine test data into a trajectory
    planned_trajectory = np.array([v, s, n, t])

    # Update the trajectory
    vehicle_twin.update_planned_desired_trajectories(planned_trajectory, planned_trajectory)

    # Perform interpolation
    current_time = state[ST.T] + 0.5  # Start time
    planning_points = 10  # Number of test points
    dt = 0.4  # Time interval
    vehicle_twin.interpolate_trajectories(current_time, planning_points, dt)

    # Visualize original data and interpolated results
    interpolated_trajectory = vehicle_twin.interpolated_planned_trajectory
    plot_trajectory_data(t, planned_trajectory, interpolated_trajectory)

def plot_trajectory_data(t, original_data, interpolated_data):
    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Plot original data and interpolated results
    labels = ['V (Velocity)', 'S (Longitudinal Displacement)', 'N (Lateral Displacement)', 'T (Time)']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(t, original_data[i], 'o-', label='Original')
        plt.plot(np.linspace(t[0], t[-1], len(interpolated_data[i])), interpolated_data[i], 'x-', label='Interpolated')
        plt.title(labels[i])
        plt.legend()

    plt.tight_layout()
    plt.show()

# Call the test function
test_particle_model_interpolation()
