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

nt=4
dt=0.02
def main():
    # Initial state for the vehicle
    initial_state = np.zeros(nt)
    initial_state[ST.V] = 5  # 5 m/s speed
    initial_state[ST.S] = 0  # initial longitudinal position
    initial_state[ST.N] = 0  # initial lateral position
    initial_state[ST.T] = 0  # initial time

    # Create a VehicleTwin instance
    vehicle_twin = VehicleTwin(initial_state)

    # Define a curved planned trajectory for the vehicle
    planned_trajectory = np.zeros((nt, 500))
    planned_trajectory[ST.T, :] = np.linspace(0, 10, 500)  # time from 0 to 10 seconds
    planned_trajectory[ST.S, :] = np.linspace(0, 50, 500)  # longitudinal position from 0 to 50 meters
    # A sinusoidal path for lateral position
    planned_trajectory[ST.N, :] = 10 * np.sin(2 * np.pi * planned_trajectory[ST.S, :] / 25)
    planned_trajectory[ST.V, :] = np.sqrt(np.diff(planned_trajectory[ST.S, :], prepend=0)**2 +
                                          np.diff(planned_trajectory[ST.N, :], prepend=0)**2) / dt  # speed

    # Update the vehicle with the planned trajectory
    vehicle_twin.update_planned_desired_trajectories(planned_trajectory, None)

    # Interpolate the trajectories
    vehicle_twin.interpolate_trajectories(0, 500, dt)  # simulate for 500 points

    # Calculate heading at different times and visualize the trajectory
    headings = []
    for current_time in np.linspace(0, 10, 500):
        psi = vehicle_twin.calculate_heading(current_time, dt)
        if psi is not None:
            headings.append(psi)

    # Convert headings to degrees for easier interpretation
    headings_degrees = np.degrees(headings)

    # Visualize the trajectory and the heading side by side
    plt.figure(figsize=(12, 6))

    # Plot the trajectory
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(planned_trajectory[ST.S, :], planned_trajectory[ST.N, :], 'b-', label='Planned trajectory')
    plt.xlabel('Longitudinal Position (m)')
    plt.ylabel('Lateral Position (m)')
    plt.title('Vehicle Planned Trajectory')
    plt.legend()
    plt.axis('equal')

    # Plot the heading
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(np.linspace(0, 10, len(headings)), headings_degrees, 'r-', label='Heading angle (degrees)')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading Angle (degrees)')
    plt.title('Vehicle Heading Angle Over Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
