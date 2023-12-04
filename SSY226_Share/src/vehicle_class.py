import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import casadi as cs
from enum import IntEnum

import warnings
warnings.simplefilter("error")

# Structure for name convenience
nt = 4  # number of variables in trajectories
class ST(IntEnum):
    V, S, N, T = range(nt)

class C(IntEnum):
    V_S, S, N, T, V_N = range(5)
# states: v_s longitudinal speed, v_n lateral speed, s longitudinal position, time, n lateral position

class C_k(IntEnum):
    X_km, Y_km, Psi, T, V_km =range(5)

#follow vehicle kinematic, define new C_K
#states: X longitudinal, Y lateral, Psi heading, T time, V velocity

class VehicleTwin:
    """Useful class to represent other vehicles, digital twin"""

    def __init__(self, state):
        self.state = state
        self.desired_trajectory = None
        self.planned_trajectory = None
        self.interpolated_desired_trajectory = None
        self.interpolated_planned_trajectory = None

    def update_state(self, state):
        """Take current state of another vehicle"""
        self.state = state

    def update_planned_desired_trajectories(self, planned_trajectory, desired_trajectory):
        """Take planned and desired trajectories of another vehicle"""
        self.desired_trajectory = desired_trajectory
        self.planned_trajectory = planned_trajectory

    def interpolate_trajectories(self, current_time, planning_points, dt):
        """Compute trajectory points of another vehicles at convenient time steps"""

        if self.planned_trajectory is not None:
            # Interpolate/extrapolate previously received
            self.interpolated_planned_trajectory = scipy.interpolate.interp1d(
                self.planned_trajectory[ST.T, :], self.planned_trajectory, fill_value='extrapolate', kind='cubic')(current_time+np.arange(planning_points)*dt)
        else:
            # Planned trajectory is either not yet received or not computed by that vehicle
            # Use a generic model to predict future points
            self.interpolated_planned_trajectory = np.zeros(
                (nt, planning_points))
            self.interpolated_planned_trajectory[ST.V] = np.ones(
                planning_points) * self.state[ST.V]
            self.interpolated_planned_trajectory[ST.S] = self.state[ST.S] + (
                current_time - self.state[ST.T]) * self.state[ST.V] + np.arange(planning_points)*self.state[ST.V]*dt
            self.interpolated_planned_trajectory[ST.N] = np.ones(
                planning_points) * self.state[ST.N]
            self.interpolated_planned_trajectory[ST.T] = current_time + np.arange(
                planning_points)*dt

        if self.desired_trajectory is not None:
            self.interpolated_desired_trajectory = scipy.interpolate.interp1d(
                self.desired_trajectory[ST.T, :], self.desired_trajectory, fill_value='extrapolate', kind='cubic')(current_time+np.arange(planning_points)*dt)
        else:
            # Desired trajectory is not computed by all vehicles at all moments so do nothing
            self.interpolated_desired_trajectory == None


class Vehicle:
    """Common class for all vehicles"""

    def __init__(self, state):
        self.vehicles = {}
        self.state = state
        self.planning_dt = 0.8*(51/31)*0.7
        self.planning_points = 31
        self.planned_trajectory = None
        self.desired_trajectory = None
        self.control_trajectory = None
        self.lane_width = 3.5 # m

        self.history_state = [np.array(self.get_state())]
        self.history_planned_trajectory = []
        self.history_desired_trajectory = []

        self.name = 'v'

    def add_vehicle_twin(self, name, state):
        self.vehicles[name] = VehicleTwin(state)

    def get_state(self):
        return self.state

    def update_state(self, state):
        """Update the vehicle state with information from CAN"""
        self.state = state

    def update_twin_state(self, name_id, state):
        self.vehicles[name_id].update_state(state)

    def receive_planned_desired_trajectories(self, name_id, planned_trajectory, desired_trajectory):
        self.vehicles[name_id].update_planned_desired_trajectories(
            planned_trajectory, desired_trajectory)
        
    def save_history(self,name):
        import pickle
        with open(name+'.pickle', 'wb') as handle:
            data = {}
            data['history_state'] = np.array(self.history_state).T
            data['history_planned_trajectory'] = self.history_planned_trajectory
            pickle.dump(data, handle)