import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import casadi as cs
from enum import IntEnum
import warnings
from vehicle_class import Vehicle, Car, Truck_CC,ST,C
from car_km_class import Car_km
def main():
    #################
    # In simulation
    nt=4
    dt = 0.02  # simulations step in seconds
    lane_width = 3.5 # m
    steps_between_replanning = 25
    # steps_between_replanning = 100
    replanning_iterations = 100
    # replanning_iterations = 10
    P_road_v1 = [0, 0.1, 0, -0.5*lane_width,
                0, 0.1, 0, 0.5*lane_width]


    # Create two independent objects to represent two vehicles

    # CC Truck
    v1 = Truck_CC([25, -416.25-70, 0, 0],dt=dt)
    v1.P_road_v = P_road_v1
    v1.name = 'v1'

    # Only used for Car (CAV)
    P_road_v = [lane_width, 0.1, 80, -1.5*lane_width,
                lane_width, 0.1, 0, -0.5*lane_width]
    v2 = Car_km([19, -397.53, -lane_width, 0],dt=dt)
    v2.P_road_v = P_road_v
    v2.lane_width = lane_width
    v2.name = 'v2'

    # Compute planned and desired trajectories of CAV without sending them to others, needed for simulation
    v1.compute_planned_desired_trajectory()
    v2.compute_planned_desired_trajectory()

    # Simulate the system for some time without updates from other vehicles
    for _ in range(steps_between_replanning):
            for v in [v1, v2]:
                v.simulate(dt)


    # Each vehicle outputs own state
    v1_state = v1.get_state()
    v2_state = v2.get_state()

    # We send and receive it over network

    # Create a twin vehicle inside v1 with the current state, CV
    v1.add_vehicle_twin('v2', v2_state)
    # Create a twin vehicle inside v2 with the current state, CAV
    v2.add_vehicle_twin('v1', v1_state)

    for _ in range(replanning_iterations):

        # Compute planned and desired trajectories
        v1_planned_trajectory, v1_desired_trajectory = v1.compute_planned_desired_trajectory()
        v2_planned_trajectory, v2_desired_trajectory = v2.compute_planned_desired_trajectory()

        # Receive planned and desired trajectories of other vehicles inside twin vehicles
        v2.receive_planned_desired_trajectories(
            'v1', v1_planned_trajectory, v1_desired_trajectory)
        v1.receive_planned_desired_trajectories(
            'v2', v2_planned_trajectory, v2_desired_trajectory)

        # Simulate all vehicles for steps_between_replanning steps
        for _ in range(steps_between_replanning):
            for v in [v1, v2]:
                v.simulate(dt)

                # Read states of all vehicles
                v1_state = v1.get_state()
                v2_state = v2.get_state()

                # Update internal twin vehicles
                v1.update_twin_state('v2', v2_state)
                v2.update_twin_state('v1', v1_state)


    # plot road
    plt.subplot(2, 1, 1)
    ss = np.linspace(-900, 2000, 1000)
    road_right = P_road_v[0]/2 * \
        (np.tanh(P_road_v[1]*(ss-P_road_v[2]))+1)+P_road_v[3]
    road_left = P_road_v[4]/2*(np.tanh(P_road_v[5]*(ss-P_road_v[6]))+1)+P_road_v[7]
    plt.plot(ss, road_right, 'k')
    plt.plot(ss, road_left, 'k')

    # plot state and history
    # v0_history_state = np.array(v0.history_state).T
    v1_history_state = np.array(v1.history_state).T
    v2_history_state = np.array(v2.history_state).T

    # v0_history_planned = v0.history_planned_trajectory
    v1_history_planned = v1.history_planned_trajectory
    v2_history_planned = v2.history_planned_trajectory

    # line_v0, = plt.plot(v0_history_state[ST.S], v0_history_state[ST.N])
    line_v1, = plt.plot(v1_history_state[ST.S], v1_history_state[ST.N])
    plt.plot(v1_history_state[ST.S][-1], v1_history_state[ST.N][-1], 'o', color=line_v1.get_color())
    line_v2, = plt.plot(v2_history_state[ST.S], v2_history_state[ST.N])
    plt.plot(v2_history_state[ST.S][-1], v2_history_state[ST.N][-1], 'o', color=line_v2.get_color())

    for vhp in v1_history_planned:
        plt.plot(vhp[ST.S, :], vhp[ST.N, :], ':',
                color=line_v1.get_color(), linewidth=0.7)
    for vhp in v2_history_planned:
        plt.plot(vhp[ST.S, :], vhp[ST.N, :], ':',
                color=line_v2.get_color(), linewidth=0.7)

    plt.subplot(2, 1, 2)
    line_v1, = plt.plot(v1_history_state[ST.S], v1_history_state[ST.V])
    line_v2, = plt.plot(v2_history_state[ST.S], v2_history_state[ST.V])
    for vhp in v1_history_planned:
        plt.plot(vhp[ST.S, :], vhp[ST.V, :], ':',
                color=line_v1.get_color(), linewidth=0.7)
    for vhp in v2_history_planned:
        plt.plot(vhp[ST.S, :], vhp[ST.V, :], ':',
                color=line_v2.get_color(), linewidth=0.7)
    if hasattr(v1, "history_v_ref"):
        plt.plot(v1_history_state[ST.S], v1.history_v_ref, '--')
    if hasattr(v2, "history_v_ref"):
        plt.plot(v2_history_state[ST.S], v2.history_v_ref, '--')
        

    # v1.save_history('v1')
    # v2.save_history('v2')

    plt.show()


if __name__ == "__main__":
    main()