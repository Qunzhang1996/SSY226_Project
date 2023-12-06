import numpy as np
import matplotlib.pyplot as plt

# Define simulation parameters and road curve parameters
dt = 0.02  # Simulation step in seconds
lane_width = 3.5  # Lane width in meters

P_road_v1 = [0, 0.1, 0, 146.818-0.5 * lane_width,
             0, 0.1, 0, 146.818+0.5 * lane_width]

P_road_v2 = [lane_width, 0.1, 80, 146.818-1.5 * lane_width,
             lane_width, 0.1, 0, 146.818-0.5 * lane_width]

# Define a range of s values to compute the road boundaries
ss = np.linspace(-900, 2000, 1000)

# Compute the road boundaries for P_road_v1
road_right_v1 = P_road_v1[0]/2 * (np.tanh(P_road_v1[1]*(ss - P_road_v1[2])) + 1) + P_road_v1[3]
road_left_v1 = P_road_v1[4]/2 * (np.tanh(P_road_v1[5]*(ss - P_road_v1[6])) + 1) + P_road_v1[7]

# Compute the road boundaries for P_road_v2
road_right_v2 = P_road_v2[0]/2 * (np.tanh(P_road_v2[1]*(ss - P_road_v2[2])) + 1) + P_road_v2[3]
road_left_v2 = P_road_v2[4]/2 * (np.tanh(P_road_v2[5]*(ss - P_road_v2[6])) + 1) + P_road_v2[7]

# Plotting the curves for P_road_v1 and P_road_v2 side by side
plt.figure(figsize=(15, 6))

# P_road_v1
plt.subplot(1, 2, 1)
plt.plot(ss, road_right_v1, 'b', label='Right Boundary v1')
plt.plot(ss, road_left_v1, 'r', label='Left Boundary v1')
plt.title('Road Boundaries for P_road_v1')
plt.xlabel('s-coordinate')
plt.ylabel('Lateral position')
plt.legend()
plt.grid(True)

# P_road_v2
plt.subplot(1, 2, 2)
plt.plot(ss, road_right_v2, 'b', label='Right Boundary v2')
plt.plot(ss, road_left_v2, 'r', label='Left Boundary v2')
plt.title('Road Boundaries for P_road_v2')
plt.xlabel('s-coordinate')
plt.ylabel('Lateral position')
plt.legend()
plt.grid(True)

# Show the plot with a tight layout
plt.tight_layout()
plt.show()
