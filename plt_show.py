# import carla
# import matplotlib.pyplot as plt

# # Connect to the CARLA server
# client = carla.Client('localhost', 2000)
# world = client.get_world()
# map = world.get_map()

# # Retrieve waypoints for the entire map
# waypoints = map.generate_waypoints(distance=1.0)  # distance between waypoints in meters

# # Initialize lists for storing data
# x_coords = []
# y_coords = []
# lane_widths = []

# # Extract the coordinates and lane width for each waypoint
# for waypoint in waypoints:
#     x_coords.append(waypoint.transform.location.x)
#     y_coords.append(waypoint.transform.location.y)
#     lane_widths.append(waypoint.lane_width)

# # Plot the waypoints
# for i in range(len(x_coords)):
#     plt.plot(x_coords[i], y_coords[i], 'o', markersize=lane_widths[i] * 2, label=f'Waypoint {i}')

# # Optional: Set labels if you want to label the waypoints
# for i, label in enumerate(lane_widths):
#     plt.text(x_coords[i], y_coords[i], f'{label:.2f}', fontsize=9)

# plt.xlabel('X coordinate')
# plt.ylabel('Y coordinate')
# plt.title('CARLA Map Waypoints and Lane Widths')
# plt.axis('equal')  # This ensures that one unit in X is the same as one unit in Y
# plt.show()
import carla

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
world = client.get_world()
map = world.get_map()

# Retrieve all the waypoints in the map at a certain distance
waypoints = map.generate_waypoints(distance=10.0)

# Process each waypoint to get road information
for waypoint in waypoints:
    road_id = waypoint.road_id
    lane_id = waypoint.lane_id
    lane_type = waypoint.lane_type
    lane_width = waypoint.lane_width

    print(f'Road ID: {road_id}, Lane ID: {lane_id}, Lane Type: {lane_type}, Lane Width: {lane_width}')

# If you want to get more detailed road information, you can also access the road's topology
# For example, to get the list of all junctions in the map:
junctions = map.get_junctions()

# Print information about each junction
for junction in junctions:
    print(f'Junction ID: {junction.id}, Bounding Box: {junction.bounding_box}')
