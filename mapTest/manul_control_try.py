import carla
import math
import random
import time
import socket

# Define a function to set throttle/acceleration and steering angle
def set_vehicle_control(vehicle, throttle, steer):
    control = carla.VehicleControl()
    control.throttle = throttle  # Throttle/acceleration value (0 to 1)
    control.steer = steer      # Steering angle (-1 to 1, where -1 is full left, 1 is full right)
    vehicle.apply_control(control)

# Define a function to send vehicle location to CarlaUE4
def send_location_to_ue4(host, port, location, rotation):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        message = f"vehicle_location {location.x} {location.y} {location.z} {rotation.pitch} {rotation.yaw} {rotation.roll}"
        client_socket.sendto(message.encode(), (host, port))
        client_socket.close()
    except Exception as e:
        print(f"Error sending location to CarlaUE4: {e}")

# Connect to the Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Set up parameters for sending location to CarlaUE4
ue4_host = 'localhost'  # Update with the host running CarlaUE4
ue4_port = 12345  # Update with the port CarlaUE4 is listening on

try:
    # Load the Town01 map
    world = client.load_world('Town01')

    # Create a vehicle at a random spawn point
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = random.choice(blueprint_library.filter('vehicle'))
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Define a desired circular trajectory
    radius = 10.0  # Radius of the circle in meters
    speed = 5.0    # Desired speed in m/s
    time_interval = 0.1  # Time interval for control updates

    while True:
        # Get the vehicle's location and rotation
        vehicle_location = vehicle.get_location()
        print(vehicle_location)
        vehicle_rotation = vehicle.get_transform().rotation

        # Send the location to CarlaUE4
        send_location_to_ue4(ue4_host, ue4_port, vehicle_location, vehicle_rotation)

        # Calculate the desired steering angle for a circular path
        angular_velocity = speed / radius
        steer = 0.5

        # Set throttle/acceleration and steering angle
        throttle = 10.0  # Full throttle
        set_vehicle_control(vehicle, throttle, steer)

        # Sleep for the time interval
        time.sleep(time_interval)

finally:
    # Cleanup
    vehicle.destroy()
