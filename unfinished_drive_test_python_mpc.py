##更换地图为Town06
#(myenv) C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\util>python config.py --map Town06

#python config.py --map Town06
#增加了control的brake输入
#更改了throttle 和 steer为非线性sin函数




# Imports

import carla
import math
import random
import time
import cv2
import sys
import numpy as np
import threading

import matplotlib.pyplot as plt
import scipy
import casadi as cs
from enum import IntEnum
import warnings
from test_mpc import C_k, Car_km
import scipy.interpolate
import scipy.optimize


# Setup

client = carla.Client('localhost', 2000)
world = client.get_world()

bp_lib = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

car_bp = bp_lib.find('vehicle.tesla.model3')
truck_bp = bp_lib.find('vehicle.carlamotors.firetruck')




for actor in world.get_actors().filter('vehicle.*'):
    actor.destroy()


# Coordinates
# Initial and destination coordinates for both the car and the truck are defined.
x_car = 78.78953552246094
y_car = 218.3877716064453 
z_car = 2.189335823059082

dest_x_car = 238.78953552246094
dest_y_car = 177.3877716064453 
dest_z_car = z_car

x_truck = 109
y_truck = 160.6210479736328
z_truck = 1.6630632877349854

dest_x_truck = 259
dest_y_truck = y_truck
dest_z_truck = z_truck


# The closest spawn points for the car, truck, and their destinations are determined based on the coordinates.
start_point_car = min(spawn_points, key=lambda spawn_point: spawn_point.location.distance(carla.Location(x_car, y_car, z_car)))
dest_point_car = min(spawn_points, key=lambda spawn_point: spawn_point.location.distance(carla.Location(dest_x_car, dest_y_car, dest_z_car)))

start_point_truck = min(spawn_points, key=lambda spawn_point: spawn_point.location.distance(carla.Location(x_truck, y_truck, z_truck)))
dest_point_truck = min(spawn_points, key=lambda spawn_point: spawn_point.location.distance(carla.Location(dest_x_truck, dest_y_truck, dest_z_truck)))

# Spawn the vehicle at the closest spawn point
car = world.try_spawn_actor(car_bp, start_point_car)
truck = world.try_spawn_actor(truck_bp, start_point_truck)
spectator = world.get_spectator()
transform = car.get_transform()
print(transform)




pref_speed = 50
speed_threshold = 2

# Define a function to generate a curve
def generate_curve(A=5, B=0.1, x_max=50):
    x = np.linspace(0, x_max, 100)
    y = A * np.sin(B * x)
    return x, y

def main(vehicle1, car_flag, route):
    # Generate the reference curve
    # x_ref, y_ref = generate_curve(A=20, B=0.05, x_max=100)
    x_ref, y_ref = visualize_route(route)
    last_index = 0
    # concate x_ref and y_ref as 2d array, shape of (N,2)
    ref_points = np.vstack([x_ref, y_ref]).T
    # Initialize car model
    car = Car_km(state=np.array([110 , 235,np.pi/4, 0]))#notice the forth element is time 
    #To change the initial velocity, change the in the class Car_km
    #TODO:  
    # self.state[C_k.V_km] = 10, #CHANGE HERE!!!!!!!!
    psi_ref = Car_km.calculate_direction(x_ref, y_ref)
    # Store the car's trajectory
    trajectory = []
    last_index = 0
    u_optimal=np.zeros(2)
    # define the car as a rectangle
    # Plot the car as a rectangle
    car_length = 4.0  # Define the car's length
    car_width = 1.0   # Define the car's width

    #todo: deadlock 不行


    plt.ion()

    for i in range(90):

        plt.cla()

        #here!    Simulate the car for one time step
        ##############################################################
        #TODO:here, u can modify it as carla_state
        carla_state=np.zeros(5)

        #TODO:Change Here!  the first elements are x,y,psi,v!!!!!!!!!!!!!!!!

        # carla_state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]=car.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]
        carla_state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]=get_state(vehicle1)
    

        u_optimal, predicted_trajectories,car.last_index = \
            car.simulate(carla_state, ref_points, psi_ref, last_index)
        ##############################################################
        # estimated_throttle = u_optimal[0]
        # steer_input = u_optimal[1]
        # # vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
        #  # 处理减速情况
        # if estimated_throttle < 0:
        #     # 如果估计的油门小于0，意味着需要减速或刹车
        #     brake_input = max(estimated_throttle, -1)  # 限制刹车输入在[-1, 0]之间
        #     throttle_input = 0  # 油门设为0
        # else:
        #     # 否则，继续使用估计的油门值
        #     brake_input = 0
        #     throttle_input = min(estimated_throttle, 1)  # 限制油门值在[0, 1]之间
        # vehicle.apply_control(carla.VehicleControl(throttle=throttle_input, steer=steer_input, brake=brake_input))
        
        
        estimated_throttle = u_optimal[0]
        steer_input = np.sin(u_optimal[1])
        # vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
         # 处理减速情况
        if estimated_throttle <0:
            # 如果估计的油门小于0，意味着需要减速或刹车
            brake_input =  np.sin(estimated_throttle) # 使用非线性函数并保证刹车输入在[0, 1]之间
            throttle_input = 0  # 油门设为0
        else:
            # 否则，继续使用估计的油门值
            throttle_input = np.sin(estimated_throttle)  # 使用非线性函数并保证油门输入在[0, 1]之间
            brake_input = 0

        vehicle1.apply_control(carla.VehicleControl(throttle=throttle_input, steer=steer_input, brake=brake_input)) 

    
        
        ###################################################################
        #vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))

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


def main2(vehicle2, car_flag, route):
    # Generate the reference curve
    # x_ref, y_ref = generate_curve(A=20, B=0.05, x_max=100)
    x_ref, y_ref = visualize_route(route)
    last_index = 0
    # concate x_ref and y_ref as 2d array, shape of (N,2)
    ref_points = np.vstack([x_ref, y_ref]).T
    # Initialize car model
    car = Car_km(state=np.array([94.8 , 150.31, 0, 0]))#notice the forth element is time 
    #To change the initial velocity, change the in the class Car_km
    #TODO:  
    # self.state[C_k.V_km] = 10, #CHANGE HERE!!!!!!!!
    psi_ref = Car_km.calculate_direction(x_ref, y_ref)
    # Store the car's trajectory
    trajectory = []
    last_index = 0
    u_optimal=np.zeros(2)
    # define the car as a rectangle
    # Plot the car as a rectangle
    car_length = 4.0  # Define the car's length
    car_width = 1.0   # Define the car's width



    plt.ion()

    for i in range(90):

        plt.cla()

        #here!    Simulate the car for one time step
        ##############################################################
        #TODO:here, u can modify it as carla_state
        carla_state=np.zeros(5)

        #TODO:Change Here!  the first elements are x,y,psi,v!!!!!!!!!!!!!!!!

        # carla_state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]=car.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]
        carla_state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]=get_state(vehicle2)
    

        u_optimal, predicted_trajectories,car.last_index = \
            car.simulate(carla_state, ref_points, psi_ref, last_index)
        ##############################################################
        estimated_throttle = u_optimal[0]
        # estimated_throttle=0
        steer_input = np.sin(u_optimal[1])
        # steer_input=0
        # vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
         # 处理减速情况
        if estimated_throttle <0:
            # 如果估计的油门小于0，意味着需要减速或刹车
            brake_input =  np.sin(estimated_throttle) # 使用非线性函数并保证刹车输入在[0, 1]之间
            throttle_input = 0  # 油门设为0
        else:
            # 否则，继续使用估计的油门值
            throttle_input = np.sin(estimated_throttle)  # 使用非线性函数并保证油门输入在[0, 1]之间
            brake_input = 0

        vehicle2.apply_control(carla.VehicleControl(throttle=throttle_input, steer=steer_input, brake=brake_input)) 

    
        
        ###################################################################
        #vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))

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

def maintain_speed(s):
    if s >= pref_speed:
        return 0
    elif s < pref_speed - speed_threshold:
        return 0.8
    else:
        return 0.4
    

def angle_between(v1, v2):
    return math.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))

def get_angle(car, wp):
    car_pos = car.get_transform()
    car_x = car_pos.location.x
    car_y = car_pos.location.y
    wp_x = wp.transform.location.x
    wp_y = wp.transform.location.y

    x = (wp_x - car_x) / ((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5
    y = (wp_y - car_y) / ((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5

    car_vector = car_pos.get_forward_vector()
    degrees = angle_between((x,y), (car_vector.x, car_vector.y))

    return degrees

# Get information about vehicle
def get_info_about_vehicle(vehicle):
    vehicle_pos = vehicle.get_transform()
    vehicle_loc = vehicle_pos.location
    vehicle_rot = vehicle_pos.rotation
    print(f'{vehicle.id} location: ', vehicle_loc, 'Vehicle rotation: ', vehicle_rot)

# get the 5 states
def get_state(vehicle):
    vehicle_pos = vehicle.get_transform()
    vehicle_loc = vehicle_pos.location
    vehicle_rot = vehicle_pos.rotation
    vehicle_vel = vehicle.get_velocity()

    # Extract relevant states
    x = vehicle_loc.x 
    y = vehicle_loc.y 
    psi = math.radians(vehicle_rot.yaw)  # Convert yaw to radians
    # v = math.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)
    v = vehicle_vel.length()  #converting it to km/hr

    return x, y, psi, v


def calculate_distance(actor1, actor2):
    location1 = actor1.get_location()
    location2 = actor2.get_location()
    distance = location1.distance(location2)
    return distance

# Route planner

def plan_route(world, start_point, dest_x, dest_y, dest_z, sampling_resolution=1, debug_draw=False, car_flag=False):
    
    # This following line needs to be changed according to YOUR installation path
    carla_path =  r"C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla"
    sys.path.append(carla_path)
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    point_a = start_point.location
    point_b = carla.Location(x=dest_x, y=dest_y, z=dest_z)

    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)

    route = grp.trace_route(point_a, point_b)
    print(route)

    if debug_draw:
        if car_flag:
            for waypoint in route:
                world.debug.draw_string(
                    waypoint[0].transform.location,
                    '^',
                    draw_shadow=False,
                    color=carla.Color(r=0, g=0, b=255),
                    life_time=60.0,
                    persistent_lines=True
            )

    return route



def plan_route2(world, start_point, dest_x, dest_y, dest_z, sampling_resolution=1, debug_draw=False, car_flag=False):
    
    # This following line needs to be changed according to YOUR installation path
    carla_path =  r"C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla"
    sys.path.append(carla_path)
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    point_a = start_point.location
    point_b = carla.Location(x=dest_x, y=dest_y, z=dest_z)

    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)

    route = grp.trace_route(point_a, point_b)
    print(route)

    if debug_draw:
        if car_flag:
            for waypoint in route:
                world.debug.draw_string(
                    waypoint[0].transform.location,
                    '^',
                    draw_shadow=False,
                    color=carla.Color(r=0, g=0, b=255),
                    life_time=60.0,
                    persistent_lines=True
            )

    return route
def visualize_route(route):
    # Extract x, y coordinates from waypoints
    x = [waypoint[0].transform.location.x for waypoint in route]
    y = [waypoint[0].transform.location.y for waypoint in route]
    return x, y

def convert_to_road_params(route, lane_width):
    # Assuming waypoints is a list of carla.Transform objects
    x = [waypoint[0].transform.location.x for waypoint in route]
    y = [waypoint[0].transform.location.y for waypoint in route]

    # Calculate parameters based on the waypoints
    P_road_v = [max(x) + lane_width/2, 0.1, 80, min(y) - lane_width/2,  # Assuming lane is on the left
                   max(x) + lane_width/2, 0.1, 0, max(y) + lane_width/2]  # Assuming lane is on the right
    
    ss = np.linspace(-900, 2000, 1000)
    road_right = P_road_v[0]/2 * (np.tanh(P_road_v[1]*(ss-P_road_v[2]))+1) + P_road_v[3]
    road_left = P_road_v[4]/2 * (np.tanh(P_road_v[5]*(ss-P_road_v[6]))+1) + P_road_v[7]


lane_width = 3.5
# convert_to_road_params(route_car, lane_width)

def drive_vehicle(vehicle, route, other_vehicle, car_flag):
    curr_wp = 0
    predicted_angle = get_angle(vehicle, route[0][0])

    while curr_wp < len(route) - 1:
        world.tick()

        # Additional conditions or user input for stopping the simulation
        # Calculate distance between the vehicles
        distance_to_other_vehicle = calculate_distance(vehicle, other_vehicle)
        print(distance_to_other_vehicle)

        # Check if the distance is below a threshold
        if distance_to_other_vehicle < 20:
            vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
            other_vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
            return


        while curr_wp < len(route) and vehicle.get_transform().location.distance(route[curr_wp][0].transform.location) < 5:
            curr_wp += 1

        if car_flag == False:
            predicted_angle = get_angle(vehicle, route[curr_wp][0])
            v = vehicle.get_velocity()
            speed = round(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2), 0)
            estimated_throttle = maintain_speed(speed)

            if predicted_angle < - 300:
                predicted_angle = predicted_angle + 360
            elif predicted_angle > 300:
                predicted_angle = predicted_angle - 360
            steer_input = predicted_angle

            if predicted_angle < -40:
                steer_input = -40
            elif predicted_angle > 40:
                steer_input = 40
            
            steer_input = steer_input / 75
            # u_optimal_mp = main()
            # estimated_throttle = u_optimal_mp[0]
            # steer_input = u_optimal_mp[1]

            vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
        else:
            main(vehicle, car_flag, route)
        

         # Get the vehicle type (Car or Truck)
        # vehicle_type = 'Car' if 'vehicle.tesla' in str(vehicle.type_id) else 'Truck'

        # print( vehicle.id , get_state(vehicle))

    vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
    
    
def drive_vehicle2(vehicle, route, other_vehicle, car_flag):
    curr_wp = 0
    predicted_angle = get_angle(vehicle, route[0][0])

    while curr_wp < len(route) - 1:
        world.tick()

        # Additional conditions or user input for stopping the simulation
        # Calculate distance between the vehicles
        distance_to_other_vehicle = calculate_distance(vehicle, other_vehicle)
        print(distance_to_other_vehicle)

        # Check if the distance is below a threshold
        if distance_to_other_vehicle < 20:
            vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
            other_vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))
            return


        while curr_wp < len(route) and vehicle.get_transform().location.distance(route[curr_wp][0].transform.location) < 5:
            curr_wp += 1

        if car_flag == False:
            predicted_angle = get_angle(vehicle, route[curr_wp][0])
            v = vehicle.get_velocity()
            speed = round(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2), 0)
            estimated_throttle = maintain_speed(speed)

            if predicted_angle < - 300:
                predicted_angle = predicted_angle + 360
            elif predicted_angle > 300:
                predicted_angle = predicted_angle - 360
            steer_input = predicted_angle

            if predicted_angle < -40:
                steer_input = -40
            elif predicted_angle > 40:
                steer_input = 40
            
            steer_input = steer_input / 75
            # u_optimal_mp = main()
            # estimated_throttle = u_optimal_mp[0]
            # steer_input = u_optimal_mp[1]

            vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
        else:
            main2(vehicle, car_flag, route)
        

         # Get the vehicle type (Car or Truck)
        # vehicle_type = 'Car' if 'vehicle.tesla' in str(vehicle.type_id) else 'Truck'

        # print( vehicle.id , get_state(vehicle))

    vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))



def create_and_start_thread(target_function, vehicle, route, other_vehicle, car_flag):
    thread = threading.Thread(target=target_function, args=(vehicle, route, other_vehicle, car_flag))
    thread.start()
    return thread

# Test for running
car_flag = True
route_car = plan_route(world, start_point_car, dest_x_car, dest_y_car, dest_z_car, sampling_resolution=1, debug_draw=True, car_flag=True)
#car_thread = threading.Thread(target=drive_vehicle, args=(car, route_car, truck, car_flag))
car_thread = create_and_start_thread(drive_vehicle, car, route_car, truck, car_flag)
car_flag = True
route_truck = plan_route2(world, start_point_truck, dest_x_truck, dest_y_truck, dest_z_truck, sampling_resolution=1, debug_draw=True, car_flag=True)
#truck_thread = threading.Thread(target=drive_vehicle2, args=(truck, route_truck, car, car_flag))
truck_thread = create_and_start_thread(drive_vehicle, truck, route_truck, car, car_flag)
# # Start the threads
car_thread.start()
truck_thread.start()

# # Wait for both threads to finish
truck_thread.join()
car_thread.join()




