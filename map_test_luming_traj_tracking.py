##change map to Town06
#python config.py --map Town06

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
from test_mpc_truck import Truck_km
import scipy.interpolate
import scipy.optimize

import sys

# Add the path to the CARLA PythonAPI directory to the system path
sys.path.append('C:\\CARLA_0.9.14\WindowsNoEditor\\PythonAPI\\carla')

# Import the VehiclePIDController
from agents.navigation.controller import VehiclePIDController

# The rest of your code follows...



def read_coordinates(file_path):
    """
    #read coordinates from file
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        coordinates = [tuple(map(float, line.split())) for line in lines]
    return coordinates

def draw_waypoints(world, coordinates, color):
    """
    在 Carla 地图上绘制坐标，使用指定的颜色。
    """
    for x, y in coordinates:
        # assume Z coordinate is 1.0, can be adjusted as needed
        location = carla.Location(x, y, 0.5)
        world.debug.draw_point(location, size=0.05, color=color, life_time=120.0)

client = carla.Client('localhost', 2000)

world = client.get_world()

for actor in world.get_actors().filter('vehicle.*'):
    actor.destroy()
#clear other vehicles in the scene 

# def draw_waypoints(waypoints, road_id=None, life_time=50.0):
 
#   for waypoint in waypoints:
 
#     if(waypoint.road_id == road_id):
#           world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
#                                    color=carla.Color(r=0, g=255, b=0), life_time=life_time,
#                                    persistent_lines=True)
 
#以距离为1的间距创建waypoints                                  
waypoints = world.get_map().generate_waypoints(distance=1.0)
#life_time 为画出的辅助标志存活时间
#draw_waypoints(waypoints, road_id=39, life_time=20)

#我期望的道路id是39.

#找到这条路上的所有waypoints
# filtered_waypoints = []
# for waypoint in waypoints:
#     if(waypoint.road_id == 39):
#       filtered_waypoints.append(waypoint)
      
# len_waypoints = len(filtered_waypoints)
# print(f"len_waypoints: {len_waypoints}")
# spawn_point = filtered_waypoints[-2].transform
# spawn_point.location.z += 2
#bp为blueprint制造出来的小车
bp_lib = world.get_blueprint_library()
car_bp = bp_lib.find('vehicle.tesla.model3')
# vehicle = world.spawn_actor(car_bp, spawn_point)
# # 假设 vehicle 是你已经创建并放置到世界中的车辆实例
# transform = vehicle.get_transform()
# location = transform.location

# # 打印车辆的坐标
# print(f"车辆坐标: x={location.x}, y={location.y}, z={location.z}")

# 确保 spawn_point 是有效的位置
spawn_point = carla.Transform(carla.Location(x=94+30, y=143.318146, z=0.300000), carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))

car = None
try:
    car = world.spawn_actor(car_bp, spawn_point)
except Exception as e:
    print(f"wrong when setting vehicle: {e}")

if car is not None:
    # wait for a tick to ensure carla has gotten the updated transform
    time.sleep(0.02)  # 等待1秒

    transform = car.get_transform()
    location_car = transform.location

    # print the position of the vehicle
    print(f"car's corrdinate: x={location_car.x}, y={location_car.y}, z={location_car.z}")
else:
    print("car is None")


truck_bp = bp_lib.find('vehicle.carlamotors.firetruck')
#truck_bp = bp_lib.find('vehicle.tesla.model3')
# x_truck, y_truck, z_truck = location_car.x+100,location_car.y-3.5, 1.6630632877349854
# start_point_truck = min(spawn_point, key=lambda spawn_point: spawn_point.location.distance(carla.Location(x_truck, y_truck, z_truck)))
# truck = world.try_spawn_actor(truck_bp, start_point_truck)
spawn_point.location.y += 3.5
spawn_point.location.x -= 100
spawn_point.rotation.yaw = 0
truck  = world.spawn_actor(truck_bp, spawn_point)
time.sleep(0.02)  # wait for a tick to ensure carla has gotten the updated transform
location_truck = truck.get_transform().location
print(f"truck的坐标: x={location_truck.x}, y={location_truck.y}, z={location_truck.z}")

# # 读取控制数据
# control_data_path = "C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\save_car_control02.txt"
# control_data = np.loadtxt(control_data_path)  # 从文件读取控制数据

# control_data_path_truck = "C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\save_truck_control02.txt"
# control_data_truck = np.loadtxt(control_data_path_truck)  # 从文件读取控制数据

velocity1 = carla.Vector3D(18, 0, 0)
#33 is to show crash, 25 is normal
velocity2 = carla.Vector3D(15, 0, 0)

#truck.set_target_velocity(velocity1)
car.set_target_velocity(velocity2)
truck.set_target_velocity(velocity1)


    
with open('C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\save_car_state.txt', 'r') as file:
   data_car = file.readlines()
#_, _, _, previous_time = data[0].split()
#previous_time = float(previous_time)
with open('C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\save_truck_state.txt', 'r') as file_truck:
   data_truck = file_truck.readlines()

files_and_colors = [
    ('C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\save_car_state.txt', carla.Color(255, 0, 0)),  # 红色
    ('C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\save_truck_state.txt', carla.Color(0, 255, 0))  # 绿色
    ]

# read coordinates from file
for file_path, color in files_and_colors:
    coordinates = read_coordinates(file_path)
    draw_waypoints(world, coordinates, color)







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
    v = math.sqrt(vehicle_vel.x**2 + vehicle_vel.y**2)
    # v = vehicle_vel.length()  #converting it to km/hr

    return x, y, psi, v


def calculate_distance(actor1, actor2):
    location1 = actor1.get_location()
    location2 = actor2.get_location()
    distance = location1.distance(location2)
    return distance

# Route planner

def plan_route(world, start_point, dest_x, dest_y, dest_z, sampling_resolution=1, debug_draw=False, car_flag=False):
    
    # This following line needs to be changed according to YOUR installation path
    carla_path = r"C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla"
    sys.path.append(carla_path)
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    point_a = start_point.location
    point_b = carla.Location(x=dest_x, y=dest_y, z=dest_z)

    grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)

    route = grp.trace_route(point_a, point_b)
    

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

    # Plot the route
    # plt.plot(x, y, 'b-', label='Route')
    # # plt.scatter(x, y, c='r', marker='o', label='Waypoints')

    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.title('Route Visualization')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

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
        # print(distance_to_other_vehicle)

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

            vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
        else:
            main(vehicle, car_flag, route)

    vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0, brake=1))













class PIDController:
    def __init__(self, kp, ki, kd, min_output, max_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_output = min_output
        self.max_output = max_output
        self.integral = 0
        self.previous_error = 0

    def reset(self):
        self.integral = 0
        self.previous_error = 0

    def run_step(self, target_speed, current_speed, dt):
        # calculate error
        error = target_speed - current_speed
        # integral
        self.integral += error * dt
        # devivative
        derivative = (error - self.previous_error) / dt
        # update error
        self.previous_error = error
        # calculate output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        # limit output
        output = max(self.min_output, min(output, self.max_output))
        return output

def main(vehicle, car_flag, route):
    past_steer=0

    # Generate the reference curve
    # x_ref, y_ref = generate_curve(A=20, B=0.05, x_max=100)
    #x_ref, y_ref = visualize_route(route)
    #x_ref is the first column of data_car
    x_ref = np.array([float(line.split()[0]) for line in data_car])
    y_ref = np.array([float(line.split()[1]) for line in data_car])  
    last_index = 0
    # concate x_ref and y_ref as 2d array, shape of (N,2)
    ref_points = np.vstack([x_ref, y_ref]).T
    # Initialize car model

    car = Car_km(state=np.array([124 , 143.318146, 0, 0]))   # notice the forth element is time 
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
    with open('car_state_actual.txt', 'w') as file:
        while True:
            #here!    Simulate the car for one time step
            ##############################################################
            #TODO:here, u can modify it as carla_state
            carla_state=np.zeros(5)

            #TODO:Change Here!  the first elements are x,y,psi,v!!!!!!!!!!!!!!!!

            # carla_state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]=car.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]
            carla_state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]=get_state(vehicle)
            x_km, y_km = carla_state[C_k.X_km], carla_state[C_k.Y_km]
            file.write(f'{x_km}\t{y_km}\n')

            u_optimal, predicted_trajectories,car.last_index = \
                car.simulate(carla_state, ref_points, psi_ref, last_index)
            ##############################################################
            # estimated_throttle = u_optimal[0]
            # steer_input = u_optimal[1]
            # print('The steering input to carla is this shit:', steer_input)
            # vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
            
            estimated_throttle = u_optimal[0]
            steer_input = np.sin(u_optimal[1])
            if steer_input>past_steer+0.1:
                steer_input=past_steer+0.1
            elif steer_input<past_steer-0.1:
                steer_input=past_steer-0.1
            #steer input should be in -1 ,1
            steer_input = max(-1, min(1, steer_input))
            past_steer=steer_input
            
            
            # vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
            if estimated_throttle <0:
                brake_input =  -np.sin(estimated_throttle)  
                throttle_input = 0  
            else:
                throttle_input = np.sin(estimated_throttle) 
                brake_input = 0

            vehicle.apply_control(carla.VehicleControl(throttle=throttle_input, steer=steer_input, brake=brake_input)) 



def main_truck( ):
    vehicle=truck
    pid_controller = PIDController(1.5, 0.1, 0.01, -1.0, 1.0)
    target_speed = 18  # 目标速度，100 km/h

    x_ref_truck = np.array([float(line.split()[0]) for line in data_truck])
    y_ref_truck = np.array([float(line.split()[1]) for line in data_truck])
    past_steer_truck=0

    Truck_last_index = 0
    # concate x_ref and y_ref as 2d array, shape of (N,2)
    Truck_ref_points = np.vstack([x_ref_truck, y_ref_truck]).T
    # Initialize car model

    truck_km = Truck_km(state=np.array([124-100 , 143.318146+3.5, 0, 0]))   # notice the forth element is time 
    #To change the initial velocity, change the in the class Car_km
    #TODO:  
    # self.state[C_k.V_km] = 10, #CHANGE HERE!!!!!!!!
    Truck_psi_ref = Truck_km.calculate_direction(x_ref_truck, y_ref_truck)
    # Store the car's trajectory
    Truck_last_index = 0
    u_optimal=np.zeros(2)
    # define the car as a rectangle
    # Plot the car as a rectangle

    with open('truck_state_actual.txt', 'w') as file:
        while True:

            #here!    Simulate the car for one time step
            ##############################################################
            #TODO:here, u can modify it as carla_state
            Truck_carla_state=np.zeros(5)

            #TODO:Change Here!  the first elements are x,y,psi,v!!!!!!!!!!!!!!!!

            # carla_state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]=car.state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]
            Truck_carla_state[[C_k.X_km, C_k.Y_km, C_k.Psi, C_k.V_km]]=get_state(vehicle)
            x_km_1, y_km_1 = Truck_carla_state[C_k.X_km], Truck_carla_state[C_k.Y_km]
            file.write(f'{x_km_1}\t{y_km_1}\n')

            u_optimal_truck, predicted_trajectories,truck_km.last_index = \
                truck_km.simulate(Truck_carla_state, Truck_ref_points, Truck_psi_ref, Truck_last_index)
            ##############################################################
            # estimated_throttle = u_optimal[0]
            # steer_input = u_optimal[1]
            # print('The steering input to carla is this shit:', steer_input)
            # vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
            
            estimated_throttle_truck = u_optimal_truck[0]
            steer_input_truck = np.sin(u_optimal_truck[1])
        
            if steer_input_truck>past_steer_truck+0.1:
                steer_input_truck=past_steer_truck+0.1
            elif steer_input_truck<past_steer_truck-0.1:
                steer_input_truck=past_steer_truck-0.1
            #steer input should be in -1 ,1
            steer_input_truck = max(-1, min(1, steer_input_truck))
            past_steer_truck=steer_input_truck
            
            
            # vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
            if estimated_throttle_truck <0:
                brake_input_truck =  -np.sin(estimated_throttle_truck)  
                throttle_input_truck = 0  
            else:
                throttle_input_truck = np.sin(estimated_throttle_truck) 
                brake_input_truck = 0

            vehicle.apply_control(carla.VehicleControl(throttle=throttle_input_truck, steer=steer_input_truck, brake=brake_input_truck)) 
            print("this is velocity of truck:",Truck_carla_state[[C_k.V_km]])




# Test for running
start_point_car=carla.Transform(carla.Location(x=94+30, y=143.318146, z=0), carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
dest_x_car, dest_y_car, dest_z_car = 527.9996547234379, 146.81831927534984, 0
start_point_truck = carla.Transform(carla.Location(x=94+30-100, y=143.318146+3.5, z=0), carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))
dest_x_truck, dest_y_truck, dest_z_truck = 459.5327666981228,146.818, 0

car_flag = True
route_car = plan_route(world, start_point_car, dest_x_car, dest_y_car, dest_z_car, sampling_resolution=1, debug_draw=False, car_flag=True)
car_thread = threading.Thread(target=drive_vehicle, args=(car, route_car, truck, car_flag))

car_flag = True
route_truck = plan_route(world, start_point_truck, dest_x_truck, dest_y_truck, dest_z_truck, sampling_resolution=1, debug_draw=True, car_flag=False)
truck_thread = threading.Thread(target=main_truck)

# Start the threads
car_thread.start()
truck_thread.start()

# Wait for both threads to finish
car_thread.join()
truck_thread.join()

