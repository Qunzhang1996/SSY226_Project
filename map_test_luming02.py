##更换地图为Town04
#python config.py --map Town04

import time
import carla
import numpy as np

client = carla.Client('localhost', 2000)

world = client.get_world()

for actor in world.get_actors().filter('vehicle.*'):
    actor.destroy()
#清理场景中的其他车辆    

def draw_waypoints(waypoints, road_id=None, life_time=50.0):
 
  for waypoint in waypoints:
 
    if(waypoint.road_id == road_id):
          world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False,
                                   color=carla.Color(r=0, g=255, b=0), life_time=life_time,
                                   persistent_lines=True)
 
#以距离为1的间距创建waypoints                                  
waypoints = world.get_map().generate_waypoints(distance=1.0)
#life_time 为画出的辅助标志存活时间
draw_waypoints(waypoints, road_id=39, life_time=20)

#我期望的道路id是39.

#找到这条路上的所有waypoints
filtered_waypoints = []
for waypoint in waypoints:
    if(waypoint.road_id == 39):
      filtered_waypoints.append(waypoint)
      
len_waypoints = len(filtered_waypoints)
print(f"len_waypoints: {len_waypoints}")
spawn_point = filtered_waypoints[-2].transform
spawn_point.location.x -= 200
spawn_point.location.z -= 1
# bp为blueprint制造出来的小车
bp_lib = world.get_blueprint_library()
car_bp = bp_lib.find('vehicle.tesla.model3')
# vehicle = world.spawn_actor(car_bp, spawn_point)
# # 假设 vehicle 是你已经创建并放置到世界中的车辆实例
# transform = vehicle.get_transform()
# location = transform.location

# # 打印车辆的坐标
# print(f"车辆坐标: x={location.x}, y={location.y}, z={location.z}")

# 确保 spawn_point 是有效的位置
#spawn_point = carla.Transform(carla.Location(x=94.830231+200, y=150.318085-7, z=0.300000), carla.Rotation(pitch=0.000000, yaw=0, roll=0.000000))

car = None
try:
    car = world.spawn_actor(car_bp, spawn_point)
except Exception as e:
    print(f"放置车辆时出错: {e}")

if car is not None:
    # 稍等片刻以确保车辆坐标已更新
    time.sleep(0.02)  # 等待1秒

    transform = car.get_transform()
    location_car = transform.location

    # 打印车辆的坐标
    print(f"car的坐标: x={location_car.x}, y={location_car.y}, z={location_car.z}")
else:
    print("车辆未成功创建")


truck_bp = bp_lib.find('vehicle.carlamotors.firetruck')
#truck_bp = bp_lib.find('vehicle.tesla.model3')
# x_truck, y_truck, z_truck = location_car.x+100,location_car.y-3.5, 1.6630632877349854
# start_point_truck = min(spawn_point, key=lambda spawn_point: spawn_point.location.distance(carla.Location(x_truck, y_truck, z_truck)))
# truck = world.try_spawn_actor(truck_bp, start_point_truck)
spawn_point.location.y += 3.5
spawn_point.location.x -= 300
spawn_point.rotation.yaw = 0
truck  = world.spawn_actor(truck_bp, spawn_point)
time.sleep(0.02)  # 等待1秒
location_truck = truck.get_transform().location
print(f"truck的坐标: x={location_truck.x}, y={location_truck.y}, z={location_truck.z}")

# 读取控制数据
state_data_path = "C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\save_car_state.txt"
state_data = np.loadtxt(state_data_path)  # 从文件读取控制数据

control_data_path_truck = "C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\save_truck_control02.txt"
control_data_truck = np.loadtxt(control_data_path_truck)  # 从文件读取控制数据

velocity1 = carla.Vector3D(50, 0, 0)
velocity2 = carla.Vector3D(18, 0, 0)

#truck.set_target_velocity(velocity1)
car.set_target_velocity(velocity2)
truck.set_target_velocity(velocity1)

# # for i in range(len(control_data[25:])):

#     # 使用i来索引control_data数组
#     #car_throttle = control_data[i + 25][0]  # 假设给Car设置的油门值
#     # vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer_input))
#         # 处理减速情况
#     if control_data[i + 25][0] <0:
#         # 如果估计的油门小于0，意味着需要减速或刹车
#         car_brake_input =  -0*np.sin(control_data[i + 25][0]) # 使用非线性函数并保证刹车输入在[0, 1]之间
#         car_throttle = 0  # 油门设为0
#     else:
#         # 否则，继续使用估计的油门值
#         car_throttle = np.tanh(control_data[i + 25][0])  # 使用非线性函数并保证油门输入在[0, 1]之间
#         car_brake_input = 0
#         if car_throttle < 0.2:
#             car_throttle = 0.2
            
    
#     car_steer_input = np.tanh(control_data[i + 25][1])   # 假设给Car设置的方向盘值
    
    
#     if control_data_truck[i + 25][0] <0:
#     # 如果估计的油门小于0，意味着需要减速或刹车
#         truck_brake_input =  -0.0*np.sin(control_data_truck[i + 25][0]) # 使用非线性函数并保证刹车输入在[0, 1]之间
#         truck_throttle = 0  # 油门设为0
#     else:
#     # 否则，继续使用估计的油门值
#         truck_throttle = control_data_truck[i + 25][0]  # 使用非线性函数并保证油门输入在[0, 1]之间
#         truck_brake_input = 0
    
#     truck_steer_input = control_data_truck[i + 25][1]   # 假设给Car设置的方向盘值


#     # 控制Car
#     car_control = carla.VehicleControl(throttle=car_throttle, steer=car_steer_input, brake=car_brake_input)
#     car.apply_control(car_control)
    
#     # 控制Truck
#     truck_control = carla.VehicleControl(throttle=truck_throttle, steer=0, brake=truck_brake_input)
#     truck.apply_control(truck_control)
#     # if i==2495:
#     # #use hard brake to stop the car
#     #     car_control = carla.VehicleControl(throttle=0, steer=0, brake=1)
#     #     car.apply_control(car_control)
#     #     truck_control = carla.VehicleControl(throttle=0, steer=0, brake=1)
#     #     truck.apply_control(truck_control)
#     #     print("stop")
    
#     time.sleep(0.01)  # 等待0.1秒

# 读取数据文件
with open('C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\examples\SSY226_Project-saeed\save_car_state.txt', 'r') as file:
    data = file.readlines()
_, _, _, previous_time = data[0].split()
previous_time = float(previous_time)

# 根据数据文件控制车辆
for line in data:
    _, x, y, wait_time = line.split()  # 忽略每行的第1个值
    current_time = float(current_time)
    wait_time = current_time - previous_time  # 计算时间差
    previous_time = current_time
    
    car.set_location(carla.Location(x=float(x), y=float(y)))
    time.sleep(float(wait_time))